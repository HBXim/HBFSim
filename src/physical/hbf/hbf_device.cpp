#include "physical/hbf/hbf_device.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <list>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_set>

namespace hbfsim::physical::hbf {
namespace {

void require_positive_count(std::uint64_t value, const char* name) {
    if (value == 0) {
        throw std::runtime_error(std::string(name) + " must be positive");
    }
}

void require_positive_timing(double value, const char* name) {
    if (!std::isfinite(value) || value <= 0.0) {
        throw std::runtime_error(std::string(name) + " must be positive and finite");
    }
}

std::uint64_t checked_mul(std::uint64_t lhs, std::uint64_t rhs, const char* name) {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
        throw std::runtime_error(std::string(name) + " overflows uint64_t");
    }
    return lhs * rhs;
}

std::uint64_t checked_add(std::uint64_t lhs, std::uint64_t rhs, const char* name) {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
        throw std::runtime_error(std::string(name) + " overflows uint64_t");
    }
    return lhs + rhs;
}

std::uint64_t checked_ceil_div(
    std::uint64_t numerator,
    std::uint64_t denominator,
    const char* name) {
    if (denominator == 0) {
        throw std::runtime_error(std::string(name) + " has zero denominator");
    }
    return numerator == 0 ? 0 : 1 + (numerator - 1) / denominator;
}

// Stable controller-side address scrambler. AI tensor/KV layouts commonly
// advance by power-of-two page strides, so each stack-local mapping group
// rotates the global page lanes before ownership is encoded in the VPN.
// SplitMix64 is only a deterministic bit mixer; it models no randomness and
// has no mutable state.
std::uint64_t placement_mix64(std::uint64_t value) {
    value += 0x9E3779B97F4A7C15ull;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
    return value ^ (value >> 31);
}

using TemporalTouch = std::pair<double, std::uint64_t>;

std::optional<TemporalTouch> latest_touch_through(
    const std::set<TemporalTouch>& touches,
    double at_ns) {
    const auto after = touches.upper_bound(TemporalTouch{
        at_ns, std::numeric_limits<std::uint64_t>::max()});
    if (after == touches.begin()) {
        return std::nullopt;
    }
    return *std::prev(after);
}

void prune_touch_history(std::set<TemporalTouch>& touches, double through_ns) {
    const auto latest = latest_touch_through(touches, through_ns);
    if (!latest) {
        return;
    }
    touches.erase(touches.begin(), touches.find(*latest));
}

double causal_finish(double start_ns, double duration_ns) {
    const double arithmetic_finish = start_ns + duration_ns;
    const double finish_ns = arithmetic_finish > start_ns ?
        arithmetic_finish :
        std::nextafter(start_ns, std::numeric_limits<double>::infinity());
    if (!std::isfinite(finish_ns)) {
        throw std::runtime_error(
            "HBF resource duration exceeds the representable time horizon");
    }
    return finish_ns;
}

std::string logic_entity(std::uint32_t stack) {
    return "stack" + std::to_string(stack) + "/logic";
}

std::string channel_entity(const HbfAddress& addr) {
    return "stack" + std::to_string(addr.stack) + "/ch" + std::to_string(addr.channel);
}

std::string die_entity(const HbfAddress& addr) {
    return channel_entity(addr) + "/die" + std::to_string(addr.die);
}

std::string plane_entity(const HbfAddress& addr) {
    return die_entity(addr) + "/plane" + std::to_string(addr.plane);
}

std::string subarray_entity(const HbfAddress& addr, std::size_t subarray) {
    return plane_entity(addr) + "/subarray" + std::to_string(subarray);
}

std::string media_lane_entity(const HbfAddress& addr, std::size_t lane) {
    return plane_entity(addr) + "/lane" + std::to_string(lane);
}

std::string page_buffer_bank_entity(const HbfAddress& addr, std::size_t bank) {
    return plane_entity(addr) + "/page_buffer_bank" + std::to_string(bank);
}

std::uint64_t metadata_lpn(std::uint64_t mapping_vpn) {
    return (std::uint64_t{1} << 63) | mapping_vpn;
}

bool is_metadata_lpn(std::uint64_t lpn) {
    return (lpn & (std::uint64_t{1} << 63)) != 0;
}

std::uint64_t metadata_vpn(std::uint64_t lpn) {
    return lpn & ~(std::uint64_t{1} << 63);
}

void trace_wait(
    std::vector<TraceSpan>* spans,
    const std::string& entity,
    double from_ns,
    double to_ns,
    const std::string& name = "scheduler_queue") {
    add_trace_span(spans, name, "queue", entity, from_ns, to_ns);
}

std::uint64_t range_overlap_bytes(
    std::uint64_t lhs_begin,
    std::uint64_t lhs_end,
    std::uint64_t rhs_begin,
    std::uint64_t rhs_end) {
    const auto begin = std::max(lhs_begin, rhs_begin);
    const auto end = std::min(lhs_end, rhs_end);
    return end > begin ? end - begin : 0;
}

std::uint64_t insert_merged_range(
    std::vector<DirtyRange>& ranges,
    DirtyRange incoming) {
    if (incoming.end <= incoming.begin) {
        return 0;
    }

    // Existing ranges are a sorted, disjoint last-writer map. Preserve the
    // source of bytes outside the incoming write, replace provenance only in
    // the overwritten interval, and coalesce adjacent ranges only when their
    // sources agree. This keeps dirty coverage and source attribution in one
    // canonical representation.
    std::uint64_t overlap = 0;
    std::vector<DirtyRange> merged;
    merged.reserve(ranges.size() + 2);
    const auto append = [&merged](DirtyRange range) {
        if (range.end <= range.begin) {
            return;
        }
        if (!merged.empty()) {
            auto& previous = merged.back();
            if (range.begin < previous.end) {
                throw std::runtime_error(
                    "HBF write-buffer provenance ranges overlap");
            }
            if (range.begin == previous.end &&
                range.heatmap_source == previous.heatmap_source) {
                previous.end = range.end;
                return;
            }
        }
        merged.push_back(range);
    };
    bool inserted = false;
    for (const auto& current : ranges) {
        if (current.end <= incoming.begin) {
            append(current);
            continue;
        }
        if (current.begin >= incoming.end) {
            if (!inserted) {
                append(incoming);
                inserted = true;
            }
            append(current);
            continue;
        }

        overlap += range_overlap_bytes(
            current.begin,
            current.end,
            incoming.begin,
            incoming.end);
        if (current.begin < incoming.begin) {
            append(DirtyRange{
                .begin = current.begin,
                .end = incoming.begin,
                .heatmap_source = current.heatmap_source,
            });
        }
        if (!inserted) {
            append(incoming);
            inserted = true;
        }
        if (current.end > incoming.end) {
            append(DirtyRange{
                .begin = incoming.end,
                .end = current.end,
                .heatmap_source = current.heatmap_source,
            });
        }
    }
    if (!inserted) {
        append(incoming);
    }
    ranges = std::move(merged);
    return overlap;
}

HeatmapTrafficSource dominant_dirty_source(
    const std::vector<DirtyRange>& ranges) {
    std::array<std::uint64_t, kHeatmapTrafficSourceCount> bytes_by_source{};
    for (const auto& range : ranges) {
        const auto source = static_cast<std::size_t>(range.heatmap_source);
        if (source >= bytes_by_source.size() || range.end <= range.begin) {
            throw std::runtime_error(
                "HBF write-buffer contains invalid source provenance");
        }
        const auto bytes = range.end - range.begin;
        if (bytes > std::numeric_limits<std::uint64_t>::max() -
                bytes_by_source[source]) {
            throw std::overflow_error(
                "HBF write-buffer source-byte accounting overflows uint64_t");
        }
        bytes_by_source[source] += bytes;
    }

    std::size_t dominant = bytes_by_source.size();
    std::uint64_t dominant_bytes = 0;
    for (std::size_t source = 0; source < bytes_by_source.size(); ++source) {
        // Iterating in enum order gives equal-byte ties a stable result.
        if (bytes_by_source[source] > dominant_bytes) {
            dominant = source;
            dominant_bytes = bytes_by_source[source];
        }
    }
    if (dominant == bytes_by_source.size()) {
        throw std::runtime_error(
            "HBF cannot flush a write-buffer entry without dirty provenance");
    }
    return static_cast<HeatmapTrafficSource>(dominant);
}

} // namespace

ResidentMappingCapacity derive_resident_mapping_capacity(
    const HbfConfig& config) {
    auto pages_per_stack = checked_mul(
        config.channels_per_stack,
        config.dies_per_channel,
        "HBF pages per stack");
    pages_per_stack = checked_mul(
        pages_per_stack,
        config.planes_per_die,
        "HBF pages per stack");
    pages_per_stack = checked_mul(
        pages_per_stack,
        config.blocks_per_plane,
        "HBF pages per stack");
    pages_per_stack = checked_mul(
        pages_per_stack,
        config.pages_per_block,
        "HBF pages per stack");
    const auto mapping_pages_per_stack = checked_ceil_div(
        pages_per_stack,
        config.mapping_entries_per_page,
        "HBF resident mapping pages per stack");
    const auto bytes_per_stack = checked_mul(
        mapping_pages_per_stack,
        config.page_size_bytes,
        "HBF resident mapping bytes per stack");
    return ResidentMappingCapacity{
        .pages_per_stack = mapping_pages_per_stack,
        .bytes_per_stack = bytes_per_stack,
        .total_bytes = checked_mul(
            bytes_per_stack,
            config.stacks,
            "HBF resident mapping table bytes"),
    };
}

ControllerDramBudget derive_controller_dram_budget(
    const HbfConfig& config,
    std::uint64_t capacity_denominator) {
    if (capacity_denominator == 0) {
        throw std::runtime_error(
            "HBF controller-DRAM capacity denominator must be positive");
    }
    auto pages_per_stack = checked_mul(
        config.channels_per_stack,
        config.dies_per_channel,
        "HBF controller-DRAM raw pages per stack");
    pages_per_stack = checked_mul(
        pages_per_stack,
        config.planes_per_die,
        "HBF controller-DRAM raw pages per stack");
    pages_per_stack = checked_mul(
        pages_per_stack,
        config.blocks_per_plane,
        "HBF controller-DRAM raw pages per stack");
    pages_per_stack = checked_mul(
        pages_per_stack,
        config.pages_per_block,
        "HBF controller-DRAM raw pages per stack");
    const auto budget_pages_per_stack =
        pages_per_stack / capacity_denominator;
    if (budget_pages_per_stack == 0) {
        throw std::runtime_error(
            "HBF controller-DRAM ratio yields less than one page per stack");
    }
    const auto bytes_per_stack = checked_mul(
        budget_pages_per_stack,
        config.page_size_bytes,
        "HBF controller-DRAM bytes per stack");
    return ControllerDramBudget{
        .bytes_per_stack = bytes_per_stack,
        .total_bytes = checked_mul(
            bytes_per_stack,
            config.stacks,
            "HBF controller-DRAM total bytes"),
    };
}

ControllerDramBudget derive_write_buffer_dram_capacity(
    const HbfConfig& config) {
    if (!config.write_coalescing_enabled) {
        return ControllerDramBudget{};
    }
    const auto bytes_per_stack = checked_mul(
        config.write_buffer_pages,
        config.page_size_bytes,
        "HBF write-buffer controller-DRAM bytes per stack");
    return ControllerDramBudget{
        .bytes_per_stack = bytes_per_stack,
        .total_bytes = checked_mul(
            bytes_per_stack,
            config.stacks,
            "HBF write-buffer controller-DRAM total bytes"),
    };
}

const char* to_string(MappingMode mode) {
    switch (mode) {
    case MappingMode::FullResident:
        return "full-resident";
    case MappingMode::Cached:
        return "cached";
    case MappingMode::Direct:
        return "direct";
    }
    throw std::runtime_error("unknown HBF mapping mode");
}

MappingMode parse_mapping_mode(std::string_view value) {
    if (value == "full-resident") {
        return MappingMode::FullResident;
    }
    if (value == "cached") {
        return MappingMode::Cached;
    }
    if (value == "direct") {
        return MappingMode::Direct;
    }
    throw std::runtime_error(
        "HBF mapping mode must be full-resident, cached, or direct");
}

std::string HbfAddress::path() const {
    std::ostringstream out;
    out << "stack" << stack << "/ch" << channel << "/die" << die
        << "/plane" << plane << "/block" << block << "/page" << page
        << "/off" << offset;
    return out.str();
}

std::optional<double> HbfStats::waf() const {
    // Raw physical programs are host writes through the append-to-publish
    // interface: their payload is in the physical numerator, so it belongs
    // in the host-written denominator too. Counting it only above inflated
    // the WAF of any session that mixes published-extent appends with
    // page-mapped writes.
    const auto host_write_bytes = checked_add(
        logical_write_bytes,
        raw_physical_program_payload_bytes,
        "HBF WAF host-write denominator");
    if (host_write_bytes == 0) {
        return std::nullopt;
    }
    return static_cast<double>(physical_write_bytes) /
        static_cast<double>(host_write_bytes);
}

double HbfStats::active_span_ns() const {
    return std::isfinite(first_arrival_ns) && finish_ns > first_arrival_ns ?
        finish_ns - first_arrival_ns : 0.0;
}

double HbfStats::media_utilization() const {
    const auto span = active_span_ns();
    return span <= 0.0 || planes == 0 ? 0.0 :
        media_busy_ns / (span * static_cast<double>(planes));
}

double HbfStats::io_utilization() const {
    // The model has independent command and data HBIO ports per stack.
    const auto span = active_span_ns();
    return span <= 0.0 || stacks == 0 ? 0.0 :
        (hb_io_command_busy_ns + hb_io_data_busy_ns) /
            (span * 2.0 * static_cast<double>(stacks));
}

double HbfStats::media_parallelism() const {
    const auto span = active_span_ns();
    return span <= 0.0 ? 0.0 : media_busy_ns / span;
}

double HbfStats::read_lane_parallelism() const {
    const auto span = active_span_ns();
    return span <= 0.0 ? 0.0 : read_lane_busy_ns / span;
}

double HbfStats::subarray_read_parallelism() const {
    const auto span = active_span_ns();
    return span <= 0.0 ? 0.0 : subarray_read_busy_ns / span;
}

double HbfStats::page_buffer_bank_parallelism() const {
    const auto span = active_span_ns();
    return span <= 0.0 ? 0.0 : page_buffer_bank_busy_ns / span;
}

double HbfStats::channel_parallelism() const {
    const auto span = active_span_ns();
    return span <= 0.0 ? 0.0 :
        (channel_command_busy_ns + channel_data_busy_ns) / span;
}

double HbfStats::hbio_parallelism() const {
    const auto span = active_span_ns();
    return span <= 0.0 ? 0.0 :
        (hb_io_command_busy_ns + hb_io_data_busy_ns) / span;
}

double HbfStats::hbio_command_utilization() const {
    const auto span = active_span_ns();
    return span <= 0.0 || stacks == 0 ? 0.0 :
        hb_io_command_busy_ns / (span * static_cast<double>(stacks));
}

double HbfStats::hbio_data_utilization() const {
    const auto span = active_span_ns();
    return span <= 0.0 || stacks == 0 ? 0.0 :
        hb_io_data_busy_ns / (span * static_cast<double>(stacks));
}

double HbfStats::sequencer_parallelism() const {
    const auto span = active_span_ns();
    return span <= 0.0 ? 0.0 : sequencer_busy_ns / span;
}

double HbfStats::ecc_issue_parallelism() const {
    const auto span = active_span_ns();
    return span <= 0.0 ? 0.0 : ecc_issue_busy_ns / span;
}

double HbfStats::ecc_issue_utilization() const {
    const auto span = active_span_ns();
    return span <= 0.0 || dies == 0 ? 0.0 :
        ecc_issue_busy_ns / (span * static_cast<double>(dies));
}

std::uint64_t HbfDevice::ResourceTimeline::next_priority() {
    // Deterministic xorshift64: priorities depend only on this calendar's
    // insertion sequence, so runs remain reproducible while the treap stays
    // balanced in expectation even for monotonically increasing gap starts.
    priority_state_ ^= priority_state_ << 13;
    priority_state_ ^= priority_state_ >> 7;
    priority_state_ ^= priority_state_ << 17;
    return priority_state_;
}

std::size_t HbfDevice::ResourceTimeline::allocate_node(Gap gap) {
    const std::size_t node = [&] {
        if (free_gap_nodes_.empty()) {
            gap_nodes_.emplace_back();
            return gap_nodes_.size() - 1;
        }
        const auto recycled = free_gap_nodes_.back();
        free_gap_nodes_.pop_back();
        return recycled;
    }();
    gap_nodes_[node] = GapNode{
        .gap = gap,
        .priority = next_priority(),
        .max_duration_ns = gap.end_ns - gap.begin_ns,
    };
    return node;
}

void HbfDevice::ResourceTimeline::recycle_node(std::size_t node) {
    if (node == kNoNode || node >= gap_nodes_.size()) {
        throw std::runtime_error(
            "HBF resource calendar recycled a node outside its arena");
    }
    gap_nodes_[node] = GapNode{};
    free_gap_nodes_.push_back(node);
}

void HbfDevice::ResourceTimeline::recycle_subtree(std::size_t root) {
    if (root == kNoNode) {
        return;
    }
    const auto left = gap_nodes_[root].left;
    const auto right = gap_nodes_[root].right;
    recycle_subtree(left);
    recycle_subtree(right);
    recycle_node(root);
}

void HbfDevice::ResourceTimeline::refresh(std::size_t node_index) {
    auto& node = gap_nodes_[node_index];
    node.max_duration_ns = node.gap.end_ns - node.gap.begin_ns;
    if (node.left != kNoNode) {
        node.max_duration_ns = std::max(
            node.max_duration_ns, gap_nodes_[node.left].max_duration_ns);
    }
    if (node.right != kNoNode) {
        node.max_duration_ns = std::max(
            node.max_duration_ns, gap_nodes_[node.right].max_duration_ns);
    }
}

void HbfDevice::ResourceTimeline::split(
    std::size_t root,
    double key,
    std::size_t& lower,
    std::size_t& upper) {
    if (root == kNoNode) {
        lower = kNoNode;
        upper = kNoNode;
        return;
    }
    if (gap_nodes_[root].gap.begin_ns < key) {
        std::size_t new_right = kNoNode;
        split(gap_nodes_[root].right, key, new_right, upper);
        gap_nodes_[root].right = new_right;
        refresh(root);
        lower = root;
    } else {
        std::size_t new_left = kNoNode;
        split(gap_nodes_[root].left, key, lower, new_left);
        gap_nodes_[root].left = new_left;
        refresh(root);
        upper = root;
    }
}

std::size_t HbfDevice::ResourceTimeline::insert_node(
    std::size_t root,
    std::size_t node) {
    if (root == kNoNode) {
        return node;
    }
    if (gap_nodes_[node].gap.begin_ns == gap_nodes_[root].gap.begin_ns) {
        throw std::runtime_error(
            "HBF resource calendar inserted two idle gaps at one instant");
    }
    if (gap_nodes_[node].priority > gap_nodes_[root].priority) {
        split(
            root,
            gap_nodes_[node].gap.begin_ns,
            gap_nodes_[node].left,
            gap_nodes_[node].right);
        refresh(node);
        return node;
    }
    if (gap_nodes_[node].gap.begin_ns < gap_nodes_[root].gap.begin_ns) {
        gap_nodes_[root].left = insert_node(gap_nodes_[root].left, node);
    } else {
        gap_nodes_[root].right = insert_node(gap_nodes_[root].right, node);
    }
    refresh(root);
    return root;
}

std::size_t HbfDevice::ResourceTimeline::merge(
    std::size_t lower,
    std::size_t upper) {
    if (lower == kNoNode) {
        return upper;
    }
    if (upper == kNoNode) {
        return lower;
    }
    if (gap_nodes_[lower].priority > gap_nodes_[upper].priority) {
        gap_nodes_[lower].right = merge(gap_nodes_[lower].right, upper);
        refresh(lower);
        return lower;
    }
    gap_nodes_[upper].left = merge(lower, gap_nodes_[upper].left);
    refresh(upper);
    return upper;
}

std::size_t HbfDevice::ResourceTimeline::erase_node(
    std::size_t root,
    double key,
    std::size_t& erased) {
    if (root == kNoNode) {
        throw std::runtime_error(
            "HBF resource calendar erased an idle gap it does not hold");
    }
    if (key == gap_nodes_[root].gap.begin_ns) {
        erased = root;
        return merge(gap_nodes_[root].left, gap_nodes_[root].right);
    }
    if (key < gap_nodes_[root].gap.begin_ns) {
        gap_nodes_[root].left = erase_node(gap_nodes_[root].left, key, erased);
    } else {
        gap_nodes_[root].right = erase_node(gap_nodes_[root].right, key, erased);
    }
    refresh(root);
    return root;
}

const HbfDevice::ResourceTimeline::GapNode*
HbfDevice::ResourceTimeline::predecessor(std::size_t root, double key) const {
    const GapNode* result = nullptr;
    while (root != kNoNode) {
        const auto& node = gap_nodes_[root];
        if (node.gap.begin_ns <= key) {
            result = &node;
            root = node.right;
        } else {
            root = node.left;
        }
    }
    return result;
}

const HbfDevice::ResourceTimeline::GapNode*
HbfDevice::ResourceTimeline::successor(std::size_t root, double key) const {
    const GapNode* result = nullptr;
    while (root != kNoNode) {
        const auto& node = gap_nodes_[root];
        if (node.gap.begin_ns >= key) {
            result = &node;
            root = node.left;
        } else {
            root = node.right;
        }
    }
    return result;
}

const HbfDevice::ResourceTimeline::GapNode*
HbfDevice::ResourceTimeline::first_fitting_from(
    std::size_t root,
    double minimum_begin_ns,
    double duration_ns) const {
    if (root == kNoNode ||
        gap_nodes_[root].max_duration_ns < duration_ns) {
        return nullptr;
    }
    const auto& node = gap_nodes_[root];
    if (node.gap.begin_ns < minimum_begin_ns) {
        return first_fitting_from(node.right, minimum_begin_ns, duration_ns);
    }
    if (const auto* in_left = first_fitting_from(
            node.left, minimum_begin_ns, duration_ns)) {
        return in_left;
    }
    if (causal_finish(node.gap.begin_ns, duration_ns) <= node.gap.end_ns) {
        return &node;
    }
    return first_fitting_from(node.right, minimum_begin_ns, duration_ns);
}

std::optional<HbfDevice::ResourceTimeline::Gap>
HbfDevice::ResourceTimeline::first_fitting_gap(
    double earliest_ns,
    double duration_ns) const {
    // These checks stay in Release builds: a reservation placed behind the
    // pruned watermark is a causality violation that must fail closed
    // rather than silently corrupt the schedule.
    if (!std::isfinite(earliest_ns) || !std::isfinite(duration_ns) ||
        earliest_ns < pruned_through_ns_ || duration_ns <= 0.0) {
        throw std::runtime_error(
            "HBF resource calendar received an invalid reservation");
    }
    if (const auto* containing = predecessor(gap_root_, earliest_ns);
        containing != nullptr &&
        causal_finish(earliest_ns, duration_ns) <=
            containing->gap.end_ns) {
        return containing->gap;
    }
    if (const auto* later = first_fitting_from(
            gap_root_, earliest_ns, duration_ns)) {
        return later->gap;
    }
    return std::nullopt;
}

bool HbfDevice::ResourceTimeline::can_reserve_exact(
    double begin_ns,
    double duration_ns) const {
    if (!std::isfinite(begin_ns) || !std::isfinite(duration_ns) ||
        begin_ns < pruned_through_ns_ || duration_ns <= 0.0 ||
        !std::isfinite(causal_finish(begin_ns, duration_ns))) {
        throw std::runtime_error(
            "HBF resource calendar received an invalid exact-reservation query");
    }
    if (begin_ns >= ready_ns) {
        return true;
    }
    const auto* containing = predecessor(gap_root_, begin_ns);
    return containing != nullptr &&
        begin_ns >= containing->gap.begin_ns &&
        causal_finish(begin_ns, duration_ns) <= containing->gap.end_ns;
}

double HbfDevice::ResourceTimeline::preview_start(
    double earliest_ns,
    double duration_ns) const {
    if (!std::isfinite(earliest_ns) || !std::isfinite(duration_ns) ||
        earliest_ns < pruned_through_ns_ || duration_ns <= 0.0) {
        throw std::runtime_error(
            "HBF resource calendar received an invalid reservation preview");
    }
    if (const auto gap = first_fitting_gap(earliest_ns, duration_ns)) {
        return std::max(gap->begin_ns, earliest_ns);
    }
    const double start = std::max(earliest_ns, ready_ns);
    if (!std::isfinite(start) ||
        !std::isfinite(causal_finish(start, duration_ns))) {
        throw std::runtime_error("HBF resource calendar preview time overflowed");
    }
    return start;
}

void HbfDevice::ResourceTimeline::insert_gap(double begin_ns, double end_ns) {
    if (!std::isfinite(begin_ns) || !std::isfinite(end_ns) ||
        begin_ns < pruned_through_ns_ || end_ns <= begin_ns) {
        throw std::runtime_error("HBF resource calendar received an invalid idle gap");
    }
    if (const auto* previous = predecessor(gap_root_, begin_ns);
        previous != nullptr && previous->gap.end_ns > begin_ns) {
        throw std::runtime_error("HBF resource calendar inserted overlapping idle gaps");
    }
    if (const auto* next = successor(gap_root_, begin_ns);
        next != nullptr && next->gap.begin_ns < end_ns) {
        throw std::runtime_error("HBF resource calendar inserted overlapping idle gaps");
    }
    const auto node = allocate_node(Gap{
        .begin_ns = begin_ns,
        .end_ns = end_ns,
    });
    gap_root_ = insert_node(gap_root_, node);
}

void HbfDevice::ResourceTimeline::consume_gap(
    const Gap& gap,
    double begin_ns,
    double end_ns) {
    if (begin_ns < gap.begin_ns || end_ns > gap.end_ns || end_ns <= begin_ns) {
        throw std::runtime_error(
            "HBF resource calendar consumed outside an idle gap (gap=[" +
            std::to_string(gap.begin_ns) + "," +
            std::to_string(gap.end_ns) + "), request=[" +
            std::to_string(begin_ns) + "," +
            std::to_string(end_ns) + "))");
    }
    std::size_t erased = kNoNode;
    gap_root_ = erase_node(gap_root_, gap.begin_ns, erased);
    if (erased == kNoNode) {
        throw std::runtime_error(
            "HBF resource calendar consumed an idle gap it does not hold");
    }
    recycle_node(erased);
    if (gap.begin_ns < begin_ns) {
        insert_gap(gap.begin_ns, begin_ns);
    }
    if (end_ns < gap.end_ns) {
        insert_gap(end_ns, gap.end_ns);
    }
}

void HbfDevice::ResourceTimeline::prune_before(double causal_watermark_ns) {
    if (!std::isfinite(causal_watermark_ns) || causal_watermark_ns < 0.0) {
        throw std::runtime_error("HBF resource calendar received an invalid causal watermark");
    }
    if (causal_watermark_ns <= pruned_through_ns_) {
        return;
    }

    std::size_t expired = kNoNode;
    std::size_t future = kNoNode;
    split(gap_root_, causal_watermark_ns, expired, future);

    // All intervals in `expired` begin before the watermark. Because gaps
    // are disjoint and ordered, only its rightmost interval can cross the
    // watermark; preserve that usable suffix exactly and discard the rest.
    std::optional<double> crossing_end_ns;
    std::size_t rightmost = expired;
    while (rightmost != kNoNode && gap_nodes_[rightmost].right != kNoNode) {
        rightmost = gap_nodes_[rightmost].right;
    }
    if (rightmost != kNoNode &&
        gap_nodes_[rightmost].gap.end_ns > causal_watermark_ns) {
        crossing_end_ns = gap_nodes_[rightmost].gap.end_ns;
    }

    gap_root_ = future;
    recycle_subtree(expired);
    pruned_through_ns_ = causal_watermark_ns;
    // A frontier behind the causal watermark represents expired idle time,
    // not future capacity. Advancing it is schedule-equivalent for all legal
    // future reservations and prevents reintroducing an expired prefix.
    ready_ns = std::max(ready_ns, causal_watermark_ns);
    if (crossing_end_ns) {
        insert_gap(causal_watermark_ns, *crossing_end_ns);
    }
}

double HbfStats::plane_media_skew() const {
    return avg_active_plane_media_busy_ns <= 0.0 ? 0.0 :
        max_plane_media_busy_ns / avg_active_plane_media_busy_ns;
}

double HbfStats::plane_op_skew() const {
    return avg_active_plane_ops <= 0.0 ? 0.0 :
        static_cast<double>(max_plane_ops) / avg_active_plane_ops;
}

double HbfStats::media_lane_skew() const {
    return avg_active_media_lane_busy_ns <= 0.0 ? 0.0 :
        max_media_lane_busy_ns / avg_active_media_lane_busy_ns;
}

double HbfStats::media_lane_read_skew() const {
    return avg_active_media_lane_reads <= 0.0 ? 0.0 :
        static_cast<double>(max_media_lane_reads) / avg_active_media_lane_reads;
}

double HbfStats::subarray_busy_skew() const {
    return avg_active_subarray_busy_ns <= 0.0 ? 0.0 :
        max_subarray_busy_ns / avg_active_subarray_busy_ns;
}

double HbfStats::subarray_read_skew() const {
    return avg_active_subarray_reads <= 0.0 ? 0.0 :
        static_cast<double>(max_subarray_reads) / avg_active_subarray_reads;
}

double HbfStats::page_buffer_bank_skew() const {
    return avg_active_page_buffer_bank_busy_ns <= 0.0 ? 0.0 :
        max_page_buffer_bank_busy_ns / avg_active_page_buffer_bank_busy_ns;
}

double HbfStats::page_buffer_bank_read_skew() const {
    return avg_active_page_buffer_bank_reads <= 0.0 ? 0.0 :
        static_cast<double>(max_page_buffer_bank_reads) / avg_active_page_buffer_bank_reads;
}

double HbfStats::channel_busy_skew() const {
    return avg_active_channel_busy_ns <= 0.0 ? 0.0 :
        max_channel_busy_ns / avg_active_channel_busy_ns;
}

double HbfStats::die_transaction_skew() const {
    return avg_active_die_transactions <= 0.0 ? 0.0 :
        static_cast<double>(max_die_transactions) / avg_active_die_transactions;
}

HbfDevice::HbfDevice(HbfConfig config, AddressHeatmap* address_heatmap)
    : config_(config), address_heatmap_(address_heatmap) {
    require_positive_count(config_.stacks, "HBF stacks");
    require_positive_count(config_.channels_per_stack, "HBF channels_per_stack");
    require_positive_count(config_.dies_per_channel, "HBF dies_per_channel");
    require_positive_count(config_.planes_per_die, "HBF planes_per_die");
    require_positive_count(config_.blocks_per_plane, "HBF blocks_per_plane");
    require_positive_count(config_.pages_per_block, "HBF pages_per_block");
    require_positive_count(config_.page_size_bytes, "HBF page_size_bytes");
    require_positive_count(config_.media_lanes_per_plane, "HBF media_lanes_per_plane");
    require_positive_count(config_.page_buffer_banks_per_plane, "HBF page_buffer_banks_per_plane");
    subarrays_per_plane_ =
        config_.subarrays_per_plane == 0 ?
        config_.media_lanes_per_plane :
        config_.subarrays_per_plane;
    require_positive_count(subarrays_per_plane_, "HBF subarrays_per_plane");
    if (config_.pages_per_block > 1024) {
        // BlockState's valid-page bitmap holds 16 x 64 bits.
        throw std::runtime_error("HBF pages_per_block must be <= 1024");
    }
    if (config_.page_size_bytes > 4096) {
        throw std::runtime_error("HBF v0 is SLC-only with page_size_bytes <= 4096");
    }
    require_positive_timing(config_.t_read_page_ns, "HBF t_read_page_ns");
    require_positive_timing(config_.t_program_page_ns, "HBF t_program_page_ns");
    require_positive_timing(config_.t_erase_block_ns, "HBF t_erase_block_ns");
    require_positive_timing(config_.t_program_verify_ns, "HBF t_program_verify_ns");
    require_positive_timing(
        config_.ecc_decode_latency_ns, "HBF ecc_decode_latency_ns");
    require_positive_timing(
        config_.ecc_encode_latency_ns, "HBF ecc_encode_latency_ns");
    require_positive_timing(
        config_.ecc_decode_raw_bandwidth_GBps_per_die,
        "HBF ecc_decode_raw_bandwidth_GBps_per_die");
    require_positive_timing(
        config_.ecc_encode_raw_bandwidth_GBps_per_die,
        "HBF ecc_encode_raw_bandwidth_GBps_per_die");
    require_positive_timing(config_.channel_bandwidth_GBps, "HBF channel_bandwidth_GBps");
    require_positive_timing(config_.hb_io_bandwidth_GBps, "HBF hb_io_bandwidth_GBps");
    require_positive_timing(config_.tsv_bandwidth_GBps, "HBF tsv_bandwidth_GBps");
    require_positive_timing(config_.media_lane_bandwidth_GBps, "HBF media_lane_bandwidth_GBps");
    require_positive_timing(config_.logic_sram_bandwidth_GBps, "HBF logic_sram_bandwidth_GBps");
    require_positive_timing(config_.page_buffer_bandwidth_GBps, "HBF page_buffer_bandwidth_GBps");
    require_positive_timing(config_.logic_scheduler_issue_ns, "HBF logic_scheduler_issue_ns");
    require_positive_timing(config_.address_generation_ns, "HBF address_generation_ns");
    require_positive_timing(config_.ctrl_dram_latency_ns, "HBF ctrl_dram_latency_ns");
    require_positive_timing(config_.ctrl_dram_issue_ns, "HBF ctrl_dram_issue_ns");
    require_positive_timing(config_.mapping_update_ns, "HBF mapping_update_ns");
    require_positive_timing(config_.free_page_allocation_ns, "HBF free_page_allocation_ns");
    require_positive_timing(config_.flash_tsu_issue_ns, "HBF flash_tsu_issue_ns");
    require_positive_count(config_.command_address_bytes, "HBF command_address_bytes");
    require_positive_count(config_.mapping_entries_per_page, "HBF mapping_entries_per_page");
    require_positive_count(
        config_.mapping_directory_entry_bytes,
        "HBF mapping_directory_entry_bytes");
    require_positive_count(
        config_.page_read_queue_depth_per_stack,
        "HBF page_read_queue_depth_per_stack");
    if (config_.page_read_queue_depth_per_stack >
        std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error(
            "HBF page_read_queue_depth_per_stack exceeds size_t range");
    }
    if (config_.oob_bytes_per_page >= config_.page_size_bytes) {
        throw std::runtime_error("HBF oob_bytes_per_page must be smaller than page_size_bytes");
    }
    const auto codeword_bytes = config_.page_size_bytes + config_.oob_bytes_per_page;
    const double decode_initiation_ns = transfer_time_ns(
        codeword_bytes, config_.ecc_decode_raw_bandwidth_GBps_per_die);
    const double encode_initiation_ns = transfer_time_ns(
        codeword_bytes, config_.ecc_encode_raw_bandwidth_GBps_per_die);
    if (config_.ecc_decode_latency_ns < decode_initiation_ns) {
        throw std::runtime_error(
            "HBF ecc_decode_latency_ns must cover the full codeword initiation interval");
    }
    if (config_.ecc_encode_latency_ns < encode_initiation_ns) {
        throw std::runtime_error(
            "HBF ecc_encode_latency_ns must cover the full codeword initiation interval");
    }
    if (config_.write_coalescing_enabled) {
        require_positive_count(config_.write_buffer_pages, "HBF write_buffer_pages");
    }
    if (config_.gc_wear_leveling_weight < 0.0 || !std::isfinite(config_.gc_wear_leveling_weight)) {
        throw std::runtime_error("HBF gc_wear_leveling_weight must be non-negative and finite");
    }
    if (config_.gc_relocation_pages_per_host_write == 0) {
        throw std::runtime_error(
            "HBF gc_relocation_pages_per_host_write must be positive");
    }
    if (config_.program_suspend_enabled) {
        require_positive_timing(
            config_.program_suspend_latency_ns, "HBF program_suspend_latency_ns");
        require_positive_timing(
            config_.program_resume_ns, "HBF program_resume_ns");
        require_positive_count(
            config_.program_max_suspends, "HBF program_max_suspends");
    } else if (config_.program_suspend_latency_ns != 0.0 ||
               config_.program_resume_ns != 0.0 ||
               config_.program_max_suspends != 0) {
        throw std::runtime_error(
            "HBF program suspend parameters require program_suspend_enabled");
    }
    if (config_.static_wear_leveling_erase_gap != 0 &&
        config_.static_wear_leveling_interval_erases == 0) {
        throw std::runtime_error(
            "HBF static_wear_leveling_interval_erases must be positive when "
            "static wear leveling is enabled");
    }
    if (config_.thermal_start_at_ceiling && !config_.thermal_enabled) {
        throw std::runtime_error(
            "HBF thermal_start_at_ceiling requires thermal_enabled: a "
            "disabled thermal model has no governor to boot at its ceiling");
    }
    if (config_.thermal_enabled) {
        require_positive_timing(
            config_.thermal_resistance_c_per_w,
            "HBF thermal_resistance_c_per_w");
        require_positive_timing(
            config_.thermal_capacitance_j_per_c,
            "HBF thermal_capacitance_j_per_c");
        require_positive_timing(
            config_.thermal_read_energy_pj_per_bit,
            "HBF thermal_read_energy_pj_per_bit");
        require_positive_timing(
            config_.thermal_program_energy_pj_per_bit,
            "HBF thermal_program_energy_pj_per_bit");
        if (!std::isfinite(config_.thermal_ambient_c)) {
            throw std::runtime_error("HBF thermal_ambient_c must be finite");
        }
        if (!std::isfinite(config_.thermal_static_power_w) ||
            config_.thermal_static_power_w < 0.0) {
            throw std::runtime_error(
                "HBF thermal_static_power_w must be non-negative and finite");
        }
        if (!std::isfinite(config_.thermal_neighbor_heat_c) ||
            config_.thermal_neighbor_heat_c < 0.0) {
            throw std::runtime_error(
                "HBF thermal_neighbor_heat_c must be non-negative and finite");
        }
        if (!std::isfinite(config_.thermal_erase_energy_uj_per_block) ||
            config_.thermal_erase_energy_uj_per_block < 0.0) {
            throw std::runtime_error(
                "HBF thermal_erase_energy_uj_per_block must be non-negative "
                "and finite");
        }
        if (!std::isfinite(config_.thermal_throttle_c) ||
            !std::isfinite(config_.thermal_release_c) ||
            config_.thermal_release_c >= config_.thermal_throttle_c) {
            throw std::runtime_error(
                "HBF thermal_release_c must be finite and strictly below "
                "thermal_throttle_c");
        }
        const double boundary_c =
            config_.thermal_ambient_c + config_.thermal_neighbor_heat_c;
        thermal_idle_temperature_c_ = boundary_c +
            config_.thermal_static_power_w * config_.thermal_resistance_c_per_w;
        if (thermal_idle_temperature_c_ >= config_.thermal_release_c) {
            throw std::runtime_error(
                "HBF idle steady-state temperature (ambient + neighbor heat "
                "+ static power * resistance) must sit strictly below "
                "thermal_release_c: the governor could never release, and "
                "permanently throttled operation must be an explicit design "
                "decision, not a silent default (policy guard, not a "
                "physical operating limit)");
        }
        if (config_.thermal_throttle_power_w != 0.0 &&
            (!std::isfinite(config_.thermal_throttle_power_w) ||
             config_.thermal_throttle_power_w < 0.0)) {
            throw std::runtime_error(
                "HBF thermal_throttle_power_w must be non-negative and finite");
        }
        thermal_pacing_power_w_ = config_.thermal_throttle_power_w > 0.0 ?
            config_.thermal_throttle_power_w :
            (config_.thermal_throttle_c - boundary_c) /
                    config_.thermal_resistance_c_per_w -
                config_.thermal_static_power_w;
        if (!(thermal_pacing_power_w_ > 0.0)) {
            throw std::runtime_error(
                "HBF thermal pacing power resolved non-positive: raise "
                "thermal_throttle_c, lower the boundary (ambient + neighbor "
                "heat) or static power, or set thermal_throttle_power_w "
                "explicitly");
        }
        stats_.thermal_boundary_temperature_c = boundary_c;
        thermal_tau_ns_ = config_.thermal_resistance_c_per_w *
            config_.thermal_capacitance_j_per_c * 1e9;
        const auto payload_bits =
            static_cast<double>(config_.page_size_bytes) * 8.0;
        thermal_read_energy_j_ =
            config_.thermal_read_energy_pj_per_bit * payload_bits * 1e-12;
        thermal_program_energy_j_ =
            config_.thermal_program_energy_pj_per_bit * payload_bits * 1e-12;
        thermal_erase_energy_j_ =
            config_.thermal_erase_energy_uj_per_block * 1e-6;
        thermal_boot_temperature_c_ = config_.thermal_start_at_ceiling ?
            config_.thermal_throttle_c : thermal_idle_temperature_c_;
    }

    const auto total_channels = checked_mul(config_.stacks, config_.channels_per_stack,
        "HBF total_channels");
    const auto total_dies = checked_mul(total_channels, config_.dies_per_channel,
        "HBF total_dies");
    const auto total_planes = checked_mul(total_dies, config_.planes_per_die,
        "HBF total_planes");
    const auto total_blocks = checked_mul(total_planes, config_.blocks_per_plane,
        "HBF total_blocks");
    total_pages_ = checked_mul(total_blocks, config_.pages_per_block, "HBF total_pages");
    (void)checked_mul(total_pages_, config_.page_size_bytes, "HBF capacity bytes");
    if (config_.gc_reserved_free_blocks_per_plane >= config_.blocks_per_plane) {
        throw std::runtime_error(
            "HBF gc_reserved_free_blocks_per_plane must be smaller than blocks_per_plane");
    }
    const auto planes_per_stack = total_planes / config_.stacks;
    if (config_.auto_gc_enabled &&
        config_.mapping_mode != MappingMode::Direct) {
        // The stack's GC-only reserve must hold one worst-case victim (and
        // the wear-leveling cold frontier); relocation draws on every plane
        // of the stack, so the requirement is per stack, not per plane.
        const auto reserve_pages = checked_mul(
            checked_mul(
                static_cast<std::uint64_t>(config_.gc_reserved_free_blocks_per_plane),
                planes_per_stack,
                "HBF GC reserve blocks per stack"),
            config_.pages_per_block,
            "HBF GC reserve pages per stack");
        const auto required = gc_reserve_requirement_pages();
        if (reserve_pages < required) {
            throw std::runtime_error(
                "HBF automatic GC requires a GC-only reserve of at least " +
                std::to_string(required) + " pages per stack "
                "(gc_reserved_free_blocks_per_plane x planes per stack x "
                "pages_per_block, currently " + std::to_string(reserve_pages) +
                "): one worst-case victim's live pages" +
                std::string(config_.mapping_mode == MappingMode::Cached ?
                    " plus the translation writebacks they induce" : "") +
                std::string(config_.static_wear_leveling_erase_gap != 0 ?
                    " plus one wear-leveling cold block" : ""));
        }
    }
    const auto usable_blocks_per_plane =
        config_.blocks_per_plane - config_.gc_reserved_free_blocks_per_plane;
    const auto usable_blocks_per_stack = checked_mul(
        planes_per_stack, usable_blocks_per_plane, "HBF usable blocks per stack");
    const auto gc_watermark_capacity = checked_mul(
        usable_blocks_per_stack, config_.pages_per_block, "HBF usable pages per stack");
    if (config_.gc_low_watermark_pages > gc_watermark_capacity) {
        throw std::runtime_error(
            "HBF gc_low_watermark_pages exceeds usable per-stack page capacity");
    }
    if (config_.gc_hard_watermark_pages > gc_watermark_capacity) {
        throw std::runtime_error(
            "HBF gc_hard_watermark_pages exceeds usable per-stack page capacity");
    }
    free_pages_ = total_pages_;
    free_pages_per_stack_.assign(config_.stacks, total_pages_ / config_.stacks);
    next_data_allocation_plane_per_stack_.assign(config_.stacks, 0);
    next_mapping_allocation_plane_per_stack_.assign(config_.stacks, 0);
    next_gc_allocation_plane_per_stack_.assign(config_.stacks, 0);
    causal_state_ready_by_stack_.assign(config_.stacks, 0.0);
    state_observation_by_stack_.assign(config_.stacks, 0.0);
    gc_active_by_stack_.assign(config_.stacks, false);
    gc_victim_by_stack_.assign(config_.stacks, std::nullopt);
    wear_leveling_by_stack_.assign(config_.stacks, std::nullopt);
    cold_block_by_stack_.assign(config_.stacks, std::nullopt);
    dirty_mapping_pages_by_stack_.assign(config_.stacks, 0);
    pending_commit_keys_by_stack_.assign(config_.stacks, {});
    materialized_ready_by_block_.assign(
        static_cast<std::size_t>(total_blocks), 0.0);
    write_buffer_lru_by_stack_.resize(config_.stacks);
    write_buffer_by_stack_.resize(config_.stacks);
    write_buffer_slot_release_by_stack_.resize(config_.stacks);
    page_read_credit_release_by_stack_.resize(config_.stacks);
    mapping_cache_lru_by_stack_.resize(config_.stacks);
    mapping_cache_by_stack_.resize(config_.stacks);
    if (config_.thermal_enabled) {
        thermal_stacks_.resize(config_.stacks);
        for (auto& stack : thermal_stacks_) {
            stack.node.temperature_c = thermal_boot_temperature_c_;
            stack.node.peak_c = thermal_boot_temperature_c_;
            stack.node.throttled = config_.thermal_start_at_ceiling;
        }
        stats_.thermal_enabled = true;
        stats_.thermal_boot_temperature_c = thermal_boot_temperature_c_;
    }

    // The logical page table has the same entry density as its persistent
    // checkpoint pages. FullResident provisions the complete footprint;
    // Cached provisions only an explicit page-cache budget.
    if (config_.ctrl_dram_capacity_denominator != 0) {
        const auto derived = derive_controller_dram_budget(
            config_, config_.ctrl_dram_capacity_denominator);
        if (config_.ctrl_dram_bytes == 0) {
            config_.ctrl_dram_bytes = derived.total_bytes;
        } else if (config_.ctrl_dram_bytes != derived.total_bytes) {
            throw std::runtime_error(
                "HBF resolved ctrl_dram_bytes differs from its physical-"
                "capacity denominator");
        }
    }
    const auto resident_mapping =
        derive_resident_mapping_capacity(config_);
    mapping_table_pages_per_stack_ =
        resident_mapping.pages_per_stack;
    mapping_table_bytes_per_stack_ =
        resident_mapping.bytes_per_stack;
    const auto required_ctrl_dram_bytes =
        resident_mapping.total_bytes;
    const auto write_buffer_dram =
        derive_write_buffer_dram_capacity(config_);
    stats_.controller_dram_budget_bytes = config_.ctrl_dram_bytes;
    stats_.controller_dram_budget_bytes_per_stack =
        config_.ctrl_dram_bytes / config_.stacks;
    stats_.write_buffer_capacity_bytes = write_buffer_dram.total_bytes;
    stats_.write_buffer_capacity_bytes_per_stack =
        write_buffer_dram.bytes_per_stack;
    stats_.mapping_table_bytes = required_ctrl_dram_bytes;
    stats_.mapping_table_bytes_per_stack = mapping_table_bytes_per_stack_;
    stats_.mapping_table_pages_per_stack = mapping_table_pages_per_stack_;
    if (config_.mapping_mode == MappingMode::Direct) {
        // The exposed-address-space mode carries no L2P state of any kind:
        // no resident table, no directory, no cache. Controller DRAM holds
        // only the configured write buffer.
        if (config_.write_coalescing_enabled) {
            throw std::runtime_error(
                "HBF direct mapping requires write coalescing disabled: "
                "buffered flushes would reintroduce FTL allocation");
        }
        mapping_table_pages_per_stack_ = 0;
        mapping_table_bytes_per_stack_ = 0;
        stats_.mapping_table_bytes = 0;
        stats_.mapping_table_bytes_per_stack = 0;
        stats_.mapping_table_pages_per_stack = 0;
        if (config_.ctrl_dram_bytes == 0) {
            config_.ctrl_dram_bytes = write_buffer_dram.total_bytes;
            stats_.controller_dram_budget_bytes = config_.ctrl_dram_bytes;
            stats_.controller_dram_budget_bytes_per_stack =
                config_.ctrl_dram_bytes / config_.stacks;
        } else if (
            config_.ctrl_dram_bytes / config_.stacks <
            write_buffer_dram.bytes_per_stack) {
            throw std::runtime_error(
                "HBF direct mapping cannot hold the configured write buffer "
                "in every stack partition");
        }
    } else if (config_.mapping_mode == MappingMode::FullResident) {
        const auto required_per_stack = checked_add(
            mapping_table_bytes_per_stack_,
            write_buffer_dram.bytes_per_stack,
            "HBF resident mapping plus write-buffer DRAM per stack");
        if (config_.ctrl_dram_bytes != 0 &&
            config_.ctrl_dram_bytes / config_.stacks < required_per_stack) {
            throw std::runtime_error(
                "HBF ctrl_dram_bytes cannot hold the complete resident L2P "
                "table and configured write buffer in every stack partition");
        }
        if (config_.ctrl_dram_bytes == 0) {
            config_.ctrl_dram_bytes = checked_add(
                required_ctrl_dram_bytes,
                write_buffer_dram.total_bytes,
                "HBF derived resident controller-DRAM budget");
            stats_.controller_dram_budget_bytes = config_.ctrl_dram_bytes;
            stats_.controller_dram_budget_bytes_per_stack =
                config_.ctrl_dram_bytes / config_.stacks;
        }
        stats_.resident_mapping_table_bytes = required_ctrl_dram_bytes;
        stats_.resident_mapping_table_bytes_per_stack =
            mapping_table_bytes_per_stack_;
        stats_.resident_mapping_pages_per_stack =
            mapping_table_pages_per_stack_;
    } else {
        if (config_.ctrl_dram_bytes == 0) {
            throw std::runtime_error(
                "HBF cached mapping requires an explicit positive "
                "ctrl_dram_bytes budget");
        }
        stats_.mapping_directory_entry_bytes =
            config_.mapping_directory_entry_bytes;
        stats_.mapping_directory_bytes_per_stack = checked_mul(
            mapping_table_pages_per_stack_,
            config_.mapping_directory_entry_bytes,
            "HBF mapping-directory bytes per stack");
        stats_.mapping_directory_bytes = checked_mul(
            stats_.mapping_directory_bytes_per_stack,
            config_.stacks,
            "HBF mapping-directory total bytes");
        const auto ctrl_dram_bytes_per_stack =
            config_.ctrl_dram_bytes / config_.stacks;
        auto minimum_cached_bytes_per_stack = checked_add(
            stats_.mapping_directory_bytes_per_stack,
            write_buffer_dram.bytes_per_stack,
            "HBF cached mapping directory plus write-buffer DRAM per stack");
        minimum_cached_bytes_per_stack = checked_add(
            minimum_cached_bytes_per_stack,
            config_.page_size_bytes,
            "HBF minimum cached-mapping DRAM per stack");
        if (ctrl_dram_bytes_per_stack < minimum_cached_bytes_per_stack) {
            throw std::runtime_error(
                "HBF cached mapping requires its complete mapping-page "
                "directory, configured write buffer, and at least one cache "
                "page per stack");
        }
        mapping_cache_pages_per_stack_ =
            (ctrl_dram_bytes_per_stack -
             stats_.mapping_directory_bytes_per_stack -
             write_buffer_dram.bytes_per_stack) /
            config_.page_size_bytes;
        stats_.mapping_cache_pages_per_stack =
            mapping_cache_pages_per_stack_;
        stats_.mapping_cache_capacity_bytes_per_stack = checked_mul(
            mapping_cache_pages_per_stack_,
            config_.page_size_bytes,
            "HBF mapping-cache bytes per stack");
        stats_.mapping_cache_capacity_bytes = checked_mul(
            stats_.mapping_cache_capacity_bytes_per_stack,
            config_.stacks,
            "HBF mapping-cache total bytes");
    }
    stats_.mapping_dram_resources = config_.stacks;

    channels_.resize(static_cast<std::size_t>(total_channels));
    dies_.resize(static_cast<std::size_t>(total_dies));
    planes_.resize(static_cast<std::size_t>(total_planes));
    logic_dies_.resize(config_.stacks);
    blocks_.resize(static_cast<std::size_t>(total_blocks));
    for (auto& plane : planes_) {
        plane.subarrays.resize(subarrays_per_plane_);
        plane.available_sense_rounds_by_subarray.resize(subarrays_per_plane_);
        plane.media_lanes.resize(config_.media_lanes_per_plane);
        plane.page_buffer_banks.resize(config_.page_buffer_banks_per_plane);
    }
    for (std::size_t block_index = 0; block_index < blocks_.size(); ++block_index) {
        auto& block = blocks_[block_index];
        block.free_pages = config_.pages_per_block;
        block.next_page = 0;
        planes_.at(block_plane_index(block_index)).free_blocks.push_back(block_index);
    }
    stats_.free_pages = free_pages_;
    stats_.gc_reserve_pages = checked_mul(
        checked_mul(
            static_cast<std::uint64_t>(config_.gc_reserved_free_blocks_per_plane),
            total_planes,
            "HBF GC reserve blocks"),
        config_.pages_per_block,
        "HBF GC reserve pages");
    // Provisional logical capacity; static/raw reservations re-derive it and
    // the first logical use freezes it.
    logical_capacity_pages_ = logical_capacity_pages();
    stats_.logical_capacity_pages = logical_capacity_pages_;
    stats_.logical_capacity_bytes = checked_mul(
        logical_capacity_pages_, config_.page_size_bytes, "HBF logical capacity bytes");
    refresh_parallel_stats();
}

std::uint64_t HbfDevice::gc_reserve_requirement_pages() const {
    if (!config_.auto_gc_enabled || config_.mapping_mode == MappingMode::Direct) {
        return 0;
    }
    // A reclaimable victim keeps at most pages_per_block - 1 live pages;
    // under cached mapping each relocated page may evict one dirty
    // translation page; a wear-leveling migration pins one cold block.
    const auto live = static_cast<std::uint64_t>(config_.pages_per_block) - 1;
    auto required = config_.mapping_mode == MappingMode::Cached ?
        checked_add(live, config_.pages_per_block, "HBF GC reserve requirement") :
        live;
    if (config_.static_wear_leveling_erase_gap != 0) {
        required = checked_add(
            required, config_.pages_per_block, "HBF GC reserve requirement");
    }
    return required;
}

std::size_t HbfDevice::active_planes_in_stack(std::size_t stack) const {
    const auto pps = planes_per_stack();
    std::size_t active = 0;
    for (std::size_t plane_index = stack * pps;
         plane_index < (stack + 1) * pps;
         ++plane_index) {
        std::uint64_t managed = 0;
        for (std::uint32_t block = 0; block < config_.blocks_per_plane; ++block) {
            const auto role = blocks_.at(
                plane_index * config_.blocks_per_plane + block).role;
            if (role != BlockRole::StaticReadOnly &&
                role != BlockRole::RawPhysical) {
                managed++;
            }
        }
        if (managed != 0) {
            active++;
        }
    }
    return active;
}

std::uint64_t HbfDevice::gc_reserve_pages(std::size_t stack) const {
    return checked_mul(
        checked_mul(
            static_cast<std::uint64_t>(config_.gc_reserved_free_blocks_per_plane),
            static_cast<std::uint64_t>(active_planes_in_stack(stack)),
            "HBF GC reserve blocks per stack"),
        config_.pages_per_block,
        "HBF GC reserve pages per stack");
}

std::uint64_t HbfDevice::derive_logical_capacity_pages() const {
    // Largest logical page count L (over the whole device, page-striped
    // across stacks so every stack serves at most ceil(L / stacks) pages)
    // such that, per stack,
    //   L_s + M(L_s) <= (usable_blocks - open_blocks) * pages_per_block - 1
    // with M(L_s) = ceil(L_s / mapping_entries_per_page) persistent mapping
    // pages, usable_blocks the managed blocks (static/raw extents excluded)
    // beyond the stack's GC-only reserve, and open_blocks the blocks that
    // can still hold free pages when the foreground stalls.
    //
    // Why this guarantees a victim. The foreground (Data or Mapping role)
    // stalls only when its own write frontier is exhausted and the GC pool
    // minus one block is below the floor (which is at most the reserve), so
    // at most reserve/ppb whole free blocks remain. Once in-flight commits
    // have landed, the only non-free blocks that are not closed are the GC
    // write frontier (one per stack: relocation opens a new block only
    // when no GC block has free pages), the Data and Mapping frontiers (one
    // each per plane with managed blocks; the stalled role's are exhausted,
    // but a fresh compact image leaves both partially filled), and the
    // pinned wear-leveling cold block. Every other usable block is closed,
    // so
    //   closed_blocks >= usable_blocks - open_blocks.
    // Every valid page belongs to exactly one live LPN or persisted mapping
    // VPN, so valid_pages <= L_s + M(L_s) < closed_blocks * pages_per_block:
    // some closed block holds an invalid page. That victim keeps fewer than
    // pages_per_block live pages and the reserve, which the foreground
    // never breaches, holds them (see gc_reserve_requirement_pages).
    const auto pps = planes_per_stack();
    const auto ppb = static_cast<std::uint64_t>(config_.pages_per_block);
    const auto entries = static_cast<std::uint64_t>(config_.mapping_entries_per_page);
    const bool ftl = config_.mapping_mode != MappingMode::Direct;
    const auto reserve = ftl ? static_cast<std::uint64_t>(
        config_.gc_reserved_free_blocks_per_plane) : 0;
    std::uint64_t minimum_stack_pages = std::numeric_limits<std::uint64_t>::max();
    for (std::size_t stack = 0; stack < config_.stacks; ++stack) {
        std::uint64_t usable_blocks = 0;
        std::uint64_t active_planes = 0;
        for (std::size_t plane_index = stack * pps;
             plane_index < (stack + 1) * pps;
             ++plane_index) {
            std::uint64_t managed = 0;
            for (std::uint32_t block = 0; block < config_.blocks_per_plane; ++block) {
                const auto role = blocks_.at(
                    plane_index * config_.blocks_per_plane + block).role;
                if (role != BlockRole::StaticReadOnly &&
                    role != BlockRole::RawPhysical) {
                    managed++;
                }
            }
            if (managed != 0) {
                usable_blocks += managed;
                active_planes++;
            }
        }
        const auto reserve_blocks = checked_mul(
            reserve, active_planes, "HBF logical capacity reserve blocks");
        usable_blocks = usable_blocks > reserve_blocks ?
            usable_blocks - reserve_blocks : 0;
        // Open blocks: the stack's GC frontier, the Data and Mapping
        // frontiers of every plane with managed blocks, and the cold block.
        const std::uint64_t open_blocks = !ftl ? 0 : checked_add(
            checked_add(
                1,
                checked_mul(2, active_planes, "HBF logical capacity open blocks"),
                "HBF logical capacity open blocks"),
            config_.static_wear_leveling_erase_gap != 0 ? 1 : 0,
            "HBF logical capacity cold frontier");
        if (usable_blocks <= open_blocks) {
            return 0;
        }
        auto budget = checked_mul(
            usable_blocks - open_blocks, ppb, "HBF logical capacity budget");
        if (!ftl) {
            minimum_stack_pages = std::min(minimum_stack_pages, budget);
            continue;
        }
        budget -= 1;
        // Largest L with L + ceil(L / entries) <= budget.
        std::uint64_t pages = budget / (entries + 1) * entries;
        const auto fits = [&](std::uint64_t candidate) {
            return candidate + checked_ceil_div(
                candidate, entries, "HBF logical capacity mapping pages") <= budget;
        };
        while (pages != 0 && !fits(pages)) {
            --pages;
        }
        while (fits(pages + 1)) {
            ++pages;
        }
        minimum_stack_pages = std::min(minimum_stack_pages, pages);
    }
    return checked_mul(
        minimum_stack_pages,
        static_cast<std::uint64_t>(config_.stacks),
        "HBF logical capacity pages");
}

std::uint64_t HbfDevice::logical_capacity_pages() const {
    if (logical_capacity_frozen_) {
        return logical_capacity_pages_;
    }
    const auto derived = derive_logical_capacity_pages();
    if (config_.logical_capacity_bytes == 0) {
        return derived;
    }
    if (config_.logical_capacity_bytes % config_.page_size_bytes != 0) {
        throw std::runtime_error(
            "HBF logical_capacity_bytes must be a multiple of page_size_bytes");
    }
    const auto pages = config_.logical_capacity_bytes / config_.page_size_bytes;
    if (pages > derived) {
        throw std::runtime_error(
            "HBF logical_capacity_bytes exceeds the maximum safe logical capacity of " +
            std::to_string(derived) + " pages for this geometry and reservations");
    }
    return pages;
}

void HbfDevice::resolve_logical_capacity() {
    if (logical_capacity_frozen_) {
        return;
    }
    // Static/raw extents may have made planes inert; the stack's remaining
    // reserve must still hold one worst-case victim.
    const auto required = gc_reserve_requirement_pages();
    for (std::size_t stack = 0; required != 0 && stack < config_.stacks; ++stack) {
        const auto reserve_pages = gc_reserve_pages(stack);
        if (reserve_pages < required) {
            throw std::runtime_error(
                "HBF stack " + std::to_string(stack) + " keeps a GC-only reserve "
                "of " + std::to_string(reserve_pages) + " pages after its "
                "static/raw reservations, below the " + std::to_string(required) +
                " pages one worst-case victim needs (raise "
                "gc_reserved_free_blocks_per_plane or leave more planes with "
                "managed blocks)");
        }
    }
    const auto capacity = logical_capacity_pages();
    if (capacity == 0) {
        throw std::runtime_error(
            "HBF geometry cannot host any logical page: every stack needs "
            "managed blocks beyond gc_reserved_free_blocks_per_plane per "
            "plane for one GC frontier, one Data and one Mapping frontier "
            "per plane" +
            std::string(config_.static_wear_leveling_erase_gap != 0 ?
                ", one wear-leveling cold block" : "") +
            ", and at least one closed block");
    }
    logical_capacity_pages_ = capacity;
    logical_capacity_frozen_ = true;
    stats_.logical_capacity_pages = logical_capacity_pages_;
    stats_.logical_capacity_bytes = checked_mul(
        logical_capacity_pages_, config_.page_size_bytes, "HBF logical capacity bytes");
}

void HbfDevice::require_logical_capacity(std::uint64_t last_lpn) {
    resolve_logical_capacity();
    if (last_lpn >= logical_capacity_pages_) {
        throw std::runtime_error(
            "HBF logical page " + std::to_string(last_lpn) +
            " is beyond the logical capacity of " +
            std::to_string(logical_capacity_pages_) + " pages (" +
            std::to_string(stats_.logical_capacity_bytes) +
            " bytes; hbf-logical-capacity-bytes)");
    }
}

HbfAddress HbfDevice::decode(std::uint64_t addr) const {
    if (addr / config_.page_size_bytes >= total_pages_) {
        throw std::runtime_error("HBF physical byte address is out of range");
    }
    HbfAddress decoded;
    decoded.offset = addr % config_.page_size_bytes;
    std::uint64_t unit = addr / config_.page_size_bytes;
    decoded.page = static_cast<std::uint32_t>(unit % config_.pages_per_block);
    unit /= config_.pages_per_block;
    decoded.block = static_cast<std::uint32_t>(unit % config_.blocks_per_plane);
    unit /= config_.blocks_per_plane;
    decoded.plane = static_cast<std::uint32_t>(unit % config_.planes_per_die);
    unit /= config_.planes_per_die;
    decoded.die = static_cast<std::uint32_t>(unit % config_.dies_per_channel);
    unit /= config_.dies_per_channel;
    decoded.channel = static_cast<std::uint32_t>(unit % config_.channels_per_stack);
    unit /= config_.channels_per_stack;
    decoded.stack = static_cast<std::uint32_t>(unit % config_.stacks);
    return decoded;
}

std::uint64_t HbfDevice::encode(const HbfAddress& addr) const {
    return encode_ppn(addr) * config_.page_size_bytes + addr.offset;
}

std::uint64_t HbfDevice::schedule_commit(
    std::size_t stack,
    double at_ns,
    std::function<void()> action) {
    if (!std::isfinite(at_ns) || at_ns < 0.0) {
        throw std::runtime_error("HBF state commit time must be finite and non-negative");
    }
    if (stack >= config_.stacks) {
        throw std::runtime_error("HBF state commit stack is out of range");
    }
    const auto sequence = next_commit_sequence_++;
    insert_pending_commit(
        at_ns,
        sequence,
        PendingCommit{.stack = stack, .action = std::move(action)});
    return sequence;
}

void HbfDevice::insert_pending_commit(
    double at_ns,
    std::uint64_t sequence,
    PendingCommit commit) {
    const auto key = std::make_pair(at_ns, sequence);
    const auto stack = commit.stack;
    if (stack >= pending_commit_keys_by_stack_.size()) {
        throw std::runtime_error("HBF pending commit names an unknown stack");
    }
    if (!pending_commits_.emplace(key, std::move(commit)).second ||
        !pending_commit_keys_by_stack_[stack].insert(key).second ||
        !pending_commit_time_by_sequence_.emplace(sequence, at_ns).second) {
        throw std::runtime_error("HBF pending commit sequence collided");
    }
}

std::function<void()> HbfDevice::take_pending_commit(
    std::map<std::pair<double, std::uint64_t>, PendingCommit>::iterator
        commit) {
    auto action = std::move(commit->second.action);
    const auto key = commit->first;
    const auto stack = commit->second.stack;
    if (pending_commit_keys_by_stack_.at(stack).erase(key) != 1 ||
        pending_commit_time_by_sequence_.erase(key.second) != 1) {
        throw std::runtime_error("HBF pending commit index lost an entry");
    }
    pending_commits_.erase(commit);
    return action;
}

void HbfDevice::apply_selected_commits_through(
    const std::unordered_set<std::uint64_t>& sequences,
    double at_ns) {
    if (sequences.empty()) {
        return;
    }
    // Resolve the selected sequences to their keys first: applying one
    // commit may enqueue or apply others, and the keys must run in time
    // order regardless of the caller's set iteration order.
    std::vector<std::pair<double, std::uint64_t>> keys;
    keys.reserve(sequences.size());
    for (const auto sequence : sequences) {
        const auto time = pending_commit_time_by_sequence_.find(sequence);
        if (time != pending_commit_time_by_sequence_.end() &&
            time->second <= at_ns) {
            keys.emplace_back(time->second, sequence);
        }
    }
    std::sort(keys.begin(), keys.end());
    for (const auto& key : keys) {
        const auto commit = pending_commits_.find(key);
        if (commit == pending_commits_.end()) {
            continue;
        }
        auto action = take_pending_commit(commit);
        action();
    }
}

void HbfDevice::apply_commits_through(double at_ns) {
    while (!pending_commits_.empty() && pending_commits_.begin()->first.first <= at_ns) {
        auto action = take_pending_commit(pending_commits_.begin());
        action();
    }
}

void HbfDevice::apply_stack_commits_through(std::size_t stack, double at_ns) {
    auto& keys = pending_commit_keys_by_stack_.at(stack);
    while (!keys.empty() && keys.begin()->first <= at_ns) {
        const auto commit = pending_commits_.find(*keys.begin());
        if (commit == pending_commits_.end()) {
            throw std::runtime_error(
                "HBF pending commit index names a missing commit");
        }
        auto action = take_pending_commit(commit);
        action();
    }
}

void HbfDevice::apply_all_commits() {
    while (!pending_commits_.empty()) {
        const double at_ns = pending_commits_.begin()->first.first;
        apply_commits_through(at_ns);
    }
}

bool HbfDevice::advance_to_next_commit(double& at_ns, std::size_t stack) {
    const auto& keys = pending_commit_keys_by_stack_.at(stack);
    if (keys.empty()) {
        return false;
    }
    at_ns = std::max(at_ns, keys.begin()->first);
    apply_stack_commits_through(stack, at_ns);
    return true;
}

std::optional<std::uint64_t> HbfDevice::visible_lpn_at(
    std::uint64_t lpn,
    double at_ns) const {
    std::optional<std::uint64_t> visible;
    if (const auto base = lpn_to_ppn_.find(lpn); base != lpn_to_ppn_.end()) {
        const auto page = programmed_pages_.find(base->second);
        const auto block_index = static_cast<std::size_t>(
            base->second / config_.pages_per_block);
        const auto& block = blocks_.at(block_index);
        if (page != programmed_pages_.end() &&
            page->second.status == PageStatus::Valid &&
            page->second.owner == PageOwner::Logical &&
            page->second.block_epoch == block.epoch &&
            !block.erase_pending) {
            visible = base->second;
        }
    }
    if (!visible) {
        visible = compact_lpn_ppn(lpn);
    }
    const auto pending = pending_lpn_updates_.find(lpn);
    if (pending == pending_lpn_updates_.end()) {
        return visible;
    }
    const PendingMappingUpdate* newest = nullptr;
    for (const auto& update : pending->second) {
        const auto block_index = static_cast<std::size_t>(
            update.new_ppn / config_.pages_per_block);
        if (blocks_.at(block_index).epoch == update.block_epoch &&
            !blocks_.at(block_index).erase_pending &&
            update.commit_ns <= at_ns &&
            (newest == nullptr || std::tie(update.commit_ns, update.sequence) >
                std::tie(newest->commit_ns, newest->sequence))) {
            newest = &update;
        }
    }
    return newest == nullptr ? visible : std::optional<std::uint64_t>{newest->new_ppn};
}

void HbfDevice::wait_for_prior_lpn_commit(
    std::uint64_t lpn,
    double& at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    bool wait_for_relocations) {
    double dependency_ready_ns = 0.0;
    if (const auto observed = materialized_ready_by_lpn_.find(lpn);
        observed != materialized_ready_by_lpn_.end()) {
        dependency_ready_ns = observed->second;
    }
    std::unordered_set<std::uint64_t> commit_sequences;
    const auto pending = pending_lpn_updates_.find(lpn);
    if (pending != pending_lpn_updates_.end()) {
        for (const auto& update : pending->second) {
            if (update.relocation && !wait_for_relocations) {
                // The old copy stays intact and carries the same bytes until
                // the victim's erase is issued; whichever publication commits
                // later already discards or invalidates the loser.
                continue;
            }
            // Per-LPN ordering survives destruction of the target: a later
            // read waits for the prior write callback and its target erase.
            dependency_ready_ns = std::max({
                dependency_ready_ns,
                update.commit_ns,
                update.destructive_ready_ns,
            });
            commit_sequences.insert(update.program_commit_sequence);
            commit_sequences.insert(update.sequence);
            const auto block_index = static_cast<std::size_t>(
                update.new_ppn / config_.pages_per_block);
            add_pending_physical_erase_commits(
                block_index, commit_sequences);
        }
    }
    const double required_ns = std::max(at_ns, dependency_ready_ns);
    if (required_ns > at_ns) {
        if (spans != nullptr) {
            trace_wait(
                spans,
                logic_entity(static_cast<std::uint32_t>(stack_for_lpn(lpn))),
                at_ns,
                required_ns,
                "wait_prior_lpn_commit");
        }
        breakdown.scheduler_queue_wait_ns += required_ns - at_ns;
        at_ns = required_ns;
    }
    const auto stack = stack_for_lpn(lpn);
    apply_selected_commits_through(commit_sequences, at_ns);
    if (dependency_ready_ns != 0.0) {
        materialized_ready_by_lpn_[lpn] = std::max(
            materialized_ready_by_lpn_[lpn], dependency_ready_ns);
    }
    state_observation_by_stack_.at(stack) = std::max(
        state_observation_by_stack_.at(stack), at_ns);
}

void HbfDevice::wait_for_pending_vpn_erase(
    std::uint64_t mapping_vpn,
    double& at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    double dependency_ready_ns = 0.0;
    if (const auto observed = materialized_ready_by_vpn_.find(mapping_vpn);
        observed != materialized_ready_by_vpn_.end()) {
        dependency_ready_ns = observed->second;
    }
    std::unordered_set<std::uint64_t> commit_sequences;
    bool has_destructive_dependency = false;
    const auto pending = pending_vpn_updates_.find(mapping_vpn);
    if (pending != pending_vpn_updates_.end()) {
        for (const auto& update : pending->second) {
            has_destructive_dependency |= update.destructive_ready_ns != 0.0;
            dependency_ready_ns = std::max(
                dependency_ready_ns, update.destructive_ready_ns);
            commit_sequences.insert(update.program_commit_sequence);
            commit_sequences.insert(update.sequence);
            const auto block_index = static_cast<std::size_t>(
                update.new_ppn / config_.pages_per_block);
            add_pending_physical_erase_commits(
                block_index, commit_sequences);
        }
    }
    if (!has_destructive_dependency && dependency_ready_ns == 0.0) {
        return;
    }
    const double required_ns = std::max(at_ns, dependency_ready_ns);
    const auto stack = stack_for_vpn(mapping_vpn);
    if (required_ns > at_ns) {
        if (spans != nullptr) {
            trace_wait(
                spans,
                logic_entity(static_cast<std::uint32_t>(stack)),
                at_ns,
                required_ns,
                "wait_pending_mapping_block_erase");
        }
        breakdown.scheduler_queue_wait_ns += required_ns - at_ns;
        at_ns = required_ns;
    }
    apply_selected_commits_through(commit_sequences, at_ns);
    materialized_ready_by_vpn_[mapping_vpn] = std::max(
        materialized_ready_by_vpn_[mapping_vpn], dependency_ready_ns);
    state_observation_by_stack_.at(stack) = std::max(
        state_observation_by_stack_.at(stack), at_ns);
}

bool HbfDevice::current_ppn_has_pending_erase(std::uint64_t lpn) const {
    std::optional<std::uint64_t> current;
    if (const auto mapping = lpn_to_ppn_.find(lpn);
        mapping != lpn_to_ppn_.end()) {
        current = mapping->second;
    } else {
        current = compact_lpn_ppn(lpn);
    }
    if (!current) {
        return false;
    }
    const auto block_index = static_cast<std::size_t>(
        *current / config_.pages_per_block);
    return blocks_.at(block_index).erase_pending ||
        pending_physical_erases_.contains(block_index);
}

void HbfDevice::wait_for_pending_block_erase(
    std::uint64_t ppn,
    double& at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    // Only a raw (host-issued) erase destroys data the mapping still names;
    // a GC erase is issued after every live page has been published
    // elsewhere and is never waited for here.
    const auto block_index = static_cast<std::size_t>(
        ppn / config_.pages_per_block);
    if (!blocks_.at(block_index).erase_pending) {
        return;
    }
    double dependency_ready_ns = materialized_ready_by_block_.at(block_index);
    std::unordered_set<std::uint64_t> commit_sequences;
    const auto pending = pending_physical_erases_.find(block_index);
    if (pending != pending_physical_erases_.end()) {
        dependency_ready_ns = std::max(
            dependency_ready_ns, pending->second.finish_ns);
        add_pending_physical_erase_commits(
            block_index, commit_sequences);
    }
    if (dependency_ready_ns == 0.0) {
        return;
    }
    const auto stack = stack_of_block(block_index);
    if (dependency_ready_ns > at_ns) {
        if (spans != nullptr) {
            trace_wait(
                spans,
                logic_entity(static_cast<std::uint32_t>(stack)),
                at_ns,
                dependency_ready_ns,
                "wait_target_block_erase");
        }
        breakdown.scheduler_queue_wait_ns += dependency_ready_ns - at_ns;
        at_ns = dependency_ready_ns;
    }
    apply_selected_commits_through(commit_sequences, at_ns);
    materialized_ready_by_block_.at(block_index) = std::max(
        materialized_ready_by_block_.at(block_index), dependency_ready_ns);
    state_observation_by_stack_.at(stack) = std::max(
        state_observation_by_stack_.at(stack), at_ns);
}

void HbfDevice::add_pending_physical_erase_commits(
    std::size_t block_index,
    std::unordered_set<std::uint64_t>& sequences) const {
    const auto pending = pending_physical_erases_.find(block_index);
    if (pending == pending_physical_erases_.end()) {
        return;
    }
    sequences.insert(
        pending->second.dependency_sequences.begin(),
        pending->second.dependency_sequences.end());
    sequences.insert(pending->second.commit_sequence);
}

void HbfDevice::tag_pending_mapping_updates_for_erase(
    std::size_t block_index,
    std::uint64_t retired_block_epoch,
    double ready_ns) {
    const auto tag = [this, block_index, retired_block_epoch, ready_ns](auto& table) {
        for (auto& [_, updates] : table) {
            for (auto& update : updates) {
                if (update.new_ppn / config_.pages_per_block == block_index &&
                    update.block_epoch == retired_block_epoch) {
                    update.destructive_ready_ns = std::max(
                        update.destructive_ready_ns, ready_ns);
                }
            }
        }
    };
    tag(pending_lpn_updates_);
    tag(pending_vpn_updates_);
}

void HbfDevice::retire_mapping_update_tombstones(
    std::size_t block_index,
    double ready_ns) {
    const auto retire = [this, block_index, ready_ns](
                            auto& table, auto& materialized_ready) {
        for (auto entry = table.begin(); entry != table.end();) {
            const bool retires_entry = std::any_of(
                entry->second.begin(),
                entry->second.end(),
                [this, block_index, ready_ns](const PendingMappingUpdate& update) {
                    return update.new_ppn / config_.pages_per_block == block_index &&
                        update.destructive_ready_ns != 0.0 &&
                        update.destructive_ready_ns <= ready_ns &&
                        update.commit_ns <= ready_ns;
                });
            std::erase_if(entry->second, [this, block_index, ready_ns](
                              const PendingMappingUpdate& update) {
                return update.new_ppn / config_.pages_per_block == block_index &&
                    update.destructive_ready_ns != 0.0 &&
                    update.destructive_ready_ns <= ready_ns &&
                    update.commit_ns <= ready_ns;
            });
            if (retires_entry) {
                materialized_ready[entry->first] = std::max(
                    materialized_ready[entry->first], ready_ns);
            }
            if (entry->second.empty()) {
                entry = table.erase(entry);
            } else {
                ++entry;
            }
        }
    };
    retire(pending_lpn_updates_, materialized_ready_by_lpn_);
    retire(pending_vpn_updates_, materialized_ready_by_vpn_);
}

std::optional<std::uint64_t> HbfDevice::visible_mapping_vpn_at(
    std::uint64_t mapping_vpn,
    double at_ns) const {
    std::optional<std::uint64_t> visible;
    if (const auto base = mapping_vpn_to_ppn_.find(mapping_vpn);
        base != mapping_vpn_to_ppn_.end()) {
        const auto page = programmed_pages_.find(base->second);
        const auto block_index = static_cast<std::size_t>(
            base->second / config_.pages_per_block);
        const auto& block = blocks_.at(block_index);
        if (page != programmed_pages_.end() &&
            page->second.status == PageStatus::Valid &&
            page->second.owner == PageOwner::Mapping &&
            page->second.block_epoch == block.epoch &&
            !block.erase_pending) {
            visible = base->second;
        }
    }
    if (!visible) {
        visible = compact_mapping_ppn(mapping_vpn);
    }
    const auto pending = pending_vpn_updates_.find(mapping_vpn);
    if (pending == pending_vpn_updates_.end()) {
        return visible;
    }
    const PendingMappingUpdate* newest = nullptr;
    for (const auto& update : pending->second) {
        const auto block_index = static_cast<std::size_t>(
            update.new_ppn / config_.pages_per_block);
        if (blocks_.at(block_index).epoch == update.block_epoch &&
            !blocks_.at(block_index).erase_pending &&
            update.commit_ns <= at_ns &&
            (newest == nullptr || std::tie(update.commit_ns, update.sequence) >
                std::tie(newest->commit_ns, newest->sequence))) {
            newest = &update;
        }
    }
    return newest == nullptr ? visible : std::optional<std::uint64_t>{newest->new_ppn};
}

void HbfDevice::schedule_lpn_mapping_commit(
    std::uint64_t lpn,
    std::uint64_t new_ppn,
    double commit_ns,
    std::uint64_t program_commit_sequence,
    std::optional<std::uint64_t> expected_old_ppn) {
    const auto sequence = next_commit_sequence_++;
    const auto block_index = static_cast<std::size_t>(
        new_ppn / config_.pages_per_block);
    const auto block_epoch = blocks_.at(block_index).epoch;
    blocks_.at(block_index).pending_mapping_publications++;
    pending_lpn_updates_[lpn].push_back(PendingMappingUpdate{
        .sequence = sequence,
        .program_commit_sequence = program_commit_sequence,
        .commit_ns = commit_ns,
        .new_ppn = new_ppn,
        .block_epoch = block_epoch,
        .relocation = expected_old_ppn.has_value(),
    });
    insert_pending_commit(
        commit_ns,
        sequence,
        PendingCommit{
        .stack = stack_for_lpn(lpn),
        .action = [this, lpn, new_ppn, block_index, block_epoch, sequence,
                      commit_ns, expected_old_ppn]() {
            const auto page = programmed_pages_.find(new_ppn);
            const bool target_survived =
                blocks_.at(block_index).epoch == block_epoch &&
                page != programmed_pages_.end() &&
                page->second.block_epoch == block_epoch &&
                page->second.status == PageStatus::Valid &&
                page->second.owner == PageOwner::Logical;
            bool source_is_current = true;
            if (expected_old_ppn) {
                if (const auto current = lpn_to_ppn_.find(lpn);
                    current != lpn_to_ppn_.end()) {
                    source_is_current = current->second == *expected_old_ppn;
                } else {
                    source_is_current =
                        compact_lpn_ppn(lpn) == expected_old_ppn;
                }
            }
            if (target_survived && source_is_current) {
                const auto current = lpn_to_ppn_.find(lpn);
                if (current != lpn_to_ppn_.end() && current->second != new_ppn) {
                    invalidate_ppn(current->second);
                } else if (current == lpn_to_ppn_.end() &&
                           compact_lpn_ppn(lpn)) {
                    retire_compact_page(lpn, PageOwner::Logical);
                }
                lpn_to_ppn_[lpn] = new_ppn;
            } else if (target_survived && expected_old_ppn) {
                // GC copied a snapshot. A foreground write that committed
                // first is authoritative; retire the stale copy instead of
                // publishing old payload over the newer version.
                invalidate_ppn(new_ppn);
            }
            if (blocks_.at(block_index).epoch == block_epoch) {
                auto& pending_publications =
                    blocks_.at(block_index).pending_mapping_publications;
                if (pending_publications == 0) {
                    throw std::runtime_error(
                        "HBF LPN publication lost its block ownership pin");
                }
                pending_publications--;
            }
            auto pending = pending_lpn_updates_.find(lpn);
            if (pending != pending_lpn_updates_.end()) {
                std::erase_if(
                    pending->second,
                    [sequence](const PendingMappingUpdate& update) {
                        return update.sequence == sequence &&
                            update.destructive_ready_ns <= update.commit_ns;
                    });
                if (pending->second.empty()) {
                    pending_lpn_updates_.erase(pending);
                }
            }
            materialized_ready_by_lpn_[lpn] = std::max(
                materialized_ready_by_lpn_[lpn], commit_ns);
        }});
}

void HbfDevice::schedule_vpn_mapping_commit(
    std::uint64_t mapping_vpn,
    std::uint64_t new_ppn,
    double commit_ns,
    std::uint64_t program_commit_sequence,
    std::optional<std::uint64_t> expected_old_ppn) {
    const auto sequence = next_commit_sequence_++;
    const auto block_index = static_cast<std::size_t>(
        new_ppn / config_.pages_per_block);
    const auto block_epoch = blocks_.at(block_index).epoch;
    blocks_.at(block_index).pending_mapping_publications++;
    pending_vpn_updates_[mapping_vpn].push_back(PendingMappingUpdate{
        .sequence = sequence,
        .program_commit_sequence = program_commit_sequence,
        .commit_ns = commit_ns,
        .new_ppn = new_ppn,
        .block_epoch = block_epoch,
        .relocation = expected_old_ppn.has_value(),
    });
    insert_pending_commit(
        commit_ns,
        sequence,
        PendingCommit{
        .stack = stack_for_vpn(mapping_vpn),
        .action = [this, mapping_vpn, new_ppn, block_index, block_epoch,
                      sequence, commit_ns, expected_old_ppn]() {
            const auto page = programmed_pages_.find(new_ppn);
            const bool target_survived =
                blocks_.at(block_index).epoch == block_epoch &&
                page != programmed_pages_.end() &&
                page->second.block_epoch == block_epoch &&
                page->second.status == PageStatus::Valid &&
                page->second.owner == PageOwner::Mapping;
            bool source_is_current = true;
            if (expected_old_ppn) {
                if (const auto current = mapping_vpn_to_ppn_.find(mapping_vpn);
                    current != mapping_vpn_to_ppn_.end()) {
                    source_is_current = current->second == *expected_old_ppn;
                } else {
                    source_is_current =
                        compact_mapping_ppn(mapping_vpn) == expected_old_ppn;
                }
            }
            if (target_survived && source_is_current) {
                const auto current = mapping_vpn_to_ppn_.find(mapping_vpn);
                if (current != mapping_vpn_to_ppn_.end() && current->second != new_ppn) {
                    invalidate_ppn(current->second);
                } else if (
                    current == mapping_vpn_to_ppn_.end() &&
                    compact_mapping_ppn(mapping_vpn)) {
                    retire_compact_page(
                        metadata_lpn(mapping_vpn),
                        PageOwner::Mapping);
                }
                mapping_vpn_to_ppn_[mapping_vpn] = new_ppn;
            } else if (target_survived && expected_old_ppn) {
                invalidate_ppn(new_ppn);
            }
            if (blocks_.at(block_index).epoch == block_epoch) {
                auto& pending_publications =
                    blocks_.at(block_index).pending_mapping_publications;
                if (pending_publications == 0) {
                    throw std::runtime_error(
                        "HBF VPN publication lost its block ownership pin");
                }
                pending_publications--;
            }
            auto pending = pending_vpn_updates_.find(mapping_vpn);
            if (pending != pending_vpn_updates_.end()) {
                std::erase_if(
                    pending->second,
                    [sequence](const PendingMappingUpdate& update) {
                        return update.sequence == sequence &&
                            update.destructive_ready_ns <= update.commit_ns;
                    });
                if (pending->second.empty()) {
                    pending_vpn_updates_.erase(pending);
                }
            }
            materialized_ready_by_vpn_[mapping_vpn] = std::max(
                materialized_ready_by_vpn_[mapping_vpn], commit_ns);
        }});
}

std::uint64_t HbfDevice::schedule_media_program_commit(
    std::uint64_t ppn,
    std::uint64_t lpn,
    PageOwner owner,
    double commit_ns) {
    const auto block_index = static_cast<std::size_t>(
        ppn / config_.pages_per_block);
    const auto block_epoch = blocks_.at(block_index).epoch;
    const auto page = programmed_pages_.find(ppn);
    if (page == programmed_pages_.end() ||
        page->second.status != PageStatus::Erased ||
        page->second.block_epoch != block_epoch) {
        throw std::runtime_error(
            "HBF media program commit was scheduled without an erased reservation");
    }
    return schedule_commit(
        stack_of_block(block_index),
        commit_ns,
        [this, ppn, lpn, owner, block_index, block_epoch, commit_ns]() {
            // An erase with a newer epoch is authoritative. A stale media
            // completion becomes a no-op instead of recreating page state.
            if (blocks_.at(block_index).epoch != block_epoch) {
                return;
            }
            const auto page = programmed_pages_.find(ppn);
            if (page == programmed_pages_.end() ||
                page->second.block_epoch != block_epoch) {
                return;
            }
            mark_programmed(ppn, lpn, owner);
            materialized_ready_by_ppn_[ppn] = std::max(
                materialized_ready_by_ppn_[ppn], commit_ns);
        });
}

void HbfDevice::schedule_physical_program_commit(
    std::uint64_t ppn,
    double commit_ns) {
    const auto page = programmed_pages_.find(ppn);
    if (page == programmed_pages_.end() || page->second.status != PageStatus::Erased) {
        throw std::runtime_error(
            "HBF physical program commit was scheduled without an erased reservation");
    }
    const auto [pending, inserted] = pending_physical_programs_.emplace(
        ppn,
        PendingPhysicalProgram{.commit_ns = commit_ns});
    if (!inserted) {
        throw std::runtime_error("HBF physical page already has an in-flight program");
    }
    try {
        const auto block_index = static_cast<std::size_t>(
            ppn / config_.pages_per_block);
        const auto block_epoch = blocks_.at(block_index).epoch;
        pending->second.commit_sequence = schedule_commit(
            stack_of_block(block_index), commit_ns,
            [this, ppn, block_index, block_epoch]() {
            const auto pending = pending_physical_programs_.find(ppn);
            if (pending == pending_physical_programs_.end()) {
                throw std::runtime_error(
                    "HBF physical program lost its in-flight reservation");
            }
            const auto page_at_commit = programmed_pages_.find(ppn);
            if (blocks_.at(block_index).epoch != block_epoch) {
                // A later physical erase already retired this incarnation;
                // the bytes were programmed and then destroyed, so only the
                // stale state publication is suppressed.
                pending_physical_programs_.erase(pending);
                return;
            }
            if (page_at_commit == programmed_pages_.end() ||
                page_at_commit->second.block_epoch != block_epoch ||
                page_at_commit->second.status != PageStatus::Erased) {
                throw std::runtime_error(
                    "HBF physical program target stopped being erased before commit");
            }
            mark_programmed(ppn, ppn, PageOwner::RawPhysical);
            materialized_ready_by_ppn_[ppn] = std::max(
                materialized_ready_by_ppn_[ppn], pending->second.commit_ns);
            pending_physical_programs_.erase(pending);
        });
    } catch (...) {
        pending_physical_programs_.erase(ppn);
        throw;
    }
}

void HbfDevice::reserve_physical_program_range(
    std::uint64_t first_ppn,
    std::uint64_t pages) {
    std::unordered_map<std::size_t, std::uint32_t> expected_next;
    std::unordered_map<std::size_t, std::uint32_t> remaining_free;

    // First pass is read-only: reject the whole multi-page request before any
    // block/page allocation state changes.
    for (std::uint64_t i = 0; i < pages; ++i) {
        const auto ppn = first_ppn + i;
        const auto block_index = static_cast<std::size_t>(
            ppn / config_.pages_per_block);
        const auto page_index = static_cast<std::uint32_t>(
            ppn % config_.pages_per_block);
        const auto& block = blocks_.at(block_index);
        if (block.role != BlockRole::Free && block.role != BlockRole::RawPhysical) {
            throw std::runtime_error(
                block.role == BlockRole::StaticReadOnly ?
                "HBF physical write targets static read-only data" :
                "HBF physical write targets a controller-owned block");
        }
        auto [next_it, next_inserted] = expected_next.emplace(
            block_index, block.next_page);
        auto [free_it, free_inserted] = remaining_free.emplace(
            block_index, block.free_pages);
        (void)next_inserted;
        (void)free_inserted;
        if (page_index != next_it->second) {
            throw std::runtime_error(
                "HBF physical program violates sequential page order");
        }
        if (free_it->second == 0) {
            throw std::runtime_error("HBF physical program selected a full block");
        }
        if (programmed_pages_.contains(ppn) ||
            pending_physical_programs_.contains(ppn)) {
            throw std::runtime_error(
                "HBF physical write targets an allocated or in-flight page");
        }
        next_it->second++;
        free_it->second--;
    }

    for (const auto& entry : expected_next) {
        const auto block_index = entry.first;
        const auto& block = blocks_.at(block_index);
        if (block.role != BlockRole::Free) {
            continue;
        }
        const auto& free_blocks = planes_.at(block_plane_index(block_index)).free_blocks;
        if (std::find(free_blocks.begin(), free_blocks.end(), block_index) ==
            free_blocks.end()) {
            throw std::runtime_error(
                "HBF physical program found a free block missing from its plane pool");
        }
    }

    // Second pass reserves capacity exactly as the FTL allocator does. Page
    // validity remains Erased until each media-completion commit executes.
    for (std::uint64_t i = 0; i < pages; ++i) {
        const auto ppn = first_ppn + i;
        const auto block_index = static_cast<std::size_t>(
            ppn / config_.pages_per_block);
        auto& block = blocks_.at(block_index);
        if (block.role == BlockRole::Free) {
            auto& free_blocks = planes_.at(block_plane_index(block_index)).free_blocks;
            const auto found = std::find(free_blocks.begin(), free_blocks.end(), block_index);
            free_blocks.erase(found);
            block.role = BlockRole::RawPhysical;
        }
        block.next_page++;
        block.free_pages--;
        block.pending_program_pages++;
        free_pages_--;
        free_pages_per_stack_.at(stack_of_block(block_index))--;
        const auto [page, inserted] = programmed_pages_.emplace(ppn, PageState{});
        if (!inserted) {
            throw std::runtime_error(
                "HBF physical program reservation unexpectedly reused page state");
        }
        page->second.status = PageStatus::Erased;
        page->second.owner = PageOwner::RawPhysical;
        page->second.lpn = 0;
        page->second.block_epoch = block.epoch;
    }
}

void HbfDevice::prepopulate_logical_pages(const std::vector<std::uint64_t>& lpns) {
    if (config_.mapping_mode == MappingMode::Direct) {
        throw std::runtime_error(
            "HBF direct mapping requires a compact logical page range");
    }
    if (compact_logical_image_) {
        throw std::runtime_error(
            "HBF cannot add materialized mappings after a compact logical "
            "image");
    }
    std::unordered_set<std::uint64_t> seen;
    for (const auto lpn : lpns) {
        if (!seen.insert(lpn).second || lpn_to_ppn_.find(lpn) != lpn_to_ppn_.end()) {
            continue;
        }
        require_logical_capacity(lpn);
        if (free_pages_ == 0) {
            throw std::runtime_error("HBF cannot prepopulate logical pages: physical capacity exhausted");
        }
        double at_ns = 0.0;
        Breakdown ignored;
        const auto ppn = allocate_free_page(
            at_ns,
            ignored,
            nullptr,
            BlockRole::Data,
            stack_for_lpn(lpn));
        lpn_to_ppn_[lpn] = ppn;
        mark_programmed(ppn, lpn);
        stats_.initial_logical_data_pages = checked_add(
            stats_.initial_logical_data_pages,
            1,
            "HBF materialized initial logical data pages");
        const auto mapping_vpn = mapping_vpn_for_lpn(lpn);
        if (mapping_vpn_to_ppn_.find(mapping_vpn) == mapping_vpn_to_ppn_.end()) {
            const auto mapping_ppn = allocate_free_page(
                at_ns,
                ignored,
                nullptr,
                BlockRole::Mapping,
                stack_for_vpn(mapping_vpn),
                mapping_plane_for_vpn(mapping_vpn));
            mapping_vpn_to_ppn_[mapping_vpn] = mapping_ppn;
            mark_programmed(
                mapping_ppn,
                metadata_lpn(mapping_vpn),
                PageOwner::Mapping);
            stats_.initial_mapping_pages = checked_add(
                stats_.initial_mapping_pages,
                1,
                "HBF materialized initial mapping pages");
        }
    }
    stats_.free_pages = free_pages_;
    refresh_parallel_stats();
}

void HbfDevice::prepopulate_read_only_logical_page_range(
    std::uint64_t first_lpn,
    std::uint64_t page_count) {
    prepopulate_compact_logical_page_range(
        first_lpn,
        page_count,
        false);
}

void HbfDevice::prepopulate_mutable_logical_page_range(
    std::uint64_t first_lpn,
    std::uint64_t page_count) {
    prepopulate_compact_logical_page_range(
        first_lpn,
        page_count,
        true);
}

void HbfDevice::prepopulate_compact_logical_page_range(
    std::uint64_t first_lpn,
    std::uint64_t page_count,
    bool mutable_image) {
    if (page_count == 0) {
        return;
    }
    const auto last_lpn = checked_add(
        first_lpn,
        page_count - 1,
        "HBF prepopulation LPN range");
    require_logical_capacity(last_lpn);
    if (compact_logical_image_ || !lpn_to_ppn_.empty() ||
        !mapping_vpn_to_ppn_.empty()) {
        throw std::runtime_error(
            "HBF compact logical image requires an empty FTL");
    }
    if (!mutable_image &&
        (!programmed_pages_.empty() || free_pages_ != total_pages_)) {
        throw std::runtime_error(
            "HBF compact read-only image must be installed into a fresh "
            "device");
    }
    if (mutable_image) {
        const bool static_only = std::all_of(
            programmed_pages_.begin(),
            programmed_pages_.end(),
            [](const auto& entry) {
                return entry.second.status == PageStatus::StaticReadOnly &&
                    entry.second.owner == PageOwner::StaticReadOnly;
            });
        const bool no_mutable_blocks = std::none_of(
            blocks_.begin(),
            blocks_.end(),
            [](const BlockState& block) {
                return block.role == BlockRole::Data ||
                    block.role == BlockRole::Mapping ||
                    block.role == BlockRole::GC ||
                    block.role == BlockRole::RawPhysical ||
                    block.pending_program_pages != 0 ||
                    block.pending_mapping_publications != 0 ||
                    block.erase_pending;
            });
        if (!static_only || !no_mutable_blocks ||
            !pending_commits_.empty() ||
            !pending_lpn_updates_.empty() ||
            !pending_vpn_updates_.empty() ||
            !dirty_mapping_vpns_.empty()) {
            throw std::runtime_error(
                "HBF compact mutable image must be installed after static "
                "fencing and before mutable FTL work");
        }
    }
    const auto entries_per_mapping_page =
        static_cast<std::uint64_t>(config_.mapping_entries_per_page);
    const auto pps = static_cast<std::uint64_t>(planes_per_stack());
    if (pps == 0) {
        throw std::runtime_error(
            "HBF compact logical image requires at least one plane per stack");
    }
    const auto cursor_was_advanced = [](const auto& cursors) {
        return std::any_of(
            cursors.begin(),
            cursors.end(),
            [](std::size_t cursor) { return cursor != 0; });
    };
    if (cursor_was_advanced(next_data_allocation_plane_per_stack_) ||
        cursor_was_advanced(next_mapping_allocation_plane_per_stack_) ||
        cursor_was_advanced(next_gc_allocation_plane_per_stack_)) {
        throw std::runtime_error(
            "HBF compact logical image requires fresh allocation cursors");
    }

    CompactLogicalImage image;
    image.mutable_image = mutable_image;
    image.first_lpn = first_lpn;
    image.page_count = page_count;
    const auto stacks = static_cast<std::uint64_t>(config_.stacks);
    const auto first_group =
        (first_lpn / stacks) / entries_per_mapping_page;
    const auto last_group =
        (last_lpn / stacks) / entries_per_mapping_page;
    image.first_vpn = checked_mul(
        first_group,
        stacks,
        "HBF compact first mapping VPN");
    const auto group_count = checked_add(
        last_group - first_group,
        1,
        "HBF compact mapping group count");
    image.vpn_slot_count = checked_mul(
        group_count,
        stacks,
        "HBF compact VPN slot count");
    if (image.vpn_slot_count > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error(
            "HBF compact logical image VPN directory exceeds size_t range");
    }
    image.data_blocks_by_plane.resize(planes_.size());
    image.mapping_ppns.resize(
        static_cast<std::size_t>(image.vpn_slot_count));
    image.vpn_ranges.resize(
        static_cast<std::size_t>(image.vpn_slot_count));
    image.vpn_offsets_by_stack.resize(config_.stacks);
    std::vector<std::uint64_t> stack_page_counts(config_.stacks, 0);

    for (std::uint64_t vpn_offset = 0;
         vpn_offset < image.vpn_slot_count;
         ++vpn_offset) {
        const auto mapping_vpn = checked_add(
            image.first_vpn, vpn_offset, "HBF compact mapping VPN");
        const auto group = mapping_vpn / stacks;
        const auto stack = stack_for_vpn(mapping_vpn);
        const auto rotation = placement_mix64(group) % stacks;
        const auto lane = static_cast<std::uint64_t>(stack) >= rotation ?
            static_cast<std::uint64_t>(stack) - rotation :
            stacks - (rotation - static_cast<std::uint64_t>(stack));
        const auto group_first_stripe = checked_mul(
            group,
            entries_per_mapping_page,
            "HBF compact mapping-group first stripe");
        const auto group_last_stripe = checked_add(
            group_first_stripe,
            entries_per_mapping_page - 1,
            "HBF compact mapping-group last stripe");
        const auto first_candidate_stripe = [&] {
            if (first_lpn <= lane) {
                return std::uint64_t{0};
            }
            const auto delta = first_lpn - lane;
            return delta / stacks + (delta % stacks == 0 ? 0 : 1);
        }();
        if (last_lpn < lane) {
            continue;
        }
        const auto last_candidate_stripe = (last_lpn - lane) / stacks;
        const auto range_first_stripe = std::max(
            group_first_stripe,
            first_candidate_stripe);
        const auto range_last_stripe = std::min(
            group_last_stripe,
            last_candidate_stripe);
        if (range_first_stripe > range_last_stripe) {
            continue;
        }
        const auto first_entry =
            range_first_stripe - group_first_stripe;
        const auto range_pages = checked_add(
            range_last_stripe - range_first_stripe,
            1,
            "HBF compact mapping-page range");
        if (range_pages > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error(
                "HBF compact mapping-page range exceeds supported page count");
        }
        const auto stack_plane_base = stack * planes_per_stack();
        const auto first_data_cursor =
            static_cast<std::uint64_t>(next_data_allocation_plane_per_stack_.at(stack));
        if (first_data_cursor != stack_page_counts.at(stack) % pps) {
            throw std::runtime_error(
                "HBF compact data-allocation cursor lost round-robin order");
        }
        image.vpn_ranges.at(static_cast<std::size_t>(vpn_offset)) =
            CompactLogicalImage::VpnRange{
            .first_entry = first_entry,
            .page_count = range_pages,
            .stack_page_offset = stack_page_counts.at(stack),
        };
        image.vpn_offsets_by_stack.at(stack).push_back(vpn_offset);

        // In the ordinary prepopulation loop the first data page is reserved
        // before the mapping page for this VPN. Preserve that ordering because
        // Data and Mapping may draw their first blocks from the same plane's
        // free-block deque.
        (void)allocate_compact_pages_on_plane(
            stack_plane_base + static_cast<std::size_t>(first_data_cursor),
            BlockRole::Data,
            1,
            &image.data_blocks_by_plane.at(
                stack_plane_base + static_cast<std::size_t>(first_data_cursor)),
            &image.live_data_pages_by_block);
        if (config_.mapping_mode != MappingMode::Direct) {
            image.mapping_ppns.at(static_cast<std::size_t>(vpn_offset)) =
                allocate_compact_pages_on_plane(
                mapping_plane_for_vpn(mapping_vpn),
                BlockRole::Mapping,
                1,
                nullptr,
                &image.live_mapping_pages_by_block);
            image.mapping_page_count = checked_add(
                image.mapping_page_count,
                1,
                "HBF compact mapping-page count");
        }

        for (std::size_t local_plane = 0;
             local_plane < planes_per_stack();
             ++local_plane) {
            const auto plane_distance =
                (local_plane + planes_per_stack() -
                    static_cast<std::size_t>(first_data_cursor)) %
                planes_per_stack();
            std::uint64_t plane_pages = range_pages > plane_distance ?
                1 + (range_pages - 1 - plane_distance) / pps : 0;
            if (local_plane == first_data_cursor) {
                --plane_pages;
            }
            if (plane_pages == 0) {
                continue;
            }
            const auto plane = stack_plane_base + local_plane;
            (void)allocate_compact_pages_on_plane(
                plane,
                BlockRole::Data,
                static_cast<std::uint32_t>(plane_pages),
                &image.data_blocks_by_plane.at(plane),
                &image.live_data_pages_by_block);
        }
        stack_page_counts.at(stack) = checked_add(
            stack_page_counts.at(stack),
            range_pages,
            "HBF compact per-stack data pages");
        next_data_allocation_plane_per_stack_.at(stack) =
            static_cast<std::size_t>(stack_page_counts.at(stack) % pps);
    }
    for (std::size_t plane = 0;
         plane < image.data_blocks_by_plane.size();
         ++plane) {
        const auto& assigned = image.data_blocks_by_plane.at(plane);
        for (std::size_t ordinal = 0; ordinal < assigned.size(); ++ordinal) {
            const bool inserted = image.data_block_locations.emplace(
                assigned[ordinal],
                CompactLogicalImage::DataBlockLocation{
                    .plane = plane,
                    .block_ordinal = ordinal,
                }).second;
            if (!inserted) {
                throw std::runtime_error(
                    "HBF compact data block appears in multiple directories");
            }
        }
    }
    for (std::size_t offset = 0;
         offset < image.mapping_ppns.size();
         ++offset) {
        if (!image.mapping_ppns[offset]) {
            continue;
        }
        const auto mapping_vpn = checked_add(
            image.first_vpn,
            offset,
            "HBF compact mapping inverse VPN");
        const bool inserted = image.mapping_vpn_by_ppn.emplace(
            *image.mapping_ppns[offset],
            mapping_vpn).second;
        if (!inserted) {
            throw std::runtime_error(
                "HBF compact mapping PPN is not unique");
        }
    }
    stats_.initial_logical_data_pages = checked_add(
        stats_.initial_logical_data_pages,
        image.page_count,
        "HBF initial logical data pages");
    stats_.initial_mapping_pages = checked_add(
        stats_.initial_mapping_pages,
        image.mapping_page_count,
        "HBF initial mapping pages");
    stats_.compact_initial_logical_data_pages = checked_add(
        stats_.compact_initial_logical_data_pages,
        image.page_count,
        "HBF compact initial logical data pages");
    stats_.compact_initial_mapping_pages = checked_add(
        stats_.compact_initial_mapping_pages,
        image.mapping_page_count,
        "HBF compact initial mapping pages");
    compact_logical_image_ = std::move(image);
    stats_.free_pages = free_pages_;
    refresh_parallel_stats();
}

void HbfDevice::reserve_static_physical_blocks(
    const std::vector<std::size_t>& block_indices) {
    if (logical_capacity_frozen_ || compact_logical_image_) {
        throw std::runtime_error(
            "HBF cannot reserve static pages after logical use");
    }
    std::unordered_set<std::size_t> touched_blocks(
        block_indices.begin(),
        block_indices.end());
    std::vector<std::uint8_t> reserve_mask(blocks_.size(), 0);
    std::vector<std::size_t> reserve_count_by_plane(planes_.size(), 0);
    std::size_t blocks_to_reserve = 0;
    for (const auto block_index : touched_blocks) {
        if (block_index >= blocks_.size()) {
            throw std::runtime_error(
                "HBF static-data block index is out of range");
        }
    }

    // Validate the entire request before changing any pool, role, bitmap, or
    // capacity counter. A mixed legal/illegal list must fail closed rather
    // than leave whichever unordered-set element happened to be visited first
    // as a partially reserved static block.
    for (const auto block_index : touched_blocks) {
        const auto& block = blocks_.at(block_index);
        if (block.role == BlockRole::StaticReadOnly) {
            continue;
        }
        if (block.role != BlockRole::Free || block.free_pages != config_.pages_per_block ||
            block.valid_pages != 0 || block.invalid_pages != 0 || block.next_page != 0 ||
            block.pending_program_pages != 0 ||
            block.pending_mapping_publications != 0) {
            throw std::runtime_error(
                "HBF static-data reservation intersects live FTL state at block " +
                std::to_string(block_index));
        }
        reserve_mask.at(block_index) = 1;
        ++blocks_to_reserve;
        ++reserve_count_by_plane.at(block_plane_index(block_index));
    }
    // Dense extents can cover hundreds of thousands of blocks. Validate pool
    // membership in one linear scan instead of performing one O(blocks-per-
    // plane) find for every reserved block.
    std::size_t pool_matches = 0;
    for (const auto& plane : planes_) {
        for (const auto block_index : plane.free_blocks) {
            pool_matches += reserve_mask.at(block_index) != 0 ? 1 : 0;
        }
    }
    if (pool_matches != blocks_to_reserve) {
        throw std::runtime_error(
            "HBF free block is missing from its plane pool");
    }

    // Reserve complete NAND blocks. Page-granular fencing would let the FTL
    // program around immutable wordlines and later collide with them, which is
    // not a valid sequential-program model.
    for (std::size_t plane_index = 0;
         plane_index < planes_.size();
         ++plane_index) {
        if (reserve_count_by_plane.at(plane_index) == 0) {
            continue;
        }
        auto& free_blocks = planes_.at(plane_index).free_blocks;
        const auto first_removed = std::remove_if(
            free_blocks.begin(),
            free_blocks.end(),
            [&](std::size_t block_index) {
                return reserve_mask.at(block_index) != 0;
            });
        const auto removed = static_cast<std::size_t>(
            std::distance(first_removed, free_blocks.end()));
        if (removed != reserve_count_by_plane.at(plane_index)) {
            throw std::runtime_error(
                "HBF static-data free-pool removal count mismatch");
        }
        free_blocks.erase(first_removed, free_blocks.end());
    }
    for (const auto block_index : touched_blocks) {
        auto& block = blocks_.at(block_index);
        if (block.role == BlockRole::StaticReadOnly) {
            continue;
        }
        block.role = BlockRole::StaticReadOnly;
        block.free_pages = 0;
        block.next_page = config_.pages_per_block;
        free_pages_ -= config_.pages_per_block;
        free_pages_per_stack_.at(stack_of_block(block_index)) -= config_.pages_per_block;
        stats_.static_reserved_pages += config_.pages_per_block;
    }
    stats_.free_pages = free_pages_;
}

void HbfDevice::reserve_static_physical_pages(
    const std::vector<std::uint64_t>& ppns) {
    std::unordered_set<std::uint64_t> unique_pages;
    std::vector<std::size_t> touched_blocks;
    touched_blocks.reserve(ppns.size());
    for (const auto ppn : ppns) {
        if (ppn >= total_pages_) {
            throw std::runtime_error("HBF static-data PPN is out of range");
        }
        if (unique_pages.insert(ppn).second) {
            touched_blocks.push_back(
                static_cast<std::size_t>(ppn / config_.pages_per_block));
        }
    }
    // Page conflicts must be checked before the block helper mutates any
    // ownership state.
    for (const auto ppn : unique_pages) {
        const auto page = programmed_pages_.find(ppn);
        if (page != programmed_pages_.end() &&
            page->second.status != PageStatus::Erased &&
            page->second.status != PageStatus::StaticReadOnly) {
            throw std::runtime_error(
                "HBF static-data reservation intersects a programmed page");
        }
    }
    reserve_static_physical_blocks(touched_blocks);
    for (const auto ppn : unique_pages) {
        auto& page = programmed_pages_[ppn];
        if (page.status == PageStatus::Erased) {
            page.status = PageStatus::StaticReadOnly;
            page.owner = PageOwner::StaticReadOnly;
            page.lpn = ppn;
            auto& block = blocks_.at(static_cast<std::size_t>(ppn / config_.pages_per_block));
            page.block_epoch = block.epoch;
            block.valid_pages++;
            block.set_valid(static_cast<std::uint32_t>(ppn % config_.pages_per_block));
        }
    }
    stats_.free_pages = free_pages_;
}

void HbfDevice::reserve_static_physical_block_indices(
    const std::vector<std::size_t>& block_indices) {
    reserve_static_physical_blocks(block_indices);
}

void HbfDevice::reserve_static_physical_block_extent(
    std::uint32_t first_block,
    std::uint32_t block_count) {
    const auto end_block = static_cast<std::uint64_t>(first_block) +
        static_cast<std::uint64_t>(block_count);
    if (end_block > config_.blocks_per_plane) {
        throw std::runtime_error(
            "HBF static-data block extent exceeds blocks-per-plane geometry");
    }
    if (block_count == 0) {
        return;
    }
    std::vector<std::size_t> block_indices;
    if (planes_.size() >
        std::numeric_limits<std::size_t>::max() / block_count) {
        throw std::runtime_error(
            "HBF static-data block extent size exceeds size_t range");
    }
    block_indices.reserve(
        planes_.size() * static_cast<std::size_t>(block_count));
    for (std::size_t plane = 0; plane < planes_.size(); ++plane) {
        const auto plane_base =
            plane * static_cast<std::size_t>(config_.blocks_per_plane);
        for (std::uint64_t block = first_block; block < end_block; ++block) {
            block_indices.push_back(
                plane_base + static_cast<std::size_t>(block));
        }
    }
    reserve_static_physical_blocks(block_indices);
}

void HbfDevice::reserve_raw_physical_block_extent(
    std::uint32_t first_block,
    std::uint32_t block_count) {
    const auto end_block = static_cast<std::uint64_t>(first_block) +
        static_cast<std::uint64_t>(block_count);
    if (end_block > config_.blocks_per_plane) {
        throw std::runtime_error(
            "HBF raw physical block extent exceeds blocks-per-plane geometry");
    }
    if (block_count == 0) {
        return;
    }
    if (logical_capacity_frozen_ || compact_logical_image_) {
        throw std::runtime_error(
            "HBF cannot reserve raw physical blocks after logical use");
    }

    std::vector<std::uint8_t> reserve_mask(blocks_.size(), 0);
    std::vector<std::size_t> reserve_count_by_plane(planes_.size(), 0);
    std::size_t blocks_to_reserve = 0;
    for (std::size_t plane = 0; plane < planes_.size(); ++plane) {
        const auto plane_base =
            plane * static_cast<std::size_t>(config_.blocks_per_plane);
        for (std::uint64_t local = first_block; local < end_block; ++local) {
            const auto block_index =
                plane_base + static_cast<std::size_t>(local);
            const auto& block = blocks_.at(block_index);
            if (block.role == BlockRole::RawPhysical) {
                continue;
            }
            if (block.role != BlockRole::Free ||
                block.free_pages != config_.pages_per_block ||
                block.valid_pages != 0 || block.invalid_pages != 0 ||
                block.next_page != 0 || block.pending_program_pages != 0 ||
                block.pending_mapping_publications != 0 ||
                block.erase_pending) {
                throw std::runtime_error(
                    "HBF raw physical reservation intersects owned media at block " +
                    std::to_string(block_index));
            }
            reserve_mask.at(block_index) = 1;
            ++reserve_count_by_plane.at(plane);
            ++blocks_to_reserve;
        }
    }

    std::size_t pool_matches = 0;
    for (const auto& plane : planes_) {
        for (const auto block_index : plane.free_blocks) {
            pool_matches += reserve_mask.at(block_index) != 0 ? 1 : 0;
        }
    }
    if (pool_matches != blocks_to_reserve) {
        throw std::runtime_error(
            "HBF raw physical free block is missing from its plane pool");
    }
    for (std::size_t plane = 0; plane < planes_.size(); ++plane) {
        if (reserve_count_by_plane.at(plane) == 0) {
            continue;
        }
        auto& free_blocks = planes_.at(plane).free_blocks;
        const auto first_removed = std::remove_if(
            free_blocks.begin(),
            free_blocks.end(),
            [&](std::size_t block_index) {
                return reserve_mask.at(block_index) != 0;
            });
        const auto removed = static_cast<std::size_t>(
            std::distance(first_removed, free_blocks.end()));
        if (removed != reserve_count_by_plane.at(plane)) {
            throw std::runtime_error(
                "HBF raw physical free-pool removal count mismatch");
        }
        free_blocks.erase(first_removed, free_blocks.end());
    }
    for (std::size_t block_index = 0; block_index < blocks_.size(); ++block_index) {
        if (reserve_mask.at(block_index) == 0) {
            continue;
        }
        blocks_.at(block_index).role = BlockRole::RawPhysical;
        stats_.raw_reserved_pages = checked_add(
            stats_.raw_reserved_pages,
            config_.pages_per_block,
            "HBF raw reserved-page accounting");
    }
}

void HbfDevice::validate_raw_physical_block_extent(
    std::uint32_t first_block,
    std::uint32_t block_count) const {
    const auto end_block = static_cast<std::uint64_t>(first_block) +
        static_cast<std::uint64_t>(block_count);
    if (block_count == 0 || end_block > config_.blocks_per_plane) {
        throw std::runtime_error(
            "HBF restored raw extent declaration is empty or out of range");
    }
    std::uint64_t raw_blocks = 0;
    for (std::size_t block_index = 0; block_index < blocks_.size(); ++block_index) {
        const auto local = static_cast<std::uint64_t>(
            block_index % config_.blocks_per_plane);
        const bool expected = local >= first_block && local < end_block;
        const bool actual = blocks_.at(block_index).role == BlockRole::RawPhysical;
        if (expected != actual) {
            throw std::runtime_error(
                "HBF restored raw physical ownership does not match the declared extent");
        }
        raw_blocks = checked_add(
            raw_blocks, actual ? 1 : 0, "HBF restored raw-block accounting");
    }
    const auto expected_blocks = checked_mul(
        static_cast<std::uint64_t>(planes_.size()),
        block_count,
        "HBF restored raw extent block count");
    if (raw_blocks != expected_blocks) {
        throw std::runtime_error(
            "HBF restored raw physical block count diverged");
    }
}

PhysicalCompletion HbfDevice::drain_pending(
    std::string id,
    double arrival_ns,
    TraceConfig trace) {
    PhysicalCompletion out;
    out.id = std::move(id);
    out.tier = Tier::HBF;
    out.op = Op::Write;
    out.arrival_ns = arrival_ns;
    out.start_ns = arrival_ns;
    out.logical_bytes = 0;
    out.resource_path = "logic/write_buffer+mapping_table";
    auto* trace_spans = trace_spans_enabled(trace) ? &out.spans : nullptr;
    const auto physical_bytes_before = stats_.physical_read_bytes + stats_.physical_write_bytes;

    if (!std::isfinite(arrival_ns) || arrival_ns < 0.0) {
        throw std::runtime_error("HBF drain arrival must be finite and non-negative");
    }
    if (last_issue_arrival_ns_ && arrival_ns < *last_issue_arrival_ns_) {
        throw std::runtime_error("HBF drain cannot precede the last issue/drain arrival");
    }
    // drain_pending is a top-level causal barrier. Recording its arrival also
    // prevents a later issue from travelling behind calendar history that the
    // drain is now allowed to reclaim.
    last_issue_arrival_ns_ = arrival_ns;
    reservation_causal_watermark_ns_ = arrival_ns;
    prune_direct_write_ready(arrival_ns);
    prune_expired_state(arrival_ns);
    double causal_ready_ns = 0.0;
    for (const auto ready_ns : causal_state_ready_by_stack_) {
        causal_ready_ns = std::max(causal_ready_ns, ready_ns);
    }
    double at_ns = std::max({arrival_ns, background_finish_ns_, causal_ready_ns});
    apply_commits_through(at_ns);
    std::fill(
        state_observation_by_stack_.begin(),
        state_observation_by_stack_.end(),
        at_ns);
    flush_all_write_buffer_entries(at_ns, out.breakdown, trace_spans);
    complete_active_relocations(at_ns, out.breakdown, trace_spans);
    flush_all_dirty_mapping_pages(at_ns, out.breakdown, trace_spans);
    // A checkpoint under mapping-role pressure may have started another
    // victim; the drain leaves no relocation in flight.
    complete_active_relocations(at_ns, out.breakdown, trace_spans);
    at_ns = std::max(at_ns, background_finish_ns_);
    if (!pending_commits_.empty()) {
        at_ns = std::max(at_ns, pending_commits_.rbegin()->first.first);
    }
    apply_commits_through(at_ns);
    out.finish_ns = at_ns;
    for (std::size_t stack = 0; stack < config_.stacks; ++stack) {
        causal_state_ready_by_stack_[stack] = std::max(
            causal_state_ready_by_stack_[stack], out.finish_ns);
        state_observation_by_stack_[stack] = std::max(
            state_observation_by_stack_[stack], out.finish_ns);
    }
    out.note = out.finish_ns > out.arrival_ns ? "drained-pending-hbf-state" : "no-pending-hbf-state";
    out.physical_bytes = stats_.physical_read_bytes + stats_.physical_write_bytes -
        physical_bytes_before;

    stats_.free_pages = free_pages_;
    stats_.stage_work += out.breakdown;
    stats_.finish_ns = std::max({stats_.finish_ns, out.finish_ns, background_finish_ns_});
    return out;
}

std::vector<HbfBlockProfileBin> HbfDevice::block_profile(
    std::size_t bin_count) const {
    if (bin_count == 0 || blocks_.size() % bin_count != 0) {
        throw std::runtime_error(
            "HBF block profile bin count must divide the block count");
    }
    const auto blocks_per_bin = blocks_.size() / bin_count;
    std::vector<HbfBlockProfileBin> profile(bin_count);
    for (std::size_t block_index = 0; block_index < blocks_.size(); ++block_index) {
        const auto& block = blocks_[block_index];
        auto& bin = profile[block_index / blocks_per_bin];
        bin.blocks++;
        bin.erase_count_sum += block.erase_count;
        switch (block.role) {
        case BlockRole::StaticReadOnly:
            bin.static_blocks++;
            bin.valid_pages += config_.pages_per_block;
            continue;
        case BlockRole::Free:
            bin.free_blocks++;
            break;
        case BlockRole::Data:
            bin.data_blocks++;
            break;
        case BlockRole::GC:
            bin.gc_blocks++;
            break;
        case BlockRole::Mapping:
            bin.mapping_blocks++;
            break;
        case BlockRole::RawPhysical:
            bin.raw_physical_blocks++;
            break;
        }
        bin.valid_pages += block.valid_pages;
        bin.invalid_pages += block.invalid_pages;
        bin.free_pages += block.free_pages;
        bin.pending_pages += block.pending_program_pages;
    }
    return profile;
}

std::vector<std::uint32_t> HbfDevice::block_erase_counts() const {
    std::vector<std::uint32_t> counts;
    counts.reserve(blocks_.size());
    for (std::size_t block_index = 0;
         block_index < blocks_.size();
         ++block_index) {
        const auto& block = blocks_[block_index];
        if (block.role == BlockRole::StaticReadOnly) {
            continue;
        }
        counts.push_back(block.erase_count);
    }
    return counts;
}

HbfQuiescenceStats HbfDevice::quiescence_stats() const {
    HbfQuiescenceStats result;
    result.dirty_mapping_pages = dirty_mapping_vpns_.size();
    for (const auto& [_, events] : pending_dirty_mapping_events_) {
        result.pending_dirty_mapping_events = checked_add(
            result.pending_dirty_mapping_events,
            events.size(),
            "HBF pending dirty-mapping event count");
    }
    for (const auto& [_, updates] : pending_lpn_updates_) {
        result.pending_lpn_updates = checked_add(
            result.pending_lpn_updates,
            updates.size(),
            "HBF pending LPN update count");
    }
    for (const auto& [_, updates] : pending_vpn_updates_) {
        result.pending_vpn_updates = checked_add(
            result.pending_vpn_updates,
            updates.size(),
            "HBF pending VPN update count");
    }
    result.pending_commits = pending_commits_.size();
    for (const auto& buffer : write_buffer_by_stack_) {
        result.write_buffer_entries = checked_add(
            result.write_buffer_entries,
            buffer.size(),
            "HBF write-buffer entry count");
    }
    for (const auto& [_, generations] : inflight_buffered_writes_) {
        result.inflight_buffered_generations = checked_add(
            result.inflight_buffered_generations,
            generations.size(),
            "HBF inflight buffered-generation count");
    }
    result.pending_physical_programs = pending_physical_programs_.size();
    result.pending_physical_erases = pending_physical_erases_.size();
    return result;
}

void HbfDevice::materialize_committed_state_through(
    double completed_frontier_ns) {
    if (!std::isfinite(completed_frontier_ns) ||
        completed_frontier_ns < 0.0) {
        throw std::runtime_error(
            "HBF completed-state frontier must be finite and non-negative");
    }
    apply_commits_through(completed_frontier_ns);
}

HbfAuditSnapshot HbfDevice::audit_snapshot() const {
    HbfAuditSnapshot snapshot;
    snapshot.free_pages = free_pages_;
    snapshot.free_pages_per_stack = free_pages_per_stack_;
    snapshot.data_allocation_cursors.assign(
        next_data_allocation_plane_per_stack_.begin(),
        next_data_allocation_plane_per_stack_.end());
    snapshot.mapping_allocation_cursors.assign(
        next_mapping_allocation_plane_per_stack_.begin(),
        next_mapping_allocation_plane_per_stack_.end());
    snapshot.gc_allocation_cursors.assign(
        next_gc_allocation_plane_per_stack_.begin(),
        next_gc_allocation_plane_per_stack_.end());

    snapshot.logical_mappings.reserve(lpn_to_ppn_.size());
    for (const auto& [lpn, ppn] : lpn_to_ppn_) {
        snapshot.logical_mappings.push_back({.key = lpn, .ppn = ppn});
    }
    std::sort(
        snapshot.logical_mappings.begin(),
        snapshot.logical_mappings.end(),
        [](const HbfAuditMapping& lhs, const HbfAuditMapping& rhs) {
            return std::tie(lhs.key, lhs.ppn) < std::tie(rhs.key, rhs.ppn);
        });

    snapshot.mapping_pages.reserve(mapping_vpn_to_ppn_.size());
    for (const auto& [vpn, ppn] : mapping_vpn_to_ppn_) {
        snapshot.mapping_pages.push_back({.key = vpn, .ppn = ppn});
    }
    std::sort(
        snapshot.mapping_pages.begin(),
        snapshot.mapping_pages.end(),
        [](const HbfAuditMapping& lhs, const HbfAuditMapping& rhs) {
            return std::tie(lhs.key, lhs.ppn) < std::tie(rhs.key, rhs.ppn);
        });

    const auto page_status_name = [](PageStatus status) -> std::string {
        switch (status) {
        case PageStatus::Erased:
            return "erased";
        case PageStatus::StaticReadOnly:
            return "static_read_only";
        case PageStatus::Valid:
            return "valid";
        case PageStatus::Invalid:
            return "invalid";
        }
        throw std::runtime_error("unknown HBF page status");
    };
    const auto page_owner_name = [](PageOwner owner) -> std::string {
        switch (owner) {
        case PageOwner::Unassigned:
            return "unassigned";
        case PageOwner::Logical:
            return "logical";
        case PageOwner::Mapping:
            return "mapping";
        case PageOwner::RawPhysical:
            return "raw_physical";
        case PageOwner::StaticReadOnly:
            return "static_read_only";
        }
        throw std::runtime_error("unknown HBF page owner");
    };
    snapshot.materialized_pages.reserve(programmed_pages_.size());
    for (const auto& [ppn, page] : programmed_pages_) {
        snapshot.materialized_pages.push_back({
            .ppn = ppn,
            .status = page_status_name(page.status),
            .owner = page_owner_name(page.owner),
            .logical_key = page.lpn,
            .block_epoch = page.block_epoch,
        });
    }
    std::sort(
        snapshot.materialized_pages.begin(),
        snapshot.materialized_pages.end(),
        [](const HbfAuditPage& lhs, const HbfAuditPage& rhs) {
            return lhs.ppn < rhs.ppn;
        });

    const auto block_role_name = [](BlockRole role) -> std::string {
        switch (role) {
        case BlockRole::Free:
            return "free";
        case BlockRole::StaticReadOnly:
            return "static_read_only";
        case BlockRole::RawPhysical:
            return "raw_physical";
        case BlockRole::Data:
            return "data";
        case BlockRole::Mapping:
            return "mapping";
        case BlockRole::GC:
            return "gc";
        }
        throw std::runtime_error("unknown HBF block role");
    };
    snapshot.blocks.reserve(blocks_.size());
    for (std::size_t index = 0; index < blocks_.size(); ++index) {
        const auto& block = blocks_[index];
        snapshot.blocks.push_back({
            .block = index,
            .role = block_role_name(block.role),
            .valid_pages = block.valid_pages,
            .invalid_pages = block.invalid_pages,
            .free_pages = block.free_pages,
            .next_page = block.next_page,
            .erase_count = block.erase_count,
            .pending_program_pages = block.pending_program_pages,
            .pending_mapping_publications =
                block.pending_mapping_publications,
            .epoch = block.epoch,
            .erase_pending = block.erase_pending ||
                pending_physical_erases_.contains(index),
        });
    }

    snapshot.dirty_mapping_vpns.assign(
        dirty_mapping_vpns_.begin(), dirty_mapping_vpns_.end());
    std::sort(
        snapshot.dirty_mapping_vpns.begin(),
        snapshot.dirty_mapping_vpns.end());
    for (const auto& [_, events] : pending_dirty_mapping_events_) {
        snapshot.pending_dirty_mapping_events += events.size();
    }
    for (const auto& [_, updates] : pending_lpn_updates_) {
        snapshot.pending_lpn_updates += updates.size();
    }
    for (const auto& [_, updates] : pending_vpn_updates_) {
        snapshot.pending_vpn_updates += updates.size();
    }
    snapshot.pending_commits = pending_commits_.size();
    for (const auto& buffer : write_buffer_by_stack_) {
        snapshot.write_buffer_entries += buffer.size();
    }
    for (const auto& [_, generations] : inflight_buffered_writes_) {
        snapshot.inflight_buffered_generations += generations.size();
    }
    snapshot.pending_physical_programs = pending_physical_programs_.size();
    snapshot.pending_physical_erases = pending_physical_erases_.size();
    return snapshot;
}

HbfPersistentImage HbfDevice::persistent_image() const {
    const auto quiescence = quiescence_stats();
    if (!quiescence.quiescent()) {
        throw std::runtime_error(
            "HBF persistent image requires quiescent media and FTL state");
    }
    HbfPersistentImage image{
        .version = 2,
        .stacks = config_.stacks,
        .planes = planes_.size(),
        .blocks_per_plane = config_.blocks_per_plane,
        .pages_per_block = config_.pages_per_block,
        .page_size_bytes = config_.page_size_bytes,
        .mapping_entries_per_page = config_.mapping_entries_per_page,
        .state = audit_snapshot(),
    };
    image.plane_state.reserve(planes_.size());
    for (const auto& plane : planes_) {
        HbfPersistentPlane persistent;
        persistent.free_blocks.reserve(plane.free_blocks.size());
        for (const auto block : plane.free_blocks) {
            persistent.free_blocks.push_back(block);
        }
        if (plane.active_data_block) {
            persistent.active_data_block = *plane.active_data_block;
        }
        if (plane.active_mapping_block) {
            persistent.active_mapping_block = *plane.active_mapping_block;
        }
        if (plane.active_gc_block) {
            persistent.active_gc_block = *plane.active_gc_block;
        }
        image.plane_state.push_back(std::move(persistent));
    }
    if (compact_logical_image_) {
        const auto& compact = *compact_logical_image_;
        HbfPersistentCompactImage persistent{
            .mutable_image = compact.mutable_image,
            .first_lpn = compact.first_lpn,
            .page_count = compact.page_count,
            .first_vpn = compact.first_vpn,
            .vpn_slot_count = compact.vpn_slot_count,
            .mapping_page_count = compact.mapping_page_count,
            .data_blocks_by_plane = compact.data_blocks_by_plane,
            .mapping_ppns = compact.mapping_ppns,
        };
        persistent.vpn_ranges.reserve(compact.vpn_ranges.size());
        for (const auto& range : compact.vpn_ranges) {
            persistent.vpn_ranges.push_back({
                .first_entry = range.first_entry,
                .page_count = range.page_count,
                .stack_page_offset = range.stack_page_offset,
            });
        }
        const auto sorted_live_blocks = [](const auto& source) {
            std::vector<HbfPersistentCompactLiveBlock> result;
            result.reserve(source.size());
            for (const auto& [block, pages] : source) {
                result.push_back({.block = block, .live_pages = pages});
            }
            std::sort(
                result.begin(), result.end(),
                [](const auto& lhs, const auto& rhs) {
                    return lhs.block < rhs.block;
                });
            return result;
        };
        persistent.live_data_pages_by_block =
            sorted_live_blocks(compact.live_data_pages_by_block);
        persistent.live_mapping_pages_by_block =
            sorted_live_blocks(compact.live_mapping_pages_by_block);
        persistent.retired_lpns.assign(
            compact.retired_lpns.begin(), compact.retired_lpns.end());
        persistent.retired_mapping_vpns.assign(
            compact.retired_mapping_vpns.begin(),
            compact.retired_mapping_vpns.end());
        std::sort(
            persistent.retired_lpns.begin(),
            persistent.retired_lpns.end());
        std::sort(
            persistent.retired_mapping_vpns.begin(),
            persistent.retired_mapping_vpns.end());
        image.compact_image = std::move(persistent);
    }
    return image;
}

void HbfDevice::restore_persistent_image(const HbfPersistentImage& image) {
    const bool fresh_blocks = std::all_of(
        blocks_.begin(),
        blocks_.end(),
        [this](const BlockState& block) {
            return block.role == BlockRole::Free &&
                block.valid_pages == 0 && block.invalid_pages == 0 &&
                block.free_pages == config_.pages_per_block &&
                block.next_page == 0 && block.erase_count == 0 &&
                block.pending_program_pages == 0 &&
                block.pending_mapping_publications == 0 &&
                block.epoch == 0 && !block.erase_pending;
        });
    if (!fresh_blocks || compact_logical_image_ || !lpn_to_ppn_.empty() ||
        !mapping_vpn_to_ppn_.empty() || !programmed_pages_.empty() ||
        free_pages_ != total_pages_ || restored_block_erases_ != 0 ||
        last_issue_arrival_ns_) {
        throw std::runtime_error(
            "HBF persistent image restore requires a fresh device");
    }
    if (image.version != 2 || image.stacks != config_.stacks ||
        image.planes != planes_.size() ||
        image.blocks_per_plane != config_.blocks_per_plane ||
        image.pages_per_block != config_.pages_per_block ||
        image.page_size_bytes != config_.page_size_bytes ||
        image.mapping_entries_per_page != config_.mapping_entries_per_page ||
        image.state.blocks.size() != blocks_.size() ||
        image.plane_state.size() != planes_.size() ||
        image.state.free_pages_per_stack.size() != config_.stacks ||
        image.state.data_allocation_cursors.size() != config_.stacks ||
        image.state.mapping_allocation_cursors.size() != config_.stacks ||
        image.state.gc_allocation_cursors.size() != config_.stacks) {
        throw std::runtime_error(
            "HBF persistent image version or geometry does not match the device");
    }
    if (!image.state.quiescent()) {
        throw std::runtime_error(
            "HBF persistent image contains volatile or pending state");
    }

    const auto block_role = [](const std::string& name) {
        if (name == "free") return BlockRole::Free;
        if (name == "static_read_only") return BlockRole::StaticReadOnly;
        if (name == "raw_physical") return BlockRole::RawPhysical;
        if (name == "data") return BlockRole::Data;
        if (name == "mapping") return BlockRole::Mapping;
        if (name == "gc") return BlockRole::GC;
        throw std::runtime_error(
            "HBF persistent image contains an unknown block role");
    };
    const auto page_status = [](const std::string& name) {
        if (name == "static_read_only") return PageStatus::StaticReadOnly;
        if (name == "valid") return PageStatus::Valid;
        if (name == "invalid") return PageStatus::Invalid;
        if (name == "erased") return PageStatus::Erased;
        throw std::runtime_error(
            "HBF persistent image contains an unknown page status");
    };
    const auto page_owner = [](const std::string& name) {
        if (name == "unassigned") return PageOwner::Unassigned;
        if (name == "logical") return PageOwner::Logical;
        if (name == "mapping") return PageOwner::Mapping;
        if (name == "raw_physical") return PageOwner::RawPhysical;
        if (name == "static_read_only") return PageOwner::StaticReadOnly;
        throw std::runtime_error(
            "HBF persistent image contains an unknown page owner");
    };

    for (std::size_t index = 0; index < image.state.blocks.size(); ++index) {
        const auto& source = image.state.blocks[index];
        if (source.block != index || source.pending_program_pages != 0 ||
            source.pending_mapping_publications != 0 || source.erase_pending) {
            throw std::runtime_error(
                "HBF persistent image contains non-quiescent block state");
        }
        auto& destination = blocks_[index];
        destination = BlockState{};
        destination.role = block_role(source.role);
        destination.valid_pages = source.valid_pages;
        destination.invalid_pages = source.invalid_pages;
        destination.free_pages = source.free_pages;
        destination.next_page = source.next_page;
        destination.erase_count = source.erase_count;
        destination.epoch = source.epoch;
    }

    lpn_to_ppn_.reserve(image.state.logical_mappings.size());
    for (const auto& mapping : image.state.logical_mappings) {
        if (!lpn_to_ppn_.emplace(mapping.key, mapping.ppn).second) {
            throw std::runtime_error(
                "HBF persistent image duplicates an L2P key");
        }
    }
    mapping_vpn_to_ppn_.reserve(image.state.mapping_pages.size());
    for (const auto& mapping : image.state.mapping_pages) {
        if (!mapping_vpn_to_ppn_.emplace(mapping.key, mapping.ppn).second) {
            throw std::runtime_error(
                "HBF persistent image duplicates a mapping-page key");
        }
    }
    programmed_pages_.reserve(image.state.materialized_pages.size());
    for (const auto& source : image.state.materialized_pages) {
        if (source.ppn >= total_pages_) {
            throw std::runtime_error(
                "HBF persistent image contains an out-of-range page");
        }
        const auto status = page_status(source.status);
        if (status == PageStatus::Erased) {
            throw std::runtime_error(
                "HBF quiescent persistent image contains an erased pending page");
        }
        const auto owner = page_owner(source.owner);
        const auto block_index = static_cast<std::size_t>(
            source.ppn / config_.pages_per_block);
        const auto page_index = static_cast<std::uint32_t>(
            source.ppn % config_.pages_per_block);
        if (source.block_epoch != blocks_[block_index].epoch ||
            !programmed_pages_.emplace(
                source.ppn,
                PageState{
                    .status = status,
                    .owner = owner,
                    .lpn = source.logical_key,
                    .block_epoch = source.block_epoch,
                }).second) {
            throw std::runtime_error(
                "HBF persistent image duplicates or mis-epochs a page");
        }
        if (status == PageStatus::Valid ||
            status == PageStatus::StaticReadOnly) {
            blocks_[block_index].set_valid(page_index);
        }
    }

    const auto pps = planes_per_stack();
    const auto restore_cursors = [pps](
                                     const std::vector<std::uint64_t>& source,
                                     std::vector<std::size_t>& destination) {
        destination.clear();
        destination.reserve(source.size());
        for (const auto cursor : source) {
            if (cursor >= pps) {
                throw std::runtime_error(
                    "HBF persistent image allocation cursor is out of range");
            }
            destination.push_back(static_cast<std::size_t>(cursor));
        }
    };
    restore_cursors(
        image.state.data_allocation_cursors,
        next_data_allocation_plane_per_stack_);
    restore_cursors(
        image.state.mapping_allocation_cursors,
        next_mapping_allocation_plane_per_stack_);
    restore_cursors(
        image.state.gc_allocation_cursors,
        next_gc_allocation_plane_per_stack_);

    for (std::size_t plane_index = 0;
         plane_index < planes_.size();
         ++plane_index) {
        auto& plane = planes_[plane_index];
        const auto& source = image.plane_state[plane_index];
        plane.free_blocks.clear();
        for (const auto block : source.free_blocks) {
            if (block >= blocks_.size() ||
                block_plane_index(static_cast<std::size_t>(block)) !=
                    plane_index) {
                throw std::runtime_error(
                    "HBF persistent image free-block order is out of plane");
            }
            plane.free_blocks.push_back(static_cast<std::size_t>(block));
        }
        const auto restore_active = [this, plane_index](
                                        std::optional<std::uint64_t> block,
                                        BlockRole expected) ->
                                        std::optional<std::size_t> {
            if (!block) return std::nullopt;
            if (*block >= blocks_.size() ||
                block_plane_index(static_cast<std::size_t>(*block)) !=
                    plane_index) {
                throw std::runtime_error(
                    "HBF persistent image active block is out of plane");
            }
            const auto& state = blocks_[static_cast<std::size_t>(*block)];
            if (state.role != expected || state.free_pages == 0) {
                throw std::runtime_error(
                    "HBF persistent image active block has the wrong role");
            }
            return static_cast<std::size_t>(*block);
        };
        plane.active_data_block = restore_active(
            source.active_data_block, BlockRole::Data);
        plane.active_mapping_block = restore_active(
            source.active_mapping_block, BlockRole::Mapping);
        plane.active_gc_block = restore_active(
            source.active_gc_block, BlockRole::GC);
    }

    if (image.compact_image) {
        const auto& source = *image.compact_image;
        const auto stacks = static_cast<std::uint64_t>(config_.stacks);
        if (source.page_count == 0 || source.page_count > total_pages_ ||
            source.data_blocks_by_plane.size() != planes_.size() ||
            source.mapping_ppns.size() != source.vpn_slot_count ||
            source.vpn_ranges.size() != source.vpn_slot_count ||
            source.mapping_page_count > source.vpn_slot_count ||
            source.retired_lpns.size() > source.page_count ||
            source.retired_mapping_vpns.size() >
                source.mapping_page_count ||
            (!source.mutable_image &&
             (!source.retired_lpns.empty() ||
              !source.retired_mapping_vpns.empty()))) {
            throw std::runtime_error(
                "HBF persistent compact image dimensions are inconsistent");
        }
        const auto last_lpn = checked_add(
            source.first_lpn,
            source.page_count - 1,
            "HBF restored compact last LPN");
        if (is_metadata_lpn(last_lpn)) {
            throw std::runtime_error(
                "HBF persistent compact image enters metadata LPN space");
        }
        const auto first_group =
            (source.first_lpn / stacks) /
            config_.mapping_entries_per_page;
        const auto last_group =
            (last_lpn / stacks) /
            config_.mapping_entries_per_page;
        const auto expected_first_vpn = checked_mul(
            first_group, stacks, "HBF restored compact first VPN");
        const auto expected_vpn_slots = checked_mul(
            checked_add(
                last_group - first_group,
                1,
                "HBF restored compact mapping groups"),
            stacks,
            "HBF restored compact VPN slots");
        if (source.first_vpn != expected_first_vpn ||
            source.vpn_slot_count != expected_vpn_slots) {
            throw std::runtime_error(
                "HBF persistent compact image VPN geometry diverged");
        }

        CompactLogicalImage compact;
        compact.mutable_image = source.mutable_image;
        compact.first_lpn = source.first_lpn;
        compact.page_count = source.page_count;
        compact.first_vpn = source.first_vpn;
        compact.vpn_slot_count = source.vpn_slot_count;
        compact.mapping_page_count = source.mapping_page_count;
        compact.data_blocks_by_plane = source.data_blocks_by_plane;
        compact.mapping_ppns = source.mapping_ppns;
        compact.vpn_ranges.reserve(source.vpn_ranges.size());
        for (const auto& range : source.vpn_ranges) {
            if (range.first_entry > config_.mapping_entries_per_page ||
                range.page_count > config_.mapping_entries_per_page ||
                range.first_entry + range.page_count >
                    config_.mapping_entries_per_page) {
                throw std::runtime_error(
                    "HBF persistent compact VPN range is out of bounds");
            }
            compact.vpn_ranges.push_back({
                .first_entry = range.first_entry,
                .page_count = range.page_count,
                .stack_page_offset = range.stack_page_offset,
            });
        }
        compact.vpn_offsets_by_stack.resize(config_.stacks);

        for (std::size_t plane = 0;
             plane < compact.data_blocks_by_plane.size();
             ++plane) {
            const auto& assigned = compact.data_blocks_by_plane[plane];
            if (assigned.size() > config_.blocks_per_plane) {
                throw std::runtime_error(
                    "HBF persistent compact data directory exceeds a plane");
            }
            for (std::size_t ordinal = 0;
                 ordinal < assigned.size();
                 ++ordinal) {
                const auto block = assigned[ordinal];
                if (block >= blocks_.size() ||
                    block_plane_index(static_cast<std::size_t>(block)) !=
                        plane ||
                    !compact.data_block_locations.emplace(
                        block,
                        CompactLogicalImage::DataBlockLocation{
                            .plane = plane,
                            .block_ordinal = ordinal,
                        }).second) {
                    throw std::runtime_error(
                        "HBF persistent compact data-block directory is invalid");
                }
            }
        }

        std::uint64_t active_mapping_pages = 0;
        std::uint64_t represented_data_pages = 0;
        for (std::size_t offset = 0;
             offset < compact.mapping_ppns.size();
             ++offset) {
            const auto& mapping_ppn = compact.mapping_ppns[offset];
            const auto& range = compact.vpn_ranges[offset];
            const bool needs_mapping = range.page_count != 0 &&
                config_.mapping_mode != MappingMode::Direct;
            if (needs_mapping != mapping_ppn.has_value()) {
                throw std::runtime_error(
                    "HBF persistent compact mapping slot activity diverged");
            }
            const auto mapping_vpn = checked_add(
                compact.first_vpn, offset, "HBF restored compact mapping VPN");
            if (range.page_count != 0) {
                compact.vpn_offsets_by_stack.at(
                    stack_for_vpn(mapping_vpn)).push_back(offset);
            }
            represented_data_pages = checked_add(
                represented_data_pages, range.page_count,
                "HBF restored compact represented data pages");
            if (!mapping_ppn) {
                continue;
            }
            if (*mapping_ppn >= total_pages_) {
                throw std::runtime_error(
                    "HBF persistent compact mapping PPN is out of range");
            }
            if (!compact.mapping_vpn_by_ppn.emplace(
                    *mapping_ppn, mapping_vpn).second) {
                throw std::runtime_error(
                    "HBF persistent compact mapping PPN is duplicated");
            }
            active_mapping_pages = checked_add(
                active_mapping_pages,
                1,
                "HBF restored compact mapping-page count");
        }
        if (active_mapping_pages != compact.mapping_page_count ||
            represented_data_pages != compact.page_count) {
            throw std::runtime_error(
                "HBF persistent compact mapping directory does not conserve pages");
        }

        const auto restore_live_blocks = [this](
            const std::vector<HbfPersistentCompactLiveBlock>& source_blocks,
            auto& destination,
            const char* description) {
            for (const auto& entry : source_blocks) {
                if (entry.block >= blocks_.size() || entry.live_pages == 0 ||
                    entry.live_pages > config_.pages_per_block ||
                    !destination.emplace(
                        entry.block, entry.live_pages).second) {
                    throw std::runtime_error(
                        std::string("HBF persistent compact ") + description +
                        " live-block index is invalid");
                }
            }
        };
        restore_live_blocks(
            source.live_data_pages_by_block,
            compact.live_data_pages_by_block,
            "data");
        restore_live_blocks(
            source.live_mapping_pages_by_block,
            compact.live_mapping_pages_by_block,
            "mapping");
        for (const auto lpn : source.retired_lpns) {
            if (lpn < compact.first_lpn ||
                lpn - compact.first_lpn >= compact.page_count ||
                !compact.retired_lpns.insert(lpn).second) {
                throw std::runtime_error(
                    "HBF persistent compact retired LPN index is invalid");
            }
        }
        for (const auto vpn : source.retired_mapping_vpns) {
            if (vpn < compact.first_vpn ||
                vpn - compact.first_vpn >= compact.vpn_slot_count ||
                !compact.mapping_ppns[
                    static_cast<std::size_t>(vpn - compact.first_vpn)] ||
                !compact.retired_mapping_vpns.insert(vpn).second) {
                throw std::runtime_error(
                    "HBF persistent compact retired VPN index is invalid");
            }
        }
        for (const auto& [lpn, _] : lpn_to_ppn_) {
            if (lpn >= compact.first_lpn &&
                lpn - compact.first_lpn < compact.page_count &&
                !compact.retired_lpns.contains(lpn)) {
                throw std::runtime_error(
                    "HBF persistent image duplicates a live compact LPN");
            }
        }
        for (const auto& [vpn, _] : mapping_vpn_to_ppn_) {
            if (vpn >= compact.first_vpn &&
                vpn - compact.first_vpn < compact.vpn_slot_count &&
                compact.mapping_ppns[
                    static_cast<std::size_t>(vpn - compact.first_vpn)] &&
                !compact.retired_mapping_vpns.contains(vpn)) {
                throw std::runtime_error(
                    "HBF persistent image duplicates a live compact VPN");
            }
        }
        const auto compact_ppn_without_validity = [this, &compact](
            std::uint64_t lpn) {
            const auto mapping_vpn = mapping_vpn_for_lpn(lpn);
            if (mapping_vpn < compact.first_vpn ||
                mapping_vpn - compact.first_vpn >=
                    compact.vpn_slot_count) {
                throw std::runtime_error(
                    "HBF restored compact LPN is outside its VPN directory");
            }
            const auto vpn_offset = mapping_vpn - compact.first_vpn;
            const auto& range = compact.vpn_ranges.at(
                static_cast<std::size_t>(vpn_offset));
            const auto local_entry =
                (lpn / config_.stacks) %
                config_.mapping_entries_per_page;
            if (local_entry < range.first_entry ||
                local_entry - range.first_entry >= range.page_count) {
                throw std::runtime_error(
                    "HBF restored compact LPN is outside its VPN range");
            }
            const auto data_index = checked_add(
                range.stack_page_offset,
                local_entry - range.first_entry,
                "HBF restored compact data index");
            const auto pps = static_cast<std::uint64_t>(planes_per_stack());
            const auto stack = stack_for_vpn(mapping_vpn);
            const auto plane = stack * planes_per_stack() +
                static_cast<std::size_t>(data_index % pps);
            const auto page_ordinal = data_index / pps;
            const auto block_ordinal =
                page_ordinal / config_.pages_per_block;
            const auto page = static_cast<std::uint32_t>(
                page_ordinal % config_.pages_per_block);
            const auto& assigned = compact.data_blocks_by_plane.at(plane);
            if (block_ordinal >= assigned.size()) {
                throw std::runtime_error(
                    "HBF restored compact LPN exceeds its data directory");
            }
            return checked_add(
                checked_mul(
                    assigned[static_cast<std::size_t>(block_ordinal)],
                    config_.pages_per_block,
                    "HBF restored compact block PPN"),
                page,
                "HBF restored compact page PPN");
        };
        std::unordered_map<std::uint64_t, std::uint32_t>
            retired_data_pages_by_block;
        std::vector<std::uint64_t> retired_data_ppns;
        retired_data_ppns.reserve(compact.retired_lpns.size());
        for (const auto lpn : compact.retired_lpns) {
            const auto ppn = compact_ppn_without_validity(lpn);
            retired_data_ppns.push_back(ppn);
            auto& count = retired_data_pages_by_block[
                ppn / config_.pages_per_block];
            if (count == std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error(
                    "HBF restored compact retired-page count overflowed");
            }
            ++count;
        }
        for (const auto& assigned : compact.data_blocks_by_plane) {
            for (const auto block_number : assigned) {
                const auto live = compact.live_data_pages_by_block.find(
                    block_number);
                const auto retired = retired_data_pages_by_block.find(
                    block_number);
                const auto original_pages = checked_add(
                    live == compact.live_data_pages_by_block.end() ?
                        0 : live->second,
                    retired == retired_data_pages_by_block.end() ?
                        0 : retired->second,
                    "HBF restored compact original block pages");
                if (original_pages > config_.pages_per_block) {
                    throw std::runtime_error(
                        "HBF restored compact block exceeds page geometry");
                }
                if (original_pages != 0) {
                    blocks_[static_cast<std::size_t>(block_number)]
                        .set_valid_range(
                            0, static_cast<std::uint32_t>(original_pages));
                }
            }
        }
        for (const auto ppn : retired_data_ppns) {
            blocks_[static_cast<std::size_t>(
                ppn / config_.pages_per_block)].clear_valid(
                    static_cast<std::uint32_t>(
                        ppn % config_.pages_per_block));
        }
        for (std::size_t offset = 0;
             offset < compact.mapping_ppns.size();
             ++offset) {
            if (!compact.mapping_ppns[offset]) {
                continue;
            }
            const auto vpn = checked_add(
                compact.first_vpn,
                offset,
                "HBF restored compact bitmap VPN");
            if (compact.retired_mapping_vpns.contains(vpn)) {
                continue;
            }
            const auto ppn = *compact.mapping_ppns[offset];
            blocks_[static_cast<std::size_t>(
                ppn / config_.pages_per_block)].set_valid(
                    static_cast<std::uint32_t>(
                        ppn % config_.pages_per_block));
        }
        compact_logical_image_ = std::move(compact);
        for (const auto lpn : source.retired_lpns) {
            record_mutated_lpn_range(lpn, 1);
        }
        // A compact-owned PPN can have been erased and reused by a later
        // materialized generation. Reapply current materialized validity
        // after reconstructing and retiring the original compact prefixes.
        for (const auto& [ppn, page] : programmed_pages_) {
            if (page.status == PageStatus::Valid ||
                page.status == PageStatus::StaticReadOnly) {
                blocks_[static_cast<std::size_t>(
                    ppn / config_.pages_per_block)].set_valid(
                        static_cast<std::uint32_t>(
                            ppn % config_.pages_per_block));
            }
        }
    }

    free_pages_ = image.state.free_pages;
    free_pages_per_stack_ = image.state.free_pages_per_stack;
    restored_block_erases_ = 0;
    stats_.static_reserved_pages = 0;
    stats_.raw_reserved_pages = 0;
    for (const auto& block : blocks_) {
        restored_block_erases_ = checked_add(
            restored_block_erases_,
            block.erase_count,
            "HBF restored block erase count");
        if (block.role == BlockRole::StaticReadOnly) {
            stats_.static_reserved_pages = checked_add(
                stats_.static_reserved_pages,
                config_.pages_per_block,
                "HBF restored static page count");
        } else if (block.role == BlockRole::RawPhysical) {
            stats_.raw_reserved_pages = checked_add(
                stats_.raw_reserved_pages,
                config_.pages_per_block,
                "HBF restored raw reserved-page count");
        }
    }
    stats_.compact_initial_logical_data_pages =
        compact_logical_image_ ? compact_logical_image_->page_count : 0;
    stats_.compact_initial_mapping_pages =
        compact_logical_image_ ? compact_logical_image_->mapping_page_count : 0;
    stats_.initial_logical_data_pages = logical_mapping_entry_count();
    const auto compact_live_mapping_pages = compact_logical_image_ ?
        compact_logical_image_->mapping_page_count -
            compact_logical_image_->retired_mapping_vpns.size() :
        0;
    stats_.initial_mapping_pages = checked_add(
        mapping_vpn_to_ppn_.size(),
        compact_live_mapping_pages,
        "HBF restored initial mapping pages");
    stats_.free_pages = free_pages_;
    // Static/raw roles came from the image: derive or validate the logical
    // capacity now and check that every restored logical page lies inside it.
    logical_capacity_pages_ = config_.logical_capacity_bytes == 0 ?
        derive_logical_capacity_pages() :
        config_.logical_capacity_bytes / config_.page_size_bytes;
    resolve_logical_capacity();
    for (const auto& [lpn, _] : lpn_to_ppn_) {
        if (lpn >= logical_capacity_pages_) {
            throw std::runtime_error(
                "HBF persistent image maps a logical page beyond the logical "
                "capacity of " + std::to_string(logical_capacity_pages_) + " pages");
        }
    }
    if (compact_logical_image_ && compact_logical_image_->page_count != 0 &&
        compact_logical_image_->first_lpn + compact_logical_image_->page_count - 1 >=
            logical_capacity_pages_) {
        throw std::runtime_error(
            "HBF persistent compact image exceeds the logical capacity of " +
            std::to_string(logical_capacity_pages_) + " pages");
    }
    // Thermal state is runtime-only: a restored image boots a fresh process
    // at the configured boot state with an empty pacing calendar.
    for (auto& stack : thermal_stacks_) {
        stack.node = ThermalNodeState{};
        stack.node.temperature_c = thermal_boot_temperature_c_;
        stack.node.peak_c = thermal_boot_temperature_c_;
        stack.node.throttled = config_.thermal_start_at_ceiling;
        stack.pacing_frontier_ns = 0.0;
    }
    refresh_parallel_stats();
}

PhysicalCompletion HbfDevice::issue(const PhysicalRequest& request) {
    if (request.tier != Tier::HBF) {
        throw std::runtime_error("HbfDevice received non-HBF request");
    }
    // Validate provenance before any request state changes. A malformed enum
    // must not be retained in a deferred buffer and fail only at a later
    // eviction or drain.
    (void)resolve_heatmap_source(
        TransactionSource::User, request.heatmap_source);
    if (request.bytes == 0 && request.op != Op::Erase) {
        throw std::runtime_error("HBF request bytes must be positive");
    }
    if (request.op != Op::Erase &&
        request.bytes - 1 > std::numeric_limits<std::uint64_t>::max() - request.addr) {
        throw std::runtime_error("HBF request address range overflows uint64_t");
    }
    if (request.op == Op::Refresh) {
        throw std::runtime_error("HBF v0 does not model background refresh");
    }
    if (!std::isfinite(request.arrival_ns) || request.arrival_ns < 0.0) {
        throw std::runtime_error("HBF request arrival must be finite and non-negative");
    }
    const bool static_request = request.address_space == AddressSpace::Static;
    const bool physical_request = request.address_space == AddressSpace::Physical ||
        request.op == Op::Erase;
    const bool direct_media_request = physical_request || static_request;
    if (compact_logical_image_ &&
        !compact_logical_image_->mutable_image &&
        (request.op != Op::Read || direct_media_request)) {
        throw std::runtime_error(
            "HBF compact read-only initial image accepts logical reads only");
    }
    if (!direct_media_request) {
        const auto last_addr = request.addr + (request.bytes - 1);
        require_logical_capacity(last_addr / config_.page_size_bytes);
    } else if (physical_request) {
        const auto capacity_bytes = checked_mul(total_pages_, config_.page_size_bytes,
            "HBF physical capacity bytes");
        if (request.addr >= capacity_bytes ||
            (request.op != Op::Erase && request.bytes > capacity_bytes - request.addr)) {
            throw std::runtime_error("HBF physical request range is out of capacity");
        }
    } else {
        const auto capacity_bytes = checked_mul(
            total_pages_, config_.page_size_bytes,
            "HBF static source capacity bytes");
        if (request.op != Op::Read || request.addr >= capacity_bytes ||
            request.bytes > capacity_bytes - request.addr) {
            throw std::runtime_error(
                "HBF static request must be an in-capacity read");
        }
        const auto first_source_page =
            request.addr / config_.page_size_bytes;
        const auto source_pages = page_count_for(
            logical_page_address(request.addr), request.bytes);
        const auto spans = static_page_run_spans(
            first_source_page, source_pages);
        for (const auto& span : spans) {
            const auto plane_base = span.plane * config_.blocks_per_plane;
            const auto first_block = span.first_page /
                config_.pages_per_block;
            const auto last_block =
                (span.first_page + span.page_count - 1) /
                config_.pages_per_block;
            for (auto block = first_block; block <= last_block; ++block) {
                if (blocks_.at(plane_base + block).role !=
                    BlockRole::StaticReadOnly) {
                    throw std::runtime_error(
                        "HBF static request escapes its reserved extent");
                }
            }
        }
    }
    if (last_issue_arrival_ns_ && request.arrival_ns < *last_issue_arrival_ns_) {
        throw std::runtime_error(
            "HBF requests and drains must be issued in nondecreasing arrival order; "
            "enqueue/sort the workload before simulation so state cannot travel backward "
            "in time");
    }
    // Once a request reaches state-dependent validation, its arrival becomes
    // a causal barrier even if the target is rejected. This prevents a caught
    // future-time error from being followed by an earlier request after prior
    // commits have already been materialized.
    last_issue_arrival_ns_ = request.arrival_ns;
    reservation_causal_watermark_ns_ = request.arrival_ns;
    prune_direct_write_ready(request.arrival_ns);
    prune_expired_state(request.arrival_ns);
    const auto request_stack = physical_request ?
        decode(request.addr).stack :
        static_cast<std::uint32_t>(
            stack_for_lpn(request.addr / config_.page_size_bytes));
    std::vector<bool> touched_stacks(config_.stacks, false);
    touched_stacks.at(request_stack) = true;
    if (request.op != Op::Erase && physical_request) {
        const auto last_stack = decode(request.addr + request.bytes - 1).stack;
        for (std::size_t stack = request_stack; stack <= last_stack; ++stack) {
            touched_stacks.at(stack) = true;
        }
    } else if (static_request) {
        const auto first_source_page =
            request.addr / config_.page_size_bytes;
        const auto source_pages = page_count_for(
            logical_page_address(request.addr), request.bytes);
        if (source_pages >= 2 * static_cast<std::uint64_t>(config_.stacks)) {
            std::fill(touched_stacks.begin(), touched_stacks.end(), true);
        } else {
            for (std::uint64_t index = 0; index < source_pages; ++index) {
                touched_stacks.at(
                    stack_for_lpn(first_source_page + index)) = true;
            }
        }
    } else if (!physical_request) {
        const auto first_lpn = request.addr / config_.page_size_bytes;
        const auto last_lpn =
            (request.addr + request.bytes - 1) / config_.page_size_bytes;
        for (auto lpn = first_lpn;; ++lpn) {
            touched_stacks.at(stack_for_lpn(lpn)) = true;
            if (lpn == last_lpn) {
                break;
            }
        }
    }
    std::vector<double> state_ready_by_stack(
        config_.stacks, request.arrival_ns);
    std::vector<std::unordered_set<std::uint64_t>> selected_commits_by_stack(
        config_.stacks);
    if (physical_request) {
        const auto first_block = block_index(decode(request.addr));
        const auto last_block = request.op == Op::Erase ? first_block :
            block_index(decode(request.addr + request.bytes - 1));
        for (auto block = first_block; block <= last_block; ++block) {
            const auto stack = stack_of_block(block);
            state_ready_by_stack[stack] = std::max(
                state_ready_by_stack[stack],
                materialized_ready_by_block_.at(block));
            const auto pending = pending_physical_erases_.find(block);
            if (pending != pending_physical_erases_.end()) {
                state_ready_by_stack[stack] = std::max(
                    state_ready_by_stack[stack], pending->second.finish_ns);
                add_pending_physical_erase_commits(
                    block, selected_commits_by_stack[stack]);
            }
            if (request.op == Op::Erase) {
                // A GC source erase depends on every relocation destination.
                // A later raw erase of one of those destination blocks must
                // therefore materialize the source chain before it retires
                // the destination epoch; otherwise the GC publication would
                // be suppressed and the source erase would lose live data.
                for (const auto& [source_block, source_erase] :
                     pending_physical_erases_) {
                    if (!source_erase.garbage_collection ||
                        !std::any_of(
                            source_erase.destination_ppns.begin(),
                            source_erase.destination_ppns.end(),
                            [this, block](std::uint64_t destination_ppn) {
                                return destination_ppn /
                                        config_.pages_per_block == block;
                            })) {
                        continue;
                    }
                    state_ready_by_stack[stack] = std::max(
                        state_ready_by_stack[stack],
                        source_erase.finish_ns);
                    add_pending_physical_erase_commits(
                        source_block,
                        selected_commits_by_stack[stack]);
                }
            }
        }
        if (request.op != Op::Erase) {
            const auto first_ppn = encode_ppn(decode(request.addr));
            const auto pages = page_count_for(decode(request.addr), request.bytes);
            for (std::uint64_t i = 0; i < pages; ++i) {
                const auto ppn = first_ppn + i;
                const auto stack = stack_of_block(static_cast<std::size_t>(
                    ppn / config_.pages_per_block));
                if (const auto observed = materialized_ready_by_ppn_.find(ppn);
                    observed != materialized_ready_by_ppn_.end()) {
                    state_ready_by_stack[stack] = std::max(
                        state_ready_by_stack[stack], observed->second);
                }
                if (const auto program = pending_physical_programs_.find(ppn);
                    program != pending_physical_programs_.end()) {
                    state_ready_by_stack[stack] = std::max(
                        state_ready_by_stack[stack], program->second.commit_ns);
                    selected_commits_by_stack[stack].insert(
                        program->second.commit_sequence);
                }
            }
        }
    } else if (!static_request) {
        const auto first_lpn = request.addr / config_.page_size_bytes;
        const auto last_lpn =
            (request.addr + request.bytes - 1) / config_.page_size_bytes;
        for (auto lpn = first_lpn; lpn <= last_lpn; ++lpn) {
            const auto stack = stack_for_lpn(lpn);
            if (const auto observed = materialized_ready_by_lpn_.find(lpn);
                observed != materialized_ready_by_lpn_.end()) {
                state_ready_by_stack[stack] = std::max(
                    state_ready_by_stack[stack], observed->second);
            }
            const auto mapping_vpn = mapping_vpn_for_lpn(lpn);
            if (const auto observed = materialized_ready_by_vpn_.find(mapping_vpn);
                observed != materialized_ready_by_vpn_.end()) {
                state_ready_by_stack[stack] = std::max(
                    state_ready_by_stack[stack], observed->second);
            }
        }
    }
    // Materialize every causally prior state transition before validating a
    // physical target. Otherwise a completed-but-not-yet-applied program
    // could make the same PPN appear erased and permit illegal reprogramming.
    // Globally materialize only events that have truly completed by host
    // arrival (or a whole-stack causal GC/drain barrier). Future block/LPN/VPN
    // dependencies apply only their selected callback chain; applying the
    // whole stack would make independent planes depend on API call order.
    apply_commits_through(request.arrival_ns);
    for (std::size_t stack = 0; stack < config_.stacks; ++stack) {
        if (!touched_stacks[stack]) {
            continue;
        }
        const double causal_ready_ns = causal_state_ready_by_stack_.at(stack);
        if (causal_ready_ns > request.arrival_ns) {
            apply_stack_commits_through(stack, causal_ready_ns);
        }
        const double ready_ns = std::max(
            state_ready_by_stack[stack], causal_ready_ns);
        state_ready_by_stack[stack] = ready_ns;
        apply_selected_commits_through(
            selected_commits_by_stack[stack], ready_ns);
        state_observation_by_stack_.at(stack) = ready_ns;
    }
    const double ingress_state_ready_ns = state_ready_by_stack.at(request_stack);
    if (physical_request && request.op == Op::Write) {
        const auto first = encode_ppn(decode(request.addr));
        const auto pages = page_count_for(decode(request.addr), request.bytes);
        reserve_physical_program_range(first, pages);
    } else if (physical_request && request.op == Op::Erase) {
        const auto block = block_index(decode(request.addr));
        if (pending_physical_erases_.contains(block) ||
            blocks_.at(block).erase_pending) {
            throw std::runtime_error("HBF block already has an in-flight physical erase");
        }
        if (blocks_.at(block).role == BlockRole::StaticReadOnly) {
            throw std::runtime_error("HBF erase targets a static read-only block");
        }
    }
    stats_.first_arrival_ns = std::min(stats_.first_arrival_ns, request.arrival_ns);

    PhysicalCompletion out;
    out.id = request.id;
    out.tier = Tier::HBF;
    out.op = request.op;
    out.arrival_ns = request.arrival_ns;
    out.logical_bytes = request.bytes;
    auto* trace_spans = trace_spans_enabled(request.trace) ? &out.spans : nullptr;
    const auto physical_bytes_before = stats_.physical_read_bytes + stats_.physical_write_bytes;
    if (ingress_state_ready_ns > request.arrival_ns) {
        out.breakdown.scheduler_queue_wait_ns +=
            ingress_state_ready_ns - request.arrival_ns;
        if (trace_spans != nullptr) {
            trace_wait(
                trace_spans,
                logic_entity(static_cast<std::uint32_t>(request_stack)),
                request.arrival_ns,
                ingress_state_ready_ns,
                "wait_stack_state_dependency");
        }
    }

    const auto ingress_stack = request_stack;
    auto& ingress_logic = logic_dies_.at(ingress_stack);
    const double request_command_done = schedule_external_request_command(
        ingress_stack,
        ingress_state_ready_ns,
        out.breakdown,
        trace_spans,
        request.id);
    const auto ingress_slot = reserve(
        request_command_done,
        config_.logic_scheduler_issue_ns,
        ingress_logic.ingress);
    const double logic_start = ingress_slot.start_ns;
    out.breakdown.ingress_queue_wait_ns = ingress_slot.wait_ns;
    double issued_ns = ingress_slot.finish_ns;
    out.breakdown.command_ns += config_.logic_scheduler_issue_ns;
    if (trace_spans != nullptr) {
        const auto entity = logic_entity(ingress_stack);
        add_trace_span(
            trace_spans,
            "logic_die_queue",
            "queue",
            entity,
            request_command_done,
            logic_start);
        add_trace_span(
            trace_spans,
            "logic_scheduler_issue",
            "logic",
            entity,
            logic_start,
            issued_ns);
    }
    out.start_ns = logic_start;
    const auto wait_for_page_stack = [&](std::size_t stack, double earliest_ns) {
        const double ready_ns = std::max(
            earliest_ns, state_ready_by_stack.at(stack));
        if (ready_ns > earliest_ns) {
            out.breakdown.scheduler_queue_wait_ns += ready_ns - earliest_ns;
            if (trace_spans != nullptr) {
                trace_wait(
                    trace_spans,
                    logic_entity(static_cast<std::uint32_t>(stack)),
                    earliest_ns,
                    ready_ns,
                    "wait_page_stack_state_dependency");
            }
        }
        return ready_ns;
    };

    if (request.op == Op::Read) {
        const auto first_lpn = physical_request ? 0 : request.addr / config_.page_size_bytes;
        const auto logical = physical_request ? decode(request.addr) : logical_page_address(request.addr);
        const auto pages = physical_request ? page_count_for(decode(request.addr), request.bytes) :
            page_count_for(logical, request.bytes);

        double finish_ns = issued_ns;
        std::uint64_t media_pages = 0;
        std::uint64_t read_buffer_pages = 0;
        std::uint64_t write_buffer_pages = 0;
        std::uint64_t erased_pages = 0;
        std::string first_path;
        if (pages > 1) {
            const double split_ready_ns = issued_ns;
            const auto split = reserve(
                split_ready_ns,
                config_.address_generation_ns,
                ingress_logic.ingress);
            out.breakdown.ingress_queue_wait_ns += split.wait_ns;
            if (trace_spans != nullptr) {
                const auto entity = logic_entity(ingress_stack);
                trace_wait(
                    trace_spans,
                    entity,
                    split_ready_ns,
                    split.start_ns,
                    "wait_read_split_ingress");
                add_trace_span(
                    trace_spans,
                    "read_split",
                    "logic",
                    entity,
                    split.start_ns,
                    split.finish_ns,
                    true,
                    std::to_string(pages) + " page transactions");
            }
            out.breakdown.address_mapping_ns += config_.address_generation_ns;
            // Address generation is real ingress work. Downstream page
            // transactions cannot observe the split before its actual
            // reservation (including any queueing) has completed.
            issued_ns = split.finish_ns;
            stats_.read_splits++;
            stats_.read_split_pages += pages;
        }
        const auto first_ppn = physical_request ?
            encode_ppn(decode(request.addr)) : 0;
        // Controller cache policy: a request larger than the read buffer
        // streams around it, whichever engine executes it and whatever the
        // geometry or observation settings are. Buffer hits/misses are
        // therefore a property of the workload, not of the acceleration.
        const bool streaming_read_buffer_bypass =
            config_.read_buffer_pages != 0 &&
            pages > config_.read_buffer_pages;
        const auto run_plan = plan_read_page_runs(
            request,
            request.address_space,
            first_lpn,
            first_ppn,
            pages,
            issued_ns,
            out.breakdown);
        std::uint64_t scalar_pages_owed = 0;
        for (const auto& scalar_segment : run_plan.scalar_segments) {
            scalar_pages_owed += scalar_segment.second;
        }
        if (run_plan.run_finish_ns) {
            finish_ns = std::max(finish_ns, *run_plan.run_finish_ns);
            media_pages = run_plan.run_pages;
            first_path = physical_request ?
                decode(request.addr).path() :
                "compact-striped-page-run";
        } else {
            stats_.scalar_read_requests = checked_add(
                stats_.scalar_read_requests,
                1,
                "HBF scalar read requests");
        }
        if (scalar_pages_owed != 0) {
            stats_.scalar_read_pages = checked_add(
                stats_.scalar_read_pages,
                scalar_pages_owed,
                "HBF scalar read pages");
        }
        // Request-granular thermal pacing: one pacing decision per touched
        // stack per scalar request, mirroring the page-run engine. Pages
        // served from buffers consume pacing budget conservatively here
        // (timing can only err slower) while heat deposits stay per actual
        // media page, so thermal_media_energy_j remains exact media work.
        std::vector<double> thermal_scalar_ready_ns;
        double thermal_budget_finish_ns = issued_ns;
        if (config_.thermal_enabled && scalar_pages_owed != 0) {
            thermal_scalar_ready_ns.assign(config_.stacks, issued_ns);
            std::vector<std::uint64_t> thermal_pages_by_stack(
                config_.stacks, 0);
            for (const auto& scalar_segment : run_plan.scalar_segments) {
                for (std::uint64_t i = scalar_segment.first;
                     i < scalar_segment.first + scalar_segment.second;
                     ++i) {
                    const auto stack = physical_request ?
                        stack_of_block(static_cast<std::size_t>(
                            (first_ppn + i) / config_.pages_per_block)) :
                        stack_for_lpn(first_lpn + i);
                    thermal_pages_by_stack.at(stack)++;
                }
            }
            for (std::size_t stack = 0;
                 stack < thermal_pages_by_stack.size();
                 ++stack) {
                const auto count = thermal_pages_by_stack[stack];
                if (count == 0) {
                    continue;
                }
                const auto admission = thermal_pace_media(
                    stack,
                    issued_ns,
                    static_cast<double>(count) * thermal_read_energy_j_,
                    count,
                    out.breakdown,
                    trace_spans);
                thermal_scalar_ready_ns[stack] = admission.ready_ns;
                thermal_budget_finish_ns = std::max(
                    thermal_budget_finish_ns, admission.budget_finish_ns);
            }
        }
        for (const auto& scalar_segment : run_plan.scalar_segments)
        for (std::uint64_t i = scalar_segment.first;
             i < scalar_segment.first + scalar_segment.second;
             ++i) {
            double page_ready_ns = issued_ns;
            const auto range_begin = i == 0 ? logical.offset : 0;
            const auto preceding_bytes = i == 0 ? std::uint64_t{0} :
                (config_.page_size_bytes - logical.offset) +
                    (i - 1) * config_.page_size_bytes;
            const auto range_bytes = std::min(
                request.bytes - preceding_bytes,
                config_.page_size_bytes - range_begin);
            const auto range_end = range_begin + range_bytes;
            const auto lpn = first_lpn + i;
            const auto page_stack = physical_request ?
                stack_of_block(static_cast<std::size_t>(
                    (encode_ppn(decode(request.addr)) + i) /
                    config_.pages_per_block)) :
                stack_for_lpn(lpn);
            if (!thermal_scalar_ready_ns.empty()) {
                page_ready_ns = std::max(
                    page_ready_ns, thermal_scalar_ready_ns[page_stack]);
            }
            page_ready_ns = wait_for_page_stack(page_stack, page_ready_ns);
            page_ready_ns = admit_foreground_page_read(
                page_stack,
                page_ready_ns,
                out.breakdown,
                trace_spans);
            const auto complete_page = [&](double page_finish_ns) {
                complete_foreground_page_read(page_stack, page_finish_ns);
                finish_ns = std::max(finish_ns, page_finish_ns);
            };
            std::optional<std::uint64_t> ppn;
            if (physical_request) {
                ppn = encode_ppn(decode(request.addr)) + i;
            } else if (static_request) {
                ppn = static_ppn_for_source_page(lpn);
            } else {
                auto& page_write_buffer = write_buffer(stack_for_lpn(lpn));
                const auto buffered = page_write_buffer.find(lpn);
                const std::vector<DirtyRange>* buffered_ranges = nullptr;
                double buffered_ready_ns = page_ready_ns;
                if (buffered != page_write_buffer.end()) {
                    buffered_ranges = &buffered->second.ranges;
                    buffered_ready_ns = buffered->second.ready_ns;
                } else if (const auto inflight = inflight_buffered_writes_.find(lpn);
                           inflight != inflight_buffered_writes_.end()) {
                    const InflightBufferedWrite* newest = nullptr;
                    for (const auto& generation : inflight->second) {
                        const auto target_block = static_cast<std::size_t>(
                            generation.target_ppn / config_.pages_per_block);
                        const auto& block = blocks_.at(target_block);
                        if (block.epoch == generation.target_block_epoch &&
                            !block.erase_pending &&
                            page_ready_ns < generation.commit_ns &&
                            (newest == nullptr || generation.generation > newest->generation)) {
                            newest = &generation;
                        }
                    }
                    if (newest != nullptr) {
                        buffered_ranges = &newest->ranges;
                        buffered_ready_ns = newest->ready_ns;
                    }
                }
                std::uint64_t buffered_overlap_bytes = 0;
                if (buffered_ranges != nullptr) {
                    for (const auto& range : *buffered_ranges) {
                        const auto lo = std::max(range.begin, range_begin);
                        const auto hi = std::min(range.end, range_end);
                        if (hi > lo) {
                            buffered_overlap_bytes += hi - lo;
                        }
                    }
                    // A dirty range elsewhere in the page is irrelevant to
                    // this sub-page read: it must not create a false WB hit,
                    // wait, overlay, or report classification.
                    if (buffered_overlap_bytes == 0) {
                        buffered_ranges = nullptr;
                    } else if (buffered_ready_ns > page_ready_ns) {
                        if (trace_spans != nullptr) {
                            trace_wait(
                                trace_spans,
                                logic_entity(static_cast<std::uint32_t>(
                                    stack_for_lpn(lpn))),
                                page_ready_ns,
                                buffered_ready_ns,
                                "wait_prior_write_buffer_stage");
                        }
                        out.breakdown.scheduler_queue_wait_ns +=
                            buffered_ready_ns - page_ready_ns;
                        page_ready_ns = buffered_ready_ns;
                    }
                }
                if (buffered_ranges != nullptr &&
                    dirty_ranges_cover(*buffered_ranges, range_begin, range_end)) {
                    complete_page(serve_read_from_write_buffer(
                        lpn,
                        range_begin,
                        range_end,
                        *buffered_ranges,
                        page_ready_ns,
                        out.breakdown,
                        trace_spans));
                    if (first_path.empty()) {
                        first_path = "write_buffer/lpn" + std::to_string(lpn);
                    }
                    write_buffer_pages++;
                    continue;
                }
                ppn = lookup_lpn(
                    lpn,
                    page_ready_ns,
                    out.breakdown,
                    trace_spans,
                    LookupIntent::ReadData);
                if (buffered_ranges != nullptr && ppn) {
                    if (first_path.empty()) {
                        first_path = decode_ppn(*ppn).path();
                    }
                    if (!streaming_read_buffer_bypass &&
                        config_.read_buffer_pages != 0 &&
                        read_buffer_contains(*ppn, page_ready_ns)) {
                        page_ready_ns = serve_read_from_read_buffer(
                            *ppn,
                            range_bytes,
                            page_ready_ns,
                            out.breakdown,
                            trace_spans,
                            false);
                        read_buffer_pages++;
                    } else {
                        stats_.read_buffer_misses++;
                        page_ready_ns = schedule_read_page(
                            *ppn,
                            page_ready_ns,
                            out.breakdown,
                            trace_spans,
                            TransactionSource::User,
                            request.heatmap_source,
                            ReadPayloadRoute::Internal,
                            0,
                            nullptr,
                            true);
                        if (!streaming_read_buffer_bypass) {
                            read_buffer_insert(*ppn, page_ready_ns);
                        }
                        media_pages++;
                    }
                    // The dirty bytes live in controller DRAM; reading them
                    // for the overlay uses the same pipelined DRAM resource
                    // the full-coverage path charges.
                    const auto overlay_detail = trace_spans == nullptr ?
                        std::string{} : "lpn" + std::to_string(lpn);
                    page_ready_ns = schedule_write_buffer_dram_access(
                        static_cast<std::uint32_t>(stack_for_lpn(lpn)),
                        buffered_overlap_bytes,
                        false,
                        page_ready_ns,
                        out.breakdown,
                        trace_spans,
                        "write_buffer_overlay_dram",
                        overlay_detail);
                    page_ready_ns = schedule_sram_transfer(
                        stack_for_lpn(lpn),
                        buffered_overlap_bytes,
                        page_ready_ns,
                        out.breakdown,
                        trace_spans,
                        "write_buffer_partial_overlay",
                        "dirty bytes over decoded backing page");
                    complete_page(schedule_external_read_egress(
                        stack_for_lpn(lpn),
                        range_bytes,
                        page_ready_ns,
                        out.breakdown,
                        trace_spans,
                        "write_buffer_partial_hbio_out",
                        "backing page + dirty overlay"));
                    stats_.write_buffer_read_hits++;
                    stats_.write_buffer_read_bytes += buffered_overlap_bytes;
                    write_buffer_pages++;
                    continue;
                } else if (buffered_ranges != nullptr) {
                    complete_page(serve_read_from_write_buffer(
                        lpn,
                        range_begin,
                        range_end,
                        *buffered_ranges,
                        page_ready_ns,
                        out.breakdown,
                        trace_spans));
                    if (first_path.empty()) {
                        first_path = "write_buffer+erased/lpn" +
                            std::to_string(lpn);
                    }
                    write_buffer_pages++;
                    continue;
                }
            }
            if (!ppn) {
                // NAND's erased value is the logical image of an unmapped
                // page. Return the exact requested decoded bytes through SRAM
                // and HBIO without inventing a flash/ECC operation.
                const auto stack = stack_for_lpn(lpn);
                page_ready_ns = schedule_sram_transfer(
                    stack,
                    range_bytes,
                    page_ready_ns,
                    out.breakdown,
                    trace_spans,
                    "unmapped_read_erased_fill",
                    "decoded erased-value payload");
                complete_page(schedule_external_read_egress(
                    stack,
                    range_bytes,
                    page_ready_ns,
                    out.breakdown,
                    trace_spans,
                    "unmapped_read_hbio_out",
                    "decoded erased-value payload"));
                if (first_path.empty()) {
                    first_path = "erased/lpn" + std::to_string(lpn);
                }
                erased_pages++;
                continue;
            }
            if (first_path.empty()) {
                first_path = decode_ppn(*ppn).path();
            }
            if (!streaming_read_buffer_bypass &&
                config_.read_buffer_pages != 0 &&
                read_buffer_contains(*ppn, page_ready_ns)) {
                complete_page(serve_read_from_read_buffer(
                    *ppn,
                    range_bytes,
                    page_ready_ns,
                    out.breakdown,
                    trace_spans,
                    true));
                read_buffer_pages++;
                continue;
            }
            stats_.read_buffer_misses++;
            double decoded_ready_ns = 0.0;
            const double page_finish = schedule_read_page(
                *ppn,
                page_ready_ns,
                out.breakdown,
                trace_spans,
                TransactionSource::User,
                request.heatmap_source,
                ReadPayloadRoute::External,
                range_bytes,
                &decoded_ready_ns,
                true);
            complete_page(page_finish);
            // The decoded page becomes cacheable when SRAM fill completes,
            // independently of how long its user's HBIO egress waits.
            if (!streaming_read_buffer_bypass) {
                read_buffer_insert(*ppn, decoded_ready_ns);
            }
            media_pages++;
        }
        // A paced request may not complete before its energy has been
        // dissipated at the pacing budget (same clamp as the run engine).
        finish_ns = std::max(finish_ns, thermal_budget_finish_ns);
        out.finish_ns = finish_ns;
        if (streaming_read_buffer_bypass) {
            // Under bypass, exactly the media pages streamed around the
            // buffer; write-buffer and erased fills never reached it.
            stats_.streaming_read_buffer_bypass_pages = checked_add(
                stats_.streaming_read_buffer_bypass_pages,
                media_pages,
                "HBF streaming read-buffer bypass pages");
        }
        if (run_plan.run_finish_ns && scalar_pages_owed == 0) {
            out.note = physical_request ?
                "static-physical-page-run-read" :
                static_request ? "static-extent-page-run-read" :
                    "compact-logical-page-run-read";
        } else if (run_plan.run_finish_ns) {
            out.note = "segmented-logical-page-run-read";
        } else if (write_buffer_pages > 0 && (media_pages > 0 || read_buffer_pages > 0)) {
            out.note = "mapped-read-with-write-buffer-overlay";
        } else if (write_buffer_pages > 0) {
            out.note = "write-buffer-read";
        } else if (erased_pages > 0 && media_pages == 0 && read_buffer_pages == 0) {
            out.note = pages == 1 ? "unmapped-erased-read" : "unmapped-erased-multi-page-read";
        } else if (erased_pages > 0) {
            out.note = "mixed-mapped-and-erased-read";
        } else if (read_buffer_pages > 0 && media_pages == 0) {
            out.note = pages == 1 ? "read-buffer-hit" : "read-buffer-multi-page-hit";
        } else {
            out.note = media_pages == 1 ? "mapped-page-read" : "mapped-multi-page-read";
        }
        if (physical_request) {
            out.resource_path = decode(request.addr).path();
        } else if (static_request) {
            out.resource_path = "static-page" + std::to_string(first_lpn) +
                "->" + decode_ppn(
                    static_ppn_for_source_page(first_lpn)).path();
        } else if (!first_path.empty()) {
            out.resource_path = "lpn" + std::to_string(first_lpn) + "->" + first_path;
        } else {
            out.resource_path = "lpn" + std::to_string(first_lpn) + "->unmapped";
        }

        stats_.read_requests++;
        if (request.address_space == AddressSpace::Logical) {
            stats_.logical_read_bytes += request.bytes;
        }
        stats_.physical_read_bytes += media_pages * config_.page_size_bytes;
        stats_.page_reads += media_pages;
    } else if (request.op == Op::Write) {
        const auto first_lpn = physical_request ? 0 : request.addr / config_.page_size_bytes;
        const auto logical = physical_request ? decode(request.addr) : logical_page_address(request.addr);
        const auto pages = physical_request ? page_count_for(decode(request.addr), request.bytes) :
            page_count_for(logical, request.bytes);

        const bool direct_mapping =
            config_.mapping_mode == MappingMode::Direct;
        if (!physical_request && !direct_mapping) {
            record_mutated_lpn_range(first_lpn, pages);
        }

        double finish_ns = issued_ns;
        std::uint64_t first_ppn = 0;
        std::string first_write_path;
        std::uint64_t remaining_bytes = request.bytes;
        for (std::uint64_t i = 0; i < pages; ++i) {
            double page_ready_ns = issued_ns;
            const auto lpn = first_lpn + i;
            const auto range_begin = i == 0 ? logical.offset : 0;
            const auto range_bytes = std::min(
                remaining_bytes, config_.page_size_bytes - range_begin);
            const auto range_end = range_begin + range_bytes;
            remaining_bytes -= range_bytes;
            const auto target_stack = physical_request ?
                stack_of_block(static_cast<std::size_t>(
                    (encode_ppn(decode(request.addr)) + i) /
                    config_.pages_per_block)) :
                stack_for_lpn(lpn);
            page_ready_ns = wait_for_page_stack(target_stack, page_ready_ns);
            const bool full_page_overwrite =
                range_begin == 0 && range_end == config_.page_size_bytes;
            const auto old_ppn = physical_request ?
                std::optional<std::uint64_t>{} :
                lookup_lpn(
                    lpn,
                    page_ready_ns,
                    out.breakdown,
                    trace_spans,
                    full_page_overwrite ?
                        LookupIntent::OverwriteFullPage :
                        LookupIntent::ReadData);
            // Physical writes name their target page directly; GC pressure
            // belongs to that page's stack, not to stack_for_lpn of a raw
            // physical address reinterpreted as an LPN.
            if (!physical_request && config_.write_coalescing_enabled) {
                double page_done = stage_write_buffer_range(
                    lpn,
                    range_begin,
                    range_end,
                    request.heatmap_source,
                    page_ready_ns,
                    out.breakdown,
                    trace_spans);
                if (config_.write_buffer_completion_requires_flush) {
                    flush_write_buffer_entry(lpn, page_done, out.breakdown, trace_spans);
                } else {
                    // Slot admission happens inside stage_write_buffer_range,
                    // before the new range becomes visible. Threshold flushes
                    // run on the device timeline in the background.
                    if (config_.write_buffer_flush_threshold_pages != 0) {
                        double background_ns = page_done;
                        const auto wb_stack = stack_for_lpn(lpn);
                        auto& die_lru = write_buffer_lru(wb_stack);
                        while (write_buffer(wb_stack).size() >=
                               config_.write_buffer_flush_threshold_pages) {
                            const auto victim_lpn = die_lru.back();
                            flush_write_buffer_entry(
                                victim_lpn,
                                background_ns,
                                out.breakdown,
                                trace_spans);
                        }
                        background_finish_ns_ = std::max(background_finish_ns_, background_ns);
                    }
                }
                if (first_write_path.empty()) {
                    first_write_path = "lpn" + std::to_string(first_lpn) + "->write_buffer";
                }
                finish_ns = std::max(finish_ns, page_done);
                continue;
            }
            if (!physical_request && old_ppn) {
                if (!full_page_overwrite) {
                    if (!direct_mapping) {
                        if (trace_spans != nullptr) {
                            add_trace_span(
                                trace_spans,
                                "partial_page_merge",
                                "translation",
                                logic_entity(ingress_stack),
                                page_ready_ns,
                                page_ready_ns + config_.mapping_update_ns,
                                true,
                                "lpn" + std::to_string(lpn));
                        }
                        out.breakdown.translation_ns +=
                            config_.mapping_update_ns;
                        page_ready_ns += config_.mapping_update_ns;
                    }
                    page_ready_ns = schedule_read_page(
                        *old_ppn,
                        page_ready_ns,
                        out.breakdown,
                        trace_spans,
                        TransactionSource::User,
                        request.heatmap_source,
                        ReadPayloadRoute::Internal,
                        0);
                    stats_.physical_read_bytes += config_.page_size_bytes;
                    stats_.page_reads++;
                }
            } else if (!full_page_overwrite) {
                // A new partial page has no old media image. NAND's erased
                // value supplies the untouched bytes explicitly; model the
                // logic-die SRAM fill instead of inventing a hidden read.
                page_ready_ns = schedule_sram_transfer(
                    target_stack,
                    config_.page_size_bytes - range_bytes,
                    page_ready_ns,
                    out.breakdown,
                    trace_spans,
                    "partial_page_erased_fill",
                    "erased-value fill before user overlay");
            }
            const auto ingress_detail = trace_spans == nullptr ?
                std::string{} :
                "request " + request.id + " page " + std::to_string(i);
            page_ready_ns = schedule_external_write_ingress(
                target_stack,
                range_bytes,
                page_ready_ns,
                out.breakdown,
                trace_spans,
                ingress_detail,
                "user/write_ingress_sram");
            if (!physical_request && !direct_mapping) {
                maybe_run_gc(
                    page_ready_ns,
                    out.breakdown,
                    trace_spans,
                    1,
                    target_stack,
                    BlockRole::Data);
            }

            if (direct_mapping && !physical_request && !old_ppn) {
                throw std::runtime_error(
                    "HBF direct-mode write targets an LPN outside the "
                    "populated compact image");
            }
            const auto new_ppn = physical_request ?
                encode_ppn(decode(request.addr)) + i :
                direct_mapping ? *old_ppn :
                allocate_free_page(
                    page_ready_ns,
                    out.breakdown,
                    trace_spans,
                    BlockRole::Data,
                    stack_for_lpn(lpn));
            if (i == 0) {
                first_ppn = new_ppn;
            }
            if (direct_mapping && !physical_request) {
                // An in-place program changes the page's contents under an
                // unchanged block epoch, so a decoded copy of the old image
                // in the logic-die read buffer must be retired when the
                // program is issued; a later read orders behind the program
                // completion and then misses to media.
                read_buffer_purge_page(new_ppn);
            }
            const double program_done = schedule_program_page(
                new_ppn,
                page_ready_ns,
                out.breakdown,
                trace_spans,
                TransactionSource::User,
                request.heatmap_source);
            double mapping_ready = program_done;
            double page_done = program_done;
            if (!physical_request && direct_mapping) {
                // In-place program over the exposed address space: no FTL
                // relocation, no mapping update, no persistence. A later
                // access of this LPN orders behind the program completion.
                auto& ready = direct_write_ready_ns_by_lpn_[lpn];
                ready = std::max(ready, program_done);
                direct_write_ready_expiry_.emplace(ready, lpn);
            } else if (!physical_request) {
                const auto program_commit_sequence = schedule_media_program_commit(
                    new_ppn,
                    lpn,
                    PageOwner::Logical,
                    program_done);
                access_mapping(
                    lpn,
                    TransactionSource::User,
                    MappingAccessKind::Update,
                    mapping_ready,
                    out.breakdown,
                    trace_spans);
                page_done = mapping_ready;
                schedule_lpn_mapping_commit(
                    lpn, new_ppn, page_done, program_commit_sequence);
                mark_mapping_page_dirty(
                    mapping_vpn_for_lpn(lpn), page_done);
                // The resident entry is authoritative immediately; its dirty
                // checkpoint page is persisted by the end-of-run drain.
            } else {
                schedule_physical_program_commit(new_ppn, page_done);
            }
            finish_ns = std::max(finish_ns, page_done);
        }
        out.finish_ns = finish_ns;
        out.resource_path = physical_request ? decode(request.addr).path() :
            (!first_write_path.empty() ? first_write_path :
                ("lpn" + std::to_string(first_lpn) + "->" + decode_ppn(first_ppn).path()));
        if (!physical_request && config_.write_coalescing_enabled) {
            out.note = pages == 1 ? "write-buffer-stage" : "multi-page-write-buffer-stage";
        } else {
            out.note = pages == 1 ? "page-program-map-update" : "multi-page-program-map-update";
        }

        stats_.program_requests++;
        if (!physical_request) {
            stats_.logical_write_bytes += request.bytes;
        }
        if (physical_request || !config_.write_coalescing_enabled) {
            const auto payload_bytes = pages * config_.page_size_bytes;
            stats_.physical_write_bytes += payload_bytes;
            stats_.data_program_payload_bytes += payload_bytes;
            stats_.data_programs += pages;
            stats_.page_programs += pages;
            if (physical_request) {
                stats_.raw_physical_program_payload_bytes += payload_bytes;
                stats_.raw_physical_programs += pages;
            }
        }
    } else if (request.op == Op::Erase) {
        const auto addr = decode(request.addr);
        const auto block = block_index(addr);
        out.finish_ns = schedule_erase_block(
            block,
            issued_ns,
            out.breakdown,
            trace_spans,
            TransactionSource::User,
            request.heatmap_source);
        out.resource_path = addr.path();
        out.note = "physical-block-erase";
        out.physical_bytes = 0;
        const auto [pending_erase, inserted] = pending_physical_erases_.emplace(
            block,
            PendingPhysicalErase{.finish_ns = out.finish_ns});
        if (!inserted) {
            throw std::runtime_error("HBF block already has an in-flight physical erase");
        }
        auto& block_state = blocks_.at(block);
        if (block_state.epoch == std::numeric_limits<std::uint64_t>::max()) {
            pending_physical_erases_.erase(block);
            throw std::runtime_error("HBF block epoch overflow");
        }
        // A raw erase may legally target an already-erased block. Such a
        // block is still present in the plane's free pool, so retire it from
        // allocation before publishing erase_pending. Otherwise a same-time
        // logical write can claim the block while its erase completion is
        // already scheduled; that completion then destroys the new page.
        auto& erased_plane = planes_.at(block_plane_index(block));
        if (block_state.role == BlockRole::Free) {
            const auto free_block = std::find(
                erased_plane.free_blocks.begin(),
                erased_plane.free_blocks.end(),
                block);
            if (free_block == erased_plane.free_blocks.end()) {
                pending_physical_erases_.erase(block);
                throw std::runtime_error(
                    "HBF free block targeted by erase is missing from its plane pool");
            }
            erased_plane.free_blocks.erase(free_block);
        }
        const auto retired_block_epoch = block_state.epoch;
        tag_pending_mapping_updates_for_erase(
            block, retired_block_epoch, out.finish_ns);
        // Capacity enforcement no longer scans every cache line looking for
        // stale epochs.  Retire this block's decoded lines at the same point
        // that the block becomes erase-pending; this is the visibility point
        // used by the former scan as well.
        read_buffer_purge_block(block);
        block_state.epoch++;
        block_state.erase_pending = true;
        const auto erase_epoch = block_state.epoch;
        if (erased_plane.active_data_block == block) {
            erased_plane.active_data_block = std::nullopt;
        }
        if (erased_plane.active_mapping_block == block) {
            erased_plane.active_mapping_block = std::nullopt;
        }
        if (erased_plane.active_gc_block == block) {
            erased_plane.active_gc_block = std::nullopt;
        }

        pending_erase->second.commit_sequence = schedule_commit(
            addr.stack, out.finish_ns, [this, block, erase_epoch]() {
            const auto pending_erase = pending_physical_erases_.find(block);
            if (pending_erase == pending_physical_erases_.end()) {
                throw std::runtime_error("HBF physical erase lost its in-flight reservation");
            }
            if (blocks_.at(block).epoch != erase_epoch ||
                !blocks_.at(block).erase_pending) {
                throw std::runtime_error(
                    "HBF physical erase lost its block-epoch ownership");
            }
            const auto block_begin = static_cast<std::uint64_t>(
                block) * config_.pages_per_block;
            for (std::uint32_t page = 0; page < config_.pages_per_block; ++page) {
                const auto ppn = block_begin + page;
                const auto it = programmed_pages_.find(ppn);
                if (it == programmed_pages_.end()) {
                    continue;
                }
                if (it->second.status == PageStatus::Valid) {
                    if (it->second.owner == PageOwner::Mapping) {
                        const auto mapping_vpn = metadata_vpn(it->second.lpn);
                        materialized_ready_by_vpn_[mapping_vpn] = std::max(
                            materialized_ready_by_vpn_[mapping_vpn],
                            pending_erase->second.finish_ns);
                        const auto found =
                            mapping_vpn_to_ppn_.find(mapping_vpn);
                        if (found != mapping_vpn_to_ppn_.end() && found->second == ppn) {
                            mapping_vpn_to_ppn_.erase(found);
                        }
                    } else if (it->second.owner == PageOwner::Logical) {
                        materialized_ready_by_lpn_[it->second.lpn] = std::max(
                            materialized_ready_by_lpn_[it->second.lpn],
                            pending_erase->second.finish_ns);
                        const auto found = lpn_to_ppn_.find(it->second.lpn);
                        if (found != lpn_to_ppn_.end() && found->second == ppn) {
                            lpn_to_ppn_.erase(found);
                        }
                    }
                }
            }
            reset_erased_block(block);
            retire_mapping_update_tombstones(
                block, pending_erase->second.finish_ns);
            materialized_ready_by_block_.at(block) = std::max(
                materialized_ready_by_block_.at(block),
                pending_erase->second.finish_ns);
            pending_physical_erases_.erase(pending_erase);
        });

        stats_.erase_requests++;
        stats_.block_erases++;
    }

    if (address_heatmap_ != nullptr && !physical_request &&
        (request.op == Op::Read || request.op == Op::Write)) {
        address_heatmap_->record(AddressTrafficRecord{
            .domain = AddressDomain::HbfLogical,
            .direction = request.op == Op::Read ?
                TrafficDirection::Read : TrafficDirection::Write,
            .source = request.heatmap_source,
            .address = request.addr,
            .bytes = request.bytes,
        });
    }
    stats_.free_pages = free_pages_;
    stats_.stage_work += out.breakdown;
    stats_.finish_ns = std::max({stats_.finish_ns, out.finish_ns, background_finish_ns_});
    if (request.op == Op::Read || request.op == Op::Write) {
        out.physical_bytes = stats_.physical_read_bytes + stats_.physical_write_bytes -
            physical_bytes_before;
    }
    return out;
}

std::size_t HbfDevice::stack_index(const HbfAddress& addr) const {
    return addr.stack;
}

std::size_t HbfDevice::channel_index(const HbfAddress& addr) const {
    return static_cast<std::size_t>(addr.stack) * config_.channels_per_stack + addr.channel;
}

std::size_t HbfDevice::die_index(const HbfAddress& addr) const {
    std::size_t index = addr.stack;
    index = index * config_.channels_per_stack + addr.channel;
    index = index * config_.dies_per_channel + addr.die;
    return index;
}

std::list<std::uint64_t>& HbfDevice::write_buffer_lru(std::size_t stack) {
    return write_buffer_lru_by_stack_.at(stack);
}

std::unordered_map<std::uint64_t, HbfDevice::WriteBufferEntry>& HbfDevice::write_buffer(
    std::size_t stack) {
    return write_buffer_by_stack_.at(stack);
}

HbfDevice::SenseRoundCandidate HbfDevice::preview_sense_round(
    PlaneState& plane,
    std::size_t subarray_index,
    double ready_ns) {
    const double t_read = config_.t_read_page_ns;
    // issue() accepts only nondecreasing arrivals, and every transaction it
    // creates is ready no earlier than its parent arrival. Therefore a round
    // before the current top-level arrival is causally unreachable by this or
    // any future issue. Reclaim only behind that proven watermark; using the
    // current command's ready time would be unsafe because a later-called
    // sibling can legitimately have an earlier ready time.
    const double causal_watermark_ns = last_issue_arrival_ns_.value_or(0.0);
    if (causal_watermark_ns > plane.sense_rounds_pruned_through_ns) {
        for (auto& available : plane.available_sense_rounds_by_subarray) {
            available.erase(available.begin(), available.lower_bound(causal_watermark_ns));
        }
        plane.sense_rounds_pruned_through_ns = causal_watermark_ns;
    }

    // Availability is indexed independently for each subarray because a
    // round has exactly one slot for each subarray. This is equivalent to a
    // used-subarray mask, but has no fixed history/width limit and provides
    // logarithmic earliest-round lookup.
    auto& subarray = plane.subarrays.at(subarray_index);
    subarray.timeline.prune_before(reservation_causal_watermark_ns_);
    plane.sense_round_calendar.prune_before(reservation_causal_watermark_ns_);
    auto& available = plane.available_sense_rounds_by_subarray.at(subarray_index);
    std::optional<double> join_start;
    for (auto round = available.lower_bound(ready_ns); round != available.end();) {
        const double start_ns = *round;
        if (subarray.timeline.can_reserve_exact(start_ns, t_read)) {
            join_start = start_ns;
            break;
        }
        // A full-plane barrier or an earlier same-subarray read consumed this
        // advertised slot. It can never become available again.
        round = available.erase(round);
    }

    double candidate = ready_ns;
    double new_round_start = 0.0;
    for (;;) {
        const double round_start = plane.sense_round_calendar.preview_start(
            candidate, t_read);
        const double subarray_start = subarray.timeline.preview_start(
            candidate, t_read);
        const double start = std::max(round_start, subarray_start);
        if (plane.sense_round_calendar.can_reserve_exact(start, t_read) &&
            subarray.timeline.can_reserve_exact(start, t_read)) {
            new_round_start = start;
            break;
        }
        if (start <= candidate) {
            throw std::runtime_error(
                "HBF batch-round preview failed to make forward progress");
        }
        candidate = start;
    }

    if (join_start && *join_start <= new_round_start) {
        return SenseRoundCandidate{
            .start_ns = *join_start,
            .joins_existing_round = true,
        };
    }
    return SenseRoundCandidate{
        .start_ns = new_round_start,
        .joins_existing_round = false,
    };
}

HbfDevice::ScheduledTransfer HbfDevice::commit_sense_round(
    PlaneState& plane,
    std::size_t subarray_index,
    double ready_ns,
    const SenseRoundCandidate& candidate) {
    const double t_read = config_.t_read_page_ns;
    auto& subarray = plane.subarrays.at(subarray_index);
    if (candidate.joins_existing_round) {
        auto& available = plane.available_sense_rounds_by_subarray.at(
            subarray_index);
        const auto found = available.find(candidate.start_ns);
        if (found == available.end()) {
            throw std::runtime_error("HBF batch round lost its subarray slot");
        }
        available.erase(found);
    } else {
        const auto round = reserve(
            candidate.start_ns, t_read, plane.sense_round_calendar);
        if (round.start_ns != candidate.start_ns) {
            throw std::runtime_error("HBF new batch round moved after preview");
        }
        for (std::size_t index = 0;
             index < plane.available_sense_rounds_by_subarray.size();
             ++index) {
            if (index != subarray_index) {
                plane.available_sense_rounds_by_subarray[index].insert(
                    candidate.start_ns);
            }
        }
    }
    const auto scheduled = reserve(
        candidate.start_ns, t_read, subarray.timeline);
    if (scheduled.start_ns != candidate.start_ns) {
        throw std::runtime_error("HBF subarray sense moved after preview");
    }
    return ScheduledTransfer{
        .start_ns = scheduled.start_ns,
        .finish_ns = scheduled.finish_ns,
        .wait_ns = std::max(0.0, scheduled.start_ns - ready_ns),
    };
}

double HbfDevice::preview_full_plane_window(
    PlaneState& plane,
    double earliest_ns,
    double duration_ns) {
    plane.sense_round_calendar.prune_before(reservation_causal_watermark_ns_);
    for (auto& subarray : plane.subarrays) {
        subarray.timeline.prune_before(reservation_causal_watermark_ns_);
    }
    for (auto& lane : plane.media_lanes) {
        lane.timeline.prune_before(reservation_causal_watermark_ns_);
    }
    for (auto& bank : plane.page_buffer_banks) {
        bank.timeline.prune_before(reservation_causal_watermark_ns_);
    }

    double candidate = earliest_ns;
    for (;;) {
        double next = plane.sense_round_calendar.preview_start(
            candidate, duration_ns);
        for (const auto& subarray : plane.subarrays) {
            next = std::max(
                next, subarray.timeline.preview_start(candidate, duration_ns));
        }
        for (const auto& lane : plane.media_lanes) {
            next = std::max(
                next, lane.timeline.preview_start(candidate, duration_ns));
        }
        for (const auto& bank : plane.page_buffer_banks) {
            next = std::max(
                next, bank.timeline.preview_start(candidate, duration_ns));
        }

        bool exact = plane.sense_round_calendar.can_reserve_exact(
            next, duration_ns);
        for (const auto& subarray : plane.subarrays) {
            exact = exact && subarray.timeline.can_reserve_exact(next, duration_ns);
        }
        for (const auto& lane : plane.media_lanes) {
            exact = exact && lane.timeline.can_reserve_exact(next, duration_ns);
        }
        for (const auto& bank : plane.page_buffer_banks) {
            exact = exact && bank.timeline.can_reserve_exact(next, duration_ns);
        }
        if (exact) {
            return next;
        }
        if (next <= candidate) {
            throw std::runtime_error(
                "HBF full-plane preview failed to make forward progress");
        }
        candidate = next;
    }
}

void HbfDevice::record_full_plane_window(
    PlaneState& plane,
    double begin_ns,
    double end_ns,
    std::uint32_t suspend_budget) {
    // Windows are disjoint and sorted, so the ones that end at or before
    // the causal watermark form a prefix that no future read can straddle.
    // Drop it once it dominates the vector (amortized O(1)).
    {
        auto& windows = plane.full_plane_windows;
        const auto dead_end = std::upper_bound(
            windows.begin(),
            windows.end(),
            reservation_causal_watermark_ns_,
            [](double watermark, const PlaneState::BusyWindow& window) {
                return watermark < window.end_ns;
            });
        const auto dead = static_cast<std::size_t>(dead_end - windows.begin());
        if (dead >= 64 && dead * 2 >= windows.size()) {
            windows.erase(windows.begin(), dead_end);
        }
    }
    const auto position = std::lower_bound(
        plane.full_plane_windows.begin(),
        plane.full_plane_windows.end(),
        begin_ns,
        [](const PlaneState::BusyWindow& window, double begin) {
            return window.begin_ns < begin;
        });
    if ((position != plane.full_plane_windows.begin() &&
         std::prev(position)->end_ns > begin_ns) ||
        (position != plane.full_plane_windows.end() &&
         position->begin_ns < end_ns)) {
        throw std::runtime_error("HBF full-plane windows overlap");
    }
    plane.full_plane_windows.insert(position, PlaneState::BusyWindow{
        .begin_ns = begin_ns,
        .end_ns = end_ns,
        .suspend_budget = suspend_budget,
        .next_suspend_ns = begin_ns,
    });
}

void HbfDevice::fold_ecc_inflight_intervals(
    DieState& die,
    double causal_arrival_watermark_ns) {
    if (die.ecc_inflight_intervals.empty()) {
        die.ecc_folded_through_ns = std::max(
            die.ecc_folded_through_ns, causal_arrival_watermark_ns);
        die.ecc_intervals_at_last_fold = 0;
        return;
    }
    // Sweep the retained intervals. Every interval alive at any time point
    // in (folded_through, watermark] is retained, because pruned intervals
    // all finished at or before the previous fold; the level at each event
    // time up to the watermark is therefore exact.
    std::vector<std::pair<double, int>> events;
    events.reserve(die.ecc_inflight_intervals.size() * 2);
    for (const auto& interval : die.ecc_inflight_intervals) {
        events.emplace_back(interval.start_ns, 1);
        events.emplace_back(interval.finish_ns, -1);
    }
    std::sort(events.begin(), events.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.first != rhs.first) {
            return lhs.first < rhs.first;
        }
        return lhs.second < rhs.second;
    });
    std::int64_t inflight = 0;
    for (const auto& [time_ns, delta] : events) {
        if (time_ns > causal_arrival_watermark_ns) {
            break;
        }
        inflight += delta;
        if (inflight < 0) {
            throw std::runtime_error(
                "HBF ECC in-flight accounting became negative while folding");
        }
        die.ecc_max_inflight_folded = std::max(
            die.ecc_max_inflight_folded, static_cast<std::uint64_t>(inflight));
    }
    std::erase_if(
        die.ecc_inflight_intervals,
        [causal_arrival_watermark_ns](const DieState::EccInflightInterval& interval) {
            return interval.finish_ns <= causal_arrival_watermark_ns;
        });
    die.ecc_folded_through_ns = std::max(
        die.ecc_folded_through_ns, causal_arrival_watermark_ns);
    die.ecc_intervals_at_last_fold = die.ecc_inflight_intervals.size();
}

std::uint64_t HbfDevice::ecc_max_inflight(const DieState& die) {
    // Retained intervals alone undercount the level at time points behind
    // the fold, but those points are covered by the folded peak, so the
    // maximum of the two is the exact all-time peak.
    std::vector<std::pair<double, int>> events;
    events.reserve(die.ecc_inflight_intervals.size() * 2);
    for (const auto& interval : die.ecc_inflight_intervals) {
        events.emplace_back(interval.start_ns, 1);
        events.emplace_back(interval.finish_ns, -1);
    }
    std::sort(events.begin(), events.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.first != rhs.first) {
            return lhs.first < rhs.first;
        }
        // A completion at t leaves before a new codeword enters at t.
        return lhs.second < rhs.second;
    });
    std::int64_t inflight = 0;
    std::uint64_t max_inflight = die.ecc_max_inflight_folded;
    for (const auto& [_, delta] : events) {
        inflight += delta;
        if (inflight < 0) {
            throw std::runtime_error(
                "HBF ECC in-flight accounting became negative");
        }
        max_inflight = std::max(
            max_inflight, static_cast<std::uint64_t>(inflight));
    }
    if (inflight != 0) {
        throw std::runtime_error(
            "HBF ECC in-flight accounting did not return to zero");
    }
    return max_inflight;
}

void HbfDevice::prune_expired_state(double causal_arrival_watermark_ns) {
    // ECC latency intervals: fold once a die has doubled its retained set
    // since the last fold and holds more than a small floor, so the sweep
    // cost is amortized over the codewords it retires.
    for (auto& die : dies_) {
        const auto retained = die.ecc_inflight_intervals.size();
        if (retained > 4096 && retained > 2 * die.ecc_intervals_at_last_fold) {
            fold_ecc_inflight_intervals(die, causal_arrival_watermark_ns);
        }
    }
    // Commit frontiers: every reader compares them against a time at or
    // after the watermark, so entries at or behind it are dead.
    const auto materialized_size = materialized_ready_by_lpn_.size() +
        materialized_ready_by_vpn_.size() + materialized_ready_by_ppn_.size();
    if (materialized_size > 1024 &&
        materialized_size > 2 * materialized_ready_prune_baseline_) {
        const auto expired = [causal_arrival_watermark_ns](const auto& entry) {
            return entry.second <= causal_arrival_watermark_ns;
        };
        std::erase_if(materialized_ready_by_lpn_, expired);
        std::erase_if(materialized_ready_by_vpn_, expired);
        std::erase_if(materialized_ready_by_ppn_, expired);
        materialized_ready_prune_baseline_ = materialized_ready_by_lpn_.size() +
            materialized_ready_by_vpn_.size() +
            materialized_ready_by_ppn_.size();
    }
}

void HbfDevice::prune_direct_write_ready(
    double causal_arrival_watermark_ns) {
    // issue() accepts only nondecreasing arrivals and every transaction it
    // creates is ready no earlier than its parent arrival, so an in-place
    // write whose completion lies at or behind the arrival watermark can
    // never again satisfy ready_ns > issued_ns for this or any future
    // request; erasing it is unobservable. A rewritten LPN leaves a stale
    // heap entry behind - the map holds the later completion, so the stale
    // pop skips the erase and the fresh entry expires it later.
    while (!direct_write_ready_expiry_.empty() &&
           direct_write_ready_expiry_.top().first <=
               causal_arrival_watermark_ns) {
        const auto lpn = direct_write_ready_expiry_.top().second;
        direct_write_ready_expiry_.pop();
        if (const auto pending = direct_write_ready_ns_by_lpn_.find(lpn);
            pending != direct_write_ready_ns_by_lpn_.end() &&
            pending->second <= causal_arrival_watermark_ns) {
            direct_write_ready_ns_by_lpn_.erase(pending);
        }
    }
}

void HbfDevice::record_plane_media_busy(
    PlaneState& plane,
    double begin_ns,
    double end_ns) {
    if (end_ns <= begin_ns) {
        return;
    }
    auto& windows = plane.media_busy_windows;
    // The union is sorted and disjoint; windows that end at or before the
    // causal watermark can never merge with a future reservation (every
    // future reservation starts at or after the watermark) and their busy
    // time is already accumulated. Drop that dead prefix once it dominates
    // the vector (amortized O(1)); the union total is unaffected.
    {
        const auto dead_end = std::upper_bound(
            windows.begin(),
            windows.end(),
            reservation_causal_watermark_ns_,
            [](double watermark, const PlaneState::BusyWindow& window) {
                return watermark < window.end_ns;
            });
        const auto dead = static_cast<std::size_t>(dead_end - windows.begin());
        if (dead >= 64 && dead * 2 >= windows.size()) {
            windows.erase(windows.begin(), dead_end);
        }
    }
    auto first = std::lower_bound(
        windows.begin(),
        windows.end(),
        begin_ns,
        [](const PlaneState::BusyWindow& window, double begin) {
            return window.end_ns < begin;
        });
    double merged_begin = begin_ns;
    double merged_end = end_ns;
    double replaced_ns = 0.0;
    auto last = first;
    while (last != windows.end() && last->begin_ns <= merged_end) {
        merged_begin = std::min(merged_begin, last->begin_ns);
        merged_end = std::max(merged_end, last->end_ns);
        replaced_ns += last->end_ns - last->begin_ns;
        ++last;
    }
    first = windows.erase(first, last);
    windows.insert(first, PlaneState::BusyWindow{
        .begin_ns = merged_begin,
        .end_ns = merged_end,
    });
    plane.media_busy_ns += merged_end - merged_begin - replaced_ns;
}

bool HbfDevice::read_buffer_contains(std::uint64_t ppn, double at_ns) {
    const auto block_index = static_cast<std::size_t>(
        ppn / config_.pages_per_block);
    const auto stack = stack_of_block(block_index);
    auto& logic_die = logic_dies_.at(stack);
    const double capacity_observation_ns =
        std::min(at_ns, state_observation_by_stack_.at(stack));
    enforce_read_buffer_capacity(logic_die, capacity_observation_ns);
    const auto found = logic_die.read_buffer.find(ppn);
    if (found == logic_die.read_buffer.end() ||
        found->second.block_epoch != blocks_.at(block_index).epoch ||
        blocks_.at(block_index).erase_pending) {
        if (found != logic_die.read_buffer.end()) {
            erase_read_buffer_entry(logic_die, ppn);
        }
        return false;
    }
    // The former capacity pass compacted every entry's history.  Compacting
    // the queried entry here retains the same temporal boundary while
    // avoiding O(cache_size) representation work on every lookup.
    prune_touch_history(found->second.touches, capacity_observation_ns);
    const double observed_ns = state_observation_by_stack_.at(stack);
    if (at_ns > observed_ns) {
        // Fills and touches that earlier-issued requests scheduled between
        // this request's causal observation point and its data-ready time
        // are exact state, not conjecture: the line may complete its own
        // fill inside that window (the request then simply waits for it),
        // and other fills may push the buffer over capacity. The former rule
        // declared a miss whenever any other fill completed in the window,
        // which discarded resident hot lines at every load level and cost a
        // full media read each time. The line is usable unless the exact
        // LRU controller would have taken it as a capacity victim by at_ns.
        if (read_buffer_evicted_before(logic_die, ppn, at_ns)) {
            return false;
        }
    }
    return true;
}

bool HbfDevice::read_buffer_evicted_before(
    const LogicDieState& logic_die,
    std::uint64_t ppn,
    double at_ns) const {
    if (!read_buffer_capacity_exceeded_at(logic_die, at_ns)) {
        return false;
    }
    // Lines ready by at_ns beyond capacity: the ready prefix holds exactly
    // capacity + 1 of them once it reports an excess, and every suffix
    // entry ready by at_ns is one more. The suffix is ordered by ready time,
    // so this walk costs O(excess).
    std::size_t excess = 1;
    for (const auto& ready_key : logic_die.read_buffer_ready_suffix) {
        if (ready_key.first > at_ns) {
            break;
        }
        ++excess;
    }
    // The controller takes victims from the least recently touched end. A
    // line whose latest touch lies beyond at_ns is pinned by an already
    // scheduled consumer and ends eviction, exactly as
    // enforce_read_buffer_capacity stops there.
    std::size_t taken = 0;
    for (const auto& lru_key : logic_die.read_buffer_lru) {
        if (taken == excess || lru_key.first.first > at_ns) {
            break;
        }
        if (lru_key.second == ppn) {
            return true;
        }
        ++taken;
    }
    return false;
}

void HbfDevice::rebalance_read_buffer_ready_index(LogicDieState& logic_die) {
    const auto total = logic_die.read_buffer_ready_prefix.size() +
        logic_die.read_buffer_ready_suffix.size();
    const auto capacity = config_.read_buffer_pages;
    const auto prefix_limit =
        capacity >= std::numeric_limits<std::size_t>::max() ?
        std::numeric_limits<std::size_t>::max() :
        static_cast<std::size_t>(capacity) + 1;
    const auto target = std::min(total, prefix_limit);

    while (logic_die.read_buffer_ready_prefix.size() > target) {
        const auto last = std::prev(logic_die.read_buffer_ready_prefix.end());
        logic_die.read_buffer_ready_suffix.insert(*last);
        logic_die.read_buffer_ready_prefix.erase(last);
    }
    while (logic_die.read_buffer_ready_prefix.size() < target &&
           !logic_die.read_buffer_ready_suffix.empty()) {
        const auto first = logic_die.read_buffer_ready_suffix.begin();
        logic_die.read_buffer_ready_prefix.insert(*first);
        logic_die.read_buffer_ready_suffix.erase(first);
    }
    while (!logic_die.read_buffer_ready_prefix.empty() &&
           !logic_die.read_buffer_ready_suffix.empty()) {
        const auto prefix_last =
            std::prev(logic_die.read_buffer_ready_prefix.end());
        const auto suffix_first = logic_die.read_buffer_ready_suffix.begin();
        if (*prefix_last <= *suffix_first) {
            break;
        }
        const auto low = *suffix_first;
        const auto high = *prefix_last;
        logic_die.read_buffer_ready_prefix.erase(prefix_last);
        logic_die.read_buffer_ready_suffix.erase(suffix_first);
        logic_die.read_buffer_ready_prefix.insert(low);
        logic_die.read_buffer_ready_suffix.insert(high);
    }
}

void HbfDevice::index_read_buffer_entry(
    LogicDieState& logic_die,
    std::uint64_t ppn) {
    const auto entry = logic_die.read_buffer.find(ppn);
    if (entry == logic_die.read_buffer.end() || entry->second.touches.empty()) {
        throw std::runtime_error(
            "HBF cannot index a missing or untouched read-buffer entry");
    }
    const auto ready_key = LogicDieState::ReadBufferReadyKey{
        entry->second.ready_ns, ppn};
    if (!logic_die.read_buffer_ready_prefix.insert(ready_key).second) {
        throw std::runtime_error("HBF duplicate read-buffer ready index entry");
    }
    const auto lru_key = LogicDieState::ReadBufferLruKey{
        *entry->second.touches.rbegin(), ppn};
    if (!logic_die.read_buffer_lru.insert(lru_key).second) {
        logic_die.read_buffer_ready_prefix.erase(ready_key);
        throw std::runtime_error("HBF duplicate read-buffer LRU index entry");
    }
    rebalance_read_buffer_ready_index(logic_die);
}

void HbfDevice::erase_read_buffer_entry(
    LogicDieState& logic_die,
    std::uint64_t ppn) {
    const auto entry = logic_die.read_buffer.find(ppn);
    if (entry == logic_die.read_buffer.end()) {
        return;
    }
    const auto ready_key = LogicDieState::ReadBufferReadyKey{
        entry->second.ready_ns, ppn};
    if (logic_die.read_buffer_ready_prefix.erase(ready_key) == 0 &&
        logic_die.read_buffer_ready_suffix.erase(ready_key) == 0) {
        throw std::runtime_error("HBF read-buffer ready index lost an entry");
    }
    const auto lru_key = LogicDieState::ReadBufferLruKey{
        *entry->second.touches.rbegin(), ppn};
    if (logic_die.read_buffer_lru.erase(lru_key) != 1) {
        throw std::runtime_error("HBF read-buffer LRU index lost an entry");
    }
    logic_die.read_buffer.erase(entry);
    rebalance_read_buffer_ready_index(logic_die);
}

void HbfDevice::touch_read_buffer_entry(
    LogicDieState& logic_die,
    std::uint64_t ppn,
    double at_ns) {
    const auto entry = logic_die.read_buffer.find(ppn);
    if (entry == logic_die.read_buffer.end() || entry->second.touches.empty()) {
        throw std::runtime_error("HBF cannot touch a missing read-buffer entry");
    }
    const auto old_key = LogicDieState::ReadBufferLruKey{
        *entry->second.touches.rbegin(), ppn};
    if (logic_die.read_buffer_lru.erase(old_key) != 1) {
        throw std::runtime_error("HBF read-buffer touch lost its LRU index");
    }
    entry->second.touches.emplace(at_ns, next_cache_touch_sequence_++);
    const auto new_key = LogicDieState::ReadBufferLruKey{
        *entry->second.touches.rbegin(), ppn};
    if (!logic_die.read_buffer_lru.insert(new_key).second) {
        throw std::runtime_error("HBF read-buffer touch duplicated its LRU index");
    }
}

bool HbfDevice::read_buffer_capacity_exceeded_at(
    const LogicDieState& logic_die,
    double at_ns) const {
    if (config_.read_buffer_pages >=
        std::numeric_limits<std::size_t>::max()) {
        return false;
    }
    const auto needed = static_cast<std::size_t>(config_.read_buffer_pages) + 1;
    return logic_die.read_buffer_ready_prefix.size() == needed &&
        std::prev(logic_die.read_buffer_ready_prefix.end())->first <= at_ns;
}

void HbfDevice::enforce_read_buffer_capacity(
    LogicDieState& logic_die,
    double at_ns) {
    while (read_buffer_capacity_exceeded_at(logic_die, at_ns)) {
        if (logic_die.read_buffer_lru.empty() ||
            logic_die.read_buffer_lru.begin()->first.first > at_ns) {
            // Every excess line has an already-scheduled future consumer.
            // Keep those lines pinned until that consumer occurs; evicting
            // now would invalidate a completion already returned to its
            // parent request.
            break;
        }
        erase_read_buffer_entry(
            logic_die, logic_die.read_buffer_lru.begin()->second);
    }
}

void HbfDevice::read_buffer_insert(std::uint64_t ppn, double ready_ns) {
    if (config_.read_buffer_pages == 0) {
        return;
    }
    auto& logic_die = logic_dies_.at(stack_of_block(
        static_cast<std::size_t>(ppn / config_.pages_per_block)));
    const auto block_index = static_cast<std::size_t>(
        ppn / config_.pages_per_block);
    const auto& block = blocks_.at(block_index);
    auto found = logic_die.read_buffer.find(ppn);
    if (found != logic_die.read_buffer.end() &&
        (found->second.block_epoch != block.epoch || block.erase_pending)) {
        erase_read_buffer_entry(logic_die, ppn);
        found = logic_die.read_buffer.end();
    }
    if (block.erase_pending) {
        return;
    }
    if (found != logic_die.read_buffer.end()) {
        const auto old_ready_key = LogicDieState::ReadBufferReadyKey{
            found->second.ready_ns, ppn};
        if (logic_die.read_buffer_ready_prefix.erase(old_ready_key) == 0 &&
            logic_die.read_buffer_ready_suffix.erase(old_ready_key) == 0) {
            throw std::runtime_error(
                "HBF read-buffer fill lost its ready index entry");
        }
        found->second.ready_ns = std::min(found->second.ready_ns, ready_ns);
        logic_die.read_buffer_ready_prefix.emplace(
            found->second.ready_ns, ppn);
        rebalance_read_buffer_ready_index(logic_die);
        touch_read_buffer_entry(logic_die, ppn, ready_ns);
        return;
    }
    LogicDieState::ReadBufferEntry entry{
        .ready_ns = ready_ns,
        .block_epoch = block.epoch,
    };
    entry.touches.emplace(ready_ns, next_cache_touch_sequence_++);
    logic_die.read_buffer.emplace(ppn, std::move(entry));
    index_read_buffer_entry(logic_die, ppn);
    // A future fill is an in-flight merge target, not a resident cache line.
    // Evicting at its future ready time here mutates present state in host
    // call order (A/B/A at the same arrival loses A before either fill). The
    // capacity check is applied only when a lookup/access actually reaches a
    // time at which those fills are ready.
}

void HbfDevice::read_buffer_purge_page(std::uint64_t ppn) {
    const auto block_index = static_cast<std::size_t>(
        ppn / config_.pages_per_block);
    auto& logic_die = logic_dies_.at(stack_of_block(block_index));
    erase_read_buffer_entry(logic_die, ppn);
}

void HbfDevice::read_buffer_purge_block(std::size_t block_index) {
    // PPNs are reused after erase+program; stale data must not be served.
    auto& logic_die = logic_dies_.at(stack_of_block(block_index));
    const auto begin = static_cast<std::uint64_t>(block_index) * config_.pages_per_block;
    const auto end = begin + config_.pages_per_block;
    for (auto ppn = begin; ppn < end; ++ppn) {
        const auto found = logic_die.read_buffer.find(ppn);
        if (found != logic_die.read_buffer.end()) {
            erase_read_buffer_entry(logic_die, ppn);
        }
    }
}

double HbfDevice::serve_read_from_read_buffer(
    std::uint64_t ppn,
    std::uint64_t bytes,
    double earliest_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    bool external_egress) {
    const auto stack = static_cast<std::uint32_t>(stack_of_block(
        static_cast<std::size_t>(ppn / config_.pages_per_block)));
    auto& logic_die = logic_dies_.at(stack);
    const auto buffered = logic_die.read_buffer.find(ppn);
    if (buffered == logic_die.read_buffer.end()) {
        throw std::runtime_error("HBF attempted to serve a missing read-buffer entry");
    }
    const auto block_index = static_cast<std::size_t>(
        ppn / config_.pages_per_block);
    if (buffered->second.block_epoch != blocks_.at(block_index).epoch ||
        blocks_.at(block_index).erase_pending) {
        throw std::runtime_error(
            "HBF attempted to serve a stale read-buffer epoch");
    }
    earliest_ns = std::max(earliest_ns, buffered->second.ready_ns);
    if (bytes == 0 || bytes > config_.page_size_bytes) {
        throw std::runtime_error("HBF read-buffer transfer size is invalid");
    }
    const double sram_ns = transfer_time_ns(bytes, config_.logic_sram_bandwidth_GBps);
    auto sram = reserve(earliest_ns, sram_ns, logic_die.sram);
    touch_read_buffer_entry(logic_die, ppn, sram.start_ns);
    breakdown.scheduler_queue_wait_ns += sram.wait_ns;
    breakdown.sram_staging_ns += sram_ns;
    if (spans != nullptr) {
        trace_wait(
            spans, logic_entity(stack), earliest_ns, sram.start_ns, "wait_sram");
        add_trace_span(
            spans,
            "read_buffer_hit",
            "sram",
            logic_entity(stack),
            sram.start_ns,
            sram.finish_ns,
            true,
            "ppn" + std::to_string(ppn));
    }
    stats_.read_buffer_hits++;
    stats_.read_buffer_read_bytes += bytes;
    if (!external_egress) {
        return sram.finish_ns;
    }
    // The hit leaves over the external interface as requested decoded bytes;
    // OOB never appears beyond ECC/SRAM.
    return schedule_external_read_egress(
        stack,
        bytes,
        sram.finish_ns,
        breakdown,
        spans,
        "read_buffer_hit_hbio_out",
        "read-buffer hit");
}

std::size_t HbfDevice::stack_for_lpn(std::uint64_t lpn) const {
    // Page-granular striping keeps a sequential window active on every HBF
    // stack. mapping_vpn_for_lpn() still groups each stack's local sequence
    // into its own mapping pages, so metadata never needs a cross-stack
    // lookup. This deliberately separates data striping from metadata
    // ownership; binding 512 consecutive global pages to one mapping page
    // serialized an entire W512 stream onto one stack.
    return stack_for_vpn(mapping_vpn_for_lpn(lpn));
}

std::size_t HbfDevice::stack_for_vpn(std::uint64_t mapping_vpn) const {
    return static_cast<std::size_t>(mapping_vpn % config_.stacks);
}

std::size_t HbfDevice::planes_per_stack() const {
    return planes_.size() / config_.stacks;
}

std::size_t HbfDevice::stack_of_plane(std::size_t plane) const {
    return plane / planes_per_stack();
}

std::size_t HbfDevice::stack_of_block(std::size_t block_index) const {
    return stack_of_plane(block_plane_index(block_index));
}

std::size_t HbfDevice::mapping_plane_for_vpn(std::uint64_t mapping_vpn) const {
    // A VPN is encoded as group * stacks + owner_stack. Divide the owner out
    // before selecting a local plane so every stack uses all of its planes.
    return stack_for_vpn(mapping_vpn) * planes_per_stack() +
        static_cast<std::size_t>(
            (mapping_vpn / config_.stacks) % planes_per_stack());
}

std::size_t HbfDevice::plane_index(const HbfAddress& addr) const {
    std::size_t index = die_index(addr);
    index = index * config_.planes_per_die + addr.plane;
    return index;
}

std::size_t HbfDevice::block_index(const HbfAddress& addr) const {
    std::size_t index = plane_index(addr);
    index = index * config_.blocks_per_plane + addr.block;
    return index;
}

std::size_t HbfDevice::block_plane_index(std::size_t block_index) const {
    return block_index / config_.blocks_per_plane;
}

std::size_t HbfDevice::subarray_index(const HbfAddress& addr) const {
    const auto page_in_plane =
        static_cast<std::uint64_t>(addr.block) * config_.pages_per_block + addr.page;
    return static_cast<std::size_t>(page_in_plane % subarrays_per_plane_);
}

std::size_t HbfDevice::media_lane_index(const HbfAddress& addr) const {
    const auto page_in_plane =
        static_cast<std::uint64_t>(addr.block) * config_.pages_per_block + addr.page;
    return static_cast<std::size_t>(page_in_plane % config_.media_lanes_per_plane);
}

std::size_t HbfDevice::page_buffer_bank_index(const HbfAddress& addr) const {
    const auto page_in_plane =
        static_cast<std::uint64_t>(addr.block) * config_.pages_per_block + addr.page;
    return static_cast<std::size_t>(page_in_plane % config_.page_buffer_banks_per_plane);
}

std::uint64_t HbfDevice::page_count_for(const HbfAddress& addr, std::uint64_t bytes) const {
    if (bytes == 0) {
        return 0;
    }
    const auto first_page_bytes = config_.page_size_bytes - addr.offset;
    if (bytes <= first_page_bytes) {
        return 1;
    }
    return 1 + div_ceil(bytes - first_page_bytes, config_.page_size_bytes);
}

HbfAddress HbfDevice::decode_ppn(std::uint64_t ppn) const {
    if (ppn >= total_pages_) {
        throw std::runtime_error("HBF PPN is out of range");
    }
    return decode(ppn * config_.page_size_bytes);
}

std::uint64_t HbfDevice::encode_ppn(const HbfAddress& addr) const {
    if (addr.stack >= config_.stacks || addr.channel >= config_.channels_per_stack ||
        addr.die >= config_.dies_per_channel || addr.plane >= config_.planes_per_die ||
        addr.block >= config_.blocks_per_plane || addr.page >= config_.pages_per_block ||
        addr.offset >= config_.page_size_bytes) {
        throw std::runtime_error("HBF address field out of range");
    }
    std::uint64_t unit = addr.stack;
    unit = unit * config_.channels_per_stack + addr.channel;
    unit = unit * config_.dies_per_channel + addr.die;
    unit = unit * config_.planes_per_die + addr.plane;
    unit = unit * config_.blocks_per_plane + addr.block;
    unit = unit * config_.pages_per_block + addr.page;
    return unit;
}

HbfAddress HbfDevice::logical_page_address(std::uint64_t logical_byte_addr) const {
    HbfAddress addr;
    addr.offset = logical_byte_addr % config_.page_size_bytes;
    return addr;
}

std::uint64_t HbfDevice::mapping_vpn_for_lpn(std::uint64_t lpn) const {
    // One global stripe contains exactly one page per stack. Rotate the lane
    // assignment once per mapping group to prevent a power-of-two logical
    // stride from pinning a tensor/KV phase to one stack, then encode the
    // owner directly in the low VPN digit:
    //
    //   vpn = local_mapping_group * stacks + owner_stack
    //
    // Each VPN therefore owns mapping_entries_per_page consecutive entries
    // from one stack's local page sequence while adjacent global LPNs remain
    // page-striped across the full fabric.
    const auto stacks = static_cast<std::uint64_t>(config_.stacks);
    const auto stripe = lpn / stacks;
    const auto lane = lpn % stacks;
    const auto group = stripe / config_.mapping_entries_per_page;
    const auto rotation = placement_mix64(group) % stacks;
    const auto stack = lane >= stacks - rotation ?
        lane - (stacks - rotation) :
        lane + rotation;
    return checked_add(
        checked_mul(group, stacks, "HBF mapping VPN group"),
        stack,
        "HBF mapping VPN");
}

std::optional<std::uint64_t> HbfDevice::compact_lpn_ppn(
    std::uint64_t lpn) const {
    if (!compact_logical_image_ ||
        lpn < compact_logical_image_->first_lpn ||
        lpn - compact_logical_image_->first_lpn >=
            compact_logical_image_->page_count) {
        return std::nullopt;
    }
    const auto& image = *compact_logical_image_;
    if (image.retired_lpns.contains(lpn)) {
        return std::nullopt;
    }
    const auto mapping_vpn = mapping_vpn_for_lpn(lpn);
    const auto stack = stack_for_vpn(mapping_vpn);
    if (mapping_vpn < image.first_vpn ||
        mapping_vpn - image.first_vpn >= image.vpn_slot_count) {
        throw std::runtime_error(
            "HBF compact LPN maps outside its VPN directory");
    }
    const auto vpn_offset = mapping_vpn - image.first_vpn;
    const auto& range = image.vpn_ranges.at(
        static_cast<std::size_t>(vpn_offset));
    const auto local_entry =
        (lpn / config_.stacks) % config_.mapping_entries_per_page;
    if (local_entry < range.first_entry ||
        local_entry - range.first_entry >= range.page_count) {
        throw std::runtime_error(
            "HBF compact LPN is outside its mapping-page range");
    }
    const auto data_index = checked_add(
        range.stack_page_offset,
        local_entry - range.first_entry,
        "HBF compact per-stack data index");
    const auto pps = static_cast<std::uint64_t>(planes_per_stack());
    const auto plane = stack * planes_per_stack() +
        static_cast<std::size_t>(data_index % pps);
    const auto page_ordinal = data_index / pps;
    const auto block_ordinal = page_ordinal / config_.pages_per_block;
    const auto page = static_cast<std::uint32_t>(
        page_ordinal % config_.pages_per_block);
    const auto& assigned_blocks = image.data_blocks_by_plane.at(plane);
    if (block_ordinal >= assigned_blocks.size()) {
        throw std::runtime_error(
            "HBF compact LPN maps beyond its data-block directory");
    }
    const auto block_index = assigned_blocks[block_ordinal];
    const auto& block = blocks_.at(static_cast<std::size_t>(block_index));
    if (block.role != BlockRole::Data || block.erase_pending ||
        !block.is_valid(page)) {
        throw std::runtime_error(
            "HBF compact LPN maps to a non-live data page");
    }
    return block_index * config_.pages_per_block + page;
}

std::optional<std::uint64_t> HbfDevice::compact_mapping_ppn(
    std::uint64_t mapping_vpn) const {
    if (!compact_logical_image_ ||
        mapping_vpn < compact_logical_image_->first_vpn ||
        mapping_vpn - compact_logical_image_->first_vpn >=
            compact_logical_image_->vpn_slot_count) {
        return std::nullopt;
    }
    if (compact_logical_image_->retired_mapping_vpns.contains(
            mapping_vpn)) {
        return std::nullopt;
    }
    const auto offset = mapping_vpn - compact_logical_image_->first_vpn;
    const auto compact_ppn = compact_logical_image_->mapping_ppns.at(
        static_cast<std::size_t>(offset));
    if (!compact_ppn) {
        return std::nullopt;
    }
    const auto ppn = *compact_ppn;
    const auto block_index = static_cast<std::size_t>(
        ppn / config_.pages_per_block);
    const auto page = static_cast<std::uint32_t>(
        ppn % config_.pages_per_block);
    const auto& block = blocks_.at(block_index);
    if (block.role != BlockRole::Mapping || block.erase_pending ||
        !block.is_valid(page)) {
        throw std::runtime_error(
            "HBF compact VPN maps to a non-live mapping page");
    }
    return ppn;
}

std::optional<std::uint64_t> HbfDevice::compact_lpn_for_ppn(
    std::uint64_t ppn) const {
    if (!compact_logical_image_ || ppn >= total_pages_) {
        return std::nullopt;
    }
    const auto& image = *compact_logical_image_;
    const auto block_index = ppn / config_.pages_per_block;
    const auto location = image.data_block_locations.find(block_index);
    if (location == image.data_block_locations.end()) {
        return std::nullopt;
    }
    const auto pps = static_cast<std::uint64_t>(planes_per_stack());
    const auto plane = location->second.plane;
    const auto stack = stack_of_plane(plane);
    const auto local_plane =
        static_cast<std::uint64_t>(plane - stack * planes_per_stack());
    const auto page_ordinal = checked_add(
        checked_mul(
            location->second.block_ordinal,
            config_.pages_per_block,
            "HBF compact inverse block ordinal"),
        ppn % config_.pages_per_block,
        "HBF compact inverse page ordinal");
    const auto data_index = checked_add(
        checked_mul(
            page_ordinal,
            pps,
            "HBF compact inverse striped data index"),
        local_plane,
        "HBF compact inverse data index");
    const auto& offsets = image.vpn_offsets_by_stack.at(stack);
    std::size_t lower = 0;
    std::size_t upper = offsets.size();
    while (lower < upper) {
        const auto middle = lower + (upper - lower) / 2;
        const auto& range = image.vpn_ranges.at(
            static_cast<std::size_t>(offsets[middle]));
        if (range.stack_page_offset <= data_index) {
            lower = middle + 1;
        } else {
            upper = middle;
        }
    }
    if (lower == 0) {
        return std::nullopt;
    }
    const auto vpn_offset = offsets[lower - 1];
    const auto& range = image.vpn_ranges.at(
        static_cast<std::size_t>(vpn_offset));
    if (data_index - range.stack_page_offset >= range.page_count) {
        return std::nullopt;
    }
    const auto mapping_vpn = checked_add(
        image.first_vpn,
        vpn_offset,
        "HBF compact inverse mapping VPN");
    const auto stacks = static_cast<std::uint64_t>(config_.stacks);
    const auto group = mapping_vpn / stacks;
    const auto rotation = placement_mix64(group) % stacks;
    const auto lane = static_cast<std::uint64_t>(stack) >= rotation ?
        static_cast<std::uint64_t>(stack) - rotation :
        stacks - (rotation - static_cast<std::uint64_t>(stack));
    const auto local_entry = checked_add(
        range.first_entry,
        data_index - range.stack_page_offset,
        "HBF compact inverse local entry");
    const auto stripe = checked_add(
        checked_mul(
            group,
            config_.mapping_entries_per_page,
            "HBF compact inverse mapping stripe"),
        local_entry,
        "HBF compact inverse stripe");
    const auto lpn = checked_add(
        checked_mul(stripe, stacks, "HBF compact inverse LPN stripe"),
        lane,
        "HBF compact inverse LPN");
    if (lpn < image.first_lpn ||
        lpn - image.first_lpn >= image.page_count ||
        image.retired_lpns.contains(lpn)) {
        return std::nullopt;
    }
    const auto forward = compact_lpn_ppn(lpn);
    if (!forward || *forward != ppn) {
        throw std::runtime_error(
            "HBF compact forward/inverse data mapping diverged");
    }
    return lpn;
}

std::optional<HbfDevice::CompactPageIdentity>
HbfDevice::compact_page_identity(std::uint64_t ppn) const {
    if (!compact_logical_image_) {
        return std::nullopt;
    }
    const auto& image = *compact_logical_image_;
    if (const auto mapping = image.mapping_vpn_by_ppn.find(ppn);
        mapping != image.mapping_vpn_by_ppn.end() &&
        !image.retired_mapping_vpns.contains(mapping->second)) {
        const auto forward = compact_mapping_ppn(mapping->second);
        if (!forward || *forward != ppn) {
            throw std::runtime_error(
                "HBF compact forward/inverse mapping-page directory "
                "diverged");
        }
        return CompactPageIdentity{
            .logical_key = metadata_lpn(mapping->second),
            .owner = PageOwner::Mapping,
        };
    }
    if (const auto lpn = compact_lpn_for_ppn(ppn)) {
        return CompactPageIdentity{
            .logical_key = *lpn,
            .owner = PageOwner::Logical,
        };
    }
    return std::nullopt;
}

void HbfDevice::retire_compact_page(
    std::uint64_t logical_key,
    PageOwner owner) {
    if (!compact_logical_image_ ||
        !compact_logical_image_->mutable_image) {
        throw std::runtime_error(
            "HBF cannot retire a non-mutable compact logical page");
    }
    auto& image = *compact_logical_image_;
    std::optional<std::uint64_t> ppn;
    std::unordered_map<std::uint64_t, std::uint32_t>* live_pages = nullptr;
    if (owner == PageOwner::Logical) {
        ppn = compact_lpn_ppn(logical_key);
        if (!ppn || !image.retired_lpns.insert(logical_key).second) {
            throw std::runtime_error(
                "HBF compact logical page was already retired");
        }
        record_mutated_lpn_range(logical_key, 1);
        live_pages = &image.live_data_pages_by_block;
    } else if (owner == PageOwner::Mapping &&
               is_metadata_lpn(logical_key)) {
        const auto mapping_vpn = metadata_vpn(logical_key);
        ppn = compact_mapping_ppn(mapping_vpn);
        if (!ppn ||
            !image.retired_mapping_vpns.insert(mapping_vpn).second) {
            throw std::runtime_error(
                "HBF compact mapping page was already retired");
        }
        live_pages = &image.live_mapping_pages_by_block;
    } else {
        throw std::runtime_error(
            "HBF compact retirement requires logical or mapping ownership");
    }

    const auto block_index = *ppn / config_.pages_per_block;
    auto live = live_pages->find(block_index);
    if (live == live_pages->end() || live->second == 0) {
        throw std::runtime_error(
            "HBF compact retirement lost its live block count");
    }
    if (--live->second == 0) {
        live_pages->erase(live);
    }
    const auto block_epoch = blocks_.at(
        static_cast<std::size_t>(block_index)).epoch;
    const bool inserted = programmed_pages_.emplace(
        *ppn,
        PageState{
            .status = PageStatus::Valid,
            .owner = owner,
            .lpn = logical_key,
            .block_epoch = block_epoch,
        }).second;
    if (!inserted) {
        throw std::runtime_error(
            "HBF compact retirement collided with materialized page state");
    }
    invalidate_ppn(*ppn);
}

std::uint64_t HbfDevice::logical_mapping_entry_count() const {
    std::uint64_t entries = lpn_to_ppn_.size();
    if (compact_logical_image_) {
        const auto compact_live =
            compact_logical_image_->page_count -
            compact_logical_image_->retired_lpns.size();
        entries = checked_add(
            entries, compact_live, "HBF compact logical mapping entries");
    }
    for (const auto& [lpn, updates] : pending_lpn_updates_) {
        const bool has_live_update = std::any_of(
            updates.begin(), updates.end(), [this](const PendingMappingUpdate& update) {
                const auto block_index = static_cast<std::size_t>(
                    update.new_ppn / config_.pages_per_block);
                return blocks_.at(block_index).epoch == update.block_epoch &&
                    !blocks_.at(block_index).erase_pending;
            });
        if (has_live_update &&
            !lpn_to_ppn_.contains(lpn) &&
            !compact_lpn_ppn(lpn)) {
            ++entries;
        }
    }
    return entries;
}

std::size_t HbfDevice::source_index(TransactionSource source) const {
    switch (source) {
    case TransactionSource::User:
        return 0;
    case TransactionSource::Mapping:
        return 1;
    case TransactionSource::GC:
        return 2;
    case TransactionSource::Prepopulate:
        return 3;
    }
    throw std::runtime_error("unknown HBF transaction source");
}

std::string HbfDevice::source_name(TransactionSource source) const {
    switch (source) {
    case TransactionSource::User:
        return "user";
    case TransactionSource::Mapping:
        return "mapping";
    case TransactionSource::GC:
        return "gc";
    case TransactionSource::Prepopulate:
        return "prepopulate";
    }
    throw std::runtime_error("unknown HBF transaction source");
}

HeatmapTrafficSource HbfDevice::resolve_heatmap_source(
    TransactionSource source,
    HeatmapTrafficSource attribution) const {
    const auto attribution_index = static_cast<std::size_t>(attribution);
    if (attribution_index >= kHeatmapTrafficSourceCount) {
        throw std::runtime_error("invalid HBF heatmap traffic source");
    }

    HeatmapTrafficSource expected = attribution;
    switch (source) {
    case TransactionSource::User:
        return attribution;
    case TransactionSource::Mapping:
        expected = HeatmapTrafficSource::Mapping;
        break;
    case TransactionSource::GC:
        expected = HeatmapTrafficSource::GarbageCollection;
        break;
    case TransactionSource::Prepopulate:
        expected = HeatmapTrafficSource::Prepopulate;
        break;
    default:
        throw std::runtime_error("unknown HBF transaction source");
    }
    if (attribution != expected) {
        throw std::runtime_error(
            "HBF transaction source and heatmap attribution disagree");
    }
    return attribution;
}

std::string HbfDevice::kind_name(TransactionKind kind) const {
    switch (kind) {
    case TransactionKind::Read:
        return "read";
    case TransactionKind::Program:
        return "program";
    case TransactionKind::Erase:
        return "erase";
    }
    throw std::runtime_error("unknown HBF transaction kind");
}

std::string HbfDevice::role_name(BlockRole role) const {
    switch (role) {
    case BlockRole::Free:
        return "free";
    case BlockRole::StaticReadOnly:
        return "static-read-only";
    case BlockRole::RawPhysical:
        return "raw-physical";
    case BlockRole::Data:
        return "data";
    case BlockRole::Mapping:
        return "mapping";
    case BlockRole::GC:
        return "gc";
    }
    throw std::runtime_error("unknown HBF block role");
}

void HbfDevice::flush_all_dirty_mapping_pages(
    double& at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    std::size_t wave = 0;
    while (!dirty_mapping_vpns_.empty()) {
        std::vector<std::uint64_t> dirty{
            dirty_mapping_vpns_.begin(), dirty_mapping_vpns_.end()};
        std::sort(dirty.begin(), dirty.end());
        // Fan out each wave like the write-buffer drain: media reservations
        // provide serialization. A mapping-page allocation may itself invoke
        // GC, whose data relocations dirty new mapping generations. Repeat
        // until those generations are also durable in this same drain call.
        const double drain_start_ns = at_ns;
        double drain_finish_ns = at_ns;
        for (const auto mapping_vpn : dirty) {
            const auto events = pending_dirty_mapping_events_.find(mapping_vpn);
            if (events == pending_dirty_mapping_events_.end() ||
                events->second.empty()) {
                clear_mapping_page_dirty(mapping_vpn);
                continue;
            }
            double entry_ns = std::max(
                drain_start_ns, events->second.rbegin()->first);
            flush_dirty_mapping_page(
                mapping_vpn, entry_ns, breakdown, spans);
            drain_finish_ns = std::max(drain_finish_ns, entry_ns);
        }
        at_ns = drain_finish_ns;
        if (++wave > blocks_.size() + 1) {
            throw std::runtime_error(
                "HBF mapping drain failed to converge after GC-generated updates");
        }
    }
}

void HbfDevice::mark_mapping_page_dirty(
    std::uint64_t mapping_vpn,
    double at_ns) {
    const auto sequence = next_cache_touch_sequence_++;
    auto& events = pending_dirty_mapping_events_[mapping_vpn];
    events.emplace(at_ns, sequence);
    // A checkpoint snapshots the newest event at or before its own time,
    // and every later query time is at or after the causal watermark, so
    // only the newest event at or behind the watermark and the events
    // beyond it can ever be selected. Older ones are unobservable; without
    // this, full-resident mapping kept one node per page write between
    // drains.
    const auto first_future = events.upper_bound(TemporalTouch{
        reservation_causal_watermark_ns_,
        std::numeric_limits<std::uint64_t>::max()});
    if (first_future != events.begin()) {
        events.erase(events.begin(), std::prev(first_future));
    }
    set_mapping_page_dirty(mapping_vpn);
}

void HbfDevice::set_mapping_page_dirty(std::uint64_t mapping_vpn) {
    if (dirty_mapping_vpns_.insert(mapping_vpn).second) {
        dirty_mapping_pages_by_stack_.at(stack_for_vpn(mapping_vpn))++;
    }
}

void HbfDevice::clear_mapping_page_dirty(std::uint64_t mapping_vpn) {
    if (dirty_mapping_vpns_.erase(mapping_vpn) != 0) {
        auto& count = dirty_mapping_pages_by_stack_.at(stack_for_vpn(mapping_vpn));
        if (count == 0) {
            throw std::runtime_error("HBF dirty mapping-page count underflowed");
        }
        count--;
    }
}

void HbfDevice::touch_mapping_cache_entry(
    std::size_t stack,
    std::uint64_t mapping_vpn) {
    auto& cache = mapping_cache_by_stack_.at(stack);
    auto found = cache.find(mapping_vpn);
    if (found == cache.end()) {
        throw std::runtime_error(
            "HBF mapping-cache touch names a nonresident page");
    }
    auto& lru = mapping_cache_lru_by_stack_.at(stack);
    lru.erase(found->second.iterator);
    lru.push_front(mapping_vpn);
    found->second.iterator = lru.begin();
}

void HbfDevice::ensure_mapping_page_cached(
    std::uint64_t mapping_vpn,
    double& at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    if (config_.mapping_mode != MappingMode::Cached) {
        throw std::runtime_error(
            "HBF mapping-cache access used outside cached mode");
    }
    const auto stack = stack_for_vpn(mapping_vpn);
    auto& cache = mapping_cache_by_stack_.at(stack);
    const auto finish_existing = [&](bool count_hit) {
        auto found = cache.find(mapping_vpn);
        if (found == cache.end()) {
            throw std::runtime_error(
                "HBF mapping-cache coalesced fill disappeared");
        }
        if (count_hit) {
            stats_.mapping_cache_hits++;
        } else {
            stats_.mapping_cache_coalesced_misses++;
        }
        if (found->second.fill_ready_ns > at_ns) {
            breakdown.scheduler_queue_wait_ns +=
                found->second.fill_ready_ns - at_ns;
            if (spans != nullptr) {
                trace_wait(
                    spans,
                    logic_entity(static_cast<std::uint32_t>(stack)),
                    at_ns,
                    found->second.fill_ready_ns,
                    count_hit ?
                        "wait_mapping_cache_fill" :
                        "wait_mapping_cache_reentrant_fill");
            }
            at_ns = found->second.fill_ready_ns;
        }
        touch_mapping_cache_entry(stack, mapping_vpn);
    };
    if (cache.contains(mapping_vpn)) {
        finish_existing(true);
        return;
    }

    stats_.mapping_cache_misses++;
    auto& lru = mapping_cache_lru_by_stack_.at(stack);
    while (cache.size() >= mapping_cache_pages_per_stack_) {
        // A dirty-victim checkpoint can invoke GC, whose mapping update may
        // recursively fill the same VPN this outer miss was resolving. That
        // nested access owns the one media fill. The outer access remains a
        // miss for access accounting, but must consume the resident result
        // instead of evicting around it and attempting a duplicate insert.
        if (cache.contains(mapping_vpn)) {
            finish_existing(false);
            return;
        }
        if (lru.empty()) {
            throw std::runtime_error(
                "HBF mapping-cache capacity and LRU state diverged");
        }
        const auto victim_vpn = lru.back();
        auto victim = cache.find(victim_vpn);
        if (victim == cache.end()) {
            throw std::runtime_error(
                "HBF mapping-cache LRU names a nonresident victim");
        }
        if (victim->second.evictable_ns > at_ns) {
            breakdown.scheduler_queue_wait_ns +=
                victim->second.evictable_ns - at_ns;
            if (spans != nullptr) {
                trace_wait(
                    spans,
                    logic_entity(static_cast<std::uint32_t>(stack)),
                    at_ns,
                    victim->second.evictable_ns,
                    "wait_mapping_cache_victim");
            }
            at_ns = victim->second.evictable_ns;
        }
        const bool dirty = dirty_mapping_vpns_.contains(victim_vpn);
        lru.pop_back();
        cache.erase(victim);
        stats_.mapping_cache_entries--;
        stats_.mapping_cache_evictions++;
        if (dirty) {
            stats_.mapping_cache_dirty_evictions++;
            flush_dirty_mapping_page(
                victim_vpn, at_ns, breakdown, spans);
        }
        // GC triggered by a dirty eviction may itself access mapping pages.
        // Recheck capacity rather than assuming that one removal left a slot.
    }

    if (cache.contains(mapping_vpn)) {
        finish_existing(false);
        return;
    }

    // The resident directory is part of the declared controller-DRAM budget,
    // not a zero-latency oracle. Every cache miss reads one 8-byte directory
    // entry through the same pipelined DRAM resource before deciding whether
    // the persistent mapping page exists or is still erased.
    auto& mapping_dram = logic_dies_.at(stack).mapping_dram_issue;
    const auto directory_issue = reserve(
        at_ns,
        config_.ctrl_dram_issue_ns,
        mapping_dram);
    if (directory_issue.wait_ns > 0.0) {
        stats_.mapping_dram_wait_ops++;
        stats_.mapping_dram_wait_ns += directory_issue.wait_ns;
        stats_.mapping_dram_wait_max_ns = std::max(
            stats_.mapping_dram_wait_max_ns,
            directory_issue.wait_ns);
        breakdown.scheduler_queue_wait_ns += directory_issue.wait_ns;
        if (spans != nullptr) {
            trace_wait(
                spans,
                logic_entity(static_cast<std::uint32_t>(stack)),
                at_ns,
                directory_issue.start_ns,
                "wait_mapping_directory_dram");
        }
    }
    const double directory_done = std::max(
        directory_issue.finish_ns,
        directory_issue.start_ns + config_.ctrl_dram_latency_ns);
    breakdown.mapping_dram_ns += config_.ctrl_dram_latency_ns;
    if (spans != nullptr) {
        add_trace_span(
            spans,
            "mapping_directory_lookup",
            "metadata",
            logic_entity(static_cast<std::uint32_t>(stack)),
            directory_issue.start_ns,
            directory_done,
            true,
            "vpn" + std::to_string(mapping_vpn));
    }
    at_ns = directory_done;

    wait_for_pending_vpn_erase(
        mapping_vpn, at_ns, breakdown, spans);
    auto backing_ppn = visible_mapping_vpn_at(mapping_vpn, at_ns);
    if (backing_ppn) {
        wait_for_pending_block_erase(
            *backing_ppn, at_ns, breakdown, spans);
        if (pending_physical_erases_.contains(static_cast<std::size_t>(
                *backing_ppn / config_.pages_per_block))) {
            // The persisted copy sits in a block whose GC erase is already
            // issued, so its relocation is published: wait for that
            // publication and read the relocated copy.
            double relocation_ready_ns = at_ns;
            std::unordered_set<std::uint64_t> sequences;
            if (const auto pending = pending_vpn_updates_.find(mapping_vpn);
                pending != pending_vpn_updates_.end()) {
                for (const auto& update : pending->second) {
                    relocation_ready_ns = std::max(
                        relocation_ready_ns, update.commit_ns);
                    sequences.insert(update.program_commit_sequence);
                    sequences.insert(update.sequence);
                }
            }
            if (relocation_ready_ns > at_ns) {
                breakdown.scheduler_queue_wait_ns += relocation_ready_ns - at_ns;
                if (spans != nullptr) {
                    trace_wait(
                        spans,
                        logic_entity(static_cast<std::uint32_t>(stack)),
                        at_ns,
                        relocation_ready_ns,
                        "wait_mapping_page_relocation");
                }
                at_ns = relocation_ready_ns;
            }
            apply_selected_commits_through(sequences, at_ns);
            backing_ppn = visible_mapping_vpn_at(mapping_vpn, at_ns);
            if (!backing_ppn) {
                throw std::runtime_error(
                    "HBF persisted mapping page vanished under a GC erase");
            }
        }
        at_ns = schedule_read_page(
            *backing_ppn,
            at_ns,
            breakdown,
            spans,
            TransactionSource::Mapping,
            HeatmapTrafficSource::Mapping,
            ReadPayloadRoute::Internal,
            0);
        stats_.physical_read_bytes = checked_add(
            stats_.physical_read_bytes,
            config_.page_size_bytes,
            "HBF mapping-cache physical read bytes");
        stats_.page_reads = checked_add(
            stats_.page_reads, 1, "HBF mapping-cache page reads");
        stats_.mapping_media_reads = checked_add(
            stats_.mapping_media_reads,
            1,
            "HBF mapping-cache media reads");
        stats_.mapping_media_read_bytes = checked_add(
            stats_.mapping_media_read_bytes,
            config_.page_size_bytes,
            "HBF mapping-cache media-read bytes");
    } else {
        // A small stack-local directory can identify mapping groups that have
        // never been persisted. Their miss installs an erased page image
        // without fabricating a NAND read.
        stats_.mapping_cache_erased_misses++;
    }

    lru.push_front(mapping_vpn);
    const auto inserted = cache.emplace(
        mapping_vpn,
        MappingCacheEntry{
            .fill_ready_ns = at_ns,
            .evictable_ns = at_ns,
            .iterator = lru.begin(),
        });
    if (!inserted.second) {
        lru.pop_front();
        throw std::runtime_error(
            "HBF mapping-cache miss raced with an existing page");
    }
    stats_.mapping_cache_entries++;
    stats_.mapping_cache_peak_entries = std::max(
        stats_.mapping_cache_peak_entries,
        stats_.mapping_cache_entries);
}

void HbfDevice::access_mapping(
    std::uint64_t lpn,
    TransactionSource source,
    MappingAccessKind kind,
    double& at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    if (source != TransactionSource::User &&
        source != TransactionSource::GC) {
        throw std::runtime_error(
            "HBF mapping access source must be foreground user or GC");
    }
    const auto stack = static_cast<std::uint32_t>(
        stack_for_lpn(lpn));

    breakdown.address_mapping_ns += config_.address_generation_ns;
    if (spans != nullptr) {
        add_trace_span(
            spans,
            kind == MappingAccessKind::Lookup ?
                "mapping_address_generation" :
                "mapping_update_address_generation",
            "translation",
            logic_entity(stack),
            at_ns,
            at_ns + config_.address_generation_ns,
            true,
            "lpn" + std::to_string(lpn));
    }
    at_ns += config_.address_generation_ns;

    const auto mapping_vpn = mapping_vpn_for_lpn(lpn);
    if (config_.mapping_mode == MappingMode::Cached) {
        ensure_mapping_page_cached(
            mapping_vpn, at_ns, breakdown, spans);
    }

    auto& mapping_dram = logic_dies_.at(stack).mapping_dram_issue;
    const auto issue = reserve(
        at_ns,
        config_.ctrl_dram_issue_ns,
        mapping_dram);
    if (issue.wait_ns > 0.0) {
        stats_.mapping_dram_wait_ops++;
        stats_.mapping_dram_wait_ns += issue.wait_ns;
        stats_.mapping_dram_wait_max_ns = std::max(
            stats_.mapping_dram_wait_max_ns,
            issue.wait_ns);
        breakdown.scheduler_queue_wait_ns += issue.wait_ns;
        if (spans != nullptr) {
            trace_wait(
                spans,
                logic_entity(stack),
                at_ns,
                issue.start_ns,
                config_.mapping_mode == MappingMode::FullResident ?
                    "wait_resident_mapping_dram" :
                    "wait_mapping_cache_dram");
        }
    }
    const double response_done = std::max(
        issue.finish_ns,
        issue.start_ns + config_.ctrl_dram_latency_ns);
    breakdown.mapping_dram_ns += config_.ctrl_dram_latency_ns;
    if (spans != nullptr) {
        const auto access_name =
            config_.mapping_mode == MappingMode::FullResident ?
                (kind == MappingAccessKind::Lookup ?
                    "resident_mapping_lookup" :
                    "resident_mapping_update_access") :
                (kind == MappingAccessKind::Lookup ?
                    "cached_mapping_lookup" :
                    "cached_mapping_update_access");
        add_trace_span(
            spans,
            access_name,
            "metadata",
            logic_entity(stack),
            issue.start_ns,
            response_done,
            true,
            "lpn" + std::to_string(lpn));
    }
    at_ns = response_done;
    if (config_.mapping_mode == MappingMode::Cached) {
        auto& cache = mapping_cache_by_stack_.at(stack);
        const auto found = cache.find(mapping_vpn);
        if (found == cache.end()) {
            throw std::runtime_error(
                "HBF mapping-cache line disappeared during DRAM access");
        }
        found->second.evictable_ns = std::max(
            found->second.evictable_ns, response_done);
    }

    if (kind == MappingAccessKind::Lookup) {
        stats_.mapping_lookup_ops++;
        if (source == TransactionSource::User) {
            stats_.mapping_user_lookup_ops++;
        } else {
            stats_.mapping_gc_lookup_ops++;
        }
    } else {
        stats_.mapping_update_ops++;
        if (source == TransactionSource::User) {
            stats_.mapping_user_update_ops++;
        } else {
            stats_.mapping_gc_update_ops++;
        }
        const double update_done = at_ns + config_.mapping_update_ns;
        breakdown.translation_ns += config_.mapping_update_ns;
        if (spans != nullptr) {
            add_trace_span(
                spans,
                source == TransactionSource::GC ?
                    (config_.mapping_mode == MappingMode::FullResident ?
                        "gc_resident_mapping_update" :
                        "gc_cached_mapping_update") :
                    (config_.mapping_mode == MappingMode::FullResident ?
                        "resident_mapping_update" :
                        "cached_mapping_update"),
                "translation",
                "logic/mapping_table",
                at_ns,
                update_done,
                true,
                "lpn" + std::to_string(lpn));
        }
        at_ns = update_done;
        if (config_.mapping_mode == MappingMode::Cached) {
            auto& cache = mapping_cache_by_stack_.at(stack);
            const auto found = cache.find(mapping_vpn);
            if (found == cache.end()) {
                throw std::runtime_error(
                    "HBF mapping-cache line disappeared during update");
            }
            found->second.evictable_ns = std::max(
                found->second.evictable_ns, update_done);
        }
    }
}

void HbfDevice::flush_dirty_mapping_page(
    std::uint64_t mapping_vpn,
    double& at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    auto events = pending_dirty_mapping_events_.find(mapping_vpn);
    if (events == pending_dirty_mapping_events_.end()) {
        clear_mapping_page_dirty(mapping_vpn);
        return;
    }
    const auto snapshot = latest_touch_through(events->second, at_ns);
    if (!snapshot) {
        return;
    }
    const auto clear_snapshot = [this, mapping_vpn, snapshot]() {
        auto pending = pending_dirty_mapping_events_.find(mapping_vpn);
        if (pending == pending_dirty_mapping_events_.end()) {
            return;
        }
        pending->second.erase(
            pending->second.begin(), pending->second.upper_bound(*snapshot));
        if (pending->second.empty()) {
            pending_dirty_mapping_events_.erase(pending);
            clear_mapping_page_dirty(mapping_vpn);
        }
    };
    const auto stack = stack_for_vpn(mapping_vpn);
    const auto preferred_plane = mapping_plane_for_vpn(mapping_vpn);
    if (!gc_active_by_stack_.at(stack)) {
        maybe_run_gc(
            at_ns,
            breakdown,
            spans,
            1,
            stack,
            BlockRole::Mapping,
            preferred_plane);
    }
    const auto new_ppn = allocate_free_page(
        at_ns,
        breakdown,
        spans,
        BlockRole::Mapping,
        stack,
        preferred_plane);
    const double program_done = schedule_program_page(
        new_ppn,
        at_ns,
        breakdown,
        spans,
        TransactionSource::Mapping,
        HeatmapTrafficSource::Mapping);

    const auto program_commit_sequence = schedule_media_program_commit(
        new_ppn,
        metadata_lpn(mapping_vpn),
        PageOwner::Mapping,
        program_done);
    schedule_vpn_mapping_commit(
        mapping_vpn, new_ppn, program_done, program_commit_sequence);
    at_ns = program_done;
    clear_snapshot();
    stats_.physical_write_bytes += config_.page_size_bytes;
    stats_.mapping_program_payload_bytes += config_.page_size_bytes;
    stats_.page_programs++;
    stats_.mapping_page_programs++;
}

std::optional<std::uint64_t> HbfDevice::lookup_lpn(
    std::uint64_t lpn,
    double& at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    LookupIntent intent) {
    if (config_.mapping_mode == MappingMode::Direct) {
        // Identity translation over the compact image: no mapping work, no
        // FTL state. Ordered memory semantics per LPN still hold through
        // the in-place write completion times.
        (void)breakdown;
        (void)spans;
        const auto compact = compact_lpn_ppn(lpn);
        if (compact) {
            if (const auto pending = direct_write_ready_ns_by_lpn_.find(lpn);
                pending != direct_write_ready_ns_by_lpn_.end()) {
                at_ns = std::max(at_ns, pending->second);
            }
        }
        return compact;
    }
    wait_for_lpn_dependencies(lpn, at_ns, breakdown, spans, intent);
    access_mapping(
        lpn,
        TransactionSource::User,
        MappingAccessKind::Lookup,
        at_ns,
        breakdown,
        spans);
    return visible_lpn_at(lpn, at_ns);
}

void HbfDevice::wait_for_lpn_dependencies(
    std::uint64_t lpn,
    double& at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    LookupIntent intent) {
    // HBF exposes ordered memory semantics per LPN: a later-arriving access
    // waits for every earlier host write to publish. A raw erase of the
    // block holding the current copy destroys it and is waited for. A GC or
    // wear-leveling relocation of the page is waited for only by a read
    // whose current copy sits in a block whose erase has already been
    // issued: until then the old copy is intact, carries the same bytes, and
    // the erase, when it is issued later, is placed behind every read
    // already issued to that block. A full-page overwrite needs no old data
    // and never waits for a relocation.
    if (const auto mapping = lpn_to_ppn_.find(lpn);
        mapping != lpn_to_ppn_.end()) {
        wait_for_pending_block_erase(
            mapping->second, at_ns, breakdown, spans);
    } else if (const auto compact = compact_lpn_ppn(lpn)) {
        wait_for_pending_block_erase(
            *compact, at_ns, breakdown, spans);
    }
    const bool wait_for_relocations =
        intent == LookupIntent::ReadData && current_ppn_has_pending_erase(lpn);
    wait_for_prior_lpn_commit(
        lpn, at_ns, breakdown, spans, wait_for_relocations);
}

bool HbfDevice::write_buffer_covers(
    const WriteBufferEntry& entry,
    std::uint64_t begin,
    std::uint64_t end) const {
    return dirty_ranges_cover(entry.ranges, begin, end);
}

bool HbfDevice::dirty_ranges_cover(
    const std::vector<DirtyRange>& ranges,
    std::uint64_t begin,
    std::uint64_t end) const {
    std::uint64_t cursor = begin;
    for (const auto& range : ranges) {
        if (range.end <= cursor) {
            continue;
        }
        if (range.begin > cursor) {
            return false;
        }
        cursor = std::max(cursor, range.end);
        if (cursor >= end) {
            return true;
        }
    }
    return cursor >= end;
}

bool HbfDevice::write_buffer_full_page(const WriteBufferEntry& entry) const {
    return write_buffer_covers(entry, 0, config_.page_size_bytes);
}

std::uint64_t HbfDevice::write_buffer_covered_bytes(const WriteBufferEntry& entry) const {
    std::uint64_t bytes = 0;
    for (const auto& range : entry.ranges) {
        bytes += range.end - range.begin;
    }
    return bytes;
}

double HbfDevice::serve_read_from_write_buffer(
    std::uint64_t lpn,
    std::uint64_t begin,
    std::uint64_t end,
    const std::vector<DirtyRange>& ranges,
    double earliest_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    const auto stack = static_cast<std::uint32_t>(stack_for_lpn(lpn));
    const auto bytes = end - begin;
    std::uint64_t buffered_bytes = 0;
    for (const auto& range : ranges) {
        const auto lo = std::max(range.begin, begin);
        const auto hi = std::min(range.end, end);
        if (hi > lo) {
            buffered_bytes += hi - lo;
        }
    }
    const auto erased_bytes = bytes - buffered_bytes;
    const auto read_detail = spans == nullptr ?
        std::string{} : "lpn" + std::to_string(lpn);
    const double dram_done = schedule_write_buffer_dram_access(
        stack,
        buffered_bytes,
        false,
        earliest_ns,
        breakdown,
        spans,
        "write_buffer_read_dram",
        read_detail);
    auto& logic_die = logic_dies_.at(stack);
    const double sram_ns = transfer_time_ns(bytes, config_.logic_sram_bandwidth_GBps);
    auto sram = reserve(dram_done, sram_ns, logic_die.sram);
    breakdown.scheduler_queue_wait_ns += sram.wait_ns;
    breakdown.sram_staging_ns += sram_ns;
    if (spans != nullptr) {
        const auto entity = logic_entity(stack);
        trace_wait(spans, entity, dram_done, sram.start_ns, "wait_sram");
        add_trace_span(
            spans,
            erased_bytes == 0 ?
                "write_buffer_read_hit" : "write_buffer_read_assemble",
            "sram",
            entity,
            sram.start_ns,
            sram.finish_ns,
            true,
            "lpn" + std::to_string(lpn) + " buffered=" +
                std::to_string(buffered_bytes) + "B erased=" +
                std::to_string(erased_bytes) + "B");
    }
    // Buffered data still crosses the stack's external interface (already
    // decoded SRAM content: data bytes only, no ECC pass).
    stats_.write_buffer_read_hits++;
    // Count only bytes the buffer actually holds: a read of a partially
    // buffered, never-mapped page is served here with less than full coverage.
    stats_.write_buffer_read_bytes += buffered_bytes;
    return schedule_external_read_egress(
        stack,
        bytes,
        sram.finish_ns,
        breakdown,
        spans,
        "write_buffer_read_hit_hbio_out",
        erased_bytes == 0 ? "buffered payload" : "buffered + erased-fill payload");
}

double HbfDevice::admit_foreground_page_read(
    std::size_t stack,
    double offered_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    auto& releases = page_read_credit_release_by_stack_.at(stack);
    const auto depth =
        static_cast<std::size_t>(config_.page_read_queue_depth_per_stack);
    double admitted_ns = offered_ns;
    if (releases.size() >= depth) {
        const auto earliest = releases.begin();
        admitted_ns = std::max(admitted_ns, *earliest);
        releases.erase(earliest);
    }
    const double wait_ns = admitted_ns - offered_ns;
    stats_.page_read_admission_events = checked_add(
        stats_.page_read_admission_events,
        1,
        "HBF page-read admission event count");
    if (wait_ns > 0.0) {
        stats_.page_read_admission_waited_pages = checked_add(
            stats_.page_read_admission_waited_pages,
            1,
            "HBF page-read admission waited-page count");
        const double updated_wait =
            stats_.page_read_admission_wait_ns + wait_ns;
        if (!std::isfinite(updated_wait)) {
            throw std::runtime_error(
                "HBF page-read admission wait work is not finite");
        }
        stats_.page_read_admission_wait_ns = updated_wait;
        stats_.page_read_admission_max_wait_ns = std::max(
            stats_.page_read_admission_max_wait_ns,
            wait_ns);
        breakdown.scheduler_queue_wait_ns += wait_ns;
        if (spans != nullptr) {
            trace_wait(
                spans,
                logic_entity(static_cast<std::uint32_t>(stack)),
                offered_ns,
                admitted_ns,
                "wait_page_read_credit");
        }
    }
    return admitted_ns;
}

void HbfDevice::complete_foreground_page_read(
    std::size_t stack,
    double finish_ns) {
    if (!std::isfinite(finish_ns) || finish_ns < 0.0) {
        throw std::runtime_error(
            "HBF page-read completion frontier is invalid");
    }
    auto& releases = page_read_credit_release_by_stack_.at(stack);
    releases.insert(finish_ns);
    if (releases.size() >
        static_cast<std::size_t>(config_.page_read_queue_depth_per_stack)) {
        throw std::runtime_error(
            "HBF page-read admission exceeded configured per-stack depth");
    }
}

double HbfDevice::schedule_external_write_ingress(
    std::size_t stack,
    std::uint64_t bytes,
    double earliest_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    const std::string& detail,
    const std::string& sram_span_name) {
    auto& logic_die = logic_dies_.at(stack);
    const double payload_ns = transfer_time_ns(bytes, config_.hb_io_bandwidth_GBps);

    // The top-level request command has already crossed the command port.
    // Only the exact payload fragment enters here; the later flash-program
    // command is generated by the logic die and starts at TSV.
    auto payload = reserve(earliest_ns, payload_ns, logic_die.hb_io_data);
    logic_die.hb_io_data_busy_ns += payload_ns;
    breakdown.scheduler_queue_wait_ns += payload.wait_ns;
    breakdown.hb_io_transfer_ns += payload_ns;
    if (spans != nullptr) {
        const auto entity = logic_entity(static_cast<std::uint32_t>(stack));
        trace_wait(
            spans,
            entity,
            earliest_ns,
            payload.start_ns,
            "wait_hbio_data");
        add_trace_span(
            spans,
            "user/data_in_hbio",
            "hbio",
            entity,
            payload.start_ns,
            payload.finish_ns,
            true,
            std::to_string(bytes) + "B payload " + detail);
    }

    const double ingress_ready_ns = payload.finish_ns;
    const double sram_ns = transfer_time_ns(bytes, config_.logic_sram_bandwidth_GBps);
    auto sram = reserve(ingress_ready_ns, sram_ns, logic_die.sram);
    breakdown.scheduler_queue_wait_ns += sram.wait_ns;
    breakdown.sram_staging_ns += sram_ns;
    if (spans != nullptr) {
        const auto entity = logic_entity(static_cast<std::uint32_t>(stack));
        trace_wait(
            spans,
            entity,
            ingress_ready_ns,
            sram.start_ns,
            "wait_sram");
        add_trace_span(
            spans,
            sram_span_name,
            "sram",
            entity,
            sram.start_ns,
            sram.finish_ns,
            true,
            std::to_string(bytes) + "B payload " + detail);
    }
    return sram.finish_ns;
}

double HbfDevice::schedule_external_request_command(
    std::size_t stack,
    double earliest_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    const std::string& detail) {
    auto& logic_die = logic_dies_.at(stack);
    const double command_ns = transfer_time_ns(
        config_.command_address_bytes, config_.hb_io_bandwidth_GBps);
    auto command = reserve(earliest_ns, command_ns, logic_die.hb_io_command);
    logic_die.hb_io_command_busy_ns += command_ns;
    breakdown.scheduler_queue_wait_ns += command.wait_ns;
    breakdown.hb_io_transfer_ns += command_ns;
    if (spans != nullptr) {
        const auto entity = logic_entity(static_cast<std::uint32_t>(stack));
        trace_wait(
            spans,
            entity,
            earliest_ns,
            command.start_ns,
            "wait_hbio_request");
        add_trace_span(
            spans,
            "user/request_hbio",
            "hbio",
            entity,
            command.start_ns,
            command.finish_ns,
            true,
            std::to_string(config_.command_address_bytes) + "B request " + detail);
    }
    return command.finish_ns;
}

double HbfDevice::schedule_sram_transfer(
    std::size_t stack,
    std::uint64_t bytes,
    double earliest_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    const std::string& name,
    const std::string& detail) {
    if (bytes == 0) {
        return earliest_ns;
    }
    auto& logic_die = logic_dies_.at(stack);
    const double duration_ns = transfer_time_ns(
        bytes, config_.logic_sram_bandwidth_GBps);
    auto transfer = reserve(earliest_ns, duration_ns, logic_die.sram);
    breakdown.scheduler_queue_wait_ns += transfer.wait_ns;
    breakdown.sram_staging_ns += duration_ns;
    if (spans != nullptr) {
        const auto entity = logic_entity(static_cast<std::uint32_t>(stack));
        trace_wait(
            spans,
            entity,
            earliest_ns,
            transfer.start_ns,
            "wait_sram");
        add_trace_span(
            spans,
            name,
            "sram",
            entity,
            transfer.start_ns,
            transfer.finish_ns,
            true,
            std::to_string(bytes) + "B " + detail);
    }
    return transfer.finish_ns;
}

double HbfDevice::schedule_write_buffer_dram_access(
    std::size_t stack,
    std::uint64_t bytes,
    bool write,
    double earliest_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    const std::string& name,
    const std::string& detail) {
    if (bytes == 0 || bytes > config_.page_size_bytes) {
        throw std::runtime_error(
            "HBF write-buffer DRAM access must contain 1..page_size bytes");
    }
    auto& logic_die = logic_dies_.at(stack);
    auto issue = reserve(
        earliest_ns,
        config_.ctrl_dram_issue_ns,
        logic_die.write_buffer_dram_issue);
    if (issue.wait_ns > 0.0) {
        stats_.write_buffer_dram_wait_ops = checked_add(
            stats_.write_buffer_dram_wait_ops,
            1,
            "HBF write-buffer DRAM waited accesses");
        stats_.write_buffer_dram_wait_ns += issue.wait_ns;
        stats_.write_buffer_dram_wait_max_ns = std::max(
            stats_.write_buffer_dram_wait_max_ns,
            issue.wait_ns);
        breakdown.scheduler_queue_wait_ns += issue.wait_ns;
        if (spans != nullptr) {
            trace_wait(
                spans,
                logic_entity(static_cast<std::uint32_t>(stack)),
                earliest_ns,
                issue.start_ns,
                "wait_write_buffer_dram");
        }
    }
    const double response_done = std::max(
        issue.finish_ns,
        issue.start_ns + config_.ctrl_dram_latency_ns);
    breakdown.write_buffer_dram_ns += config_.ctrl_dram_latency_ns;
    if (write) {
        stats_.write_buffer_dram_write_ops = checked_add(
            stats_.write_buffer_dram_write_ops,
            1,
            "HBF write-buffer DRAM write accesses");
        stats_.write_buffer_dram_write_bytes = checked_add(
            stats_.write_buffer_dram_write_bytes,
            bytes,
            "HBF write-buffer DRAM write bytes");
    } else {
        stats_.write_buffer_dram_read_ops = checked_add(
            stats_.write_buffer_dram_read_ops,
            1,
            "HBF write-buffer DRAM read accesses");
        stats_.write_buffer_dram_read_bytes = checked_add(
            stats_.write_buffer_dram_read_bytes,
            bytes,
            "HBF write-buffer DRAM read bytes");
    }
    if (spans != nullptr) {
        add_trace_span(
            spans,
            name,
            "controller_dram",
            logic_entity(static_cast<std::uint32_t>(stack)),
            issue.start_ns,
            response_done,
            true,
            std::to_string(bytes) + "B " + detail);
    }
    return response_done;
}

double HbfDevice::schedule_external_read_egress(
    std::size_t stack,
    std::uint64_t bytes,
    double earliest_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    const std::string& name,
    const std::string& detail) {
    if (bytes == 0 || bytes > config_.page_size_bytes) {
        throw std::runtime_error(
            "HBF external read egress must contain 1..page_size payload bytes");
    }
    auto& logic_die = logic_dies_.at(stack);
    const double duration_ns = transfer_time_ns(
        bytes, config_.hb_io_bandwidth_GBps);
    auto transfer = reserve(earliest_ns, duration_ns, logic_die.hb_io_data);
    logic_die.hb_io_data_busy_ns += duration_ns;
    breakdown.scheduler_queue_wait_ns += transfer.wait_ns;
    breakdown.hb_io_transfer_ns += duration_ns;
    if (spans != nullptr) {
        const auto entity = logic_entity(static_cast<std::uint32_t>(stack));
        trace_wait(
            spans,
            entity,
            earliest_ns,
            transfer.start_ns,
            "wait_hbio_data");
        add_trace_span(
            spans,
            name,
            "hbio",
            entity,
            transfer.start_ns,
            transfer.finish_ns,
            true,
            std::to_string(bytes) + "B payload " + detail);
    }
    return transfer.finish_ns;
}

double HbfDevice::stage_write_buffer_range(
    std::uint64_t lpn,
    std::uint64_t begin,
    std::uint64_t end,
    HeatmapTrafficSource heatmap_source,
    double earliest_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    const auto stack = static_cast<std::uint32_t>(stack_for_lpn(lpn));
    const auto bytes = end - begin;
    auto& die_write_buffer = write_buffer(stack);
    auto& die_lru = write_buffer_lru(stack);
    auto found = die_write_buffer.find(lpn);
    double admit_ns = earliest_ns;
    if (found == die_write_buffer.end()) {
        // A new entry needs a free controller-DRAM slot. Occupancy counts both live
        // entries and flushed pages whose programs are still in flight; at
        // capacity the admit waits for the oldest program to complete (the
        // real backpressure of a sustained overload). An overwrite of a live
        // entry consumes no new slot and never waits here.
        auto& slot_releases = write_buffer_slot_release_by_stack_.at(stack);
        const auto release_completed_slots = [&slot_releases](double through_ns) {
            slot_releases.erase(slot_releases.begin(), slot_releases.upper_bound(through_ns));
        };
        const auto slot_capacity_reached = [&]() {
            const auto live = static_cast<std::uint64_t>(die_write_buffer.size());
            const auto inflight = static_cast<std::uint64_t>(slot_releases.size());
            if (live > config_.write_buffer_pages ||
                inflight > config_.write_buffer_pages - live) {
                throw std::runtime_error("HBF write-buffer slot accounting exceeded capacity");
            }
            return live + inflight == config_.write_buffer_pages;
        };

        release_completed_slots(admit_ns);
        while (slot_capacity_reached()) {
            if (!slot_releases.empty()) {
                admit_ns = std::max(admit_ns, *slot_releases.begin());
                release_completed_slots(admit_ns);
                continue;
            }
            if (die_lru.empty()) {
                throw std::runtime_error(
                    "HBF write-buffer capacity is full but has no live or in-flight owner");
            }

            // Every occupied slot is still live. Evict the LRU entry first;
            // flush_write_buffer_entry removes it from the live buffer and
            // records the media-completion release. Only after that release
            // is reached may the new LPN be installed below.
            const auto victim_lpn = die_lru.back();
            double eviction_done_ns = admit_ns;
            flush_write_buffer_entry(
                victim_lpn, eviction_done_ns, breakdown, spans);
            admit_ns = std::max(admit_ns, eviction_done_ns);
            release_completed_slots(admit_ns);
        }
        if (admit_ns > earliest_ns) {
            stats_.write_buffer_slot_wait_ops++;
            stats_.write_buffer_slot_wait_ns += admit_ns - earliest_ns;
            breakdown.scheduler_queue_wait_ns += admit_ns - earliest_ns;
            if (spans != nullptr) {
                trace_wait(
                    spans,
                    logic_entity(stack),
                    earliest_ns,
                    admit_ns,
                    "wait_write_buffer_slot");
            }
        }
        if (slot_capacity_reached()) {
            throw std::runtime_error("HBF write-buffer admission did not release a slot");
        }
        die_lru.push_front(lpn);
        found = die_write_buffer.emplace(lpn, WriteBufferEntry{
            .lpn = lpn,
            .ranges = {},
            .iterator = die_lru.begin(),
            .ready_ns = 0.0,
        }).first;
        stats_.write_buffer_misses++;
    } else {
        die_lru.splice(die_lru.begin(), die_lru, found->second.iterator);
        found->second.iterator = die_lru.begin();
        stats_.write_buffer_hits++;
    }

    const auto overlapped = insert_merged_range(
        found->second.ranges,
        DirtyRange{
            .begin = begin,
            .end = end,
            .heatmap_source = heatmap_source,
        });
    stats_.write_buffer_merged_bytes += overlapped;

    // This is the one external payload crossing for a coalesced write. The
    // logic-die SRAM is only an ingress datapath; durable-in-controller
    // buffering is charged to and written into the shared controller DRAM.
    // Later destage must not charge a second full-page HBIO transfer.
    const auto stage_detail = spans == nullptr ? std::string{} :
        "buffered lpn" + std::to_string(lpn) + " dirty=" +
            std::to_string(write_buffer_covered_bytes(found->second)) + "/" +
            std::to_string(config_.page_size_bytes) + "B";
    const double ingress_done_ns = schedule_external_write_ingress(
        stack,
        bytes,
        admit_ns,
        breakdown,
        spans,
        stage_detail,
        "write_buffer_stage");
    const auto dram_detail = spans == nullptr ?
        std::string{} : "lpn" + std::to_string(lpn);
    const double stage_done_ns = schedule_write_buffer_dram_access(
        stack,
        bytes,
        true,
        ingress_done_ns,
        breakdown,
        spans,
        "write_buffer_stage_dram",
        dram_detail);
    found->second.ready_ns = std::max(found->second.ready_ns, stage_done_ns);
    return stage_done_ns;
}

void HbfDevice::flush_write_buffer_entry(
    std::uint64_t lpn,
    double& at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    const auto stack = static_cast<std::uint32_t>(stack_for_lpn(lpn));
    auto& die_write_buffer = write_buffer(stack);
    auto found = die_write_buffer.find(lpn);
    if (found == die_write_buffer.end()) {
        return;
    }
    const auto entry = found->second;
    // NAND programs are atomic at page granularity, while buffered writes may
    // combine byte ranges from several request classes. Preserve byte-level
    // last-writer provenance in the entry, then conservatively label the one
    // physical page operation with the source owning the most dirty bytes.
    // Equal-byte ties resolve by HeatmapTrafficSource enum order. This keeps
    // physical byte/access conservation exact without relabeling a deferred
    // flush from whichever request (or drain) happens to trigger it.
    const auto entry_heatmap_source = dominant_dirty_source(entry.ranges);
    at_ns = std::max(at_ns, entry.ready_ns);
    const auto flush_detail = spans == nullptr ?
        std::string{} : "lpn" + std::to_string(lpn);
    at_ns = schedule_write_buffer_dram_access(
        stack,
        write_buffer_covered_bytes(entry),
        false,
        at_ns,
        breakdown,
        spans,
        "write_buffer_flush_dram",
        flush_detail);
    const double flush_ready_ns = at_ns;
    auto& logic_die = logic_dies_.at(stack);
    // The dirty bytes fetched from controller DRAM are staged into logic-die
    // SRAM at the SRAM payload rate; the full-page program stage that
    // follows reserves the SRAM again for its own page transfer.
    const double flush_sram_ns = transfer_time_ns(
        write_buffer_covered_bytes(entry), config_.logic_sram_bandwidth_GBps);
    const auto flush_sram = reserve(
        flush_ready_ns,
        flush_sram_ns,
        logic_die.sram);
    breakdown.scheduler_queue_wait_ns += flush_sram.wait_ns;
    if (spans != nullptr) {
        const auto entity = logic_entity(stack);
        trace_wait(
            spans,
            entity,
            flush_ready_ns,
            flush_sram.start_ns,
            "wait_write_buffer_flush_sram");
        add_trace_span(
            spans,
            "write_buffer_flush",
            "sram",
            entity,
            flush_sram.start_ns,
            flush_sram.finish_ns,
            true,
            "lpn" + std::to_string(lpn));
    }
    breakdown.sram_staging_ns += flush_sram_ns;
    at_ns = flush_sram.finish_ns;

    // Resolve the mapping at flush time: GC may have relocated the page since
    // the write was staged. State-only read; translation time was charged at stage.
    wait_for_lpn_dependencies(
        lpn, at_ns, breakdown, spans,
        write_buffer_full_page(entry) ?
            LookupIntent::OverwriteFullPage : LookupIntent::ReadData);
    const auto current_ppn = visible_lpn_at(lpn, at_ns);

    if (current_ppn && !write_buffer_full_page(entry)) {
        // The old media image is read and merged whether or not the run is
        // traced: tracing must never gate timing or accounting work.
        if (spans != nullptr) {
            add_trace_span(
                spans,
                "partial_page_merge_on_flush",
                "translation",
                logic_entity(stack),
                at_ns,
                at_ns + config_.mapping_update_ns,
                true,
                "lpn" + std::to_string(lpn));
        }
        breakdown.translation_ns += config_.mapping_update_ns;
        at_ns += config_.mapping_update_ns;
        at_ns = schedule_read_page(
            *current_ppn,
            at_ns,
            breakdown,
            spans,
            TransactionSource::User,
            entry_heatmap_source,
            ReadPayloadRoute::Internal,
            0);
        stats_.physical_read_bytes += config_.page_size_bytes;
        stats_.page_reads++;
        at_ns = schedule_sram_transfer(
            stack,
            write_buffer_covered_bytes(entry),
            at_ns,
            breakdown,
            spans,
            "partial_page_buffer_overlay",
            "dirty bytes over old media image");
    } else if (!current_ppn && !write_buffer_full_page(entry)) {
        // Explicit erased-value initialization supplies bytes that have never
        // existed in the logical image. This is a modeled NAND behavior, not
        // an implicit zero-filled backing page or an uncharged media read.
        at_ns = schedule_sram_transfer(
            stack,
            config_.page_size_bytes - write_buffer_covered_bytes(entry),
            at_ns,
            breakdown,
            spans,
            "partial_page_erased_fill",
            "erased-value fill for unmapped buffered page");
    }

    maybe_run_gc(
        at_ns,
        breakdown,
        spans,
        1,
        stack_for_lpn(lpn),
        BlockRole::Data);
    const auto new_ppn = allocate_free_page(
        at_ns,
        breakdown,
        spans,
        BlockRole::Data,
        stack_for_lpn(lpn));

    const double program_done = schedule_program_page(
        new_ppn,
        at_ns,
        breakdown,
        spans,
        TransactionSource::User,
        entry_heatmap_source);
    const auto program_commit_sequence = schedule_media_program_commit(
        new_ppn,
        lpn,
        PageOwner::Logical,
        program_done);
    double mapping_ready = program_done;
    access_mapping(
        lpn,
        TransactionSource::User,
        MappingAccessKind::Update,
        mapping_ready,
        breakdown,
        spans);
    const double page_done = mapping_ready;
    schedule_lpn_mapping_commit(
        lpn, new_ppn, page_done, program_commit_sequence);
    mark_mapping_page_dirty(mapping_vpn_for_lpn(lpn), page_done);
    at_ns = page_done;
    // The dirty resident entry persists through its checkpoint at drain.

    const auto generation = next_buffer_generation_++;
    inflight_buffered_writes_[lpn].push_back(InflightBufferedWrite{
        .generation = generation,
        .ranges = entry.ranges,
        .ready_ns = entry.ready_ns,
        .commit_ns = page_done,
        .target_ppn = new_ppn,
        .target_block_epoch = blocks_.at(
            static_cast<std::size_t>(new_ppn / config_.pages_per_block)).epoch,
    });
    (void)schedule_commit(stack, page_done, [this, lpn, generation]() {
        const auto found_inflight = inflight_buffered_writes_.find(lpn);
        if (found_inflight == inflight_buffered_writes_.end()) {
            return;
        }
        std::erase_if(
            found_inflight->second,
            [generation](const InflightBufferedWrite& write) {
                return write.generation == generation;
            });
        if (found_inflight->second.empty()) {
            inflight_buffered_writes_.erase(found_inflight);
        }
    });

    write_buffer_lru(stack).erase(entry.iterator);
    die_write_buffer.erase(found);
    // The generation remains readable from controller DRAM until the logical mapping
    // commits. Releasing its physical slot at program_done allowed a new LPN
    // to reuse the same one-page buffer while the old generation was still
    // reported as a DRAM hit. Keep ownership and readability on one deadline.
    write_buffer_slot_release_by_stack_.at(stack).insert(page_done);
    stats_.physical_write_bytes += config_.page_size_bytes;
    stats_.data_program_payload_bytes += config_.page_size_bytes;
    stats_.data_programs++;
    stats_.page_programs++;
    stats_.write_buffer_flushes++;
}

void HbfDevice::flush_all_write_buffer_entries(
    double& at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    // Drain fans out: every entry's flush is issued from the drain start and
    // the media resource reservations do the real serialization; a single
    // chained cursor would serialize the issue times themselves.
    const double drain_start_ns = at_ns;
    double drain_finish_ns = at_ns;
    for (std::size_t stack = 0; stack < config_.stacks; ++stack) {
        auto& die_lru = write_buffer_lru(stack);
        while (!die_lru.empty()) {
            const auto victim_lpn = die_lru.back();
            double entry_ns = drain_start_ns;
            flush_write_buffer_entry(victim_lpn, entry_ns, breakdown, spans);
            drain_finish_ns = std::max(drain_finish_ns, entry_ns);
        }
    }
    at_ns = drain_finish_ns;
}

std::uint64_t HbfDevice::allocate_free_page(
    double& at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    BlockRole role,
    std::size_t stack,
    std::optional<std::size_t> preferred_plane) {
    if (free_pages_per_stack_.at(stack) == 0) {
        throw std::runtime_error(
            "HBF stack " + std::to_string(stack) +
            " free-page pool exhausted before GC could free space (shared-nothing "
            "FTL: stacks cannot borrow pages from each other)");
    }
    breakdown.address_mapping_ns += config_.free_page_allocation_ns;
    if (spans != nullptr) {
        add_trace_span(
            spans,
            "free_page_alloc",
            "translation",
            "logic/free_page_allocator",
            at_ns,
            at_ns + config_.free_page_allocation_ns,
            true,
            role_name(role));
    }
    at_ns += config_.free_page_allocation_ns;

    const auto take_page = [&](std::size_t plane_candidate) {
        auto& active = active_block_ref(planes_.at(plane_candidate), role);
        const auto ppn = allocate_page_from_block(*active);
        if (blocks_.at(*active).free_pages == 0) {
            active = std::nullopt;
        }
        return ppn;
    };
    const auto active_has_room = [&](std::size_t plane_candidate) {
        auto& active = active_block_ref(planes_.at(plane_candidate), role);
        if (active && blocks_.at(*active).free_pages != 0) {
            return true;
        }
        active = std::nullopt;
        return false;
    };
    const auto pps = planes_per_stack();
    const auto stack_base = stack * pps;
    if (preferred_plane &&
        (*preferred_plane < stack_base || *preferred_plane >= stack_base + pps)) {
        throw std::runtime_error(
            "HBF preferred allocation plane belongs to another stack");
    }

    if (role == BlockRole::GC) {
        // Relocation fills the stack's open GC frontier wherever it is
        // before opening another block, so at most one GC block per stack
        // holds free pages and the reserve is consumed one block at a time.
        // A new frontier opens in the victim's plane when it has a free
        // block (copy-back locality), else in the plane with the deepest
        // free pool; GC may open reserved blocks.
        if (preferred_plane && active_has_room(*preferred_plane)) {
            return take_page(*preferred_plane);
        }
        for (std::size_t plane_candidate = stack_base;
             plane_candidate < stack_base + pps;
             ++plane_candidate) {
            if (active_has_room(plane_candidate)) {
                return take_page(plane_candidate);
            }
        }
        std::optional<std::size_t> open_plane;
        if (preferred_plane && !planes_.at(*preferred_plane).free_blocks.empty()) {
            open_plane = preferred_plane;
        } else {
            for (std::size_t plane_candidate = stack_base;
                 plane_candidate < stack_base + pps;
                 ++plane_candidate) {
                const auto depth = planes_.at(plane_candidate).free_blocks.size();
                if (depth != 0 &&
                    (!open_plane ||
                     depth > planes_.at(*open_plane).free_blocks.size())) {
                    open_plane = plane_candidate;
                }
            }
        }
        if (open_plane) {
            active_block_ref(planes_.at(*open_plane), role) =
                allocate_block_from_plane(*open_plane, role);
            return take_page(*open_plane);
        }
    } else {
        // Foreground roles keep one write frontier per plane and open a new
        // block only while the GC pool minus that block stays at or above
        // the stack's reserve.
        if (preferred_plane && active_has_room(*preferred_plane)) {
            return take_page(*preferred_plane);
        }
        const auto first = choose_allocation_plane(stack, role);
        for (std::size_t offset = 0; offset < pps; ++offset) {
            const auto plane_candidate = stack_base + (first + offset) % pps;
            if (active_has_room(plane_candidate)) {
                return take_page(plane_candidate);
            }
        }
        const auto pool = gc_headroom(stack, BlockRole::Data).relocation_pages;
        const bool above_floor = pool >= checked_add(
            gc_reserve_requirement_pages(), config_.pages_per_block,
            "HBF foreground block admission");
        if (above_floor ||
            (role == BlockRole::Mapping && gc_active_by_stack_.at(stack))) {
            std::optional<std::size_t> open_plane;
            if (preferred_plane && !planes_.at(*preferred_plane).free_blocks.empty()) {
                open_plane = preferred_plane;
            }
            for (std::size_t offset = 0; !open_plane && offset < pps; ++offset) {
                const auto plane_candidate = stack_base + (first + offset) % pps;
                if (!planes_.at(plane_candidate).free_blocks.empty()) {
                    open_plane = plane_candidate;
                }
            }
            if (open_plane) {
                active_block_ref(planes_.at(*open_plane), role) =
                    allocate_block_from_plane(*open_plane, role);
                return take_page(*open_plane);
            }
        }
    }

    throw std::runtime_error(
        "HBF cannot allocate a page for role " + role_name(role) +
        ": free pages exist only in the GC-only reserve or other roles' active "
        "blocks (relocation_capacity=" +
        std::to_string(gc_relocation_capacity(stack)) +
        " pages; the foreground blocks until GC returns a block, or automatic "
        "GC is disabled)");
}

std::size_t HbfDevice::choose_allocation_plane(
    std::size_t stack,
    BlockRole role) {
    const auto pps = planes_per_stack();
    if (pps == 0) {
        throw std::runtime_error("HBF has no planes");
    }
    std::vector<std::size_t>* cursors = nullptr;
    switch (role) {
    case BlockRole::Data:
        cursors = &next_data_allocation_plane_per_stack_;
        break;
    case BlockRole::Mapping:
        cursors = &next_mapping_allocation_plane_per_stack_;
        break;
    case BlockRole::GC:
        cursors = &next_gc_allocation_plane_per_stack_;
        break;
    case BlockRole::Free:
    case BlockRole::StaticReadOnly:
    case BlockRole::RawPhysical:
        throw std::runtime_error(
            "HBF allocation cursor requires a Data, Mapping, or GC role");
    }
    auto& cursor = cursors->at(stack);
    const auto selected = cursor % pps;
    cursor = (cursor + 1) % pps;
    return selected;
}

std::size_t HbfDevice::allocate_block_from_plane(std::size_t plane_index, BlockRole role) {
    auto& plane = planes_.at(plane_index);
    if (plane.free_blocks.empty()) {
        throw std::runtime_error("HBF plane free-block pool exhausted");
    }
    const auto block_index = plane.free_blocks.front();
    plane.free_blocks.pop_front();
    auto& block = blocks_.at(block_index);
    if (block.role != BlockRole::Free || block.free_pages != config_.pages_per_block ||
        block.valid_pages != 0 || block.invalid_pages != 0 || block.next_page != 0 ||
        block.pending_program_pages != 0 || block.erase_pending ||
        block.pending_mapping_publications != 0) {
        throw std::runtime_error("HBF free-block pool contains a non-erased block");
    }
    block.role = role;
    return block_index;
}

std::optional<std::size_t>& HbfDevice::active_block_ref(PlaneState& plane, BlockRole role) {
    switch (role) {
    case BlockRole::Data:
        return plane.active_data_block;
    case BlockRole::Mapping:
        return plane.active_mapping_block;
    case BlockRole::GC:
        return plane.active_gc_block;
    case BlockRole::RawPhysical:
        throw std::runtime_error("HBF FTL cannot allocate a page from a raw-physical block");
    case BlockRole::StaticReadOnly:
        throw std::runtime_error("HBF cannot allocate a page from a static-data block");
    case BlockRole::Free:
        throw std::runtime_error("HBF cannot allocate a page from a free-role block");
    }
    throw std::runtime_error("unknown HBF block role");
}

std::uint64_t HbfDevice::allocate_page_from_block(std::size_t block_index) {
    auto& block = blocks_.at(block_index);
    if (block.role == BlockRole::Free || block.free_pages == 0 ||
        block.next_page >= config_.pages_per_block) {
        throw std::runtime_error("HBF allocator selected a block with no programmable pages");
    }
    const auto page = block.next_page++;
    block.free_pages--;
    block.pending_program_pages++;
    free_pages_--;
    free_pages_per_stack_.at(stack_of_block(block_index))--;
    const auto ppn = static_cast<std::uint64_t>(block_index) * config_.pages_per_block + page;
    const auto inserted = programmed_pages_.emplace(ppn, PageState{});
    if (!inserted.second && inserted.first->second.status != PageStatus::Erased) {
        throw std::runtime_error("HBF allocator selected a programmed physical page");
    }
    inserted.first->second.status = PageStatus::Erased;
    inserted.first->second.owner = PageOwner::Unassigned;
    inserted.first->second.lpn = 0;
    inserted.first->second.block_epoch = block.epoch;
    return ppn;
}

std::uint64_t HbfDevice::allocate_compact_pages_on_plane(
    std::size_t plane_index,
    BlockRole role,
    std::uint32_t page_count,
    std::vector<std::uint64_t>* assigned_blocks,
    std::unordered_map<std::uint64_t, std::uint32_t>*
        live_pages_by_block) {
    if (page_count == 0 ||
        (role != BlockRole::Data && role != BlockRole::Mapping)) {
        throw std::runtime_error(
            "HBF compact allocation requires positive Data/Mapping pages");
    }
    auto& plane = planes_.at(plane_index);
    std::optional<std::uint64_t> first_ppn;
    auto remaining = page_count;
    while (remaining != 0) {
        auto& active = active_block_ref(plane, role);
        if (!active || blocks_.at(*active).free_pages == 0) {
            active.reset();
            const auto stack = stack_of_plane(plane_index);
            if (plane.free_blocks.empty() ||
                gc_headroom(stack, BlockRole::Data).relocation_pages <
                    checked_add(
                        gc_reserve_requirement_pages(), config_.pages_per_block,
                        "HBF compact image block admission")) {
                throw std::runtime_error(
                    "HBF compact image exhausted an allocatable plane block");
            }
            active = allocate_block_from_plane(plane_index, role);
            if (assigned_blocks != nullptr) {
                assigned_blocks->push_back(*active);
            }
        }
        auto& block = blocks_.at(*active);
        if (assigned_blocks != nullptr &&
            (assigned_blocks->empty() || assigned_blocks->back() != *active)) {
            throw std::runtime_error(
                "HBF compact data-block directory lost allocator order");
        }
        const auto take = std::min<std::uint32_t>(remaining, block.free_pages);
        const auto first_page = block.next_page;
        const auto ppn =
            static_cast<std::uint64_t>(*active) * config_.pages_per_block +
            first_page;
        if (!first_ppn) {
            first_ppn = ppn;
        }
        block.set_valid_range(first_page, take);
        block.next_page += take;
        block.free_pages -= take;
        block.valid_pages += take;
        if (live_pages_by_block != nullptr) {
            auto& live = (*live_pages_by_block)[*active];
            if (take > std::numeric_limits<std::uint32_t>::max() - live) {
                throw std::runtime_error(
                    "HBF compact per-block live-page count overflowed");
            }
            live += take;
        }
        free_pages_ -= take;
        free_pages_per_stack_.at(stack_of_block(*active)) -= take;
        remaining -= take;
        if (block.free_pages == 0) {
            active.reset();
        }
    }
    return *first_ppn;
}

std::uint64_t HbfDevice::dirty_cached_mapping_pages(std::size_t stack) const {
    if (config_.mapping_mode != MappingMode::Cached) {
        return 0;
    }
    return dirty_mapping_pages_by_stack_.at(stack);
}

std::uint64_t HbfDevice::induced_translation_writebacks(
    std::size_t block_index,
    std::uint64_t dirty_cached_mapping_pages) const {
    const auto& block = blocks_.at(block_index);
    const auto live_pages = static_cast<std::uint64_t>(block.valid_pages);
    if (config_.mapping_mode != MappingMode::Cached ||
        block.role == BlockRole::Mapping) {
        return 0;
    }
    // Relocating a data page updates its translation through the mapping
    // cache. A miss on a full cache evicts the LRU line, which costs one
    // checkpoint program only when that line is dirty. Lines the relocation
    // itself dirties are most-recently-used and can be evicted only after
    // mapping_cache_pages_per_stack_ later misses, so at most
    //   dirty_now + max(0, live - cache_pages)
    // dirty evictions, and never more than one per relocated page, can be
    // induced by the pages that are live now.
    const auto overflow = live_pages > mapping_cache_pages_per_stack_ ?
        live_pages - mapping_cache_pages_per_stack_ : 0;
    return std::min(
        live_pages,
        checked_add(
            dirty_cached_mapping_pages,
            overflow,
            "HBF GC induced translation writebacks"));
}

std::uint64_t HbfDevice::gc_victim_relocation_demand(
    std::size_t block_index,
    std::uint64_t dirty_cached_mapping_pages) const {
    const auto stack = stack_of_block(block_index);
    const auto live_pages = static_cast<std::uint64_t>(
        blocks_.at(block_index).valid_pages);
    if (blocks_.at(block_index).role == BlockRole::Mapping) {
        return mapping_allocation_pool_demand(stack, live_pages);
    }
    return checked_add(
        live_pages,
        mapping_allocation_pool_demand(
            stack,
            induced_translation_writebacks(block_index, dirty_cached_mapping_pages)),
        "HBF GC victim relocation demand");
}

std::uint64_t HbfDevice::mapping_allocation_pool_demand(
    std::size_t stack,
    std::uint64_t pages) const {
    const auto pps = planes_per_stack();
    for (std::size_t plane_index = stack * pps;
         plane_index < (stack + 1) * pps;
         ++plane_index) {
        if (const auto active = planes_.at(plane_index).active_mapping_block) {
            const auto available = blocks_.at(*active).free_pages;
            pages -= std::min<std::uint64_t>(pages, available);
        }
    }
    return checked_mul(
        checked_ceil_div(pages, config_.pages_per_block,
            "HBF mapping allocation blocks"),
        config_.pages_per_block,
        "HBF mapping allocation pool demand");
}

bool HbfDevice::mapping_allocation_preserves_relocation(std::size_t stack) const {
    const auto& victim = gc_victim_by_stack_.at(stack);
    if (!victim) {
        return true;
    }
    auto pool = gc_relocation_capacity(stack);
    std::uint64_t mapping_pages = 0;
    const auto pps = planes_per_stack();
    for (std::size_t plane_index = stack * pps;
         plane_index < (stack + 1) * pps;
         ++plane_index) {
        if (const auto active = planes_.at(plane_index).active_mapping_block) {
            mapping_pages += blocks_.at(*active).free_pages;
        }
    }
    if (mapping_pages == 0) {
        if (pool < config_.pages_per_block) {
            return false;
        }
        pool -= config_.pages_per_block;
        mapping_pages = config_.pages_per_block;
    }
    mapping_pages--;
    const auto& block = blocks_.at(victim->block);
    std::uint64_t remaining = 0;
    for (auto page = victim->scan_page; page < config_.pages_per_block; ++page) {
        remaining += block.is_valid(page) ? 1 : 0;
    }
    const bool metadata = block.role == BlockRole::Mapping;
    const auto data_pages = metadata ? 0 : remaining;
    const auto checkpoints = metadata || config_.mapping_mode == MappingMode::Cached ?
        remaining : 0;
    const auto uncovered = checkpoints > mapping_pages ?
        checkpoints - mapping_pages : 0;
    const auto mapping_pool = checked_mul(
        checked_ceil_div(uncovered, config_.pages_per_block,
            "HBF active relocation mapping blocks"),
        config_.pages_per_block,
        "HBF active relocation mapping pool");
    return pool >= checked_add(
        data_pages, mapping_pool, "HBF active relocation allocation floor");
}

std::uint64_t HbfDevice::worst_gc_victim_demand(std::size_t stack) const {
    // A reclaimable victim keeps at most pages_per_block - 1 live pages.
    const auto live_pages =
        static_cast<std::uint64_t>(config_.pages_per_block) - 1;
    if (config_.mapping_mode != MappingMode::Cached) {
        return live_pages;
    }
    const auto overflow = live_pages > mapping_cache_pages_per_stack_ ?
        live_pages - mapping_cache_pages_per_stack_ : 0;
    return checked_add(
        live_pages,
        mapping_allocation_pool_demand(
            stack,
            std::min(
                live_pages,
                checked_add(
                    dirty_cached_mapping_pages(stack),
                    overflow,
                    "HBF worst GC induced writebacks"))),
        "HBF worst GC victim demand");
}

std::uint64_t HbfDevice::gc_relocation_capacity(std::size_t stack) const {
    auto capacity = gc_headroom(stack, BlockRole::Data).relocation_pages;
    // An active wear-leveling migration relocates cold data into the pinned
    // cold block, but the translation writebacks it induces under cached
    // mapping still draw on the GC pool.
    if (const auto& migration = wear_leveling_by_stack_.at(stack)) {
        const auto owed = mapping_allocation_pool_demand(
            stack,
            induced_translation_writebacks(
                migration->block, dirty_cached_mapping_pages(stack)));
        capacity = capacity > owed ? capacity - owed : 0;
    }
    return capacity;
}

std::uint64_t HbfDevice::stack_minimum_erase_count(std::size_t stack) const {
    const auto pps = planes_per_stack();
    const auto block_begin = stack * pps * config_.blocks_per_plane;
    const auto block_end = (stack + 1) * pps * config_.blocks_per_plane;
    std::uint64_t minimum = std::numeric_limits<std::uint64_t>::max();
    for (std::size_t i = block_begin; i < block_end; ++i) {
        const auto& block = blocks_[i];
        if (block.role == BlockRole::StaticReadOnly ||
            block.role == BlockRole::RawPhysical) {
            continue;
        }
        minimum = std::min(
            minimum, static_cast<std::uint64_t>(block.erase_count));
    }
    return minimum == std::numeric_limits<std::uint64_t>::max() ? 0 : minimum;
}

HbfDevice::GcHeadroom HbfDevice::gc_headroom(
    std::size_t stack,
    BlockRole allocation_role) const {
    if (allocation_role != BlockRole::Data &&
        allocation_role != BlockRole::Mapping) {
        throw std::runtime_error(
            "HBF GC headroom requires a Data or Mapping allocation role");
    }

    GcHeadroom headroom;
    const auto pps = planes_per_stack();
    std::uint64_t free_blocks = 0;
    std::uint64_t returning_blocks = 0;
    for (std::size_t plane_index = stack * pps;
         plane_index < (stack + 1) * pps;
         ++plane_index) {
        const auto& plane = planes_.at(plane_index);
        const auto validate_active = [&](const std::optional<std::size_t>& active,
                                         BlockRole expected_role,
                                         const char* name)
            -> std::uint64_t {
            if (!active) {
                return 0;
            }
            const auto& block = blocks_.at(*active);
            if (block_plane_index(*active) != plane_index ||
                block.role != expected_role || block.erase_pending ||
                block.free_pages == 0) {
                throw std::runtime_error(std::string("HBF invalid active ") + name +
                    " block while computing GC headroom");
            }
            return block.free_pages;
        };

        if ((plane.active_data_block && plane.active_mapping_block &&
                *plane.active_data_block == *plane.active_mapping_block) ||
            (plane.active_data_block && plane.active_gc_block &&
                *plane.active_data_block == *plane.active_gc_block) ||
            (plane.active_mapping_block && plane.active_gc_block &&
                *plane.active_mapping_block == *plane.active_gc_block)) {
            throw std::runtime_error(
                "HBF active Data/Mapping/GC block roles alias each other");
        }

        const auto active_data_pages = validate_active(
            plane.active_data_block, BlockRole::Data, "Data");
        const auto active_mapping_pages = validate_active(
            plane.active_mapping_block, BlockRole::Mapping, "Mapping");
        const auto active_gc_pages = validate_active(
            plane.active_gc_block, BlockRole::GC, "GC");
        const auto active_role_pages = allocation_role == BlockRole::Data ?
            active_data_pages : active_mapping_pages;

        headroom.foreground_pages = checked_add(
            headroom.foreground_pages,
            active_role_pages,
            "HBF foreground active-block headroom");
        headroom.relocation_pages = checked_add(
            headroom.relocation_pages,
            active_gc_pages,
            "HBF active-GC relocation headroom");
        free_blocks = checked_add(
            free_blocks,
            static_cast<std::uint64_t>(plane.free_blocks.size()),
            "HBF stack free blocks");
    }
    headroom.relocation_pages = checked_add(
        headroom.relocation_pages,
        checked_mul(
            free_blocks,
            config_.pages_per_block,
            "HBF whole-free-block GC headroom"),
        "HBF total GC relocation headroom");
    // Blocks whose relocation erase is in flight return to the pool.
    for (const auto& [block_index, erase] : pending_physical_erases_) {
        if (erase.garbage_collection && stack_of_block(block_index) == stack) {
            returning_blocks++;
        }
    }
    // Whole blocks the foreground may open right now: those above the
    // floor one worst-case victim needs. Pages above the configured
    // reserve (pooled across the stack's planes) drive preventive GC.
    const auto floor = gc_reserve_requirement_pages();
    const auto reserve = gc_reserve_pages(stack);
    const auto spare_blocks = [&](std::uint64_t pool, std::uint64_t blocks) {
        if (pool < floor) {
            return std::uint64_t{0};
        }
        return std::min(blocks, (pool - floor) / config_.pages_per_block);
    };
    const auto above_reserve = [&](std::uint64_t pool) {
        return pool > reserve ? pool - reserve : 0;
    };
    const auto own_frontier_pages = headroom.foreground_pages;
    const auto spare_now = spare_blocks(headroom.relocation_pages, free_blocks);
    headroom.foreground_pages = checked_add(
        own_frontier_pages,
        checked_mul(
            spare_now,
            config_.pages_per_block,
            "HBF foreground whole-free-block headroom"),
        "HBF total foreground GC headroom");
    headroom.preventive_pages = checked_add(
        own_frontier_pages,
        above_reserve(headroom.relocation_pages),
        "HBF preventive GC headroom");
    const auto pool_after = checked_add(
        headroom.relocation_pages,
        checked_mul(
            returning_blocks,
            config_.pages_per_block,
            "HBF returning GC pool pages"),
        "HBF GC pool after in-flight erases");
    const auto spare_after = spare_blocks(
        pool_after,
        checked_add(free_blocks, returning_blocks, "HBF returning free blocks"));
    headroom.returning_pages = checked_mul(
        spare_after - spare_now,
        config_.pages_per_block,
        "HBF returning foreground headroom");
    headroom.returning_preventive_pages =
        above_reserve(pool_after) - above_reserve(headroom.relocation_pages);
    return headroom;
}

std::optional<std::size_t> HbfDevice::choose_gc_victim(std::size_t stack) const {
    // Victim score: net pages reclaimed (invalid pages, less the dirty
    // translation pages the relocation is bounded to write back under
    // cached mapping), less the configured number of pages per erase the
    // block sits above the stack's least-worn block. For a closed block
    // under full-resident mapping this is greedy minimum-valid selection
    // with a wear bias whose unit is documented in configs/README.md; ties
    // keep the lowest block index. Only victims whose demand fits the GC
    // pool right now are eligible.
    const auto relocation_capacity = gc_relocation_capacity(stack);
    const auto dirty = dirty_cached_mapping_pages(stack);
    const auto minimum_erase_count = stack_minimum_erase_count(stack);
    const auto& active_victim = gc_victim_by_stack_.at(stack);
    const auto& migration = wear_leveling_by_stack_.at(stack);
    const auto cold_block = cold_block_by_stack_.at(stack);
    const auto block_begin = stack * planes_per_stack() * config_.blocks_per_plane;
    const auto block_end = (stack + 1) * planes_per_stack() * config_.blocks_per_plane;
    std::optional<std::size_t> best;
    double best_score = -std::numeric_limits<double>::infinity();
    for (std::size_t i = block_begin; i < block_end; ++i) {
        const auto& block = blocks_[i];
        if (block.role == BlockRole::Free || block.role == BlockRole::StaticReadOnly ||
            block.role == BlockRole::RawPhysical || block.erase_pending ||
            pending_physical_erases_.contains(i) ||
            block.pending_program_pages != 0 ||
            block.pending_mapping_publications != 0 ||
            block.free_pages != 0 || block.invalid_pages == 0) {
            continue;
        }
        const auto plane = block_plane_index(i);
        const auto& plane_state = planes_.at(plane);
        if (plane_state.active_data_block == i || plane_state.active_mapping_block == i ||
            plane_state.active_gc_block == i || cold_block == i ||
            (active_victim && active_victim->block == i) ||
            (migration && migration->block == i)) {
            continue;
        }
        const auto induced = induced_translation_writebacks(i, dirty);
        if (gc_victim_relocation_demand(i, dirty) > relocation_capacity) {
            continue;
        }
        const double score = static_cast<double>(block.invalid_pages) -
            static_cast<double>(induced) -
            config_.gc_wear_leveling_weight *
                static_cast<double>(block.erase_count - minimum_erase_count);
        if (!best || score > best_score) {
            best = i;
            best_score = score;
        }
    }
    return best;
}

std::optional<HbfDevice::ActiveRelocation> HbfDevice::start_relocation(
    std::size_t block_index,
    RelocationPurpose purpose,
    double& at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    const auto& block = blocks_.at(block_index);
    if (pending_physical_erases_.contains(block_index) || block.erase_pending ||
        block.pending_program_pages != 0 ||
        block.pending_mapping_publications != 0 ||
        block.free_pages != 0 ||
        checked_add(
            static_cast<std::uint64_t>(block.valid_pages),
            static_cast<std::uint64_t>(block.invalid_pages),
            "HBF relocation source page accounting") != config_.pages_per_block) {
        throw std::runtime_error(
            "HBF relocation source is not a closed, fully committed block");
    }
    const bool wear_leveling = purpose == RelocationPurpose::StaticWearLeveling;
    breakdown.maintenance_ns += 50.0;
    const double select_done = at_ns + 50.0;
    if (spans != nullptr) {
        add_trace_span(
            spans,
            wear_leveling ? "static_wear_leveling_select" : "gc_victim_select",
            "maintenance",
            "logic/maintenance",
            at_ns,
            select_done,
            true,
            "block" + std::to_string(block_index));
    }
    at_ns = select_done;
    return ActiveRelocation{
        .block = block_index,
        .purpose = purpose,
        .invalid_pages_at_start = block.invalid_pages,
    };
}

bool HbfDevice::relocation_scan_complete(
    const ActiveRelocation& relocation) const {
    const auto& block = blocks_.at(relocation.block);
    for (std::uint32_t page = relocation.scan_page;
         page < config_.pages_per_block;
         ++page) {
        if (block.is_valid(page)) {
            return false;
        }
    }
    return true;
}

std::optional<std::size_t> HbfDevice::ensure_cold_block(
    std::size_t stack,
    std::uint32_t minimum_erase_count) {
    auto& cold = cold_block_by_stack_.at(stack);
    if (cold && blocks_.at(*cold).free_pages != 0) {
        return cold;
    }
    cold.reset();
    // The cold write frontier is opened on the most worn free block of the
    // stack once the GC pool holds the floor, which is sized for the cold
    // block (gc_reserve_requirement_pages). It must lead the migrating
    // block by at least half the gap; otherwise the migration would only
    // create another cold block.
    if (gc_headroom(stack, BlockRole::Data).relocation_pages <
        gc_reserve_requirement_pages()) {
        return std::nullopt;
    }
    const auto pps = planes_per_stack();
    std::optional<std::size_t> destination;
    for (std::size_t plane_index = stack * pps;
         plane_index < (stack + 1) * pps;
         ++plane_index) {
        const auto& plane = planes_.at(plane_index);
        for (const auto block_index : plane.free_blocks) {
            const auto& block = blocks_.at(block_index);
            if (block.erase_pending || pending_physical_erases_.contains(block_index) ||
                block.erase_count < minimum_erase_count) {
                continue;
            }
            if (!destination ||
                block.erase_count > blocks_.at(*destination).erase_count) {
                destination = block_index;
            }
        }
    }
    if (!destination) {
        return std::nullopt;
    }
    auto& plane = planes_.at(block_plane_index(*destination));
    const auto position = std::find(
        plane.free_blocks.begin(), plane.free_blocks.end(), *destination);
    if (position == plane.free_blocks.end()) {
        throw std::runtime_error(
            "HBF cold-block destination left the free pool");
    }
    plane.free_blocks.erase(position);
    auto& block = blocks_.at(*destination);
    if (block.role != BlockRole::Free || block.free_pages != config_.pages_per_block ||
        block.valid_pages != 0 || block.invalid_pages != 0 || block.next_page != 0 ||
        block.pending_program_pages != 0 || block.pending_mapping_publications != 0) {
        throw std::runtime_error(
            "HBF cold-block destination is not an erased block");
    }
    block.role = BlockRole::GC;
    cold = destination;
    return cold;
}

std::uint32_t HbfDevice::relocate_live_pages(
    ActiveRelocation& relocation,
    double at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    std::uint32_t max_pages) {
    const auto victim_block = relocation.block;
    const auto victim_stack = stack_of_block(victim_block);
    const bool wear_leveling =
        relocation.purpose == RelocationPurpose::StaticWearLeveling;
    const auto block_begin =
        static_cast<std::uint64_t>(victim_block) * config_.pages_per_block;
    const auto first_sequence = next_commit_sequence_;
    std::uint32_t relocated = 0;
    auto page = relocation.scan_page;
    for (; page < config_.pages_per_block && relocated < max_pages; ++page) {
        const auto& victim = blocks_.at(victim_block);
        if (!victim.is_valid(page)) {
            continue;
        }
        const auto old_ppn = block_begin + page;
        std::uint64_t lpn = 0;
        PageOwner owner = PageOwner::Unassigned;
        if (const auto materialized = programmed_pages_.find(old_ppn);
            materialized != programmed_pages_.end()) {
            if (materialized->second.status != PageStatus::Valid ||
                (materialized->second.owner != PageOwner::Logical &&
                 materialized->second.owner != PageOwner::Mapping)) {
                throw std::runtime_error(
                    "HBF relocation found invalid materialized victim ownership");
            }
            lpn = materialized->second.lpn;
            owner = materialized->second.owner;
        } else if (const auto compact = compact_page_identity(old_ppn)) {
            lpn = compact->logical_key;
            owner = compact->owner;
        } else {
            throw std::runtime_error(
                "HBF relocation valid bitmap page has no logical owner");
        }
        const bool metadata_page = owner == PageOwner::Mapping;
        if (metadata_page != is_metadata_lpn(lpn)) {
            throw std::runtime_error(
                "HBF relocation logical key disagrees with page ownership");
        }

        std::uint64_t pool_need = wear_leveling || metadata_page ? 0 : 1;
        std::uint64_t mapping_programs = !wear_leveling && metadata_page ? 1 : 0;
        if (config_.mapping_mode == MappingMode::Cached && !metadata_page) {
            const auto mapping_vpn = mapping_vpn_for_lpn(lpn);
            const auto& cache = mapping_cache_by_stack_.at(victim_stack);
            const auto& lru = mapping_cache_lru_by_stack_.at(victim_stack);
            if (!cache.contains(mapping_vpn) &&
                cache.size() >= mapping_cache_pages_per_stack_ &&
                !lru.empty() && dirty_mapping_vpns_.contains(lru.back())) {
                mapping_programs++;
            }
        }
        pool_need = checked_add(
            pool_need,
            mapping_allocation_pool_demand(victim_stack, mapping_programs),
            "HBF relocation page admission");
        if (pool_need != 0 &&
            gc_headroom(victim_stack, BlockRole::Data).relocation_pages <
                pool_need) {
            break;
        }
        std::optional<std::size_t> cold_destination;
        if (wear_leveling) {
            cold_destination = ensure_cold_block(
                victim_stack, relocation.cold_destination_minimum_erases);
            if (!cold_destination) {
                break;
            }
        }

        const double read_done = schedule_read_page(
            old_ppn,
            at_ns,
            breakdown,
            spans,
            TransactionSource::GC,
            HeatmapTrafficSource::GarbageCollection,
            ReadPayloadRoute::Internal,
            0);
        double alloc_ready = read_done;
        const auto new_ppn = cold_destination ?
            allocate_page_from_pinned_block(
                *cold_destination, alloc_ready, breakdown, spans) :
            allocate_free_page(
                alloc_ready,
                breakdown,
                spans,
                metadata_page ? BlockRole::Mapping : BlockRole::GC,
                victim_stack,
                block_plane_index(victim_block));
        const double program_done = schedule_program_page(
            new_ppn,
            alloc_ready,
            breakdown,
            spans,
            TransactionSource::GC,
            HeatmapTrafficSource::GarbageCollection);
        const auto program_commit_sequence = schedule_media_program_commit(
            new_ppn,
            lpn,
            owner,
            program_done);
        double mapping_ready = program_done;
        if (!metadata_page) {
            // Relocating a data page updates the same authoritative L2P entry
            // as a user write. In cached mode, the corresponding checkpoint
            // page may need cache admission before it becomes dirty.
            state_observation_by_stack_.at(victim_stack) = std::max(
                state_observation_by_stack_.at(victim_stack), mapping_ready);
            access_mapping(
                lpn,
                TransactionSource::GC,
                MappingAccessKind::Update,
                mapping_ready,
                breakdown,
                spans);
            state_observation_by_stack_.at(victim_stack) = std::max(
                state_observation_by_stack_.at(victim_stack), mapping_ready);
        } else {
            const double mapping_done =
                mapping_ready + config_.mapping_update_ns;
            breakdown.translation_ns += config_.mapping_update_ns;
            if (spans != nullptr) {
                add_trace_span(
                    spans,
                    "gc_mapping_checkpoint_relocate",
                    "translation",
                    "logic/mapping_table",
                    mapping_ready,
                    mapping_done,
                    true,
                    "vpn" + std::to_string(metadata_vpn(lpn)));
            }
            mapping_ready = mapping_done;
        }
        const double mapping_done = mapping_ready;
        if (metadata_page) {
            schedule_vpn_mapping_commit(
                metadata_vpn(lpn),
                new_ppn,
                mapping_done,
                program_commit_sequence,
                old_ppn);
        } else {
            schedule_lpn_mapping_commit(
                lpn,
                new_ppn,
                mapping_done,
                program_commit_sequence,
                old_ppn);
            mark_mapping_page_dirty(mapping_vpn_for_lpn(lpn), mapping_done);
        }
        relocation.destination_ppns.push_back(new_ppn);
        relocation.chain_ready_ns = std::max(
            relocation.chain_ready_ns, mapping_done);
        relocation.relocated_pages++;
        relocated++;
        stats_.physical_read_bytes += config_.page_size_bytes;
        stats_.physical_write_bytes += config_.page_size_bytes;
        stats_.page_reads++;
        stats_.page_programs++;
        if (wear_leveling) {
            stats_.static_wear_leveling_relocation_payload_bytes +=
                config_.page_size_bytes;
            stats_.static_wear_leveling_relocations++;
        } else {
            stats_.gc_relocation_payload_bytes += config_.page_size_bytes;
            stats_.gc_relocations++;
            if (metadata_page) {
                stats_.gc_mapping_relocations++;
            } else {
                stats_.gc_data_relocations++;
            }
        }
    }
    relocation.scan_page = page;
    // Every commit scheduled by this step (program completions, conditional
    // publications, induced checkpoints) is a dependency of the erase.
    for (auto sequence = first_sequence; sequence < next_commit_sequence_; ++sequence) {
        relocation.dependency_sequences.push_back(sequence);
    }
    return relocated;
}

double HbfDevice::schedule_relocation_erase(
    ActiveRelocation relocation,
    double at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    const auto victim_block = relocation.block;
    const auto victim_stack = stack_of_block(victim_block);
    const bool wear_leveling =
        relocation.purpose == RelocationPurpose::StaticWearLeveling;
    if (!relocation_scan_complete(relocation)) {
        throw std::runtime_error(
            "HBF relocation erase was scheduled with live source pages");
    }
    if (pending_physical_erases_.contains(victim_block)) {
        throw std::runtime_error(
            "HBF relocation source already has an in-flight erase");
    }
    // The erase command is issued only after the last publication of this
    // chain, so its commit finds no live page; the media placement also
    // waits for every read already issued to the block.
    const double finish_ns = schedule_erase_block(
        victim_block,
        std::max(at_ns, relocation.chain_ready_ns),
        breakdown,
        spans,
        TransactionSource::GC,
        HeatmapTrafficSource::GarbageCollection);
    PendingPhysicalErase pending{
        .finish_ns = finish_ns,
        .garbage_collection = true,
        .static_wear_leveling = wear_leveling,
        .dependency_sequences = std::move(relocation.dependency_sequences),
        .destination_ppns = std::move(relocation.destination_ppns),
    };
    const auto [pending_erase, inserted] = pending_physical_erases_.emplace(
        victim_block, std::move(pending));
    if (!inserted) {
        throw std::runtime_error(
            "HBF relocation source already has an in-flight erase");
    }
    pending_erase->second.commit_sequence = schedule_commit(
        victim_stack, finish_ns, [this, victim_block]() {
        const auto pending = pending_physical_erases_.find(victim_block);
        if (pending == pending_physical_erases_.end()) {
            throw std::runtime_error(
                "HBF GC erase lost its in-flight reservation");
        }
        const auto finish_ns = pending->second.finish_ns;
        const auto block_begin = static_cast<std::uint64_t>(
            victim_block) * config_.pages_per_block;
        for (std::uint32_t page = 0;
             page < config_.pages_per_block;
             ++page) {
            if (blocks_.at(victim_block).is_valid(page)) {
                throw std::runtime_error(
                    "HBF GC erase reached commit with a live source page");
            }
            const auto ppn = block_begin + page;
            if (const auto state = programmed_pages_.find(ppn);
                state != programmed_pages_.end() &&
                state->second.status == PageStatus::Valid) {
                throw std::runtime_error(
                    "HBF GC source page table remained live at erase commit");
            }
        }
        reset_erased_block(victim_block);
        materialized_ready_by_block_.at(victim_block) = std::max(
            materialized_ready_by_block_.at(victim_block), finish_ns);
        pending_physical_erases_.erase(pending);
    });
    background_finish_ns_ = std::max(background_finish_ns_, finish_ns);
    stats_.block_erases++;
    const auto reclaimed = static_cast<std::uint64_t>(config_.pages_per_block) -
        relocation.relocated_pages;
    if (wear_leveling) {
        stats_.static_wear_leveling_runs++;
        stats_.static_wear_leveling_reclaimed_invalid_pages = checked_add(
            stats_.static_wear_leveling_reclaimed_invalid_pages,
            reclaimed,
            "HBF static wear-leveling reclaimed invalid-page accounting");
    } else {
        stats_.gc_runs++;
        stats_.gc_reclaimed_invalid_pages = checked_add(
            stats_.gc_reclaimed_invalid_pages,
            reclaimed,
            "HBF GC reclaimed invalid-page accounting");
    }
    return finish_ns;
}

std::optional<std::size_t> HbfDevice::pending_gc_erase(std::size_t stack) const {
    // Earliest-finishing relocation erase of the stack (GC victim or a
    // fully migrated wear-leveling source): the cheapest capacity a blocked
    // allocation can wait for, since its relocation work is already done.
    std::optional<std::size_t> earliest;
    for (const auto& [block_index, erase] : pending_physical_erases_) {
        if (!erase.garbage_collection ||
            stack_of_block(block_index) != stack) {
            continue;
        }
        if (!earliest || erase.finish_ns <
                pending_physical_erases_.at(*earliest).finish_ns) {
            earliest = block_index;
        }
    }
    return earliest;
}

void HbfDevice::maybe_run_gc(
    double& at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    std::uint64_t required_pages,
    std::size_t stack,
    BlockRole allocation_role,
    std::optional<std::size_t> preferred_plane) {
    (void)preferred_plane;
    if (!config_.auto_gc_enabled || required_pages == 0) {
        return;
    }
    if (required_pages != 1) {
        throw std::runtime_error(
            "HBF GC allocation preview currently requires one page");
    }
    if (gc_active_by_stack_.at(stack)) {
        // A checkpoint evicted inside a relocation step allocates from the
        // GC pool; it never collects recursively.
        return;
    }
    gc_active_by_stack_[stack] = true;
    struct GcActiveGuard {
        std::vector<bool>& active;
        std::size_t stack;
        ~GcActiveGuard() { active[stack] = false; }
    } active_guard{gc_active_by_stack_, stack};

    const auto low_watermark = config_.gc_low_watermark_pages == 0 ?
        static_cast<std::uint64_t>(config_.pages_per_block) :
        config_.gc_low_watermark_pages;
    const auto soft_threshold = checked_add(
        required_pages, low_watermark, "HBF GC soft threshold");
    const auto hard_threshold = checked_add(
        required_pages, config_.gc_hard_watermark_pages, "HBF GC hard threshold");
    const auto allocation_ready = [&]() {
        return gc_headroom(stack, allocation_role).foreground_pages >= hard_threshold &&
            (allocation_role != BlockRole::Mapping ||
             mapping_allocation_preserves_relocation(stack));
    };
    auto& victim = gc_victim_by_stack_.at(stack);
    const auto finish_victim = [&](double cursor_ns) {
        const auto hot_plane = block_plane_index(victim->block);
        (void)schedule_relocation_erase(*victim, cursor_ns, breakdown, spans);
        // The P/E cycle was counted when the erase was scheduled.
        const auto hot_erase_count = blocks_.at(victim->block).erase_count;
        victim.reset();
        maybe_start_static_wear_leveling(
            cursor_ns, breakdown, spans, stack, hot_plane, hot_erase_count);
    };

    auto headroom = gc_headroom(stack, allocation_role);
    if (allocation_ready()) {
        if (allocation_role == BlockRole::Mapping) {
            return;
        }
        // Pages already promised by in-flight relocation erases count: a
        // victim whose erase is scheduled has done its relocation work, and
        // starting the next one before it lands would spend reserve blocks
        // twice over. Preventive GC keeps the pool above the configured
        // reserve; the runway it paces against is what the foreground can
        // still write before it blocks on the floor.
        const auto credited = checked_add(
            headroom.preventive_pages,
            headroom.returning_preventive_pages,
            "HBF GC credited preventive headroom");
        const auto runway_pages = checked_add(
            headroom.foreground_pages,
            headroom.returning_pages,
            "HBF GC credited runway");
        if (credited > soft_threshold) {
            // No reclaim pressure: only a wear-leveling migration advances.
            // Its cold frontier may take the plane block this allocation
            // was counting on, so the foreground is rechecked below.
            advance_static_wear_leveling(at_ns, breakdown, spans, stack);
            headroom = gc_headroom(stack, allocation_role);
            if (headroom.foreground_pages >= hard_threshold) {
                return;
            }
        } else {
            // Soft pressure: one paced step at the caller's cursor. The
            // pace rises as the credited runway shrinks so the victim is
            // erased before the foreground blocks.
            double cursor_ns = at_ns;
            if (!victim) {
                const auto selected = choose_gc_victim(stack);
                if (!selected) {
                    return;
                }
                victim = start_relocation(
                    *selected,
                    RelocationPurpose::GarbageCollection,
                    cursor_ns,
                    breakdown,
                    spans);
            }
            const auto runway = runway_pages - hard_threshold;
            const auto remaining = static_cast<std::uint64_t>(
                blocks_.at(victim->block).valid_pages);
            const auto needed_pace = runway == 0 ?
                remaining : checked_ceil_div(remaining, runway, "HBF GC pace");
            const auto pace = std::max<std::uint64_t>(
                config_.gc_relocation_pages_per_host_write,
                std::min<std::uint64_t>(needed_pace, config_.pages_per_block));
            (void)relocate_live_pages(
                *victim, cursor_ns, breakdown, spans,
                static_cast<std::uint32_t>(pace));
            if (relocation_scan_complete(*victim)) {
                finish_victim(cursor_ns);
            }
            // The step's GC frontier may have opened the block this
            // allocation was counting on (relocation prefers the victim's
            // plane, where the erase returns it); if so the foreground is
            // blocked until that erase lands and is handled below.
            if (gc_headroom(stack, allocation_role).foreground_pages >= hard_threshold) {
                return;
            }
        }
    }

    // Hard pressure: the allocation cannot proceed. Collect until it can,
    // materializing only this stack's reclaim chain. Each iteration first
    // banks any relocation erase already in flight (its capacity costs no
    // further relocation work), then relocates as much of the current
    // victim as the GC pool holds, and schedules its erase once no live
    // page remains.
    stats_.gc_user_blocked_runs++;
    const double blocked_begin_ns = at_ns;
    const auto wait_for_erase = [&](std::size_t pending_victim, const char* detail) {
        const double wait_begin_ns = at_ns;
        std::unordered_set<std::uint64_t> sequences;
        add_pending_physical_erase_commits(pending_victim, sequences);
        at_ns = std::max(
            at_ns, pending_physical_erases_.at(pending_victim).finish_ns);
        apply_selected_commits_through(sequences, at_ns);
        if (at_ns > wait_begin_ns) {
            breakdown.scheduler_queue_wait_ns += at_ns - wait_begin_ns;
            if (spans != nullptr) {
                add_trace_span(
                    spans,
                    "gc/foreground_block",
                    "maintenance",
                    "logic/maintenance",
                    wait_begin_ns,
                    at_ns,
                    true,
                    detail);
            }
        }
    };
    const auto stuck = [&](const std::string& reason) {
        return std::runtime_error(
            "HBF GC cannot make progress for role " + role_name(allocation_role) +
            ": " + reason + " (relocation_capacity=" +
            std::to_string(gc_relocation_capacity(stack)) +
            " pages, no relocation erase in flight, no commit pending); raise "
            "hbf-gc-reserved-free-blocks-per-plane or lower "
            "hbf-logical-capacity-bytes");
    };
    const auto iteration_limit = checked_add(
        checked_mul(
            static_cast<std::uint64_t>(planes_per_stack()) * config_.blocks_per_plane,
            8,
            "HBF GC iteration guard"),
        256,
        "HBF GC iteration guard");
    for (std::uint64_t iteration = 0;; ++iteration) {
        if (allocation_ready()) {
            break;
        }
        if (iteration > iteration_limit) {
            throw std::runtime_error(
                "HBF GC exceeded its iteration guard without freeing a "
                "foreground page for role " + role_name(allocation_role) +
                " (relocation_capacity=" +
                std::to_string(gc_relocation_capacity(stack)) + " pages)");
        }
        if (const auto pending = pending_gc_erase(stack)) {
            wait_for_erase(*pending, "wait for in-flight relocation erase");
            continue;
        }
        if (!victim) {
            const auto selected = choose_gc_victim(stack);
            if (!selected) {
                // A closed block may still be committing; a pending commit
                // can also retire a live page or a raw erase.
                const double wait_begin_ns = at_ns;
                if (!advance_to_next_commit(at_ns, stack)) {
                    throw stuck(
                        "no closed block with an invalid page fits the GC-only "
                        "reserve");
                }
                if (at_ns > wait_begin_ns) {
                    breakdown.scheduler_queue_wait_ns += at_ns - wait_begin_ns;
                    if (spans != nullptr) {
                        add_trace_span(
                            spans,
                            "gc/foreground_block",
                            "maintenance",
                            "logic/maintenance",
                            wait_begin_ns,
                            at_ns,
                            true,
                            "wait for reclaimable committed state");
                    }
                }
                continue;
            }
            victim = start_relocation(
                *selected,
                RelocationPurpose::GarbageCollection,
                at_ns,
                breakdown,
                spans);
        }
        const auto relocated = relocate_live_pages(
            *victim, at_ns, breakdown, spans, config_.pages_per_block);
        if (relocation_scan_complete(*victim)) {
            finish_victim(at_ns);
            continue;
        }
        if (relocated == 0) {
            // Stalled on GC-pool capacity mid-victim with no relocation
            // erase in flight: only a pending commit (a raw erase, or a
            // publication that retires a stale copy) can still help.
            const double wait_begin_ns = at_ns;
            if (!advance_to_next_commit(at_ns, stack)) {
                throw stuck(
                    "the GC-only reserve cannot hold victim " +
                     std::to_string(victim->block) + " at page " +
                     std::to_string(victim->scan_page) + " with " +
                     std::to_string(blocks_.at(victim->block).valid_pages) +
                     " live pages");
            }
            if (at_ns > wait_begin_ns) {
                breakdown.scheduler_queue_wait_ns += at_ns - wait_begin_ns;
                if (spans != nullptr) {
                    add_trace_span(
                        spans,
                        "gc/foreground_block",
                        "maintenance",
                        "logic/maintenance",
                        wait_begin_ns,
                        at_ns,
                        true,
                        "wait for GC pool capacity");
                }
            }
        }
    }
    if (at_ns > blocked_begin_ns && spans != nullptr) {
        add_trace_span(
            spans,
            "gc/foreground_block",
            "maintenance",
            "logic/maintenance",
            blocked_begin_ns,
            at_ns,
            true,
            "foreground hard-watermark GC");
    }
}

void HbfDevice::complete_active_relocations(
    double& at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    // A drain finishes every relocation in flight so the conservation
    // identities (relocations + reclaimed == runs x pages_per_block) close
    // within the process and the persistent image never carries a
    // half-collected victim.
    for (std::size_t stack = 0; stack < config_.stacks; ++stack) {
        for (auto* slot : {&gc_victim_by_stack_.at(stack),
                           &wear_leveling_by_stack_.at(stack)}) {
            if (!*slot) {
                continue;
            }
            gc_active_by_stack_[stack] = true;
            struct GcActiveGuard {
                std::vector<bool>& active;
                std::size_t stack;
                ~GcActiveGuard() { active[stack] = false; }
            } active_guard{gc_active_by_stack_, stack};
            auto& relocation = **slot;
            for (;;) {
                (void)relocate_live_pages(
                    relocation, at_ns, breakdown, spans, config_.pages_per_block);
                if (relocation_scan_complete(relocation)) {
                    break;
                }
                if (const auto pending = pending_gc_erase(stack)) {
                    std::unordered_set<std::uint64_t> sequences;
                    add_pending_physical_erase_commits(*pending, sequences);
                    at_ns = std::max(
                        at_ns, pending_physical_erases_.at(*pending).finish_ns);
                    apply_selected_commits_through(sequences, at_ns);
                } else if (!advance_to_next_commit(at_ns, stack)) {
                    throw std::runtime_error(
                        "HBF drain cannot complete an active relocation: the "
                        "GC-only reserve cannot hold its remaining pages");
                }
            }
            const double finish_ns = schedule_relocation_erase(
                relocation, at_ns, breakdown, spans);
            slot->reset();
            at_ns = std::max(at_ns, finish_ns);
        }
    }
}

std::uint64_t HbfDevice::allocate_page_from_pinned_block(
    std::size_t block_index,
    double& at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    // Same controller cost as the free-page allocator, but the destination
    // is the stack's cold write frontier rather than the FIFO free pool.
    breakdown.address_mapping_ns += config_.free_page_allocation_ns;
    if (spans != nullptr) {
        add_trace_span(
            spans,
            "free_page_alloc",
            "translation",
            "logic/free_page_allocator",
            at_ns,
            at_ns + config_.free_page_allocation_ns,
            true,
            "static_wear_leveling");
    }
    at_ns += config_.free_page_allocation_ns;
    return allocate_page_from_block(block_index);
}

void HbfDevice::maybe_start_static_wear_leveling(
    double at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    std::size_t stack,
    std::size_t hot_plane,
    std::uint32_t hot_erase_count) {
    const auto gap = config_.static_wear_leveling_erase_gap;
    if (gap == 0) {
        return;
    }
    auto& plane = planes_.at(hot_plane);
    // Rate limit: one cold-block search per interval of this plane's erases.
    if (plane.erase_count < checked_add(
            plane.static_wear_leveling_checked_erase_count,
            config_.static_wear_leveling_interval_erases,
            "HBF static wear-leveling check interval")) {
        return;
    }
    plane.static_wear_leveling_checked_erase_count = plane.erase_count;
    stats_.static_wear_leveling_checks++;
    if (hot_erase_count < gap ||
        hot_erase_count < config_.static_wear_leveling_start_erases ||
        wear_leveling_by_stack_.at(stack)) {
        return;
    }

    // Coldest closed block in the stack that still carries live pages. A
    // fully invalid block is ordinary GC work; static read-only and raw
    // extents are outside the FTL and cannot move. Ties prefer the block
    // with fewer live pages because it is cheaper to migrate.
    const auto pps = planes_per_stack();
    const auto block_begin = stack * pps * config_.blocks_per_plane;
    const auto block_end = (stack + 1) * pps * config_.blocks_per_plane;
    const auto& victim = gc_victim_by_stack_.at(stack);
    const auto cold_frontier = cold_block_by_stack_.at(stack);
    std::optional<std::size_t> cold;
    for (std::size_t i = block_begin; i < block_end; ++i) {
        const auto& block = blocks_[i];
        if ((block.role != BlockRole::Data && block.role != BlockRole::Mapping &&
             block.role != BlockRole::GC) ||
            block.erase_pending || block.pending_program_pages != 0 ||
            block.pending_mapping_publications != 0 ||
            block.free_pages != 0 || block.valid_pages == 0 ||
            checked_add(
                static_cast<std::uint64_t>(block.valid_pages),
                static_cast<std::uint64_t>(block.invalid_pages),
                "HBF static wear-leveling candidate pages") !=
                config_.pages_per_block ||
            pending_physical_erases_.contains(i) ||
            (victim && victim->block == i) || cold_frontier == i) {
            continue;
        }
        if (cold) {
            const auto& best = blocks_[*cold];
            if (best.erase_count < block.erase_count ||
                (best.erase_count == block.erase_count &&
                 best.valid_pages <= block.valid_pages)) {
                continue;
            }
        }
        const auto& owner_plane = planes_.at(block_plane_index(i));
        if (owner_plane.active_data_block == i ||
            owner_plane.active_mapping_block == i ||
            owner_plane.active_gc_block == i) {
            continue;
        }
        cold = i;
    }
    if (!cold) {
        return;
    }
    const auto cold_count = blocks_[*cold].erase_count;
    if (checked_add(
            static_cast<std::uint64_t>(cold_count),
            static_cast<std::uint64_t>(gap),
            "HBF static wear-leveling gap") > hot_erase_count) {
        return;
    }
    // Capacity: cold data lands in the cold frontier (one whole block when
    // none is open yet), and its induced translation writebacks draw on the
    // GC pool, which must still hold the worst reclaimable victim afterwards
    // or GC could wedge behind the migration.
    const auto induced = mapping_allocation_pool_demand(
        stack,
        induced_translation_writebacks(*cold, dirty_cached_mapping_pages(stack)));
    const auto& open_cold = cold_block_by_stack_.at(stack);
    const std::uint64_t cold_block_pages =
        open_cold && blocks_.at(*open_cold).free_pages != 0 ?
        0 : config_.pages_per_block;
    if (gc_relocation_capacity(stack) <
        checked_add(
            checked_add(induced, cold_block_pages,
                "HBF static wear-leveling capacity check"),
            worst_gc_victim_demand(stack),
            "HBF static wear-leveling capacity check")) {
        return;
    }
    const auto minimum_destination_count = checked_add(
        static_cast<std::uint64_t>(cold_count),
        static_cast<std::uint64_t>((gap + 1) / 2),
        "HBF static wear-leveling destination floor");
    if (minimum_destination_count > std::numeric_limits<std::uint32_t>::max()) {
        return;
    }
    if (!ensure_cold_block(
            stack, static_cast<std::uint32_t>(minimum_destination_count))) {
        return;
    }
    double cursor_ns = at_ns;
    auto migration = start_relocation(
        *cold,
        RelocationPurpose::StaticWearLeveling,
        cursor_ns,
        breakdown,
        spans);
    migration->cold_destination_minimum_erases =
        static_cast<std::uint32_t>(minimum_destination_count);
    wear_leveling_by_stack_.at(stack) = std::move(migration);
}

void HbfDevice::advance_static_wear_leveling(
    double at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    std::size_t stack) {
    auto& migration = wear_leveling_by_stack_.at(stack);
    if (!migration) {
        return;
    }
    // Background work at the caller's cursor, paced like preventive GC and
    // never inside the hard-pressure loop.
    (void)relocate_live_pages(
        *migration, at_ns, breakdown, spans,
        config_.gc_relocation_pages_per_host_write);
    if (relocation_scan_complete(*migration)) {
        (void)schedule_relocation_erase(*migration, at_ns, breakdown, spans);
        migration.reset();
    }
}

void HbfDevice::invalidate_ppn(std::uint64_t ppn) {
    auto found = programmed_pages_.find(ppn);
    if (found == programmed_pages_.end() || found->second.status != PageStatus::Valid) {
        return;
    }
    found->second.status = PageStatus::Invalid;
    auto& block = blocks_.at(ppn / config_.pages_per_block);
    if (block.valid_pages == 0) {
        // A valid page always counts in its block; an underflow here means
        // the page table and the block counters disagree, which the audit
        // exists to catch rather than hide.
        throw std::runtime_error(
            "HBF invalidated a page in a block with no valid pages");
    }
    block.valid_pages--;
    block.clear_valid(static_cast<std::uint32_t>(ppn % config_.pages_per_block));
    block.invalid_pages++;
    stats_.invalidations++;
}

void HbfDevice::mark_programmed(
    std::uint64_t ppn,
    std::uint64_t lpn,
    PageOwner owner) {
    if (ppn >= total_pages_) {
        throw std::runtime_error("HBF program target PPN is out of range");
    }
    const auto block_index = static_cast<std::size_t>(ppn / config_.pages_per_block);
    const auto page_index = static_cast<std::uint32_t>(ppn % config_.pages_per_block);
    auto& block = blocks_.at(block_index);
    auto page_state = programmed_pages_.find(ppn);
    if (page_state == programmed_pages_.end()) {
        throw std::runtime_error(
            "HBF program completion has no allocator/raw reservation");
    }
    if (page_state->second.block_epoch != block.epoch || block.erase_pending) {
        throw std::runtime_error("HBF program completion targets a retired block epoch");
    }
    const bool role_matches =
        (owner == PageOwner::Logical &&
            (block.role == BlockRole::Data || block.role == BlockRole::GC)) ||
        (owner == PageOwner::Mapping &&
            (block.role == BlockRole::Mapping || block.role == BlockRole::GC)) ||
        (owner == PageOwner::RawPhysical && block.role == BlockRole::RawPhysical);
    if (!role_matches) {
        throw std::runtime_error("HBF program owner does not match block role");
    }
    if (page_state->second.status != PageStatus::Erased) {
        throw std::runtime_error("HBF attempted to program a non-erased physical page");
    }
    if (block.pending_program_pages == 0) {
        throw std::runtime_error(
            "HBF program completion lost its block ownership pin");
    }
    // The block epoch changes on erase, but not when an erased page becomes
    // programmed. Retire any decoded erased-value cache line exactly when the
    // media program completes so a later physical read cannot observe the old
    // contents under the still-current block epoch.
    read_buffer_purge_page(ppn);
    page_state->second.status = PageStatus::Valid;
    page_state->second.owner = owner;
    page_state->second.lpn = lpn;
    block.pending_program_pages--;
    block.valid_pages++;
    block.set_valid(page_index);
}

void HbfDevice::reset_erased_block(std::size_t block_index) {
    const auto& retiring_block = blocks_.at(block_index);
    if (!retiring_block.erase_pending &&
        (retiring_block.pending_program_pages != 0 ||
         retiring_block.pending_mapping_publications != 0)) {
        throw std::runtime_error(
            "HBF GC attempted to erase a block with in-flight ownership");
    }
    read_buffer_purge_block(block_index);
    const auto block_begin = static_cast<std::uint64_t>(block_index) * config_.pages_per_block;
    for (std::uint32_t page = 0; page < config_.pages_per_block; ++page) {
        programmed_pages_.erase(block_begin + page);
    }

    auto& block = blocks_.at(block_index);
    const bool was_free = block.role == BlockRole::Free;
    const bool was_pending_free_erase = was_free && block.erase_pending;
    const auto next_epoch = block.erase_pending ? block.epoch : block.epoch + 1;
    const auto newly_free_pages = config_.pages_per_block - block.free_pages;
    free_pages_ += newly_free_pages;
    free_pages_per_stack_.at(stack_of_block(block_index)) += newly_free_pages;
    // The P/E cycle was counted when the erase was scheduled.
    const auto erase_count = block.erase_count;
    block = BlockState{};
    block.role = BlockRole::Free;
    block.free_pages = config_.pages_per_block;
    block.erase_count = erase_count;
    block.epoch = next_epoch;

    auto& plane = planes_.at(block_plane_index(block_index));
    if (plane.active_data_block == block_index) {
        plane.active_data_block = std::nullopt;
    }
    if (plane.active_mapping_block == block_index) {
        plane.active_mapping_block = std::nullopt;
    }
    if (plane.active_gc_block == block_index) {
        plane.active_gc_block = std::nullopt;
    }
    if (!was_free || was_pending_free_erase) {
        plane.free_blocks.push_back(block_index);
    }
}

HbfDevice::ScheduledTransfer HbfDevice::reserve(
    double earliest_ns,
    double duration_ns,
    ResourceTimeline& timeline) {
    // Validate the complete reservation before pruning mutates the calendar.
    // Legal internal work is never ready before its top-level issue/drain
    // arrival; violating that invariant must fail without consuming history.
    if (!std::isfinite(reservation_causal_watermark_ns_) ||
        reservation_causal_watermark_ns_ < 0.0 ||
        !std::isfinite(earliest_ns) ||
        earliest_ns < reservation_causal_watermark_ns_ ||
        !std::isfinite(duration_ns) || duration_ns <= 0.0) {
        throw std::runtime_error("HBF resource calendar received an invalid reservation");
    }
    const double updated_reserved_work_ns =
        timeline.reserved_work_ns + duration_ns;
    if (!std::isfinite(updated_reserved_work_ns)) {
        throw std::runtime_error(
            "HBF resource reservation work exceeds finite double range");
    }
    timeline.prune_before(reservation_causal_watermark_ns_);
    // Use the earliest exact idle interval that fits. No still-usable gap is
    // evicted: losing one changes the simulated schedule based on call order
    // and creates phantom queueing even when the physical resource is idle.
    if (const auto gap = timeline.first_fitting_gap(earliest_ns, duration_ns)) {
        const double start = std::max(gap->begin_ns, earliest_ns);
        const double finish = causal_finish(start, duration_ns);
        timeline.consume_gap(*gap, start, finish);
        timeline.reserved_work_ns = updated_reserved_work_ns;
        return ScheduledTransfer{
            .start_ns = start,
            .finish_ns = finish,
            .wait_ns = std::max(0.0, start - earliest_ns),
        };
    }
    const double start = std::max(earliest_ns, timeline.ready_ns);
    const double finish = causal_finish(start, duration_ns);
    if (!std::isfinite(start) || !std::isfinite(finish)) {
        throw std::runtime_error("HBF resource calendar reservation time overflowed");
    }
    if (start > timeline.ready_ns) {
        timeline.insert_gap(timeline.ready_ns, start);
    }
    timeline.ready_ns = finish;
    timeline.reserved_work_ns = updated_reserved_work_ns;
    return ScheduledTransfer{
        .start_ns = start,
        .finish_ns = finish,
        .wait_ns = std::max(0.0, start - earliest_ns),
    };
}

HbfDevice::ScheduledTransfer HbfDevice::schedule_flash_transaction(
    const HbfAddress& addr,
    double earliest_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    TransactionSource source,
    TransactionKind kind) {
    auto& die = dies_.at(die_index(addr));
    const auto source_slot = source_index(source);
    stats_.flash_scheduler_enqueues++;

    auto slot = reserve(
        earliest_ns, config_.flash_tsu_issue_ns, die.source_queues.at(source_slot));
    breakdown.scheduler_queue_wait_ns += slot.wait_ns;
    const double source_ready = slot.start_ns;

    auto issue = reserve(source_ready, config_.flash_tsu_issue_ns, die.sequencer);
    breakdown.scheduler_queue_wait_ns += issue.wait_ns;
    breakdown.command_ns += config_.flash_tsu_issue_ns;
    if (spans != nullptr) {
        const auto source_label = source_name(source);
        const auto kind_label = kind_name(kind);
        trace_wait(
            spans,
            die_entity(addr),
            earliest_ns,
            slot.start_ns,
            "wait_flash_scheduler_" + source_label);
        trace_wait(
            spans,
            die_entity(addr),
            source_ready,
            issue.start_ns,
            "wait_flash_sequencer");
        add_trace_span(
            spans,
            source_label + "/flash_scheduler_issue_" + kind_label,
            "sequencer",
            die_entity(addr),
            issue.start_ns,
            issue.finish_ns,
            true,
            source_label + " " + kind_label);
    }

    die.sequencer_busy_ns += config_.flash_tsu_issue_ns;
    die.transaction_count++;
    stats_.flash_scheduler_issues++;
    return issue;
}

double HbfDevice::schedule_command_path(
    const HbfAddress& addr,
    double earliest_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    TransactionSource source) {
    auto& logic_die = logic_dies_.at(stack_index(addr));
    auto& channel = channels_.at(channel_index(addr));
    const double tsv_ns = transfer_time_ns(config_.command_address_bytes, config_.tsv_bandwidth_GBps);
    const double channel_ns = transfer_time_ns(config_.command_address_bytes, config_.channel_bandwidth_GBps);

    breakdown.tsv_transfer_ns += tsv_ns;
    auto tsv = reserve(earliest_ns, tsv_ns, logic_die.tsv);
    breakdown.scheduler_queue_wait_ns += tsv.wait_ns;

    auto command = reserve(tsv.finish_ns, channel_ns, channel.command);
    breakdown.scheduler_queue_wait_ns += command.wait_ns;
    breakdown.channel_transfer_ns += channel_ns;
    channel.command_busy_ns += channel_ns;
    channel.command_count++;
    if (spans != nullptr) {
        const auto source_label = source_name(source);
        const auto tsv_entity =
            "stack" + std::to_string(addr.stack) + "/tsv";
        trace_wait(
            spans,
            tsv_entity,
            earliest_ns,
            tsv.start_ns,
            "wait_tsv_cmd");
        add_trace_span(
            spans,
            source_label + "/cmd_addr_tsv",
            "tsv",
            tsv_entity,
            tsv.start_ns,
            tsv.finish_ns,
            true,
            std::to_string(config_.command_address_bytes) + "B " + source_label);
        trace_wait(
            spans,
            channel_entity(addr),
            tsv.finish_ns,
            command.start_ns,
            "wait_channel_cmd");
        add_trace_span(
            spans,
            source_label + "/cmd_addr_channel",
            "flash_channel",
            channel_entity(addr),
            command.start_ns,
            command.finish_ns,
            true,
            std::to_string(config_.command_address_bytes) + "B " + source_label);
    }
    return command.finish_ns;
}

double HbfDevice::schedule_ecc(
    const HbfAddress& addr,
    double earliest_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    TransactionSource source,
    bool decode) {
    auto& die = dies_.at(die_index(addr));
    const double latency_ns = decode ?
        config_.ecc_decode_latency_ns : config_.ecc_encode_latency_ns;
    const double raw_bandwidth_GBps = decode ?
        config_.ecc_decode_raw_bandwidth_GBps_per_die :
        config_.ecc_encode_raw_bandwidth_GBps_per_die;
    const double initiation_ns = transfer_time_ns(page_wire_bytes(), raw_bandwidth_GBps);

    // A codeword consumes only an initiation slot. Its response latency may
    // overlap later codewords in the same per-die pipeline. Decode and encode
    // conservatively share this issue port, so mixed traffic still contends.
    const auto issue = reserve(earliest_ns, initiation_ns, die.ecc_issue);
    const double finish_ns = issue.start_ns + latency_ns;
    if (!std::isfinite(finish_ns)) {
        throw std::runtime_error("HBF ECC completion time overflowed");
    }

    breakdown.ecc_queue_wait_ns += issue.wait_ns;
    breakdown.ecc_latency_ns += latency_ns;
    stats_.ecc_issue_busy_ns += initiation_ns;
    stats_.ecc_codeword_bytes = checked_add(
        stats_.ecc_codeword_bytes, page_wire_bytes(), "HBF ECC codeword bytes");
    die.ecc_issue_busy_ns += initiation_ns;
    die.ecc_inflight_intervals.push_back(DieState::EccInflightInterval{
        .start_ns = issue.start_ns,
        .finish_ns = finish_ns,
    });
    if (decode) {
        stats_.ecc_decode_ops++;
        stats_.ecc_decode_queue_wait_ns += issue.wait_ns;
        stats_.ecc_decode_latency_work_ns += latency_ns;
        stats_.ecc_decode_issue_busy_ns += initiation_ns;
        stats_.ecc_decode_codeword_bytes = checked_add(
            stats_.ecc_decode_codeword_bytes,
            page_wire_bytes(),
            "HBF ECC decode codeword bytes");
        die.ecc_decode_ops++;
    } else {
        stats_.ecc_encode_ops++;
        stats_.ecc_encode_queue_wait_ns += issue.wait_ns;
        stats_.ecc_encode_latency_work_ns += latency_ns;
        stats_.ecc_encode_issue_busy_ns += initiation_ns;
        stats_.ecc_encode_codeword_bytes = checked_add(
            stats_.ecc_encode_codeword_bytes,
            page_wire_bytes(),
            "HBF ECC encode codeword bytes");
        die.ecc_encode_ops++;
    }

    if (spans != nullptr) {
        const auto source_label = source_name(source);
        const auto operation =
            decode ? std::string{"decode"} : std::string{"encode"};
        const auto die_label = die_entity(addr);
        const auto issue_entity = die_label + "/ecc_issue_port";
        const auto pipeline_entity =
            die_label + "/ecc_" + operation + "_pipeline";
        trace_wait(
            spans,
            issue_entity,
            earliest_ns,
            issue.start_ns,
            "wait_ecc_" + operation + "_issue");
        const auto detail = std::to_string(page_wire_bytes()) +
            "B raw, II=" + fixed(initiation_ns, 6) + "ns";
        add_trace_span(
            spans,
            source_label + "/ecc_" + operation + "_issue",
            "ecc_issue",
            issue_entity,
            issue.start_ns,
            issue.finish_ns,
            false,
            detail);
        add_trace_span(
            spans,
            source_label + "/ecc_" + operation + "_latency",
            "ecc_latency",
            pipeline_entity,
            issue.start_ns,
            finish_ns,
            true,
            detail);
    }
    return finish_ns;
}

std::vector<HbfDevice::PageRunSpan>
HbfDevice::static_page_run_spans(
    std::uint64_t first_source_page,
    std::uint64_t pages) const {
    if (pages == 0) {
        throw std::runtime_error("HBF static page run cannot be empty");
    }
    const auto end_source_page = checked_add(
        first_source_page, pages, "HBF static page-run source end");
    if (end_source_page > total_pages_) {
        throw std::runtime_error(
            "HBF static page run exceeds physical page capacity");
    }
    const auto plane_count = static_cast<std::uint64_t>(planes_.size());
    std::vector<PageRunSpan> spans;
    std::vector<std::optional<std::size_t>> last_span_by_plane(
        planes_.size());
    const auto append = [&](std::size_t plane,
                            std::uint64_t first_page,
                            std::uint64_t count) {
        if (count == 0) {
            return;
        }
        auto& last = last_span_by_plane.at(plane);
        if (last &&
            spans.at(*last).first_page + spans.at(*last).page_count ==
                first_page) {
            spans.at(*last).page_count = checked_add(
                spans.at(*last).page_count,
                count,
                "HBF static page-run merged span");
            return;
        }
        last = spans.size();
        spans.push_back(PageRunSpan{
            .plane = plane,
            .first_page = first_page,
            .page_count = count,
        });
    };
    const auto append_source_page = [&](std::uint64_t source_page) {
        const auto ppn = static_ppn_for_source_page(source_page);
        const auto pages_per_physical_plane = checked_mul(
            config_.blocks_per_plane,
            config_.pages_per_block,
            "HBF static pages per plane");
        const auto plane = static_cast<std::size_t>(
            ppn / pages_per_physical_plane);
        append(plane, ppn % pages_per_physical_plane, 1);
    };

    auto cursor = first_source_page;
    if (cursor % plane_count != 0) {
        const auto next_cycle = checked_mul(
            checked_add(
                cursor / plane_count,
                1,
                "HBF static leading cycle"),
            plane_count,
            "HBF static leading cycle boundary");
        const auto leading_end = std::min(end_source_page, next_cycle);
        while (cursor < leading_end) {
            append_source_page(cursor++);
        }
    }
    if (cursor < end_source_page) {
        const auto full_cycle_end =
            end_source_page / plane_count * plane_count;
        if (cursor < full_cycle_end) {
            const auto first_page_in_plane = cursor / plane_count;
            const auto cycles = (full_cycle_end - cursor) / plane_count;
            for (std::size_t plane = 0; plane < planes_.size(); ++plane) {
                append(plane, first_page_in_plane, cycles);
            }
            cursor = full_cycle_end;
        }
        while (cursor < end_source_page) {
            append_source_page(cursor++);
        }
    }
    std::uint64_t represented_pages = 0;
    for (const auto& span : spans) {
        represented_pages = checked_add(
            represented_pages,
            span.page_count,
            "HBF static page-run represented pages");
    }
    if (represented_pages != pages) {
        throw std::runtime_error(
            "HBF static page-run span accounting did not conserve pages");
    }
    return spans;
}

std::uint64_t HbfDevice::static_ppn_for_source_page(
    std::uint64_t source_page) const {
    if (source_page >= total_pages_) {
        throw std::runtime_error(
            "HBF static source page exceeds physical page capacity");
    }
    const auto stack = stack_for_lpn(source_page);
    const auto stack_page_index = source_page / config_.stacks;
    const auto local_plane = stack_page_index % planes_per_stack();
    const auto page_in_plane = stack_page_index / planes_per_stack();
    const auto plane = stack * planes_per_stack() + local_plane;
    const auto pages_per_physical_plane = checked_mul(
        config_.blocks_per_plane,
        config_.pages_per_block,
        "HBF static pages per plane");
    return checked_add(
        checked_mul(
            plane,
            pages_per_physical_plane,
            "HBF static PPN plane base"),
        page_in_plane,
        "HBF static PPN");
}

std::vector<HbfDevice::PageRunSpan>
HbfDevice::compact_logical_page_run_spans(
    std::uint64_t first_lpn,
    std::uint64_t pages) const {
    if (!compact_logical_image_ || pages == 0) {
        throw std::runtime_error(
            "HBF compact page-run span construction has no source image");
    }
    const auto& image = *compact_logical_image_;
    const auto end_lpn = checked_add(
        first_lpn, pages, "HBF compact page-run LPN end");
    const auto image_end = checked_add(
        image.first_lpn,
        image.page_count,
        "HBF compact image LPN end");
    if (first_lpn < image.first_lpn || end_lpn > image_end) {
        throw std::runtime_error(
            "HBF compact page-run escapes its dense initial image");
    }

    std::vector<PageRunSpan> spans;
    std::vector<std::optional<std::size_t>> last_span_by_plane(
        planes_.size());
    const auto append_physical_span = [&](std::size_t plane,
                                          std::uint64_t first_page,
                                          std::uint64_t count) {
        if (count == 0) {
            return;
        }
        auto& last = last_span_by_plane.at(plane);
        if (last &&
            spans.at(*last).first_page + spans.at(*last).page_count ==
                first_page) {
            spans.at(*last).page_count = checked_add(
                spans.at(*last).page_count,
                count,
                "HBF compact page-run merged span");
            return;
        }
        last = spans.size();
        spans.push_back(PageRunSpan{
            .plane = plane,
            .first_page = first_page,
            .page_count = count,
        });
    };
    const auto append_data_index_interval = [&](std::size_t stack,
                                                 std::uint64_t first_index,
                                                 std::uint64_t count) {
        const auto local_planes = static_cast<std::uint64_t>(
            planes_per_stack());
        for (std::uint64_t local_plane = 0;
             local_plane < local_planes;
             ++local_plane) {
            const auto residue = first_index % local_planes;
            const auto delta = local_plane >= residue ?
                local_plane - residue : local_planes - (residue - local_plane);
            if (delta >= count) {
                continue;
            }
            const auto assigned_pages =
                1 + (count - 1 - delta) / local_planes;
            auto assigned_page = (first_index + delta) / local_planes;
            auto remaining = assigned_pages;
            const auto plane = stack * planes_per_stack() +
                static_cast<std::size_t>(local_plane);
            const auto& assigned_blocks = image.data_blocks_by_plane.at(plane);
            while (remaining != 0) {
                const auto block_ordinal =
                    assigned_page / config_.pages_per_block;
                const auto page = assigned_page % config_.pages_per_block;
                if (block_ordinal >= assigned_blocks.size()) {
                    throw std::runtime_error(
                        "HBF compact page-run exceeds its block directory");
                }
                const auto in_block = std::min<std::uint64_t>(
                    remaining, config_.pages_per_block - page);
                const auto block_index = assigned_blocks.at(
                    static_cast<std::size_t>(block_ordinal));
                if (block_plane_index(static_cast<std::size_t>(block_index)) !=
                    plane) {
                    throw std::runtime_error(
                        "HBF compact page-run block belongs to another plane");
                }
                const auto local_block =
                    block_index % config_.blocks_per_plane;
                append_physical_span(
                    plane,
                    checked_add(
                        checked_mul(
                            local_block,
                            config_.pages_per_block,
                            "HBF compact page-run block base"),
                        page,
                        "HBF compact page-run page"),
                    in_block);
                assigned_page += in_block;
                remaining -= in_block;
            }
        }
    };

    const auto stacks = static_cast<std::uint64_t>(config_.stacks);
    const auto group_width = checked_mul(
        stacks,
        config_.mapping_entries_per_page,
        "HBF compact page-run mapping group width");
    auto group = first_lpn / group_width;
    const auto last_group = (end_lpn - 1) / group_width;
    for (;; ++group) {
        const auto group_begin = checked_mul(
            group, group_width, "HBF compact page-run group base");
        const auto group_end = checked_add(
            group_begin, group_width, "HBF compact page-run group end");
        const auto lower = std::max(first_lpn, group_begin);
        const auto upper = std::min(end_lpn, group_end);
        for (std::uint64_t lane = 0; lane < stacks; ++lane) {
            const auto lane_begin = checked_add(
                group_begin, lane, "HBF compact page-run lane base");
            if (upper <= lane_begin) {
                continue;
            }
            const auto relative_lower = lower > lane_begin ?
                lower - lane_begin : 0;
            const auto relative_upper = upper - lane_begin;
            const auto first_entry = checked_ceil_div(
                relative_lower, stacks, "HBF compact page-run first entry");
            const auto end_entry = checked_ceil_div(
                relative_upper, stacks, "HBF compact page-run end entry");
            if (first_entry >= end_entry ||
                first_entry >= config_.mapping_entries_per_page) {
                continue;
            }
            const auto bounded_end = std::min<std::uint64_t>(
                end_entry, config_.mapping_entries_per_page);
            const auto representative_lpn = checked_add(
                lane_begin,
                checked_mul(
                    first_entry,
                    stacks,
                    "HBF compact page-run representative entry"),
                "HBF compact page-run representative LPN");
            const auto mapping_vpn = mapping_vpn_for_lpn(representative_lpn);
            const auto stack = stack_for_vpn(mapping_vpn);
            if (mapping_vpn < image.first_vpn ||
                mapping_vpn - image.first_vpn >= image.vpn_slot_count) {
                throw std::runtime_error(
                    "HBF compact page-run maps outside its VPN directory");
            }
            const auto& range = image.vpn_ranges.at(
                static_cast<std::size_t>(mapping_vpn - image.first_vpn));
            if (first_entry < range.first_entry ||
                bounded_end - range.first_entry > range.page_count) {
                throw std::runtime_error(
                    "HBF compact page-run crosses an inactive VPN boundary");
            }
            append_data_index_interval(
                stack,
                checked_add(
                    range.stack_page_offset,
                    first_entry - range.first_entry,
                    "HBF compact page-run data index"),
                bounded_end - first_entry);
        }
        if (group == last_group) {
            break;
        }
    }

    std::uint64_t represented_pages = 0;
    for (const auto& span : spans) {
        represented_pages = checked_add(
            represented_pages,
            span.page_count,
            "HBF compact page-run represented pages");
    }
    if (represented_pages != pages) {
        throw std::runtime_error(
            "HBF compact page-run span accounting did not conserve pages");
    }
    return spans;
}

void HbfDevice::record_mutated_lpn_range(
    std::uint64_t first_lpn,
    std::uint64_t pages) {
    if (pages == 0) {
        throw std::runtime_error("HBF mutation range cannot be empty");
    }
    auto begin = first_lpn;
    auto end = checked_add(first_lpn, pages, "HBF mutation range end");
    auto next = mutated_lpn_ranges_.lower_bound(begin);
    if (next != mutated_lpn_ranges_.begin()) {
        auto previous = std::prev(next);
        if (previous->second >= begin) {
            begin = previous->first;
            end = std::max(end, previous->second);
            next = mutated_lpn_ranges_.erase(previous);
        }
    }
    while (next != mutated_lpn_ranges_.end() && next->first <= end) {
        end = std::max(end, next->second);
        next = mutated_lpn_ranges_.erase(next);
    }
    const auto inserted = mutated_lpn_ranges_.emplace(begin, end).second;
    if (!inserted) {
        throw std::runtime_error("HBF mutation range insertion collided");
    }
}

bool HbfDevice::mutated_lpn_range_overlaps(
    std::uint64_t first_lpn,
    std::uint64_t pages) const {
    if (pages == 0) {
        throw std::runtime_error("HBF mutation overlap range cannot be empty");
    }
    const auto end = checked_add(
        first_lpn, pages, "HBF mutation overlap range end");
    auto next = mutated_lpn_ranges_.lower_bound(first_lpn);
    if (next != mutated_lpn_ranges_.begin()) {
        const auto previous = std::prev(next);
        if (previous->second > first_lpn) {
            return true;
        }
    }
    return next != mutated_lpn_ranges_.end() && next->first < end;
}

bool HbfDevice::read_run_is_ecc_limited(
    const std::vector<PageRunSpan>& spans,
    bool logical) const {
    const auto wire_bytes = page_wire_bytes();
    const double ecc_ns = transfer_time_ns(
        wire_bytes, config_.ecc_decode_raw_bandwidth_GBps_per_die);
    const double lane_ns = transfer_time_ns(wire_bytes, config_.media_lane_bandwidth_GBps);
    const double buffer_ns = transfer_time_ns(wire_bytes, config_.page_buffer_bandwidth_GBps);
    const double channel_command_ns = transfer_time_ns(
        config_.command_address_bytes, config_.channel_bandwidth_GBps);
    const double channel_data_ns = transfer_time_ns(wire_bytes, config_.channel_bandwidth_GBps);
    const double tsv_command_ns = transfer_time_ns(
        config_.command_address_bytes, config_.tsv_bandwidth_GBps);
    const double tsv_data_ns = transfer_time_ns(wire_bytes, config_.tsv_bandwidth_GBps);
    const double sram_ns = transfer_time_ns(config_.page_size_bytes, config_.logic_sram_bandwidth_GBps);
    const double hbio_ns = transfer_time_ns(config_.page_size_bytes, config_.hb_io_bandwidth_GBps);
    if (config_.flash_tsu_issue_ns > ecc_ns) {
        return false;
    }
    std::vector<std::uint64_t> plane_pages(planes_.size(), 0);
    std::vector<std::uint64_t> die_pages(dies_.size(), 0);
    std::vector<std::uint64_t> channel_pages(channels_.size(), 0);
    std::vector<std::uint64_t> stack_pages(config_.stacks, 0);
    for (const auto& span : spans) {
        const auto die = span.plane / config_.planes_per_die;
        const auto channel = die / config_.dies_per_channel;
        const auto stack = channel / config_.channels_per_stack;
        plane_pages.at(span.plane) += span.page_count;
        die_pages.at(die) += span.page_count;
        channel_pages.at(channel) += span.page_count;
        stack_pages.at(stack) += span.page_count;
    }
    std::vector<double> channel_ecc_work(channels_.size(), 0.0);
    std::vector<double> stack_ecc_work(config_.stacks, 0.0);
    for (std::size_t die = 0; die < die_pages.size(); ++die) {
        const auto channel = die / config_.dies_per_channel;
        const auto stack = channel / config_.channels_per_stack;
        const double work_ns = die_pages[die] * ecc_ns;
        channel_ecc_work[channel] = std::max(channel_ecc_work[channel], work_ns);
        stack_ecc_work[stack] = std::max(stack_ecc_work[stack], work_ns);
    }
    for (std::size_t plane = 0; plane < plane_pages.size(); ++plane) {
        const double ecc_work_ns = die_pages[plane / config_.planes_per_die] * ecc_ns;
        const double pages = plane_pages[plane];
        if (pages * config_.t_read_page_ns / subarrays_per_plane_ > ecc_work_ns ||
            pages * lane_ns / config_.media_lanes_per_plane > ecc_work_ns ||
            pages * buffer_ns / config_.page_buffer_banks_per_plane > ecc_work_ns) {
            return false;
        }
    }
    for (std::size_t channel = 0; channel < channel_pages.size(); ++channel) {
        if (channel_pages[channel] * std::max(channel_command_ns, channel_data_ns) >
            channel_ecc_work[channel]) {
            return false;
        }
    }
    const double mapping_ns = logical ?
        config_.address_generation_ns + config_.ctrl_dram_latency_ns : 0.0;
    const double pipeline_ns = tsv_command_ns + channel_command_ns +
        config_.flash_tsu_issue_ns + 2.0 * config_.t_read_page_ns + lane_ns +
        buffer_ns + channel_data_ns + tsv_data_ns + config_.ecc_decode_latency_ns +
        sram_ns + hbio_ns + mapping_ns;
    const double stack_interval_ns = std::max({
        tsv_command_ns + tsv_data_ns, sram_ns, hbio_ns,
        logical ? config_.ctrl_dram_issue_ns : 0.0});
    for (std::size_t stack = 0; stack < stack_pages.size(); ++stack) {
        if (stack_pages[stack] == 0) {
            continue;
        }
        if (stack_pages[stack] * stack_interval_ns > stack_ecc_work[stack] ||
            pipeline_ns * stack_pages[stack] >
                config_.page_read_queue_depth_per_stack * stack_ecc_work[stack]) {
            return false;
        }
    }
    return true;
}

double HbfDevice::schedule_read_page_run_resources(
    const std::vector<PageRunSpan>& spans,
    const std::vector<std::uint64_t>& mapping_pages_by_stack,
    std::uint64_t total_pages,
    double issued_ns,
    bool logical,
    Breakdown& breakdown,
    const std::vector<double>* mapping_ready_by_stack) {
    std::vector<std::uint64_t> pages_by_plane(planes_.size(), 0);
    std::vector<std::uint64_t> pages_by_die(dies_.size(), 0);
    std::vector<std::uint64_t> pages_by_channel(channels_.size(), 0);
    std::vector<std::uint64_t> pages_by_stack(config_.stacks, 0);
    std::vector<std::vector<std::uint64_t>> subarray_counts(
        planes_.size(), std::vector<std::uint64_t>(subarrays_per_plane_, 0));
    std::vector<std::vector<std::uint64_t>> lane_counts(
        planes_.size(),
        std::vector<std::uint64_t>(config_.media_lanes_per_plane, 0));
    std::vector<std::vector<std::uint64_t>> bank_counts(
        planes_.size(),
        std::vector<std::uint64_t>(config_.page_buffer_banks_per_plane, 0));

    const auto distribute = [](std::vector<std::uint64_t>& counts,
                               std::uint64_t first,
                               std::uint64_t count) {
        const auto width = static_cast<std::uint64_t>(counts.size());
        const auto quotient = count / width;
        const auto remainder = count % width;
        for (auto& value : counts) {
            value += quotient;
        }
        for (std::uint64_t index = 0; index < remainder; ++index) {
            counts.at(static_cast<std::size_t>(
                (first + index) % width))++;
        }
    };

    for (const auto& span : spans) {
        if (span.page_count == 0 || span.plane >= planes_.size()) {
            throw std::runtime_error("HBF page-run contains an invalid span");
        }
        const auto plane_base = checked_mul(
            span.plane,
            config_.blocks_per_plane,
            "HBF page-run plane block base");
        const auto first_block = span.first_page / config_.pages_per_block;
        const auto last_block =
            (span.first_page + span.page_count - 1) /
            config_.pages_per_block;
        if (last_block >= config_.blocks_per_plane) {
            throw std::runtime_error("HBF page-run span escapes its plane");
        }
        for (auto block = first_block; block <= last_block; ++block) {
            blocks_.at(static_cast<std::size_t>(plane_base + block))
                .issued_media_ready_ns = std::max(
                    blocks_.at(static_cast<std::size_t>(plane_base + block))
                        .issued_media_ready_ns,
                    issued_ns);
        }
        pages_by_plane.at(span.plane) = checked_add(
            pages_by_plane.at(span.plane),
            span.page_count,
            "HBF page-run plane pages");
        const auto first = span.first_page;
        distribute(
            subarray_counts.at(span.plane), first, span.page_count);
        distribute(lane_counts.at(span.plane), first, span.page_count);
        distribute(bank_counts.at(span.plane), first, span.page_count);
    }

    std::uint64_t counted_pages = 0;
    for (std::size_t plane_index_value = 0;
         plane_index_value < pages_by_plane.size();
         ++plane_index_value) {
        const auto count = pages_by_plane[plane_index_value];
        if (count == 0) {
            continue;
        }
        counted_pages = checked_add(
            counted_pages, count, "HBF page-run total plane pages");
        const auto stack = stack_of_plane(plane_index_value);
        const auto planes_per_die = static_cast<std::size_t>(
            config_.planes_per_die);
        const auto die = plane_index_value / planes_per_die;
        const auto channel = die / config_.dies_per_channel;
        pages_by_die.at(die) = checked_add(
            pages_by_die.at(die), count, "HBF page-run die pages");
        pages_by_channel.at(channel) = checked_add(
            pages_by_channel.at(channel), count, "HBF page-run channel pages");
        pages_by_stack.at(stack) = checked_add(
            pages_by_stack.at(stack), count, "HBF page-run stack pages");
    }
    if (counted_pages != total_pages ||
        mapping_pages_by_stack.size() != config_.stacks) {
        throw std::runtime_error(
            "HBF page-run hierarchy accounting did not conserve pages");
    }

    // A stack's media work cannot start before its share of the run has been
    // translated: cached-mapping misses resolved during admission lift that
    // stack's ready time here.
    std::vector<double> stack_ready_ns(config_.stacks, issued_ns);
    if (mapping_ready_by_stack != nullptr) {
        if (mapping_ready_by_stack->size() != config_.stacks) {
            throw std::runtime_error(
                "HBF page-run mapping-ready vector does not match stacks");
        }
        for (std::size_t stack = 0; stack < config_.stacks; ++stack) {
            stack_ready_ns[stack] = std::max(
                stack_ready_ns[stack], (*mapping_ready_by_stack)[stack]);
        }
    }

    // Thermal admission is per stack: a throttled stack delays and
    // rate-caps only its own share of the run. The budget finish below is
    // the earliest instant the paced stacks may have dissipated this run's
    // energy, so run completion can never beat the pacing power budget.
    double thermal_budget_finish_ns = issued_ns;
    if (config_.thermal_enabled) {
        for (std::size_t stack = 0; stack < pages_by_stack.size(); ++stack) {
            const auto count = pages_by_stack[stack];
            if (count == 0) {
                continue;
            }
            const auto admission = thermal_admit_media(
                stack,
                stack_ready_ns[stack],
                static_cast<double>(count) * thermal_read_energy_j_,
                count,
                breakdown,
                nullptr);
            stack_ready_ns[stack] = admission.ready_ns;
            thermal_budget_finish_ns = std::max(
                thermal_budget_finish_ns, admission.budget_finish_ns);
        }
    }

    double resource_finish_ns = issued_ns;
    // The shared breakdown accumulates across a request's run segments; the
    // directional ECC ledger below must only receive this call's own waits.
    const double ecc_queue_wait_before_ns = breakdown.ecc_queue_wait_ns;
    // A run's work on one resource is `units` back-to-back operations of
    // `unit_ns` each. Reserving them as a single block could not use the
    // idle gaps that earlier scattered requests left behind the resource's
    // frontier, so a run issued after a handful of small reads finished up
    // to 17% later than the scalar engine, which interleaves page by page.
    // The units are placed gap by gap instead: each idle interval that
    // holds at least one whole unit takes as many as fit, and the remainder
    // appends at the frontier. The cost is one calendar operation per gap
    // touched, never per page, so the compression is preserved while the
    // envelope matches the scalar engine's use of the same idle capacity.
    const auto reserve_work = [&](double earliest_ns,
                                  double unit_ns,
                                  std::uint64_t units,
                                  ResourceTimeline& timeline,
                                  double& queue_wait_ns) {
        if (units == 0 || unit_ns <= 0.0) {
            return earliest_ns;
        }
        if (!std::isfinite(reservation_causal_watermark_ns_) ||
            earliest_ns < reservation_causal_watermark_ns_ ||
            !std::isfinite(earliest_ns) || !std::isfinite(unit_ns)) {
            throw std::runtime_error(
                "HBF page-run resource calendar received an invalid reservation");
        }
        timeline.prune_before(reservation_causal_watermark_ns_);
        double cursor_ns = earliest_ns;
        double first_start_ns = -1.0;
        double finish_ns = earliest_ns;
        std::uint64_t remaining = units;
        while (remaining != 0) {
            const auto gap = timeline.first_fitting_gap(cursor_ns, unit_ns);
            if (!gap) {
                break;
            }
            const double start_ns = std::max(gap->begin_ns, cursor_ns);
            auto fit = static_cast<std::uint64_t>(
                std::floor((gap->end_ns - start_ns) / unit_ns));
            fit = std::min(fit, remaining);
            double end_ns = start_ns + static_cast<double>(fit) * unit_ns;
            // Guard the floating-point boundary of the gap itself.
            while (fit != 0 && end_ns > gap->end_ns) {
                --fit;
                end_ns = start_ns + static_cast<double>(fit) * unit_ns;
            }
            if (fit == 0) {
                cursor_ns = gap->end_ns;
                continue;
            }
            timeline.consume_gap(*gap, start_ns, end_ns);
            if (first_start_ns < 0.0) {
                first_start_ns = start_ns;
            }
            finish_ns = std::max(finish_ns, end_ns);
            cursor_ns = end_ns;
            remaining -= fit;
        }
        if (remaining != 0) {
            const double start_ns = std::max(cursor_ns, timeline.ready_ns);
            const double end_ns =
                start_ns + static_cast<double>(remaining) * unit_ns;
            if (!std::isfinite(end_ns)) {
                throw std::runtime_error(
                    "HBF page-run resource reservation time overflowed");
            }
            if (start_ns > timeline.ready_ns) {
                timeline.insert_gap(timeline.ready_ns, start_ns);
            }
            timeline.ready_ns = end_ns;
            if (first_start_ns < 0.0) {
                first_start_ns = start_ns;
            }
            finish_ns = std::max(finish_ns, end_ns);
        }
        timeline.reserved_work_ns += static_cast<double>(units) * unit_ns;
        if (!std::isfinite(timeline.reserved_work_ns)) {
            throw std::runtime_error(
                "HBF resource reservation work exceeds finite double range");
        }
        queue_wait_ns += std::max(0.0, first_start_ns - earliest_ns);
        resource_finish_ns = std::max(resource_finish_ns, finish_ns);
        return finish_ns;
    };

    const double lane_ns = transfer_time_ns(
        page_wire_bytes(), config_.media_lane_bandwidth_GBps);
    const double page_buffer_ns = transfer_time_ns(
        page_wire_bytes(), config_.page_buffer_bandwidth_GBps);
    double aggregate_queue_wait_ns = 0.0;
    for (std::size_t plane_index_value = 0;
         plane_index_value < pages_by_plane.size();
         ++plane_index_value) {
        const auto count = pages_by_plane[plane_index_value];
        if (count == 0) {
            continue;
        }
        auto& plane = planes_.at(plane_index_value);
        const double plane_ready_ns =
            stack_ready_ns[stack_of_plane(plane_index_value)];
        const auto rounds = *std::max_element(
            subarray_counts[plane_index_value].begin(),
            subarray_counts[plane_index_value].end());
        const auto sense_finish = reserve_work(
            plane_ready_ns,
            config_.t_read_page_ns,
            rounds,
            plane.sense_round_calendar,
            aggregate_queue_wait_ns);
        record_plane_media_busy(
            plane,
            sense_finish - static_cast<double>(rounds) *
                config_.t_read_page_ns,
            sense_finish);
        plane.read_count = checked_add(
            plane.read_count, count, "HBF page-run plane reads");
        for (std::size_t index = 0; index < plane.subarrays.size(); ++index) {
            const auto operations = subarray_counts[plane_index_value][index];
            if (operations == 0) {
                continue;
            }
            auto& subarray = plane.subarrays[index];
            reserve_work(
                plane_ready_ns,
                config_.t_read_page_ns,
                operations,
                subarray.timeline,
                aggregate_queue_wait_ns);
            subarray.busy_ns +=
                static_cast<double>(operations) * config_.t_read_page_ns;
            subarray.read_count = checked_add(
                subarray.read_count,
                operations,
                "HBF page-run subarray reads");
        }
        for (std::size_t index = 0; index < plane.media_lanes.size(); ++index) {
            const auto operations = lane_counts[plane_index_value][index];
            if (operations == 0) {
                continue;
            }
            auto& lane = plane.media_lanes[index];
            reserve_work(
                plane_ready_ns,
                lane_ns,
                operations,
                lane.timeline,
                aggregate_queue_wait_ns);
            lane.busy_ns += static_cast<double>(operations) * lane_ns;
            lane.read_count = checked_add(
                lane.read_count, operations, "HBF page-run media-lane reads");
        }
        for (std::size_t index = 0;
             index < plane.page_buffer_banks.size();
             ++index) {
            const auto operations = bank_counts[plane_index_value][index];
            if (operations == 0) {
                continue;
            }
            auto& bank = plane.page_buffer_banks[index];
            reserve_work(
                plane_ready_ns,
                page_buffer_ns,
                operations,
                bank.timeline,
                aggregate_queue_wait_ns);
            bank.busy_ns +=
                static_cast<double>(operations) * page_buffer_ns;
            bank.read_count = checked_add(
                bank.read_count,
                operations,
                "HBF page-run page-buffer reads");
        }
    }

    const double command_tsv_ns = transfer_time_ns(
        config_.command_address_bytes, config_.tsv_bandwidth_GBps);
    const double data_tsv_ns = transfer_time_ns(
        page_wire_bytes(), config_.tsv_bandwidth_GBps);
    const double command_channel_ns = transfer_time_ns(
        config_.command_address_bytes, config_.channel_bandwidth_GBps);
    const double data_channel_ns = transfer_time_ns(
        page_wire_bytes(), config_.channel_bandwidth_GBps);
    const double ecc_issue_ns = transfer_time_ns(
        page_wire_bytes(), config_.ecc_decode_raw_bandwidth_GBps_per_die);
    const double sram_ns = transfer_time_ns(
        config_.page_size_bytes, config_.logic_sram_bandwidth_GBps);
    const double hbio_ns = transfer_time_ns(
        config_.page_size_bytes, config_.hb_io_bandwidth_GBps);

    for (std::size_t index = 0; index < pages_by_die.size(); ++index) {
        const auto count = pages_by_die[index];
        if (count == 0) {
            continue;
        }
        auto& die = dies_.at(index);
        const double die_ready_ns = stack_ready_ns[
            index / (static_cast<std::size_t>(config_.dies_per_channel) *
                     config_.channels_per_stack)];
        reserve_work(
            die_ready_ns,
            config_.flash_tsu_issue_ns,
            count,
            die.source_queues.at(source_index(TransactionSource::User)),
            aggregate_queue_wait_ns);
        reserve_work(
            die_ready_ns,
            config_.flash_tsu_issue_ns,
            count,
            die.sequencer,
            aggregate_queue_wait_ns);
        const auto ecc_finish = reserve_work(
            die_ready_ns,
            ecc_issue_ns,
            count,
            die.ecc_issue,
            breakdown.ecc_queue_wait_ns);
        (void)ecc_finish;
        die.sequencer_busy_ns +=
            static_cast<double>(count) * config_.flash_tsu_issue_ns;
        die.transaction_count = checked_add(
            die.transaction_count,
            count,
            "HBF page-run die transactions");
        die.ecc_issue_busy_ns += static_cast<double>(count) * ecc_issue_ns;
        die.ecc_decode_ops = checked_add(
            die.ecc_decode_ops, count, "HBF page-run die ECC decodes");
        die.page_run_max_ecc_inflight = std::max<std::uint64_t>(
            die.page_run_max_ecc_inflight,
            static_cast<std::uint64_t>(std::ceil(
                config_.ecc_decode_latency_ns / ecc_issue_ns)));
    }
    for (std::size_t index = 0; index < pages_by_channel.size(); ++index) {
        const auto count = pages_by_channel[index];
        if (count == 0) {
            continue;
        }
        auto& channel = channels_.at(index);
        const double channel_ready_ns =
            stack_ready_ns[index / config_.channels_per_stack];
        reserve_work(
            channel_ready_ns,
            command_channel_ns,
            count,
            channel.command,
            aggregate_queue_wait_ns);
        reserve_work(
            channel_ready_ns,
            data_channel_ns,
            count,
            channel.data,
            aggregate_queue_wait_ns);
        channel.command_busy_ns +=
            static_cast<double>(count) * command_channel_ns;
        channel.data_busy_ns +=
            static_cast<double>(count) * data_channel_ns;
        channel.command_count = checked_add(
            channel.command_count,
            count,
            "HBF page-run channel commands");
        channel.data_count = checked_add(
            channel.data_count,
            count,
            "HBF page-run channel transfers");
    }
    for (std::size_t stack = 0; stack < pages_by_stack.size(); ++stack) {
        const auto count = pages_by_stack[stack];
        if (count == 0) {
            continue;
        }
        auto& logic = logic_dies_.at(stack);
        const double logic_ready_ns = stack_ready_ns[stack];
        reserve_work(
            logic_ready_ns,
            command_tsv_ns + data_tsv_ns,
            count,
            logic.tsv,
            aggregate_queue_wait_ns);
        reserve_work(
            logic_ready_ns,
            sram_ns,
            count,
            logic.sram,
            aggregate_queue_wait_ns);
        reserve_work(
            logic_ready_ns,
            hbio_ns,
            count,
            logic.hb_io_data,
            aggregate_queue_wait_ns);
        logic.hb_io_data_busy_ns += static_cast<double>(count) * hbio_ns;
        if (logical) {
            const auto mapping_count = mapping_pages_by_stack.at(stack);
            if (mapping_count != count) {
                throw std::runtime_error(
                    "HBF logical page-run mapping count diverged by stack");
            }
            reserve_work(
                logic_ready_ns,
                config_.ctrl_dram_issue_ns,
                mapping_count,
                logic.mapping_dram_issue,
                aggregate_queue_wait_ns);
        }
    }

    const auto wire_bytes = checked_mul(
        total_pages, page_wire_bytes(), "HBF page-run wire bytes");
    stats_.flash_scheduler_enqueues = checked_add(
        stats_.flash_scheduler_enqueues,
        total_pages,
        "HBF page-run scheduler enqueues");
    stats_.flash_scheduler_issues = checked_add(
        stats_.flash_scheduler_issues,
        total_pages,
        "HBF page-run scheduler issues");
    stats_.ecc_decode_ops = checked_add(
        stats_.ecc_decode_ops, total_pages, "HBF page-run ECC decodes");
    stats_.ecc_codeword_bytes = checked_add(
        stats_.ecc_codeword_bytes, wire_bytes, "HBF page-run ECC bytes");
    stats_.ecc_decode_codeword_bytes = checked_add(
        stats_.ecc_decode_codeword_bytes,
        wire_bytes,
        "HBF page-run ECC decode bytes");
    stats_.ecc_issue_busy_ns += static_cast<double>(total_pages) * ecc_issue_ns;
    stats_.ecc_decode_issue_busy_ns +=
        static_cast<double>(total_pages) * ecc_issue_ns;
    stats_.ecc_decode_latency_work_ns +=
        static_cast<double>(total_pages) * config_.ecc_decode_latency_ns;
    stats_.ecc_decode_queue_wait_ns +=
        breakdown.ecc_queue_wait_ns - ecc_queue_wait_before_ns;

    const auto pages_as_double = static_cast<double>(total_pages);
    breakdown.scheduler_queue_wait_ns += aggregate_queue_wait_ns;
    breakdown.command_ns +=
        pages_as_double * config_.flash_tsu_issue_ns;
    breakdown.array_read_ns += pages_as_double * config_.t_read_page_ns;
    breakdown.media_lane_transfer_ns += pages_as_double * lane_ns;
    breakdown.page_buffer_ns += pages_as_double * page_buffer_ns;
    breakdown.channel_transfer_ns +=
        pages_as_double * (command_channel_ns + data_channel_ns);
    breakdown.tsv_transfer_ns +=
        pages_as_double * (command_tsv_ns + data_tsv_ns);
    breakdown.ecc_latency_ns +=
        pages_as_double * config_.ecc_decode_latency_ns;
    breakdown.sram_staging_ns += pages_as_double * sram_ns;
    breakdown.hb_io_transfer_ns += pages_as_double * hbio_ns;
    if (logical) {
        breakdown.address_mapping_ns +=
            pages_as_double * config_.address_generation_ns;
        breakdown.mapping_dram_ns +=
            pages_as_double * config_.ctrl_dram_latency_ns;
        stats_.mapping_lookup_ops = checked_add(
            stats_.mapping_lookup_ops,
            total_pages,
            "HBF page-run mapping lookups");
        stats_.mapping_user_lookup_ops = checked_add(
            stats_.mapping_user_lookup_ops,
            total_pages,
            "HBF page-run user mapping lookups");
        stats_.mapping_dram_issue_busy_ns +=
            pages_as_double * config_.ctrl_dram_issue_ns;
    }

    // Calibrated startup/drain term of the exact scalar pipeline. For two or
    // more pages on an otherwise idle plane, scalar completion is the first
    // decoded-page path, one additional batch-sense round, then one ECC
    // initiation interval per remaining page. Aggregate resource frontiers
    // above replace only that regular middle; this fixed term preserves the
    // same 4 KiB event boundaries at both ends.
    const double first_page_path_ns =
        command_tsv_ns + command_channel_ns + config_.flash_tsu_issue_ns +
        config_.t_read_page_ns + lane_ns + page_buffer_ns + data_channel_ns +
        data_tsv_ns + config_.ecc_decode_latency_ns + sram_ns + hbio_ns;
    const double pipeline_offset_ns = std::max(
        0.0,
        first_page_path_ns + config_.t_read_page_ns - 2.0 * ecc_issue_ns +
            (logical ? config_.address_generation_ns +
                    config_.ctrl_dram_latency_ns + 3.0 * ecc_issue_ns : 0.0));
    // The pacing budget clamp is a wall-clock lower bound on when this run's
    // energy can have been dissipated; pipeline startup/drain proceeds
    // concurrently with that dissipation, so the calibrated offset extends
    // only the resource frontiers and never stacks on top of the budget.
    const auto finish_ns = std::max(
        resource_finish_ns + pipeline_offset_ns, thermal_budget_finish_ns);
    if (!std::isfinite(finish_ns)) {
        throw std::runtime_error("HBF page-run completion time overflowed");
    }
    for (const auto& span : spans) {
        const auto plane_base = span.plane * config_.blocks_per_plane;
        const auto first_block = span.first_page / config_.pages_per_block;
        const auto last_block =
            (span.first_page + span.page_count - 1) /
            config_.pages_per_block;
        for (auto block = first_block; block <= last_block; ++block) {
            auto& state = blocks_.at(plane_base + block);
            state.issued_media_ready_ns = std::max(
                state.issued_media_ready_ns, finish_ns);
        }
    }
    return finish_ns;
}

std::vector<double> HbfDevice::admit_run_mapping_pages(
    std::uint64_t first_lpn,
    std::uint64_t pages,
    double issued_ns,
    std::vector<std::uint64_t>& touched_vpns,
    Breakdown& breakdown) {
    // Run-granular admission of the cached L2P. A page-striped run touches
    // each stack's mapping pages in ascending group order, exactly the
    // per-stack sequence the scalar engine produces when it walks LPNs one
    // by one, so LRU/miss/fill state stays identical to scalar execution.
    // The per-page DRAM lookup work itself is accounted in bulk by
    // schedule_read_page_run_resources.
    std::vector<double> ready_by_stack(config_.stacks, issued_ns);
    const auto stacks = static_cast<std::uint64_t>(config_.stacks);
    const auto entries_per_mapping_page =
        static_cast<std::uint64_t>(config_.mapping_entries_per_page);
    const auto last_lpn = checked_add(
        first_lpn, pages - 1, "HBF run mapping-admission last LPN");
    const auto first_group =
        (first_lpn / stacks) / entries_per_mapping_page;
    const auto last_group =
        (last_lpn / stacks) / entries_per_mapping_page;
    std::uint64_t distinct_vpns = 0;
    for (auto group = first_group; group <= last_group; ++group) {
        const auto group_first_stripe = checked_mul(
            group,
            entries_per_mapping_page,
            "HBF run mapping-group first stripe");
        const auto group_last_stripe = group_first_stripe +
            entries_per_mapping_page - 1;
        for (std::uint64_t lane = 0; lane < stacks; ++lane) {
            if (last_lpn < lane) {
                continue;
            }
            const auto first_candidate_stripe = first_lpn <= lane ?
                std::uint64_t{0} :
                (first_lpn - lane + stacks - 1) / stacks;
            const auto last_candidate_stripe = (last_lpn - lane) / stacks;
            const auto probe_stripe = std::max(
                group_first_stripe, first_candidate_stripe);
            if (probe_stripe > std::min(
                    group_last_stripe, last_candidate_stripe)) {
                continue;
            }
            const auto probe_lpn = probe_stripe * stacks + lane;
            const auto mapping_vpn = mapping_vpn_for_lpn(probe_lpn);
            const auto stack = stack_for_vpn(mapping_vpn);
            double at_ns = issued_ns;
            ensure_mapping_page_cached(
                mapping_vpn, at_ns, breakdown, nullptr);
            ready_by_stack.at(stack) = std::max(
                ready_by_stack.at(stack), at_ns);
            touched_vpns.push_back(mapping_vpn);
            ++distinct_vpns;
        }
    }
    if (distinct_vpns == 0 || distinct_vpns > pages) {
        throw std::runtime_error(
            "HBF run mapping admission did not conserve mapping pages");
    }
    // The scalar engine touches a run's mapping page once per data page; the
    // first touch was accounted by ensure_mapping_page_cached above and the
    // remaining touches of an already-resident page are hits.
    stats_.mapping_cache_hits = checked_add(
        stats_.mapping_cache_hits,
        pages - distinct_vpns,
        "HBF run mapping-cache folded hits");
    return ready_by_stack;
}

HbfDevice::ReadRunPlan HbfDevice::plan_read_page_runs(
    const PhysicalRequest& request,
    AddressSpace address_space,
    std::uint64_t first_lpn,
    std::uint64_t first_ppn,
    std::uint64_t pages,
    double issued_ns,
    Breakdown& breakdown) {
    constexpr std::uint64_t kMinimumPhysicalRunPages = 128;
    // Below ~1024 pages the compressed run's calibrated startup/drain floor
    // dominates its aggregate frontiers and over-predicts completion by more
    // than the differential-oracle bound (26% at 256 pages, 0.9% at 768,
    // 0.3% at 1024, exact by 2048). Shorter stretches stay on the scalar
    // engine.
    constexpr std::uint64_t kMinimumLogicalRunPages = 1024;
    const bool physical_request = address_space == AddressSpace::Physical;
    const bool static_request = address_space == AddressSpace::Static;
    const bool logical_request = address_space == AddressSpace::Logical;
    const auto minimum_run_pages = (physical_request || static_request) ?
        kMinimumPhysicalRunPages : kMinimumLogicalRunPages;
    ReadRunPlan scalar_plan;
    scalar_plan.scalar_segments.emplace_back(0, pages);
    if (!config_.page_run_acceleration || !config_.batch_activation ||
        pages < minimum_run_pages ||
        request.addr % config_.page_size_bytes != 0 ||
        request.bytes % config_.page_size_bytes != 0 ||
        request.trace.retain_completion_diagnostics ||
        trace_spans_enabled(request.trace) ||
        (config_.read_buffer_pages != 0 &&
         pages <= config_.read_buffer_pages)) {
        return scalar_plan;
    }

    // The queue stores only enough descriptors to cover the real pipeline
    // bandwidth-delay product; page-run scheduling does not relax that
    // architectural bound. Configurations too shallow for even two full
    // batch-activation rounds remain on the scalar credit engine.
    const auto minimum_credits = checked_mul(
        2,
        subarrays_per_plane_,
        "HBF page-run minimum read credits");
    if (config_.page_read_queue_depth_per_stack < minimum_credits) {
        return scalar_plan;
    }

    std::vector<PageRunSpan> whole_request_spans;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> run_segments;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> scalar_segments;
    if (physical_request) {
        const auto first = decode_ppn(first_ppn);
        const auto last = decode_ppn(first_ppn + pages - 1);
        if (plane_index(first) != plane_index(last)) {
            return scalar_plan;
        }
        const auto first_block = first_ppn / config_.pages_per_block;
        const auto last_block =
            (first_ppn + pages - 1) / config_.pages_per_block;
        for (auto block = first_block; block <= last_block; ++block) {
            if (blocks_.at(static_cast<std::size_t>(block)).role !=
                BlockRole::StaticReadOnly) {
                return scalar_plan;
            }
        }
        whole_request_spans.push_back(PageRunSpan{
            .plane = plane_index(first),
            .first_page =
                static_cast<std::uint64_t>(first.block) *
                    config_.pages_per_block + first.page,
            .page_count = pages,
        });
        run_segments.emplace_back(0, pages);
    } else if (static_request) {
        whole_request_spans = static_page_run_spans(first_lpn, pages);
        for (const auto& span : whole_request_spans) {
            const auto plane_base = span.plane * config_.blocks_per_plane;
            const auto first_block = span.first_page /
                config_.pages_per_block;
            const auto last_block =
                (span.first_page + span.page_count - 1) /
                config_.pages_per_block;
            for (auto block = first_block; block <= last_block; ++block) {
                if (blocks_.at(plane_base + block).role !=
                    BlockRole::StaticReadOnly) {
                    throw std::runtime_error(
                        "HBF static page run escapes its reserved extent");
                }
            }
        }
        run_segments.emplace_back(0, pages);
    } else {
        // The compact logical image is striped by mapping-entry lane and then
        // by stack. compact_logical_page_run_spans() clips both ends of every
        // partial stripe, so power-of-two stack counts are not a requirement:
        // the published six-stack backing topology is covered by the same
        // page-conservation proof as four and eight stacks.
        if (!compact_logical_image_ || planes_per_stack() != 64 ||
            first_lpn < compact_logical_image_->first_lpn ||
            pages > compact_logical_image_->page_count ||
            first_lpn - compact_logical_image_->first_lpn >
                compact_logical_image_->page_count - pages) {
            return scalar_plan;
        }
        // Mutated stretches fall back to the scalar engine: their pages live
        // wherever the FTL rewrote them, outside the pristine compact image
        // the run compiler enumerates. The clean stretches between them keep
        // the analytic engine. Segments are request-relative page indices.
        const auto append_scalar = [&](std::uint64_t begin_index,
                                       std::uint64_t count) {
            if (count == 0) {
                return;
            }
            if (!scalar_segments.empty() &&
                scalar_segments.back().first + scalar_segments.back().second ==
                    begin_index) {
                scalar_segments.back().second += count;
                return;
            }
            scalar_segments.emplace_back(begin_index, count);
        };
        const auto append_clean = [&](std::uint64_t begin_lpn,
                                      std::uint64_t end_lpn) {
            if (end_lpn <= begin_lpn) {
                return;
            }
            const auto count = end_lpn - begin_lpn;
            if (count >= kMinimumLogicalRunPages) {
                run_segments.emplace_back(begin_lpn - first_lpn, count);
            } else {
                append_scalar(begin_lpn - first_lpn, count);
            }
        };
        const auto request_end = checked_add(
            first_lpn, pages, "HBF page-run request end");
        std::uint64_t cursor = first_lpn;
        auto overlap = mutated_lpn_ranges_.lower_bound(first_lpn);
        if (overlap != mutated_lpn_ranges_.begin()) {
            const auto previous = std::prev(overlap);
            if (previous->second > first_lpn) {
                overlap = previous;
            }
        }
        for (; overlap != mutated_lpn_ranges_.end() &&
               overlap->first < request_end;
             ++overlap) {
            const auto dirty_begin = std::max(overlap->first, cursor);
            const auto dirty_end = std::min(overlap->second, request_end);
            if (dirty_end <= cursor) {
                continue;
            }
            append_clean(cursor, dirty_begin);
            append_scalar(dirty_begin - first_lpn, dirty_end - dirty_begin);
            cursor = dirty_end;
        }
        append_clean(cursor, request_end);
        if (run_segments.empty()) {
            return scalar_plan;
        }
    }

    // The compressed resource envelope cannot consume an unused subarray slot
    // in a sense round opened by an earlier scalar request. Continuing with
    // acceleration in that state made completion depend on the approximation
    // (up to 6.5% in the scattered-read regression). Keep the fast path only
    // when it is observationally equivalent: if any touched plane advertises
    // a joinable current/future round, use the exact scalar controller for the
    // whole request. Stale rounds before this request's issue time cannot be
    // joined and do not disable acceleration.
    const auto has_joinable_sense_round = [&](const auto& spans) {
        for (const auto& span : spans) {
            const auto& plane = planes_.at(span.plane);
            for (const auto& available :
                 plane.available_sense_rounds_by_subarray) {
                if (available.lower_bound(issued_ns) != available.end()) {
                    return true;
                }
            }
        }
        return false;
    };
    const bool charge_mapping = logical_request &&
        config_.mapping_mode != MappingMode::Direct;
    if (logical_request) {
        for (const auto& [segment_offset, segment_pages] : run_segments) {
            const auto spans = compact_logical_page_run_spans(
                first_lpn + segment_offset, segment_pages);
            if (has_joinable_sense_round(spans) ||
                !read_run_is_ecc_limited(spans, charge_mapping)) {
                return scalar_plan;
            }
        }
    } else if (has_joinable_sense_round(whole_request_spans) ||
               !read_run_is_ecc_limited(whole_request_spans, false)) {
        return scalar_plan;
    }

    ReadRunPlan plan;
    plan.scalar_segments = std::move(scalar_segments);
    for (const auto& [segment_offset, segment_pages] : run_segments) {
        std::vector<PageRunSpan> spans;
        std::vector<std::uint64_t> mapping_pages_by_stack(config_.stacks, 0);
        if (logical_request) {
            spans = compact_logical_page_run_spans(
                first_lpn + segment_offset, segment_pages);
            if (charge_mapping) {
                for (const auto& span : spans) {
                    mapping_pages_by_stack.at(stack_of_plane(span.plane)) =
                        checked_add(
                            mapping_pages_by_stack.at(
                                stack_of_plane(span.plane)),
                            span.page_count,
                            "HBF logical page-run mapping pages");
                }
            }
        } else {
            spans = std::move(whole_request_spans);
        }
        std::vector<std::uint64_t> touched_vpns;
        std::vector<double> mapping_ready_by_stack;
        if (logical_request && config_.mapping_mode == MappingMode::Cached) {
            mapping_ready_by_stack = admit_run_mapping_pages(
                first_lpn + segment_offset,
                segment_pages,
                issued_ns,
                touched_vpns,
                breakdown);
        } else if (
            logical_request &&
            config_.mapping_mode == MappingMode::Direct &&
            !direct_write_ready_ns_by_lpn_.empty()) {
            // In-place writes keep the image pristine, so runs stay whole;
            // read-after-write ordering is preserved by lifting the touched
            // stacks to the pending in-place program completions. The table
            // is LPN-ordered, so only the overlapped interval is visited.
            const auto segment_first = first_lpn + segment_offset;
            const auto segment_end = segment_first + segment_pages;
            for (auto pending = direct_write_ready_ns_by_lpn_.lower_bound(
                     segment_first);
                 pending != direct_write_ready_ns_by_lpn_.end() &&
                 pending->first < segment_end;
                 ++pending) {
                if (pending->second > issued_ns) {
                    if (mapping_ready_by_stack.empty()) {
                        mapping_ready_by_stack.assign(
                            config_.stacks, issued_ns);
                    }
                    auto& slot = mapping_ready_by_stack.at(
                        stack_for_lpn(pending->first));
                    slot = std::max(slot, pending->second);
                }
            }
        }
        const auto segment_finish_ns = schedule_read_page_run_resources(
            spans,
            mapping_pages_by_stack,
            segment_pages,
            issued_ns,
            charge_mapping,
            breakdown,
            mapping_ready_by_stack.empty() ?
                nullptr : &mapping_ready_by_stack);
        if (address_heatmap_ != nullptr) {
            const auto source = resolve_heatmap_source(
                TransactionSource::User, request.heatmap_source);
            for (const auto& span : spans) {
                const auto plane_first_ppn = checked_mul(
                    static_cast<std::uint64_t>(span.plane),
                    static_cast<std::uint64_t>(config_.blocks_per_plane) *
                        config_.pages_per_block,
                    "HBF page-run heatmap plane offset");
                const auto first_ppn = checked_add(
                    plane_first_ppn,
                    span.first_page,
                    "HBF page-run heatmap first PPN");
                address_heatmap_->record_contiguous_accesses(
                    AddressTrafficRecord{
                        .domain = AddressDomain::HbfPhysical,
                        .direction = TrafficDirection::Read,
                        .source = source,
                        .address = checked_mul(
                            first_ppn,
                            config_.page_size_bytes,
                            "HBF page-run heatmap first address"),
                        .bytes = config_.page_size_bytes,
                    },
                    span.page_count);
            }
        }
        // The scalar engine advances a mapping line's evictable horizon with
        // every page it translates, through roughly the request's completion.
        // Pin the run's lines to the same horizon so eviction pressure from
        // concurrent misses cannot recycle them earlier than scalar execution
        // would allow.
        for (const auto mapping_vpn : touched_vpns) {
            auto& cache = mapping_cache_by_stack_.at(stack_for_vpn(mapping_vpn));
            const auto found = cache.find(mapping_vpn);
            if (found != cache.end()) {
                found->second.evictable_ns = std::max(
                    found->second.evictable_ns, segment_finish_ns);
            }
        }
        plan.run_pages = checked_add(
            plan.run_pages, segment_pages, "HBF page-run planned pages");
        plan.run_finish_ns = std::max(
            plan.run_finish_ns.value_or(segment_finish_ns),
            segment_finish_ns);
        stats_.page_run_pages = checked_add(
            stats_.page_run_pages, segment_pages, "HBF page-run pages");
        stats_.read_buffer_misses = checked_add(
            stats_.read_buffer_misses,
            segment_pages,
            "HBF page-run buffer misses");
        stats_.page_read_admission_events = checked_add(
            stats_.page_read_admission_events,
            segment_pages,
            "HBF page-run admission events");
    }
    stats_.page_run_requests = checked_add(
        stats_.page_run_requests, 1, "HBF page-run requests");
    if (physical_request) {
        stats_.page_run_physical_requests = checked_add(
            stats_.page_run_physical_requests,
            1,
            "HBF physical page-run requests");
    } else if (static_request) {
        stats_.page_run_static_requests = checked_add(
            stats_.page_run_static_requests,
            1,
            "HBF static page-run requests");
    } else {
        stats_.page_run_logical_requests = checked_add(
            stats_.page_run_logical_requests,
            1,
            "HBF logical page-run requests");
    }
    return plan;
}

void HbfDevice::thermal_advance(
    ThermalNodeState& node,
    double t_ns,
    bool record) const {
    const double target_ns = std::max(t_ns, node.clock_ns);
    const double idle_c = thermal_idle_temperature_c_;
    const auto decay_to = [&](double until_ns) {
        double dt_ns = until_ns - node.clock_ns;
        if (dt_ns <= 0.0) {
            return;
        }
        if (node.throttled) {
            // Only deposits raise temperature, so a decay segment can only
            // cool: solve the exact release-threshold crossing instead of
            // sampling it.
            if (node.temperature_c > config_.thermal_release_c) {
                const double gap = node.temperature_c - idle_c;
                const double release_gap =
                    config_.thermal_release_c - idle_c;
                const double crossing_ns =
                    thermal_tau_ns_ * std::log(gap / release_gap);
                if (crossing_ns >= dt_ns) {
                    if (record) {
                        stats_.thermal_throttled_span_ns += dt_ns;
                    }
                    node.temperature_c = idle_c +
                        gap * std::exp(-dt_ns / thermal_tau_ns_);
                    node.clock_ns = until_ns;
                    return;
                }
                if (record) {
                    stats_.thermal_throttled_span_ns += crossing_ns;
                }
                node.temperature_c = config_.thermal_release_c;
                node.clock_ns += crossing_ns;
                dt_ns -= crossing_ns;
            }
            node.throttled = false;
        }
        node.temperature_c = idle_c +
            (node.temperature_c - idle_c) * std::exp(-dt_ns / thermal_tau_ns_);
        node.clock_ns = until_ns;
    };
    decay_to(target_ns);
}

HbfDevice::ThermalAdmission HbfDevice::thermal_pace_media(
    std::size_t stack,
    double earliest_ns,
    double energy_j,
    std::uint64_t media_ops,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    if (!config_.thermal_enabled || energy_j <= 0.0) {
        return ThermalAdmission{earliest_ns, earliest_ns};
    }
    auto& state = thermal_stacks_.at(stack);
    // The sensor is advanced to the admission instant before this work's
    // own heat is considered: firmware reacts to already dissipated work,
    // never to the command it is about to issue.
    thermal_advance(state.node, earliest_ns, true);
    if (!state.node.throttled) {
        return ThermalAdmission{earliest_ns, earliest_ns};
    }
    const double slot_ns = energy_j / thermal_pacing_power_w_ * 1e9;
    const double start_ns =
        std::max(earliest_ns, state.pacing_frontier_ns);
    state.pacing_frontier_ns = start_ns + slot_ns;
    const double wait_ns = start_ns - earliest_ns;
    breakdown.scheduler_queue_wait_ns += wait_ns;
    if (spans != nullptr) {
        trace_wait(
            spans,
            "stack" + std::to_string(stack) + "/thermal",
            earliest_ns,
            start_ns,
            "wait_thermal_pacing");
    }
    stats_.thermal_throttled_media_ops = checked_add(
        stats_.thermal_throttled_media_ops,
        media_ops,
        "HBF thermal paced media ops");
    stats_.thermal_throttle_wait_ns += wait_ns;
    stats_.thermal_pacing_busy_ns += slot_ns;
    return ThermalAdmission{start_ns, state.pacing_frontier_ns};
}

void HbfDevice::thermal_deposit_media(
    std::size_t stack,
    double at_ns,
    double energy_j) {
    if (!config_.thermal_enabled || energy_j <= 0.0) {
        return;
    }
    auto& state = thermal_stacks_.at(stack);
    thermal_advance(state.node, at_ns, true);
    stats_.thermal_media_energy_j += energy_j;
    state.node.temperature_c +=
        energy_j / config_.thermal_capacitance_j_per_c;
    state.node.peak_c =
        std::max(state.node.peak_c, state.node.temperature_c);
    if (!state.node.throttled &&
        state.node.temperature_c >= config_.thermal_throttle_c) {
        state.node.throttled = true;
        stats_.thermal_throttle_engagements = checked_add(
            stats_.thermal_throttle_engagements,
            1,
            "HBF thermal throttle engagements");
    }
}

HbfDevice::ThermalAdmission HbfDevice::thermal_admit_media(
    std::size_t stack,
    double earliest_ns,
    double energy_j,
    std::uint64_t media_ops,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    const auto admission = thermal_pace_media(
        stack, earliest_ns, energy_j, media_ops, breakdown, spans);
    thermal_deposit_media(stack, admission.ready_ns, energy_j);
    return admission;
}

double HbfDevice::schedule_read_page(
    std::uint64_t ppn,
    double earliest_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    TransactionSource source,
    HeatmapTrafficSource heatmap_attribution,
    ReadPayloadRoute route,
    std::uint64_t external_payload_bytes,
    double* decoded_ready_ns,
    bool thermally_preadmitted) {
    const auto addr = decode_ppn(ppn);
    auto& logic_die = logic_dies_.at(stack_index(addr));
    auto& channel = channels_.at(channel_index(addr));
    auto& plane = planes_.at(plane_index(addr));
    std::string source_label;
    if (spans != nullptr) {
        source_label = source_name(source);
    }
    const auto subarray_index_value = subarray_index(addr);
    const auto lane_index = media_lane_index(addr);
    const auto buffer_bank_index = page_buffer_bank_index(addr);
    auto& subarray = plane.subarrays.at(subarray_index_value);
    auto& lane = plane.media_lanes.at(lane_index);
    auto& buffer_bank = plane.page_buffer_banks.at(buffer_bank_index);
    if (thermally_preadmitted) {
        // The enclosing scalar request already paced this stack; the page
        // contributes its exact media heat here.
        thermal_deposit_media(
            stack_index(addr), earliest_ns, thermal_read_energy_j_);
    } else {
        earliest_ns = thermal_admit_media(
            stack_index(addr),
            earliest_ns,
            thermal_read_energy_j_,
            1,
            breakdown,
            spans).ready_ns;
    }
    const double command_done = schedule_command_path(
        addr, earliest_ns, breakdown, spans, source);
    auto tsu = schedule_flash_transaction(
        addr,
        command_done,
        breakdown,
        spans,
        source,
        TransactionKind::Read);

    const double lane_ns = transfer_time_ns(
        page_wire_bytes(), config_.media_lane_bandwidth_GBps);
    const double page_buffer_ns = transfer_time_ns(
        page_wire_bytes(), config_.page_buffer_bandwidth_GBps);
    ScheduledTransfer sense;
    ScheduledTransfer lane_transfer;
    ScheduledTransfer page_buffer;
    double local_ready_ns = tsu.finish_ns;
    for (;;) {
        subarray.timeline.prune_before(reservation_causal_watermark_ns_);
        lane.timeline.prune_before(reservation_causal_watermark_ns_);
        buffer_bank.timeline.prune_before(reservation_causal_watermark_ns_);

        SenseRoundCandidate batch_candidate;
        double sense_start_ns = 0.0;
        if (config_.batch_activation) {
            batch_candidate = preview_sense_round(
                plane, subarray_index_value, local_ready_ns);
            sense_start_ns = batch_candidate.start_ns;
        } else {
            sense_start_ns = subarray.timeline.preview_start(
                local_ready_ns, config_.t_read_page_ns);
        }
        const double sense_finish_ns = sense_start_ns + config_.t_read_page_ns;
        const double lane_start_ns = lane.timeline.preview_start(
            sense_finish_ns, lane_ns);
        const double lane_finish_ns = lane_start_ns + lane_ns;
        const double buffer_start_ns = buffer_bank.timeline.preview_start(
            lane_finish_ns, page_buffer_ns);
        const double buffer_finish_ns = buffer_start_ns + page_buffer_ns;

        const auto conflict = std::lower_bound(
            plane.full_plane_windows.begin(),
            plane.full_plane_windows.end(),
            sense_start_ns,
            [](const PlaneState::BusyWindow& window, double start) {
                return window.end_ns <= start;
            });
        if (conflict != plane.full_plane_windows.end() &&
            conflict->begin_ns < buffer_finish_ns) {
            local_ready_ns = std::max(local_ready_ns, conflict->end_ns);
            continue;
        }

        if (config_.batch_activation) {
            sense = commit_sense_round(
                plane,
                subarray_index_value,
                local_ready_ns,
                batch_candidate);
        } else {
            sense = reserve(
                sense_start_ns, config_.t_read_page_ns, subarray.timeline);
            if (sense.start_ns != sense_start_ns) {
                throw std::runtime_error(
                    "HBF independent sense moved after path preview");
            }
            sense.wait_ns = std::max(0.0, sense.start_ns - tsu.finish_ns);
        }
        lane_transfer = reserve(lane_start_ns, lane_ns, lane.timeline);
        page_buffer = reserve(
            buffer_start_ns, page_buffer_ns, buffer_bank.timeline);
        if (lane_transfer.start_ns != lane_start_ns ||
            page_buffer.start_ns != buffer_start_ns) {
            throw std::runtime_error(
                "HBF read local data path moved after atomic preview");
        }
        // Exact commits use the already-previewed start as their reservation
        // key, so reserve() correctly reports zero wait. Preserve the actual
        // pipeline queueing relative to the producer instead: otherwise the
        // trace shows lane/page-buffer stalls that disappear from Breakdown.
        lane_transfer.wait_ns = std::max(
            0.0, lane_transfer.start_ns - sense.finish_ns);
        page_buffer.wait_ns = std::max(
            0.0, page_buffer.start_ns - lane_transfer.finish_ns);
        break;
    }
    const double media_start = sense.start_ns;
    breakdown.scheduler_queue_wait_ns += std::max(0.0, media_start - tsu.finish_ns);
    const double sense_done = media_start + config_.t_read_page_ns;
    breakdown.array_read_ns += config_.t_read_page_ns;
    if (spans != nullptr) {
        const auto entity = subarray_entity(addr, subarray_index_value);
        trace_wait(
            spans, entity, tsu.finish_ns, media_start, "wait_subarray");
        add_trace_span(
            spans,
            source_label + "/array_read",
            "flash_array",
            entity,
            media_start,
            sense_done,
            true,
            "plane" + std::to_string(addr.plane) +
                " subarray" + std::to_string(subarray_index_value) +
                " lane" + std::to_string(lane_index));
    }
    subarray.busy_ns += config_.t_read_page_ns;
    subarray.read_count++;

    breakdown.scheduler_queue_wait_ns += lane_transfer.wait_ns;
    breakdown.media_lane_transfer_ns += lane_ns;
    if (spans != nullptr) {
        const auto entity = media_lane_entity(addr, lane_index);
        trace_wait(
            spans,
            entity,
            sense_done,
            lane_transfer.start_ns,
            "wait_media_lane");
        add_trace_span(
            spans,
            source_label + "/array_to_page_buffer",
            "media_lane",
            entity,
            lane_transfer.start_ns,
            lane_transfer.finish_ns,
            true,
            std::to_string(page_wire_bytes()) + "B raw from subarray" +
                std::to_string(subarray_index_value));
    }
    lane.busy_ns += lane_ns;
    lane.read_count++;
    record_plane_media_busy(plane, media_start, sense_done);
    plane.read_count++;

    breakdown.scheduler_queue_wait_ns += page_buffer.wait_ns;
    breakdown.page_buffer_ns += page_buffer_ns;
    buffer_bank.busy_ns += page_buffer_ns;
    buffer_bank.read_count++;
    if (spans != nullptr) {
        const auto entity = page_buffer_bank_entity(addr, buffer_bank_index);
        trace_wait(
            spans,
            entity,
            lane_transfer.finish_ns,
            page_buffer.start_ns,
            "wait_page_buffer_bank");
        add_trace_span(
            spans,
            source_label + "/page_buffer_out",
            "page_buffer",
            entity,
            page_buffer.start_ns,
            page_buffer.finish_ns,
            true,
            std::to_string(page_wire_bytes()) + "B raw bank" +
                std::to_string(buffer_bank_index));
    }
    auto& block = blocks_.at(static_cast<std::size_t>(
        ppn / config_.pages_per_block));
    block.issued_media_ready_ns = std::max(
        block.issued_media_ready_ns, page_buffer.finish_ns);

    const double channel_ns = transfer_time_ns(page_wire_bytes(), config_.channel_bandwidth_GBps);
    auto channel_transfer = reserve(page_buffer.finish_ns, channel_ns, channel.data);
    breakdown.scheduler_queue_wait_ns += channel_transfer.wait_ns;
    breakdown.channel_transfer_ns += channel_ns;
    channel.data_busy_ns += channel_ns;
    channel.data_count++;
    if (spans != nullptr) {
        const auto entity = channel_entity(addr);
        trace_wait(
            spans,
            entity,
            page_buffer.finish_ns,
            channel_transfer.start_ns,
            "wait_channel_data");
        add_trace_span(
            spans,
            source_label + "/data_out_channel",
            "flash_channel",
            entity,
            channel_transfer.start_ns,
            channel_transfer.finish_ns,
            true,
            std::to_string(page_wire_bytes()) + "B raw");
    }

    const double tsv_ns = transfer_time_ns(page_wire_bytes(), config_.tsv_bandwidth_GBps);
    breakdown.tsv_transfer_ns += tsv_ns;
    auto tsv = reserve(channel_transfer.finish_ns, tsv_ns, logic_die.tsv);
    breakdown.scheduler_queue_wait_ns += tsv.wait_ns;
    if (spans != nullptr) {
        const auto entity =
            "stack" + std::to_string(addr.stack) + "/tsv";
        trace_wait(
            spans,
            entity,
            channel_transfer.finish_ns,
            tsv.start_ns,
            "wait_tsv_data");
        add_trace_span(
            spans,
            source_label + "/data_out_tsv",
            "tsv",
            entity,
            tsv.start_ns,
            tsv.finish_ns,
            true,
            std::to_string(page_wire_bytes()) + "B raw");
    }

    // Raw codeword bytes cross the flash channel and TSV. Decode happens on
    // the logic die before decoded payload enters SRAM and leaves over the
    // external HBIO; parity/OOB bytes never consume external payload BW.
    const double ecc_done = schedule_ecc(
        addr, tsv.finish_ns, breakdown, spans, source, true);
    const double sram_ns = transfer_time_ns(
        config_.page_size_bytes, config_.logic_sram_bandwidth_GBps);
    auto sram = reserve(ecc_done, sram_ns, logic_die.sram);
    breakdown.scheduler_queue_wait_ns += sram.wait_ns;
    breakdown.sram_staging_ns += sram_ns;
    if (spans != nullptr) {
        const auto entity = logic_entity(addr.stack);
        trace_wait(spans, entity, ecc_done, sram.start_ns, "wait_sram");
        add_trace_span(
            spans,
            source_label + "/sram_stage_read",
            "sram",
            entity,
            sram.start_ns,
            sram.finish_ns,
            true,
            std::to_string(config_.page_size_bytes) + "B payload");
    }
    if (address_heatmap_ != nullptr) {
        address_heatmap_->record(AddressTrafficRecord{
            .domain = AddressDomain::HbfPhysical,
            .direction = TrafficDirection::Read,
            .source = resolve_heatmap_source(source, heatmap_attribution),
            .address = checked_mul(
                ppn,
                config_.page_size_bytes,
                "HBF heatmap page-read address"),
            .bytes = config_.page_size_bytes,
        });
    }
    if (decoded_ready_ns != nullptr) {
        *decoded_ready_ns = sram.finish_ns;
    }
    if (route == ReadPayloadRoute::Internal) {
        if (external_payload_bytes != 0) {
            throw std::runtime_error(
                "HBF internal read cannot request external payload bytes");
        }
        return sram.finish_ns;
    }
    if (spans != nullptr) {
        return schedule_external_read_egress(
            addr.stack,
            external_payload_bytes,
            sram.finish_ns,
            breakdown,
            spans,
            source_label + "/data_out_hbio",
            "decoded page read");
    }
    return schedule_external_read_egress(
        addr.stack,
        external_payload_bytes,
        sram.finish_ns,
        breakdown,
        spans,
        {},
        {});
}

double HbfDevice::schedule_program_page(
    std::uint64_t ppn,
    double earliest_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    TransactionSource source,
    HeatmapTrafficSource heatmap_attribution) {
    const auto addr = decode_ppn(ppn);
    auto& logic_die = logic_dies_.at(stack_index(addr));
    auto& channel = channels_.at(channel_index(addr));
    auto& plane = planes_.at(plane_index(addr));
    std::string source_label;
    if (spans != nullptr) {
        source_label = source_name(source);
    }
    earliest_ns = thermal_admit_media(
        stack_index(addr),
        earliest_ns,
        thermal_program_energy_j_,
        1,
        breakdown,
        spans).ready_ns;
    const double command_done = schedule_command_path(
        addr, earliest_ns, breakdown, spans, source);

    // The payload is already resident in logic-die SRAM: user writes crossed
    // HBIO at request ingress, while mapping/GC data was produced internally.
    // Programming reads one full decoded page into ECC, then sends the raw
    // page+OOB codeword over TSV and the flash channel.
    const double sram_ns = transfer_time_ns(
        config_.page_size_bytes, config_.logic_sram_bandwidth_GBps);
    auto sram = reserve(command_done, sram_ns, logic_die.sram);
    breakdown.scheduler_queue_wait_ns += sram.wait_ns;
    breakdown.sram_staging_ns += sram_ns;
    if (spans != nullptr) {
        const auto entity = logic_entity(addr.stack);
        trace_wait(
            spans, entity, command_done, sram.start_ns, "wait_sram");
        add_trace_span(
            spans,
            source_label + "/sram_stage_write",
            "sram",
            entity,
            sram.start_ns,
            sram.finish_ns,
            true,
            std::to_string(config_.page_size_bytes) + "B payload");
    }

    const double ecc_done = schedule_ecc(
        addr, sram.finish_ns, breakdown, spans, source, false);

    const double tsv_ns = transfer_time_ns(page_wire_bytes(), config_.tsv_bandwidth_GBps);
    breakdown.tsv_transfer_ns += tsv_ns;
    auto tsv = reserve(ecc_done, tsv_ns, logic_die.tsv);
    breakdown.scheduler_queue_wait_ns += tsv.wait_ns;
    if (spans != nullptr) {
        const auto entity =
            "stack" + std::to_string(addr.stack) + "/tsv";
        trace_wait(spans, entity, ecc_done, tsv.start_ns, "wait_tsv_data");
        add_trace_span(
            spans,
            source_label + "/data_in_tsv",
            "tsv",
            entity,
            tsv.start_ns,
            tsv.finish_ns,
            true,
            std::to_string(page_wire_bytes()) + "B raw");
    }

    const double channel_ns = transfer_time_ns(page_wire_bytes(), config_.channel_bandwidth_GBps);
    auto channel_transfer = reserve(tsv.finish_ns, channel_ns, channel.data);
    breakdown.scheduler_queue_wait_ns += channel_transfer.wait_ns;
    breakdown.channel_transfer_ns += channel_ns;
    channel.data_busy_ns += channel_ns;
    channel.data_count++;
    if (spans != nullptr) {
        const auto entity = channel_entity(addr);
        trace_wait(
            spans,
            entity,
            tsv.finish_ns,
            channel_transfer.start_ns,
            "wait_channel_data");
        add_trace_span(
            spans,
            source_label + "/data_in_channel",
            "flash_channel",
            entity,
            channel_transfer.start_ns,
            channel_transfer.finish_ns,
            true,
            std::to_string(page_wire_bytes()) + "B raw");
    }

    auto tsu = schedule_flash_transaction(
        addr,
        channel_transfer.finish_ns,
        breakdown,
        spans,
        source,
        TransactionKind::Program);

    const double barrier_ns =
        config_.t_program_page_ns + config_.t_program_verify_ns;
    const double media_start = preview_full_plane_window(
        plane, tsu.finish_ns, barrier_ns);
    breakdown.scheduler_queue_wait_ns += std::max(0.0, media_start - tsu.finish_ns);
    const double program_done = media_start + config_.t_program_page_ns;
    // Summed exactly as the calendar reservations below (media_start +
    // barrier_ns): floating-point addition is not associative, and the
    // recorded full-plane window must equal the reserved interval to the
    // last ulp or back-to-back programs read as overlapping.
    const double verify_done = media_start + barrier_ns;
    breakdown.array_program_ns += config_.t_program_page_ns;
    breakdown.program_verify_ns += config_.t_program_verify_ns;
    if (spans != nullptr) {
        const auto entity = plane_entity(addr);
        trace_wait(
            spans, entity, tsu.finish_ns, media_start, "wait_plane_array");
        add_trace_span(
            spans,
            source_label + "/array_program",
            "flash_array",
            entity,
            media_start,
            program_done);
        add_trace_span(
            spans,
            source_label + "/program_verify",
            "flash_array",
            entity,
            program_done,
            verify_done);
    }
    const auto round_barrier = reserve(
        media_start, barrier_ns, plane.sense_round_calendar);
    if (round_barrier.start_ns != media_start) {
        throw std::runtime_error(
            "HBF program barrier diverged from batch-round calendar");
    }
    for (auto& subarray : plane.subarrays) {
        const auto barrier = reserve(media_start, barrier_ns, subarray.timeline);
        if (barrier.start_ns != media_start) {
            throw std::runtime_error(
                "HBF program barrier diverged across subarray calendars");
        }
    }
    // Program is a non-preemptible full-plane operation. The previous
    // suspend path could move this frontier after returning a completion,
    // violating causality, so it has been removed until a preemptible event
    // primitive with revisable completion dependencies exists.
    for (auto& lane : plane.media_lanes) {
        const auto barrier = reserve(media_start, barrier_ns, lane.timeline);
        if (barrier.start_ns != media_start) {
            throw std::runtime_error(
                "HBF program barrier diverged across media-lane calendars");
        }
    }
    for (auto& bank : plane.page_buffer_banks) {
        const auto barrier = reserve(media_start, barrier_ns, bank.timeline);
        if (barrier.start_ns != media_start) {
            throw std::runtime_error(
                "HBF program barrier diverged across page-buffer calendars");
        }
    }
    record_full_plane_window(plane, media_start, verify_done);
    record_plane_media_busy(plane, media_start, verify_done);
    plane.program_count++;
    auto& block = blocks_.at(static_cast<std::size_t>(
        ppn / config_.pages_per_block));
    block.issued_media_ready_ns = std::max(
        block.issued_media_ready_ns, verify_done);
    if (address_heatmap_ != nullptr) {
        address_heatmap_->record(AddressTrafficRecord{
            .domain = AddressDomain::HbfPhysical,
            .direction = TrafficDirection::Write,
            .source = resolve_heatmap_source(source, heatmap_attribution),
            .address = checked_mul(
                ppn,
                config_.page_size_bytes,
                "HBF heatmap page-program address"),
            .bytes = config_.page_size_bytes,
        });
    }
    return verify_done;
}

double HbfDevice::schedule_erase_block(
    std::size_t block,
    double earliest_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    TransactionSource source,
    HeatmapTrafficSource heatmap_attribution) {
    const auto ppn = static_cast<std::uint64_t>(block) * config_.pages_per_block;
    const auto addr = decode_ppn(ppn);
    auto& plane = planes_.at(plane_index(addr));
    std::string source_label;
    if (spans != nullptr) {
        source_label = source_name(source);
    }

    if (blocks_.at(block).issued_media_ready_ns > earliest_ns) {
        if (spans != nullptr) {
            trace_wait(
                spans,
                addr.path(),
                earliest_ns,
                blocks_.at(block).issued_media_ready_ns,
                "wait_target_block_access");
        }
        breakdown.scheduler_queue_wait_ns +=
            blocks_.at(block).issued_media_ready_ns - earliest_ns;
        earliest_ns = blocks_.at(block).issued_media_ready_ns;
    }

    earliest_ns = thermal_admit_media(
        stack_index(addr),
        earliest_ns,
        thermal_erase_energy_j_,
        1,
        breakdown,
        spans).ready_ns;
    const double command_done = schedule_command_path(
        addr, earliest_ns, breakdown, spans, source);
    auto tsu = schedule_flash_transaction(
        addr,
        command_done,
        breakdown,
        spans,
        source,
        TransactionKind::Erase);

    const double barrier_ns = config_.t_erase_block_ns;
    const double media_start = preview_full_plane_window(
        plane, tsu.finish_ns, barrier_ns);
    breakdown.scheduler_queue_wait_ns += std::max(0.0, media_start - tsu.finish_ns);
    const double erase_done = media_start + config_.t_erase_block_ns;
    breakdown.array_erase_ns += config_.t_erase_block_ns;
    if (spans != nullptr) {
        const auto entity = plane_entity(addr);
        trace_wait(
            spans, entity, tsu.finish_ns, media_start, "wait_plane_array");
        add_trace_span(
            spans,
            source_label + "/block_erase",
            "flash_array",
            entity,
            media_start,
            erase_done,
            true,
            "block" + std::to_string(addr.block));
    }
    const auto round_barrier = reserve(
        media_start, barrier_ns, plane.sense_round_calendar);
    if (round_barrier.start_ns != media_start) {
        throw std::runtime_error(
            "HBF erase barrier diverged from batch-round calendar");
    }
    for (auto& subarray : plane.subarrays) {
        const auto barrier = reserve(media_start, barrier_ns, subarray.timeline);
        if (barrier.start_ns != media_start) {
            throw std::runtime_error(
                "HBF erase barrier diverged across subarray calendars");
        }
    }
    for (auto& lane : plane.media_lanes) {
        const auto barrier = reserve(media_start, barrier_ns, lane.timeline);
        if (barrier.start_ns != media_start) {
            throw std::runtime_error(
                "HBF erase barrier diverged across media-lane calendars");
        }
    }
    for (auto& bank : plane.page_buffer_banks) {
        const auto barrier = reserve(media_start, barrier_ns, bank.timeline);
        if (barrier.start_ns != media_start) {
            throw std::runtime_error(
                "HBF erase barrier diverged across page-buffer calendars");
        }
    }
    record_full_plane_window(plane, media_start, erase_done);
    record_plane_media_busy(plane, media_start, erase_done);
    plane.erase_count++;
    // A scheduled erase is committed to happen and the block is already
    // unallocatable, so its P/E cycle is counted here, at issue, exactly like
    // stats_.block_erases; the later commit only resets the page state.
    auto& erased_block = blocks_.at(block);
    if (erased_block.erase_count == std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("HBF block erase count overflowed");
    }
    erased_block.erase_count++;
    if (address_heatmap_ != nullptr) {
        address_heatmap_->record(AddressTrafficRecord{
            .domain = AddressDomain::HbfPhysical,
            .direction = TrafficDirection::Erase,
            .source = resolve_heatmap_source(source, heatmap_attribution),
            .address = checked_mul(
                ppn,
                config_.page_size_bytes,
                "HBF heatmap block-erase address"),
            .bytes = checked_mul(
                config_.pages_per_block,
                config_.page_size_bytes,
                "HBF heatmap block-erase bytes"),
        });
    }
    return erase_done;
}

void HbfDevice::refresh_accounting_stats() const {
    std::uint64_t free_pages = 0;
    std::uint64_t valid_pages = 0;
    std::uint64_t invalid_pages = 0;
    std::uint64_t pending_program_pages = 0;
    std::uint64_t pending_mapping_publications = 0;
    std::uint64_t static_unmaterialized_pages = 0;
    std::uint64_t raw_reserved_pages = 0;
    std::uint64_t writable_blocks = 0;
    std::uint64_t worn_blocks = 0;
    std::uint64_t block_erase_count_sum = 0;
    long double block_erase_count_sum_squares = 0.0L;
    std::uint32_t min_block_erase_count =
        std::numeric_limits<std::uint32_t>::max();
    std::map<std::uint32_t, std::uint64_t> block_erase_count_histogram;
    std::uint32_t max_block_erase_count = 0;
    std::vector<std::uint64_t> free_pages_per_stack(config_.stacks, 0);
    std::vector<std::uint8_t> free_pool_membership(blocks_.size(), 0);

    double mapping_dram_issue_busy_ns = 0.0;
    double write_buffer_dram_issue_busy_ns = 0.0;
    for (const auto& logic_die : logic_dies_) {
        mapping_dram_issue_busy_ns +=
            logic_die.mapping_dram_issue.reserved_work_ns;
        write_buffer_dram_issue_busy_ns +=
            logic_die.write_buffer_dram_issue.reserved_work_ns;
    }
    stats_.mapping_dram_issue_busy_ns = mapping_dram_issue_busy_ns;
    stats_.write_buffer_dram_issue_busy_ns =
        write_buffer_dram_issue_busy_ns;
    const auto mapping_table_bytes = checked_mul(
        mapping_table_bytes_per_stack_,
        config_.stacks,
        "HBF mapping-table accounting");
    if (stats_.mapping_table_bytes != mapping_table_bytes ||
        stats_.mapping_table_bytes_per_stack !=
            mapping_table_bytes_per_stack_ ||
        stats_.mapping_table_pages_per_stack !=
            mapping_table_pages_per_stack_ ||
        stats_.mapping_dram_resources != config_.stacks) {
        throw std::runtime_error(
            "HBF mapping-table footprint accounting diverged");
    }
    const auto expected_write_buffer_dram =
        derive_write_buffer_dram_capacity(config_);
    if (stats_.controller_dram_budget_bytes != config_.ctrl_dram_bytes ||
        stats_.controller_dram_budget_bytes_per_stack !=
            config_.ctrl_dram_bytes / config_.stacks ||
        stats_.write_buffer_capacity_bytes !=
            expected_write_buffer_dram.total_bytes ||
        stats_.write_buffer_capacity_bytes_per_stack !=
            expected_write_buffer_dram.bytes_per_stack) {
        throw std::runtime_error(
            "HBF shared controller-DRAM capacity accounting diverged");
    }
    if (config_.mapping_mode == MappingMode::FullResident) {
        const auto required_ctrl_dram_per_stack = checked_add(
            mapping_table_bytes_per_stack_,
            expected_write_buffer_dram.bytes_per_stack,
            "HBF resident controller-DRAM audit");
        if (stats_.resident_mapping_table_bytes != mapping_table_bytes ||
            stats_.resident_mapping_table_bytes_per_stack !=
                mapping_table_bytes_per_stack_ ||
            stats_.resident_mapping_pages_per_stack !=
                mapping_table_pages_per_stack_ ||
            config_.ctrl_dram_bytes / config_.stacks <
                required_ctrl_dram_per_stack ||
            stats_.mapping_directory_entry_bytes != 0 ||
            stats_.mapping_directory_bytes != 0 ||
            stats_.mapping_directory_bytes_per_stack != 0 ||
            stats_.mapping_cache_capacity_bytes != 0 ||
            stats_.mapping_cache_capacity_bytes_per_stack != 0 ||
            stats_.mapping_cache_pages_per_stack != 0 ||
            stats_.mapping_cache_entries != 0 ||
            stats_.mapping_cache_peak_entries != 0 ||
            stats_.mapping_cache_hits != 0 ||
            stats_.mapping_cache_misses != 0 ||
            stats_.mapping_cache_erased_misses != 0 ||
            stats_.mapping_cache_coalesced_misses != 0 ||
            stats_.mapping_cache_evictions != 0 ||
            stats_.mapping_cache_dirty_evictions != 0 ||
            stats_.mapping_media_reads != 0 ||
            stats_.mapping_media_read_bytes != 0) {
            throw std::runtime_error(
                "HBF full-resident mapping accounting diverged");
        }
    } else if (config_.mapping_mode == MappingMode::Direct) {
        if (mapping_table_bytes != 0 ||
            stats_.mapping_table_bytes != 0 ||
            stats_.resident_mapping_table_bytes != 0 ||
            stats_.mapping_directory_bytes != 0 ||
            stats_.mapping_cache_capacity_bytes != 0 ||
            stats_.mapping_cache_entries != 0 ||
            stats_.mapping_cache_hits != 0 ||
            stats_.mapping_cache_misses != 0 ||
            stats_.mapping_media_reads != 0 ||
            stats_.mapping_lookup_ops != 0 ||
            stats_.mapping_update_ops != 0 ||
            stats_.mapping_dram_issue_busy_ns != 0.0 ||
            stats_.initial_mapping_pages != 0 ||
            !mapping_vpn_to_ppn_.empty()) {
            throw std::runtime_error(
                "HBF direct mapping accounting diverged: the exposed "
                "address space must carry no L2P state or work");
        }
    } else {
        std::uint64_t cache_entries = 0;
        for (std::size_t stack = 0; stack < config_.stacks; ++stack) {
            const auto& cache = mapping_cache_by_stack_.at(stack);
            const auto& lru = mapping_cache_lru_by_stack_.at(stack);
            if (cache.size() != lru.size() ||
                cache.size() > mapping_cache_pages_per_stack_) {
                throw std::runtime_error(
                    "HBF mapping-cache state exceeds its stack partition");
            }
            for (auto position = lru.begin(); position != lru.end();
                 ++position) {
                const auto found = cache.find(*position);
                if (found == cache.end() || found->second.iterator != position) {
                    throw std::runtime_error(
                        "HBF mapping-cache LRU index diverged");
                }
            }
            cache_entries = checked_add(
                cache_entries,
                cache.size(),
                "HBF mapping-cache entry accounting");
        }
        const auto cache_bytes_per_stack = checked_mul(
            mapping_cache_pages_per_stack_,
            config_.page_size_bytes,
            "HBF mapping-cache bytes per stack audit");
        const auto cache_bytes = checked_mul(
            cache_bytes_per_stack,
            config_.stacks,
            "HBF mapping-cache total bytes audit");
        const auto directory_bytes_per_stack = checked_mul(
            mapping_table_pages_per_stack_,
            config_.mapping_directory_entry_bytes,
            "HBF mapping-directory bytes per stack audit");
        const auto directory_bytes = checked_mul(
            directory_bytes_per_stack,
            config_.stacks,
            "HBF mapping-directory total bytes audit");
        auto used_ctrl_dram_bytes_per_stack = checked_add(
            directory_bytes_per_stack,
            cache_bytes_per_stack,
            "HBF cached-mapping used DRAM per stack");
        used_ctrl_dram_bytes_per_stack = checked_add(
            used_ctrl_dram_bytes_per_stack,
            expected_write_buffer_dram.bytes_per_stack,
            "HBF cached mapping plus write-buffer DRAM per stack");
        const auto mapping_accesses = checked_add(
            stats_.mapping_lookup_ops,
            stats_.mapping_update_ops,
            "HBF cached mapping accesses");
        if (stats_.resident_mapping_table_bytes != 0 ||
            stats_.resident_mapping_table_bytes_per_stack != 0 ||
            stats_.resident_mapping_pages_per_stack != 0 ||
            stats_.mapping_directory_entry_bytes !=
                config_.mapping_directory_entry_bytes ||
            stats_.mapping_directory_bytes != directory_bytes ||
            stats_.mapping_directory_bytes_per_stack !=
                directory_bytes_per_stack ||
            used_ctrl_dram_bytes_per_stack >
                config_.ctrl_dram_bytes / config_.stacks ||
            stats_.mapping_cache_capacity_bytes != cache_bytes ||
            stats_.mapping_cache_capacity_bytes_per_stack !=
                cache_bytes_per_stack ||
            stats_.mapping_cache_pages_per_stack !=
                mapping_cache_pages_per_stack_ ||
            stats_.mapping_cache_entries != cache_entries ||
            stats_.mapping_cache_peak_entries < cache_entries ||
            stats_.mapping_cache_peak_entries >
                checked_mul(
                    mapping_cache_pages_per_stack_,
                    config_.stacks,
                    "HBF mapping-cache peak capacity") ||
            checked_add(
                stats_.mapping_cache_hits,
                stats_.mapping_cache_misses,
                "HBF mapping-cache hit/miss accounting") !=
                mapping_accesses ||
            checked_add(
                checked_add(
                    stats_.mapping_media_reads,
                    stats_.mapping_cache_erased_misses,
                    "HBF mapping-cache direct miss classes"),
                stats_.mapping_cache_coalesced_misses,
                "HBF mapping-cache coalesced miss classes") !=
                stats_.mapping_cache_misses ||
            stats_.mapping_cache_dirty_evictions >
                stats_.mapping_cache_evictions ||
            stats_.mapping_cache_coalesced_misses >
                stats_.mapping_cache_misses ||
            stats_.mapping_media_read_bytes != checked_mul(
                stats_.mapping_media_reads,
                config_.page_size_bytes,
                "HBF mapping-cache media-read accounting")) {
            throw std::runtime_error(
                "HBF cached mapping accounting diverged");
        }
    }
    if (stats_.mapping_lookup_ops != checked_add(
            stats_.mapping_user_lookup_ops,
            stats_.mapping_gc_lookup_ops,
            "HBF mapping lookup-source accounting") ||
        stats_.mapping_update_ops != checked_add(
            stats_.mapping_user_update_ops,
            stats_.mapping_gc_update_ops,
            "HBF mapping update-source accounting")) {
        throw std::runtime_error(
            "HBF mapping user/GC source accounting diverged");
    }
    const auto valid_mapping_wait = [](std::uint64_t operations,
                                       double total_ns,
                                       double max_ns) {
        if (!std::isfinite(total_ns) || !std::isfinite(max_ns) ||
            total_ns < 0.0 || max_ns < 0.0 || max_ns > total_ns) {
            return false;
        }
        return operations == 0 ?
            total_ns == 0.0 && max_ns == 0.0 :
            total_ns > 0.0 && max_ns > 0.0;
    };
    const double expected_mapping_issue_busy_ns =
        static_cast<double>(checked_add(
            checked_add(
                stats_.mapping_lookup_ops,
                stats_.mapping_update_ops,
                "HBF mapping access accounting"),
            stats_.mapping_cache_misses -
                stats_.mapping_cache_coalesced_misses,
            "HBF mapping directory access accounting")) *
        config_.ctrl_dram_issue_ns;
    const double mapping_busy_tolerance = std::max(
        1e-9,
        std::abs(expected_mapping_issue_busy_ns) * 1e-12);
    if (!valid_mapping_wait(
            stats_.mapping_dram_wait_ops,
            stats_.mapping_dram_wait_ns,
            stats_.mapping_dram_wait_max_ns) ||
        !std::isfinite(stats_.mapping_dram_issue_busy_ns) ||
        std::abs(
            stats_.mapping_dram_issue_busy_ns -
            expected_mapping_issue_busy_ns) > mapping_busy_tolerance) {
        throw std::runtime_error(
            "HBF mapping DRAM telemetry diverged");
    }
    const auto write_buffer_dram_ops = checked_add(
        stats_.write_buffer_dram_read_ops,
        stats_.write_buffer_dram_write_ops,
        "HBF write-buffer DRAM access accounting");
    const double expected_write_buffer_issue_busy_ns =
        static_cast<double>(write_buffer_dram_ops) *
        config_.ctrl_dram_issue_ns;
    const double write_buffer_busy_tolerance = std::max(
        1e-9,
        std::abs(expected_write_buffer_issue_busy_ns) * 1e-12);
    if (!valid_mapping_wait(
            stats_.write_buffer_dram_wait_ops,
            stats_.write_buffer_dram_wait_ns,
            stats_.write_buffer_dram_wait_max_ns) ||
        !std::isfinite(stats_.write_buffer_dram_issue_busy_ns) ||
        std::abs(
            stats_.write_buffer_dram_issue_busy_ns -
            expected_write_buffer_issue_busy_ns) >
            write_buffer_busy_tolerance ||
        (!config_.write_coalescing_enabled &&
         (write_buffer_dram_ops != 0 ||
          stats_.write_buffer_dram_read_bytes != 0 ||
          stats_.write_buffer_dram_write_bytes != 0))) {
        throw std::runtime_error(
            "HBF write-buffer DRAM telemetry diverged");
    }

    for (std::size_t plane_index = 0; plane_index < planes_.size(); ++plane_index) {
        const auto& plane = planes_.at(plane_index);
        for (const auto block_index : plane.free_blocks) {
            if (block_index >= blocks_.size() ||
                block_plane_index(block_index) != plane_index) {
                throw std::runtime_error(
                    "HBF free-block pool contains an out-of-plane block");
            }
            if (free_pool_membership.at(block_index) != 0) {
                throw std::runtime_error(
                    "HBF free-block pool contains a duplicate block");
            }
            const auto& block = blocks_.at(block_index);
            if (block.role != BlockRole::Free || block.erase_pending) {
                throw std::runtime_error(
                    "HBF free-block pool contains an owned or pending-erase block");
            }
            free_pool_membership[block_index] = 1;
        }
    }

    for (std::size_t block_index = 0; block_index < blocks_.size(); ++block_index) {
        const auto& block = blocks_.at(block_index);
        if (block.role != BlockRole::StaticReadOnly) {
            // block_erases and block.erase_count both count an erase when it
            // is scheduled, so the two views agree in every observation
            // window without patching for in-flight erases.
            const auto effective_erase_count =
                static_cast<std::uint64_t>(block.erase_count);
            writable_blocks = checked_add(
                writable_blocks, 1, "HBF writable-block accounting");
            if (effective_erase_count != 0) {
                worn_blocks = checked_add(
                    worn_blocks, 1, "HBF worn-block accounting");
            }
            block_erase_count_sum = checked_add(
                block_erase_count_sum,
                effective_erase_count,
                "HBF block erase-count accounting");
            const auto erase_count =
                static_cast<long double>(effective_erase_count);
            block_erase_count_sum_squares += erase_count * erase_count;
            min_block_erase_count = std::min(
                min_block_erase_count,
                static_cast<std::uint32_t>(effective_erase_count));
            max_block_erase_count = std::max(
                max_block_erase_count,
                static_cast<std::uint32_t>(effective_erase_count));
            block_erase_count_histogram[
                static_cast<std::uint32_t>(effective_erase_count)]++;
        }
        const auto block_valid = static_cast<std::uint64_t>(block.valid_pages);
        const auto block_invalid = static_cast<std::uint64_t>(block.invalid_pages);
        const auto block_pending =
            static_cast<std::uint64_t>(block.pending_program_pages);
        const auto block_free = static_cast<std::uint64_t>(block.free_pages);
        if (block_valid > config_.pages_per_block ||
            block_invalid > config_.pages_per_block ||
            block_pending > config_.pages_per_block ||
            block_free > config_.pages_per_block) {
            throw std::runtime_error(
                "HBF block page-state counter exceeds block geometry");
        }

        std::uint64_t bitmap_valid = 0;
        if (block.valid_bitmap) {
            for (const auto word : *block.valid_bitmap) {
                bitmap_valid = checked_add(
                    bitmap_valid,
                    static_cast<std::uint64_t>(std::popcount(word)),
                    "HBF valid-page bitmap accounting");
            }
        }
        if (bitmap_valid != block_valid) {
            throw std::runtime_error(
                "HBF valid-page bitmap diverged from the block counter");
        }

        const bool allocatable_free =
            block.role == BlockRole::Free && !block.erase_pending;
        if (static_cast<bool>(free_pool_membership.at(block_index)) !=
            allocatable_free) {
            throw std::runtime_error(
                "HBF block role/pending-erase state diverged from its free pool");
        }

        if (block.role == BlockRole::StaticReadOnly) {
            if (block_invalid != 0 || block_pending != 0 || block_free != 0 ||
                block.next_page != config_.pages_per_block || block.erase_pending) {
                throw std::runtime_error(
                    "HBF static block has mutable or free page state");
            }
            static_unmaterialized_pages = checked_add(
                static_unmaterialized_pages,
                config_.pages_per_block - block_valid,
                "HBF static unmaterialized-page accounting");
        } else {
            if (block.role == BlockRole::RawPhysical) {
                raw_reserved_pages = checked_add(
                    raw_reserved_pages,
                    config_.pages_per_block,
                    "HBF raw reserved-page accounting audit");
            }
            const auto used = checked_add(
                checked_add(
                    block_valid,
                    block_invalid,
                    "HBF block valid/invalid accounting"),
                block_pending,
                "HBF block programmed/pending accounting");
            if (checked_add(
                    used, block_free, "HBF block total page accounting") !=
                    config_.pages_per_block ||
                block.next_page != used) {
                throw std::runtime_error(
                    "HBF block valid/invalid/pending/free pages do not conserve capacity");
            }
        }

        free_pages = checked_add(
            free_pages, block_free, "HBF global free-page accounting");
        valid_pages = checked_add(
            valid_pages, block_valid, "HBF global valid-page accounting");
        invalid_pages = checked_add(
            invalid_pages, block_invalid, "HBF global invalid-page accounting");
        pending_program_pages = checked_add(
            pending_program_pages,
            block_pending,
            "HBF global pending-program accounting");
        pending_mapping_publications = checked_add(
            pending_mapping_publications,
            block.pending_mapping_publications,
            "HBF global pending-mapping-publication accounting");
        auto& stack_free = free_pages_per_stack.at(stack_of_block(block_index));
        stack_free = checked_add(
            stack_free, block_free, "HBF per-stack free-page accounting");
    }

    std::uint64_t page_map_valid = 0;
    std::uint64_t page_map_invalid = 0;
    std::uint64_t page_map_pending = 0;
    for (const auto& [ppn, page] : programmed_pages_) {
        if (ppn >= total_pages_) {
            throw std::runtime_error("HBF page-state table contains an out-of-range PPN");
        }
        const auto block_index = static_cast<std::size_t>(
            ppn / config_.pages_per_block);
        const auto page_index = static_cast<std::uint32_t>(
            ppn % config_.pages_per_block);
        const auto& block = blocks_.at(block_index);
        switch (page.status) {
        case PageStatus::Erased:
            page_map_pending++;
            break;
        case PageStatus::StaticReadOnly:
        case PageStatus::Valid:
            if (!block.is_valid(page_index)) {
                throw std::runtime_error(
                    "HBF valid page-state entry is absent from its block bitmap");
            }
            page_map_valid++;
            break;
        case PageStatus::Invalid:
            if (block.is_valid(page_index)) {
                throw std::runtime_error(
                    "HBF invalid page-state entry remains in its block bitmap");
            }
            page_map_invalid++;
            break;
        }
    }
    std::uint64_t compact_valid_pages = 0;
    if (compact_logical_image_) {
        const auto& image = *compact_logical_image_;
        if (image.data_blocks_by_plane.size() != planes_.size() ||
            image.mapping_ppns.size() != image.vpn_slot_count ||
            image.vpn_ranges.size() != image.vpn_slot_count ||
            image.vpn_offsets_by_stack.size() != config_.stacks) {
            throw std::runtime_error(
                "HBF compact image directory dimensions are inconsistent");
        }
        std::vector<std::uint8_t> compact_data_block_seen(blocks_.size(), 0);
        std::uint64_t compact_live_data_pages = 0;
        for (std::size_t plane = 0;
             plane < image.data_blocks_by_plane.size();
             ++plane) {
            const auto& assigned = image.data_blocks_by_plane[plane];
            for (std::size_t ordinal = 0;
                 ordinal < assigned.size();
                 ++ordinal) {
                const auto block_number = assigned[ordinal];
                if (block_number >= blocks_.size() ||
                    block_plane_index(static_cast<std::size_t>(block_number)) != plane ||
                    compact_data_block_seen.at(
                        static_cast<std::size_t>(block_number)) != 0) {
                    throw std::runtime_error(
                        "HBF compact data-block directory is invalid or duplicated");
                }
                compact_data_block_seen[static_cast<std::size_t>(block_number)] = 1;
                const auto location =
                    image.data_block_locations.find(block_number);
                if (location == image.data_block_locations.end() ||
                    location->second.plane != plane ||
                    location->second.block_ordinal != ordinal) {
                    throw std::runtime_error(
                        "HBF compact data-block reverse index diverged");
                }
                const auto& block = blocks_.at(
                    static_cast<std::size_t>(block_number));
                const auto live = image.live_data_pages_by_block.find(
                    block_number);
                const auto live_pages =
                    live == image.live_data_pages_by_block.end() ?
                    0 : static_cast<std::uint64_t>(live->second);
                if (live_pages > block.valid_pages ||
                    (live_pages != 0 &&
                     (block.role != BlockRole::Data ||
                      block.erase_pending))) {
                    throw std::runtime_error(
                        "HBF compact data-block live count references "
                        "non-live physical state");
                }
                compact_live_data_pages = checked_add(
                    compact_live_data_pages,
                    live_pages,
                    "HBF compact live data-page accounting");
            }
        }
        for (const auto& [block_number, live_pages] :
             image.live_data_pages_by_block) {
            if (live_pages == 0 ||
                block_number >= compact_data_block_seen.size() ||
                compact_data_block_seen[
                    static_cast<std::size_t>(block_number)] == 0) {
                throw std::runtime_error(
                    "HBF compact live data-block index is not canonical");
            }
        }
        if (image.retired_lpns.size() > image.page_count ||
            compact_live_data_pages !=
                image.page_count - image.retired_lpns.size()) {
            throw std::runtime_error(
                "HBF compact live/retired data pages do not conserve the "
                "logical image");
        }
        std::unordered_set<std::uint64_t> compact_mapping_seen;
        compact_mapping_seen.reserve(
            static_cast<std::size_t>(image.mapping_page_count));
        std::uint64_t compact_range_pages = 0;
        std::uint64_t compact_live_mapping_pages = 0;
        for (std::size_t index = 0;
             index < image.mapping_ppns.size();
             ++index) {
            const auto& range = image.vpn_ranges[index];
            const auto& compact_ppn = image.mapping_ppns[index];
            const bool needs_mapping = range.page_count != 0 &&
                config_.mapping_mode != MappingMode::Direct;
            if (needs_mapping != compact_ppn.has_value()) {
                throw std::runtime_error(
                    "HBF compact mapping slot activity is inconsistent");
            }
            compact_range_pages = checked_add(
                compact_range_pages,
                range.page_count,
                "HBF compact mapping-range page accounting");
            if (!compact_ppn) {
                continue;
            }
            const auto ppn = *compact_ppn;
            if (ppn >= total_pages_) {
                throw std::runtime_error(
                    "HBF compact mapping-page directory is out of range");
            }
            const auto block_index = static_cast<std::size_t>(
                ppn / config_.pages_per_block);
            const auto page_index = static_cast<std::uint32_t>(
                ppn % config_.pages_per_block);
            const auto mapping_vpn = checked_add(
                image.first_vpn,
                index,
                "HBF compact accounting mapping VPN");
            const auto inverse = image.mapping_vpn_by_ppn.find(ppn);
            if (!compact_mapping_seen.insert(ppn).second ||
                inverse == image.mapping_vpn_by_ppn.end() ||
                inverse->second != mapping_vpn) {
                throw std::runtime_error(
                    "HBF compact mapping-page reverse index diverged");
            }
            if (!image.retired_mapping_vpns.contains(mapping_vpn)) {
                const auto& block = blocks_.at(block_index);
                if (block.role != BlockRole::Mapping ||
                    block.erase_pending ||
                    !block.is_valid(page_index)) {
                    throw std::runtime_error(
                        "HBF compact mapping-page directory references a "
                        "non-live page");
                }
                ++compact_live_mapping_pages;
            }
        }
        if (compact_range_pages != image.page_count ||
            compact_mapping_seen.size() != image.mapping_page_count ||
            image.mapping_vpn_by_ppn.size() != image.mapping_page_count ||
            image.retired_mapping_vpns.size() >
                image.mapping_page_count ||
            compact_live_mapping_pages !=
                image.mapping_page_count -
                    image.retired_mapping_vpns.size()) {
            throw std::runtime_error(
                "HBF compact mapping slots do not conserve the logical image");
        }
        std::uint64_t indexed_live_mapping_pages = 0;
        for (const auto& [block_number, live_pages] :
             image.live_mapping_pages_by_block) {
            if (live_pages == 0 || block_number >= blocks_.size()) {
                throw std::runtime_error(
                    "HBF compact live mapping-block index is invalid");
            }
            indexed_live_mapping_pages = checked_add(
                indexed_live_mapping_pages,
                live_pages,
                "HBF compact indexed live mapping pages");
        }
        if (indexed_live_mapping_pages != compact_live_mapping_pages) {
            throw std::runtime_error(
                "HBF compact mapping live-page block index diverged");
        }
        compact_valid_pages = checked_add(
            compact_live_data_pages,
            compact_live_mapping_pages,
            "HBF compact valid-page accounting");
        stats_.compact_live_logical_data_pages =
            compact_live_data_pages;
        stats_.compact_live_mapping_pages =
            compact_live_mapping_pages;
        stats_.compact_retired_logical_data_pages =
            image.retired_lpns.size();
        stats_.compact_retired_mapping_pages =
            image.retired_mapping_vpns.size();
    } else {
        stats_.compact_live_logical_data_pages = 0;
        stats_.compact_live_mapping_pages = 0;
        stats_.compact_retired_logical_data_pages = 0;
        stats_.compact_retired_mapping_pages = 0;
    }
    if (stats_.compact_initial_logical_data_pages >
            stats_.initial_logical_data_pages ||
        stats_.compact_initial_mapping_pages >
            stats_.initial_mapping_pages ||
        checked_add(
            stats_.compact_live_logical_data_pages,
            stats_.compact_retired_logical_data_pages,
            "HBF compact initial data-page lifecycle") !=
            stats_.compact_initial_logical_data_pages ||
        checked_add(
            stats_.compact_live_mapping_pages,
            stats_.compact_retired_mapping_pages,
            "HBF compact initial mapping-page lifecycle") !=
            stats_.compact_initial_mapping_pages) {
        throw std::runtime_error(
            "HBF compact initial-image lifecycle accounting diverged");
    }
    if (checked_add(
            page_map_valid,
            compact_valid_pages,
            "HBF materialized/compact valid-page accounting") != valid_pages ||
        page_map_invalid != invalid_pages ||
        page_map_pending != pending_program_pages) {
        throw std::runtime_error(
            "HBF page-state table diverged from block page counters");
    }
    const auto validate_mapping_target = [this](
                                             std::uint64_t ppn,
                                             std::uint64_t expected_lpn,
                                             PageOwner expected_owner,
                                             const char* name) {
        const auto page = programmed_pages_.find(ppn);
        if (ppn >= total_pages_ || page == programmed_pages_.end() ||
            page->second.status != PageStatus::Valid ||
            page->second.owner != expected_owner ||
            page->second.lpn != expected_lpn) {
            throw std::runtime_error(
                std::string("HBF ") + name +
                " points to a non-live or wrongly owned physical page");
        }
    };
    for (const auto& [lpn, ppn] : lpn_to_ppn_) {
        validate_mapping_target(ppn, lpn, PageOwner::Logical, "L2P mapping");
    }
    for (const auto& [mapping_vpn, ppn] : mapping_vpn_to_ppn_) {
        validate_mapping_target(
            ppn,
            metadata_lpn(mapping_vpn),
            PageOwner::Mapping,
            "mapping-page directory");
    }
    // Between media-program and mapping-publication callbacks, both old and
    // new versions may legitimately be valid. Once all program/publication
    // pins are gone, every live logical or mapping page must be reachable in
    // the corresponding directory; this catches orphaned GC relocations.
    if (pending_program_pages == 0 && pending_mapping_publications == 0) {
        for (const auto& [ppn, page] : programmed_pages_) {
            if (page.status != PageStatus::Valid) {
                continue;
            }
            if (page.owner == PageOwner::Logical) {
                const auto mapping = lpn_to_ppn_.find(page.lpn);
                if (mapping == lpn_to_ppn_.end() || mapping->second != ppn) {
                    throw std::runtime_error(
                        "HBF quiescent logical page is unreachable from L2P");
                }
            } else if (page.owner == PageOwner::Mapping) {
                const auto mapping_vpn = metadata_vpn(page.lpn);
                const auto mapping = mapping_vpn_to_ppn_.find(mapping_vpn);
                if (mapping == mapping_vpn_to_ppn_.end() || mapping->second != ppn) {
                    throw std::runtime_error(
                        "HBF quiescent mapping page is unreachable from its directory");
                }
            } else if (page.owner != PageOwner::RawPhysical) {
                throw std::runtime_error(
                    "HBF valid page has an invalid owner at quiescence");
            }
        }
    }
    if (free_pages != free_pages_ || free_pages_per_stack != free_pages_per_stack_) {
        throw std::runtime_error(
            "HBF global/per-stack free-page counters do not match block state");
    }
    const auto accounted_pages = checked_add(
        checked_add(
            checked_add(
                free_pages, valid_pages, "HBF free/valid capacity accounting"),
            invalid_pages,
            "HBF free/valid/invalid capacity accounting"),
        checked_add(
            pending_program_pages,
            static_unmaterialized_pages,
            "HBF pending/static capacity accounting"),
        "HBF total physical-page accounting");
    if (accounted_pages != total_pages_) {
        throw std::runtime_error(
            "HBF physical page states do not conserve total capacity");
    }
    if (block_erase_count_sum != checked_add(
            restored_block_erases_,
            stats_.block_erases,
            "HBF restored/current block erase accounting")) {
        throw std::runtime_error(
            "HBF per-block erase counts diverged from block_erases");
    }

    const auto classified_programs = checked_add(
        checked_add(
            checked_add(
                stats_.data_programs,
                stats_.mapping_page_programs,
                "HBF data/mapping program accounting"),
            stats_.gc_relocations,
            "HBF classified program accounting"),
        stats_.static_wear_leveling_relocations,
        "HBF wear-leveling program accounting");
    const auto classified_payload_bytes = checked_add(
        checked_add(
            checked_add(
                stats_.data_program_payload_bytes,
                stats_.mapping_program_payload_bytes,
                "HBF data/mapping payload-byte accounting"),
            stats_.gc_relocation_payload_bytes,
            "HBF classified payload-byte accounting"),
        stats_.static_wear_leveling_relocation_payload_bytes,
        "HBF wear-leveling payload-byte accounting");
    std::uint64_t active_gc_relocations = 0;
    std::uint64_t active_wear_relocations = 0;
    for (std::size_t stack = 0; stack < config_.stacks; ++stack) {
        if (const auto& victim = gc_victim_by_stack_.at(stack)) {
            active_gc_relocations = checked_add(
                active_gc_relocations, victim->relocated_pages,
                "HBF active GC relocation accounting");
        }
        if (const auto& migration = wear_leveling_by_stack_.at(stack)) {
            active_wear_relocations = checked_add(
                active_wear_relocations, migration->relocated_pages,
                "HBF active wear-leveling relocation accounting");
        }
    }
    std::uint64_t pending_relocation_publications = 0;
    for (const auto* pending_updates : {&pending_lpn_updates_, &pending_vpn_updates_}) {
        for (const auto& entry : *pending_updates) {
            for (const auto& update : entry.second) {
                if (update.relocation &&
                    pending_commit_time_by_sequence_.contains(update.sequence)) {
                    pending_relocation_publications = checked_add(
                        pending_relocation_publications, 1,
                        "HBF pending relocation publication accounting");
                }
            }
        }
    }
    if (stats_.page_programs != classified_programs ||
        stats_.raw_physical_programs > stats_.data_programs ||
        stats_.raw_physical_program_payload_bytes != checked_mul(
            stats_.raw_physical_programs,
            config_.page_size_bytes,
            "HBF raw physical program payload-byte accounting") ||
        stats_.raw_physical_program_payload_bytes >
            stats_.data_program_payload_bytes ||
        stats_.data_program_payload_bytes != checked_mul(
            stats_.data_programs,
            config_.page_size_bytes,
            "HBF data-program payload-byte accounting") ||
        stats_.mapping_program_payload_bytes != checked_mul(
            stats_.mapping_page_programs,
            config_.page_size_bytes,
            "HBF mapping-program payload-byte accounting") ||
        stats_.gc_relocation_payload_bytes != checked_mul(
            stats_.gc_relocations,
            config_.page_size_bytes,
            "HBF GC-relocation payload-byte accounting") ||
        stats_.static_wear_leveling_relocation_payload_bytes != checked_mul(
            stats_.static_wear_leveling_relocations,
            config_.page_size_bytes,
            "HBF wear-leveling relocation payload-byte accounting") ||
        checked_add(
            stats_.static_wear_leveling_relocations,
            stats_.static_wear_leveling_reclaimed_invalid_pages,
            "HBF wear-leveling relocated/reclaimed page accounting") !=
            checked_add(
                checked_mul(
                    stats_.static_wear_leveling_runs,
                    config_.pages_per_block,
                    "HBF wear-leveling victim-page accounting"),
                active_wear_relocations,
                "HBF completed/active wear-leveling accounting") ||
        stats_.physical_write_bytes != classified_payload_bytes ||
        stats_.physical_write_bytes != checked_mul(
            stats_.page_programs,
            config_.page_size_bytes,
            "HBF physical-write byte accounting") ||
        stats_.physical_read_bytes != checked_mul(
            stats_.page_reads,
            config_.page_size_bytes,
            "HBF physical-read byte accounting")) {
        throw std::runtime_error(
            "HBF program/read byte accounting identities diverged");
    }
    if (stats_.gc_relocations != checked_add(
            stats_.gc_data_relocations,
            stats_.gc_mapping_relocations,
            "HBF GC relocation owner accounting")) {
        throw std::runtime_error(
            "HBF GC data/mapping relocation counts do not conserve relocations");
    }
    const auto gc_victim_pages = checked_add(
        checked_mul(stats_.gc_runs, config_.pages_per_block,
            "HBF GC victim-page accounting"),
        active_gc_relocations,
        "HBF completed/active GC accounting");
    if (checked_add(
            stats_.gc_relocations,
            stats_.gc_reclaimed_invalid_pages,
            "HBF GC relocated/reclaimed page accounting") != gc_victim_pages ||
        stats_.block_erases != checked_add(
            checked_add(
                stats_.erase_requests,
                stats_.gc_runs,
                "HBF user/GC erase accounting"),
            stats_.static_wear_leveling_runs,
            "HBF user/GC/wear-leveling erase accounting") ||
        checked_add(
            stats_.invalidations, pending_relocation_publications,
            "HBF committed/pending relocation invalidation accounting") < checked_add(
            stats_.gc_relocations,
            stats_.static_wear_leveling_relocations,
            "HBF relocation invalidation accounting")) {
        throw std::runtime_error(
            "HBF GC victim, erase, or invalidation accounting diverged");
    }
    const auto flash_transactions = checked_add(
        checked_add(
            stats_.page_reads,
            stats_.page_programs,
            "HBF flash read/program accounting"),
        stats_.block_erases,
        "HBF flash transaction accounting");
    if (stats_.flash_scheduler_enqueues != flash_transactions ||
        stats_.flash_scheduler_issues != flash_transactions) {
        throw std::runtime_error(
            "HBF flash scheduler counts do not conserve media transactions");
    }

    stats_.total_pages = total_pages_;
    stats_.free_pages = free_pages;
    stats_.valid_pages = valid_pages;
    stats_.invalid_pages = invalid_pages;
    stats_.pending_program_pages = pending_program_pages;
    stats_.pending_mapping_publications = pending_mapping_publications;
    stats_.static_unmaterialized_pages = static_unmaterialized_pages;
    stats_.raw_reserved_pages = raw_reserved_pages;
    stats_.accounting_verified = true;
    stats_.writable_blocks = writable_blocks;
    stats_.writable_pages = checked_mul(
        writable_blocks,
        config_.pages_per_block,
        "HBF writable-page accounting");
    stats_.writable_payload_bytes = checked_mul(
        stats_.writable_pages,
        config_.page_size_bytes,
        "HBF writable-payload accounting");
    stats_.worn_blocks = worn_blocks;
    stats_.block_erase_count_sum = block_erase_count_sum;
    stats_.block_erase_count_sum_squares = block_erase_count_sum_squares;
    stats_.min_block_erase_count = writable_blocks == 0 ?
        0 : min_block_erase_count;
    stats_.max_block_erase_count = max_block_erase_count;
    stats_.block_erase_count_histogram = std::move(block_erase_count_histogram);
}

void HbfDevice::refresh_parallel_stats() const {
    refresh_accounting_stats();
    stats_.mapping_entries = logical_mapping_entry_count();
    stats_.thermal_peak_temperature_c = 0.0;
    stats_.thermal_final_temperature_c = 0.0;
    stats_.thermal_throttled_stacks = 0;
    if (config_.thermal_enabled) {
        // Project every stack to the finish frontier on a copy: reading
        // stats must never advance governor state.
        const double horizon_ns = std::max(stats_.finish_ns, 0.0);
        double peak_c = thermal_boot_temperature_c_;
        double final_c = thermal_boot_temperature_c_;
        std::uint64_t throttled_stacks = 0;
        for (const auto& stack : thermal_stacks_) {
            auto node = stack.node;
            thermal_advance(node, horizon_ns, false);
            peak_c = std::max(peak_c, node.peak_c);
            final_c = std::max(final_c, node.temperature_c);
            throttled_stacks += node.throttled ? 1 : 0;
        }
        stats_.thermal_peak_temperature_c = peak_c;
        stats_.thermal_final_temperature_c = final_c;
        stats_.thermal_throttled_stacks = throttled_stacks;
    }
    // The same physical work is accumulated in request order for the public
    // total and in resource order for directional/per-die totals. Their
    // round-off bound grows with the number of additions; a fixed 32-epsilon
    // tolerance falsely rejected long, otherwise exact replays.
    const auto accumulation_terms = std::max<std::uint64_t>(
        32,
        checked_add(
            checked_add(
                checked_add(
                    stats_.read_requests,
                    stats_.program_requests,
                    "HBF conservation request terms"),
                stats_.erase_requests,
                "HBF conservation request/erase terms"),
            checked_add(
                checked_add(
                    stats_.page_reads,
                    stats_.page_programs,
                    "HBF conservation media terms"),
                checked_add(
                    stats_.read_buffer_hits,
                    stats_.write_buffer_read_hits,
                    "HBF conservation buffer terms"),
                "HBF conservation media/buffer terms"),
            "HBF conservation total terms"));
    const auto work_conserved = [accumulation_terms](
                                    double total, double decode, double encode) {
        const double expected = decode + encode;
        const double scale = std::max({1.0, std::abs(total), std::abs(expected)});
        return std::abs(total - expected) <=
            8.0 * static_cast<double>(accumulation_terms) *
                std::numeric_limits<double>::epsilon() * scale;
    };
    if (stats_.ecc_decode_ops != stats_.page_reads) {
        throw std::runtime_error(
            "HBF ECC decode operation count diverged from physical page reads");
    }
    if (stats_.ecc_encode_ops != stats_.page_programs) {
        throw std::runtime_error(
            "HBF ECC encode operation count diverged from physical page programs");
    }
    const auto expected_decode_bytes = checked_mul(
        stats_.ecc_decode_ops, page_wire_bytes(), "HBF ECC decode conservation");
    const auto expected_encode_bytes = checked_mul(
        stats_.ecc_encode_ops, page_wire_bytes(), "HBF ECC encode conservation");
    if (stats_.ecc_decode_codeword_bytes != expected_decode_bytes ||
        stats_.ecc_encode_codeword_bytes != expected_encode_bytes ||
        stats_.ecc_codeword_bytes != checked_add(
            expected_decode_bytes, expected_encode_bytes,
            "HBF ECC total conservation")) {
        throw std::runtime_error(
            "HBF ECC codeword-byte accounting did not conserve page operations");
    }
    if (!work_conserved(
            stats_.stage_work.ecc_queue_wait_ns,
            stats_.ecc_decode_queue_wait_ns,
            stats_.ecc_encode_queue_wait_ns) ||
        !work_conserved(
            stats_.stage_work.ecc_latency_ns,
            stats_.ecc_decode_latency_work_ns,
            stats_.ecc_encode_latency_work_ns) ||
        !work_conserved(
            stats_.ecc_issue_busy_ns,
            stats_.ecc_decode_issue_busy_ns,
            stats_.ecc_encode_issue_busy_ns)) {
        throw std::runtime_error(
            "HBF ECC directional work accounting did not conserve totals");
    }
    stats_.stacks = config_.stacks;
    stats_.channels = channels_.size();
    stats_.dies = dies_.size();
    stats_.planes = planes_.size();
    stats_.media_lanes = checked_mul(
        static_cast<std::uint64_t>(planes_.size()),
        config_.media_lanes_per_plane,
        "HBF stats media_lanes");
    stats_.subarrays = checked_mul(
        static_cast<std::uint64_t>(planes_.size()),
        subarrays_per_plane_,
        "HBF stats subarrays");
    stats_.page_buffer_banks = checked_mul(
        static_cast<std::uint64_t>(planes_.size()),
        config_.page_buffer_banks_per_plane,
        "HBF stats page_buffer_banks");
    stats_.active_channels = 0;
    stats_.active_dies = 0;
    stats_.active_planes = 0;
    stats_.active_media_lanes = 0;
    stats_.active_subarrays = 0;
    stats_.active_page_buffer_banks = 0;
    stats_.max_plane_ops = 0;
    stats_.max_media_lane_reads = 0;
    stats_.max_subarray_reads = 0;
    stats_.max_page_buffer_bank_reads = 0;
    stats_.max_die_transactions = 0;
    stats_.max_plane_media_busy_ns = 0.0;
    stats_.avg_active_plane_media_busy_ns = 0.0;
    stats_.avg_active_plane_ops = 0.0;
    stats_.read_lane_busy_ns = 0.0;
    stats_.subarray_read_busy_ns = 0.0;
    stats_.page_buffer_bank_busy_ns = 0.0;
    stats_.max_media_lane_busy_ns = 0.0;
    stats_.avg_active_media_lane_busy_ns = 0.0;
    stats_.avg_active_media_lane_reads = 0.0;
    stats_.max_subarray_busy_ns = 0.0;
    stats_.avg_active_subarray_busy_ns = 0.0;
    stats_.avg_active_subarray_reads = 0.0;
    stats_.max_page_buffer_bank_busy_ns = 0.0;
    stats_.avg_active_page_buffer_bank_busy_ns = 0.0;
    stats_.avg_active_page_buffer_bank_reads = 0.0;
    stats_.max_channel_busy_ns = 0.0;
    stats_.avg_active_channel_busy_ns = 0.0;
    stats_.avg_active_die_transactions = 0.0;
    stats_.sequencer_busy_ns = 0.0;
    stats_.hb_io_command_busy_ns = 0.0;
    stats_.hb_io_data_busy_ns = 0.0;
    stats_.logic_ingress_busy_ns = 0.0;
    stats_.tsv_busy_ns = 0.0;
    stats_.sram_busy_ns = 0.0;
    stats_.flash_source_queue_busy_ns = 0.0;
    stats_.channel_command_busy_ns = 0.0;
    stats_.channel_data_busy_ns = 0.0;
    stats_.logic_ingress_resources = logic_dies_.size();
    stats_.tsv_resources = logic_dies_.size();
    stats_.sram_resources = logic_dies_.size();
    stats_.flash_source_queue_resources = checked_mul(
        static_cast<std::uint64_t>(dies_.size()),
        static_cast<std::uint64_t>(std::tuple_size_v<
            decltype(DieState::source_queues)>),
        "HBF stats flash source queue resources");
    stats_.channel_command_resources = channels_.size();
    stats_.channel_data_resources = channels_.size();
    stats_.active_ecc_dies = 0;
    stats_.max_ecc_inflight_per_die = 0;
    stats_.max_ecc_issue_busy_ns = 0.0;
    stats_.avg_active_ecc_issue_busy_ns = 0.0;

    double active_plane_busy_ns = 0.0;
    std::uint64_t active_plane_ops = 0;
    for (const auto& plane : planes_) {
        const auto ops = plane.read_count + plane.program_count + plane.erase_count;
        if (ops == 0 && plane.media_busy_ns <= 0.0) {
            continue;
        }
        stats_.active_planes++;
        active_plane_busy_ns += plane.media_busy_ns;
        active_plane_ops += ops;
        stats_.max_plane_media_busy_ns =
            std::max(stats_.max_plane_media_busy_ns, plane.media_busy_ns);
        stats_.max_plane_ops = std::max(stats_.max_plane_ops, ops);

        for (const auto& lane : plane.media_lanes) {
            if (lane.read_count == 0 && lane.busy_ns <= 0.0) {
                continue;
            }
            stats_.active_media_lanes++;
            stats_.read_lane_busy_ns += lane.busy_ns;
            stats_.max_media_lane_busy_ns =
                std::max(stats_.max_media_lane_busy_ns, lane.busy_ns);
            stats_.max_media_lane_reads =
                std::max(stats_.max_media_lane_reads, lane.read_count);
        }

        for (const auto& subarray : plane.subarrays) {
            if (subarray.read_count == 0 && subarray.busy_ns <= 0.0) {
                continue;
            }
            stats_.active_subarrays++;
            stats_.subarray_read_busy_ns += subarray.busy_ns;
            stats_.max_subarray_busy_ns =
                std::max(stats_.max_subarray_busy_ns, subarray.busy_ns);
            stats_.max_subarray_reads =
                std::max(stats_.max_subarray_reads, subarray.read_count);
        }

        for (const auto& bank : plane.page_buffer_banks) {
            if (bank.read_count == 0 && bank.busy_ns <= 0.0) {
                continue;
            }
            stats_.active_page_buffer_banks++;
            stats_.page_buffer_bank_busy_ns += bank.busy_ns;
            stats_.max_page_buffer_bank_busy_ns =
                std::max(stats_.max_page_buffer_bank_busy_ns, bank.busy_ns);
            stats_.max_page_buffer_bank_reads =
                std::max(stats_.max_page_buffer_bank_reads, bank.read_count);
        }
    }
    stats_.media_busy_ns = active_plane_busy_ns;
    if (stats_.active_planes != 0) {
        stats_.avg_active_plane_media_busy_ns =
            active_plane_busy_ns / static_cast<double>(stats_.active_planes);
        stats_.avg_active_plane_ops =
            static_cast<double>(active_plane_ops) / static_cast<double>(stats_.active_planes);
    }
    if (stats_.active_media_lanes != 0) {
        stats_.avg_active_media_lane_busy_ns =
            stats_.read_lane_busy_ns / static_cast<double>(stats_.active_media_lanes);
        std::uint64_t active_lane_reads = 0;
        for (const auto& plane : planes_) {
            for (const auto& lane : plane.media_lanes) {
                if (lane.read_count != 0 || lane.busy_ns > 0.0) {
                    active_lane_reads += lane.read_count;
                }
            }
        }
        stats_.avg_active_media_lane_reads =
            static_cast<double>(active_lane_reads) /
            static_cast<double>(stats_.active_media_lanes);
    }
    if (stats_.active_subarrays != 0) {
        stats_.avg_active_subarray_busy_ns =
            stats_.subarray_read_busy_ns / static_cast<double>(stats_.active_subarrays);
        std::uint64_t active_subarray_reads = 0;
        for (const auto& plane : planes_) {
            for (const auto& subarray : plane.subarrays) {
                if (subarray.read_count != 0 || subarray.busy_ns > 0.0) {
                    active_subarray_reads += subarray.read_count;
                }
            }
        }
        stats_.avg_active_subarray_reads =
            static_cast<double>(active_subarray_reads) /
            static_cast<double>(stats_.active_subarrays);
    }
    if (stats_.active_page_buffer_banks != 0) {
        double active_bank_busy_ns = 0.0;
        std::uint64_t active_bank_reads = 0;
        for (const auto& plane : planes_) {
            for (const auto& bank : plane.page_buffer_banks) {
                if (bank.read_count != 0 || bank.busy_ns > 0.0) {
                    active_bank_busy_ns += bank.busy_ns;
                    active_bank_reads += bank.read_count;
                }
            }
        }
        stats_.avg_active_page_buffer_bank_busy_ns =
            active_bank_busy_ns / static_cast<double>(stats_.active_page_buffer_banks);
        stats_.avg_active_page_buffer_bank_reads =
            static_cast<double>(active_bank_reads) /
            static_cast<double>(stats_.active_page_buffer_banks);
    }

    double active_channel_busy_ns = 0.0;
    for (const auto& channel : channels_) {
        stats_.channel_command_busy_ns += channel.command_busy_ns;
        stats_.channel_data_busy_ns += channel.data_busy_ns;
        const double busy_ns = channel.command_busy_ns + channel.data_busy_ns;
        if (channel.command_count == 0 && channel.data_count == 0 && busy_ns <= 0.0) {
            continue;
        }
        stats_.active_channels++;
        active_channel_busy_ns += busy_ns;
        stats_.max_channel_busy_ns = std::max(stats_.max_channel_busy_ns, busy_ns);
    }
    if (stats_.active_channels != 0) {
        stats_.avg_active_channel_busy_ns =
            active_channel_busy_ns / static_cast<double>(stats_.active_channels);
    }
    if (!work_conserved(
            stats_.stage_work.channel_transfer_ns,
            stats_.channel_command_busy_ns,
            stats_.channel_data_busy_ns)) {
        throw std::runtime_error(
            "HBF channel command/data resource work did not conserve stage work");
    }

    for (const auto& logic_die : logic_dies_) {
        stats_.logic_ingress_busy_ns += logic_die.ingress.reserved_work_ns;
        stats_.tsv_busy_ns += logic_die.tsv.reserved_work_ns;
        stats_.sram_busy_ns += logic_die.sram.reserved_work_ns;
        stats_.hb_io_command_busy_ns += logic_die.hb_io_command_busy_ns;
        stats_.hb_io_data_busy_ns += logic_die.hb_io_data_busy_ns;
    }
    if (!work_conserved(
            stats_.stage_work.hb_io_transfer_ns,
            stats_.hb_io_command_busy_ns,
            stats_.hb_io_data_busy_ns)) {
        throw std::runtime_error(
            "HBF HBIO command/data work accounting did not conserve totals");
    }
    if (!work_conserved(
            stats_.stage_work.tsv_transfer_ns,
            stats_.tsv_busy_ns,
            0.0)) {
        throw std::runtime_error(
            "HBF TSV resource work did not conserve stage work");
    }
    if (!work_conserved(
            stats_.stage_work.sram_staging_ns,
            stats_.sram_busy_ns,
            0.0)) {
        throw std::runtime_error(
            "HBF SRAM resource work did not conserve stage work");
    }

    std::uint64_t active_die_transactions = 0;
    double active_ecc_issue_busy_ns = 0.0;
    std::uint64_t die_ecc_decode_ops = 0;
    std::uint64_t die_ecc_encode_ops = 0;
    double die_ecc_issue_busy_ns = 0.0;
    for (const auto& die : dies_) {
        stats_.sequencer_busy_ns += die.sequencer_busy_ns;
        for (const auto& queue : die.source_queues) {
            stats_.flash_source_queue_busy_ns += queue.reserved_work_ns;
        }
        die_ecc_decode_ops = checked_add(
            die_ecc_decode_ops, die.ecc_decode_ops, "HBF die ECC decode ops");
        die_ecc_encode_ops = checked_add(
            die_ecc_encode_ops, die.ecc_encode_ops, "HBF die ECC encode ops");
        die_ecc_issue_busy_ns += die.ecc_issue_busy_ns;
        if (die.ecc_decode_ops != 0 || die.ecc_encode_ops != 0 ||
            die.ecc_issue_busy_ns > 0.0) {
            stats_.active_ecc_dies++;
            active_ecc_issue_busy_ns += die.ecc_issue_busy_ns;
            stats_.max_ecc_issue_busy_ns =
                std::max(stats_.max_ecc_issue_busy_ns, die.ecc_issue_busy_ns);

            // Exact reservation calendars may backfill issue slots, so call
            // order is not time order. Sweep the retained latency intervals
            // and combine them with the folded peak of retired ones.
            std::uint64_t max_inflight = ecc_max_inflight(die);
            max_inflight = std::max(
                max_inflight, die.page_run_max_ecc_inflight);
            stats_.max_ecc_inflight_per_die =
                std::max(stats_.max_ecc_inflight_per_die, max_inflight);
        }
        if (die.transaction_count == 0 && die.sequencer_busy_ns <= 0.0) {
            continue;
        }
        stats_.active_dies++;
        active_die_transactions += die.transaction_count;
        stats_.max_die_transactions =
            std::max(stats_.max_die_transactions, die.transaction_count);
    }
    if (stats_.active_dies != 0) {
        stats_.avg_active_die_transactions =
            static_cast<double>(active_die_transactions) / static_cast<double>(stats_.active_dies);
    }
    if (stats_.active_ecc_dies != 0) {
        stats_.avg_active_ecc_issue_busy_ns =
            active_ecc_issue_busy_ns / static_cast<double>(stats_.active_ecc_dies);
    }
    if (die_ecc_decode_ops != stats_.ecc_decode_ops ||
        die_ecc_encode_ops != stats_.ecc_encode_ops ||
        !work_conserved(stats_.ecc_issue_busy_ns, die_ecc_issue_busy_ns, 0.0)) {
        throw std::runtime_error(
            "HBF ECC per-die accounting did not conserve aggregate work");
    }
    if (stats_.read_splits > stats_.read_requests) {
        throw std::runtime_error(
            "HBF read-split count exceeds read-request count");
    }
    if (checked_add(
            stats_.page_run_requests,
            stats_.scalar_read_requests,
            "HBF read-engine request accounting") != stats_.read_requests ||
        stats_.page_run_pages > stats_.page_reads ||
        checked_add(
            checked_add(
                stats_.page_run_physical_requests,
                stats_.page_run_static_requests,
                "HBF physical/static page-run request classes"),
            stats_.page_run_logical_requests,
            "HBF page-run request classes") != stats_.page_run_requests ||
        stats_.streaming_read_buffer_bypass_pages > stats_.page_reads) {
        throw std::runtime_error(
            "HBF page-run/scalar read-engine accounting did not conserve requests");
    }
    const auto expected_page_admissions = checked_add(
        stats_.read_split_pages,
        stats_.read_requests - stats_.read_splits,
        "HBF expected page-read admissions");
    if (stats_.page_read_admission_events != expected_page_admissions ||
        stats_.page_read_admission_waited_pages >
            stats_.page_read_admission_events ||
        (stats_.page_read_admission_waited_pages == 0) !=
            (stats_.page_read_admission_wait_ns == 0.0) ||
        (stats_.page_read_admission_waited_pages == 0) !=
            (stats_.page_read_admission_max_wait_ns == 0.0) ||
        stats_.page_read_admission_max_wait_ns >
            stats_.page_read_admission_wait_ns) {
        throw std::runtime_error(
            "HBF page-read admission accounting did not conserve requests");
    }
}

} // namespace hbfsim::physical::hbf
