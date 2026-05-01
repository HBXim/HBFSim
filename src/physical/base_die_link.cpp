#include "physical/base_die_link.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace hbfsim::physical {
namespace {

std::uint64_t checked_add(
    std::uint64_t lhs,
    std::uint64_t rhs,
    const char* description) {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
        throw std::runtime_error(
            std::string(description) + " overflows uint64_t");
    }
    return lhs + rhs;
}

} // namespace

double BaseDieLinkStats::active_span_ns() const {
    return std::isfinite(first_arrival_ns) && finish_ns > first_arrival_ns ?
        finish_ns - first_arrival_ns : 0.0;
}

double BaseDieLinkStats::read_utilization() const {
    const auto span = active_span_ns();
    return span <= 0.0 || links == 0 ?
        0.0 : read_busy_ns / (span * static_cast<double>(links));
}

double BaseDieLinkStats::write_utilization() const {
    const auto span = active_span_ns();
    return span <= 0.0 || links == 0 ?
        0.0 : write_busy_ns / (span * static_cast<double>(links));
}

BaseDieLink::BaseDieLink(BaseDieLinkConfig config, std::string entity)
    : config_(config), entity_(std::move(entity)) {
    if (!(config_.read_bandwidth_GBps > 0.0) ||
        !std::isfinite(config_.read_bandwidth_GBps) ||
        !(config_.write_bandwidth_GBps > 0.0) ||
        !std::isfinite(config_.write_bandwidth_GBps) ||
        config_.latency_ns < 0.0 || !std::isfinite(config_.latency_ns)) {
        throw std::runtime_error(
            "base-die link bandwidth must be positive and finite; latency must "
            "be non-negative and finite");
    }
}

PhysicalCompletion BaseDieLink::issue(
    std::string id,
    Op op,
    double arrival_ns,
    std::uint64_t bytes,
    TraceConfig trace) {
    if (op != Op::Read && op != Op::Write) {
        throw std::runtime_error("base-die link only supports read/write transfers");
    }
    if (!std::isfinite(arrival_ns) || arrival_ns < 0.0) {
        throw std::runtime_error("base-die link arrival must be finite and non-negative");
    }
    if (bytes == 0) {
        throw std::runtime_error("base-die link transfer bytes must be positive");
    }

    stats_.first_arrival_ns = std::min(stats_.first_arrival_ns, arrival_ns);
    const bool is_read = op == Op::Read;
    auto& ready = is_read ? read_ready_ns_ : write_ready_ns_;
    auto& queue_wait = is_read ? stats_.read_queue_wait_ns : stats_.write_queue_wait_ns;
    auto& busy = is_read ? stats_.read_busy_ns : stats_.write_busy_ns;
    auto& fixed_latency_work = is_read ?
        stats_.read_fixed_latency_work_ns :
        stats_.write_fixed_latency_work_ns;
    auto& transfer_count = is_read ? stats_.read_transfers : stats_.write_transfers;
    auto& byte_count = is_read ? stats_.read_bytes : stats_.write_bytes;
    const double bandwidth = is_read ? config_.read_bandwidth_GBps : config_.write_bandwidth_GBps;
    const double serialization_ns = transfer_time_ns(bytes, bandwidth);
    const double start_ns = std::max(arrival_ns, ready);
    const double finish_ns = start_ns + serialization_ns + config_.latency_ns;
    if (!std::isfinite(serialization_ns) || !std::isfinite(finish_ns)) {
        throw std::runtime_error("base-die link timing exceeds finite double range");
    }
    const double wait_ns = start_ns - arrival_ns;
    ready = start_ns + serialization_ns;

    queue_wait += wait_ns;
    busy += serialization_ns;
    fixed_latency_work += config_.latency_ns;
    transfer_count = checked_add(
        transfer_count, 1, "base-die link transfer count");
    byte_count = checked_add(
        byte_count, bytes, "base-die link byte count");
    stats_.finish_ns = std::max(stats_.finish_ns, finish_ns);

    PhysicalCompletion out;
    out.id = std::move(id);
    out.tier = Tier::HBF;
    out.op = op;
    out.arrival_ns = arrival_ns;
    out.start_ns = start_ns;
    out.finish_ns = finish_ns;
    out.logical_bytes = bytes;
    out.physical_bytes = bytes;
    out.resource_path = entity_;
    out.note = is_read ? "hbf-to-hbm-staging-link" : "hbm-to-hbf-backing-link";
    out.breakdown.scheduler_queue_wait_ns = wait_ns;
    out.breakdown.hb_io_transfer_ns = serialization_ns;
    out.breakdown.command_ns = config_.latency_ns;
    auto* spans = trace_spans_enabled(trace) ? &out.spans : nullptr;
    add_trace_span(
        spans,
        is_read ? "base_die_link_read_serialize" : "base_die_link_write_serialize",
        "base_die_link",
        entity_,
        start_ns,
        start_ns + serialization_ns,
        true,
        std::to_string(bytes) + "B");
    add_trace_span(
        spans,
        is_read ? "base_die_link_read_latency" : "base_die_link_write_latency",
        "base_die_link",
        entity_,
        start_ns + serialization_ns,
        finish_ns,
        true,
        std::to_string(bytes) + "B");
    return out;
}

BaseDieLinkStats aggregate_base_die_link_stats(
    const std::vector<BaseDieLink>& links) {
    BaseDieLinkStats total;
    total.links = links.size();
    for (const auto& link : links) {
        const auto& stats = link.stats();
        total.read_transfers = checked_add(
            total.read_transfers,
            stats.read_transfers,
            "aggregate base-die read transfers");
        total.write_transfers = checked_add(
            total.write_transfers,
            stats.write_transfers,
            "aggregate base-die write transfers");
        total.read_bytes = checked_add(
            total.read_bytes,
            stats.read_bytes,
            "aggregate base-die read bytes");
        total.write_bytes = checked_add(
            total.write_bytes,
            stats.write_bytes,
            "aggregate base-die write bytes");
        total.read_queue_wait_ns += stats.read_queue_wait_ns;
        total.write_queue_wait_ns += stats.write_queue_wait_ns;
        total.read_busy_ns += stats.read_busy_ns;
        total.write_busy_ns += stats.write_busy_ns;
        total.read_fixed_latency_work_ns += stats.read_fixed_latency_work_ns;
        total.write_fixed_latency_work_ns += stats.write_fixed_latency_work_ns;
        total.first_arrival_ns = std::min(
            total.first_arrival_ns, stats.first_arrival_ns);
        total.finish_ns = std::max(total.finish_ns, stats.finish_ns);
    }
    return total;
}

} // namespace hbfsim::physical
