#include "physical/external/external_backing_device.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

namespace hbfsim::physical::external {
namespace {

std::uint64_t checked_add(
    std::uint64_t lhs,
    std::uint64_t rhs,
    const char* name) {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
        throw std::runtime_error(std::string(name) + " overflows uint64_t");
    }
    return lhs + rhs;
}

void validate_positive_finite(double value, const char* name) {
    if (!std::isfinite(value) || value <= 0.0) {
        throw std::runtime_error(
            std::string(name) + " must be positive and finite");
    }
}

void validate_nonnegative_finite(double value, const char* name) {
    if (!std::isfinite(value) || value < 0.0) {
        throw std::runtime_error(
            std::string(name) + " must be non-negative and finite");
    }
}

double resource_span(double first_busy_ns, double last_busy_ns) {
    return std::isfinite(first_busy_ns) && last_busy_ns > first_busy_ns ?
        last_busy_ns - first_busy_ns : 0.0;
}

void observe_busy_interval(
    double start_ns,
    double finish_ns,
    double& first_busy_ns,
    double& last_busy_ns) {
    if (finish_ns <= start_ns) {
        return;
    }
    first_busy_ns = std::min(first_busy_ns, start_ns);
    last_busy_ns = std::max(last_busy_ns, finish_ns);
}

std::string traffic_detail(
    std::uint64_t wire_bytes,
    std::uint64_t payload_bytes,
    std::uint64_t protocol_bytes) {
    return std::to_string(wire_bytes) + "B wire (" +
        std::to_string(payload_bytes) + "B payload+" +
        std::to_string(protocol_bytes) + "B protocol)";
}

} // namespace

const char* to_string(ExternalBackingKind kind) {
    switch (kind) {
    case ExternalBackingKind::OnPackageLpddr:
        return "on-package-lpddr";
    case ExternalBackingKind::HostDram:
        return "host-dram";
    case ExternalBackingKind::CxlMemory:
        return "cxl-memory";
    case ExternalBackingKind::NvmeSsd:
        return "nvme-ssd";
    case ExternalBackingKind::CxlSsd:
        return "cxl-ssd";
    }
    throw std::runtime_error("unknown external-backing kind");
}

ExternalBackingKind parse_external_backing_kind(const std::string& value) {
    if (value == "on-package-lpddr") {
        return ExternalBackingKind::OnPackageLpddr;
    }
    if (value == "host-dram") {
        return ExternalBackingKind::HostDram;
    }
    if (value == "cxl-memory") {
        return ExternalBackingKind::CxlMemory;
    }
    if (value == "nvme-ssd") {
        return ExternalBackingKind::NvmeSsd;
    }
    if (value == "cxl-ssd") {
        return ExternalBackingKind::CxlSsd;
    }
    throw std::runtime_error(
        "external-backing-kind must be on-package-lpddr, host-dram, "
        "cxl-memory, nvme-ssd, or cxl-ssd");
}

ExternalBackingConfig on_package_lpddr_profile() {
    auto config = ExternalBackingConfig{};
    config.kind = ExternalBackingKind::OnPackageLpddr;
    config.capacity_bytes = 2ull << 40; // 2 TiB
    config.media_channels = 16;
    config.max_outstanding_requests = 512;
    config.controller_issue_ns = 2.0;
    config.controller_processing_ns = 20.0;
    config.media_read_latency_ns = 100.0;
    config.media_write_latency_ns = 100.0;
    config.media_read_bandwidth_GBps = 2'000.0;
    config.media_write_bandwidth_GBps = 2'000.0;
    config.m2s_bandwidth_GBps = 2'000.0;
    config.s2m_bandwidth_GBps = 2'000.0;
    config.one_way_propagation_ns = 0.0;
    return config;
}

ExternalBackingConfig cxl_memory_profile() {
    return ExternalBackingConfig{};
}

ExternalBackingConfig host_dram_profile() {
    // Host DRAM has its own canonical identity. These starting values remain
    // an explicitly provisional sensitivity envelope; platform measurements
    // should override every timing/bandwidth field in a calibration overlay.
    auto config = ExternalBackingConfig{};
    config.kind = ExternalBackingKind::HostDram;
    return config;
}

// cxl_ssd_profile() lives in cxl_ssd.cpp with the rest of the CXL-SSD
// device module.

ExternalBackingConfig nvme_ssd_profile() {
    auto config = ExternalBackingConfig{};
    config.kind = ExternalBackingKind::NvmeSsd;
    config.capacity_bytes = 4ull << 40; // 4 TiB
    config.media_channels = 4;
    config.max_outstanding_requests = 512;
    config.controller_issue_ns = 20.0;
    config.controller_processing_ns = 500.0;
    config.media_read_latency_ns = 80'000.0;
    config.media_write_latency_ns = 100'000.0;
    config.media_read_bandwidth_GBps = 14.0;
    config.media_write_bandwidth_GBps = 7.0;
    config.m2s_bandwidth_GBps = 16.0;
    config.s2m_bandwidth_GBps = 16.0;
    config.one_way_propagation_ns = 250.0;
    return config;
}

double ExternalBackingStats::active_span_ns() const {
    return std::isfinite(first_arrival_ns) && finish_ns > first_arrival_ns ?
        finish_ns - first_arrival_ns : 0.0;
}

double ExternalBackingStats::controller_active_span_ns() const {
    return resource_span(
        controller_first_busy_ns, controller_last_busy_ns);
}

double ExternalBackingStats::media_active_span_ns() const {
    return resource_span(media_first_busy_ns, media_last_busy_ns);
}

double ExternalBackingStats::m2s_active_span_ns() const {
    return resource_span(m2s_first_busy_ns, m2s_last_busy_ns);
}

double ExternalBackingStats::s2m_active_span_ns() const {
    return resource_span(s2m_first_busy_ns, s2m_last_busy_ns);
}

double ExternalBackingStats::controller_utilization() const {
    const auto span = controller_active_span_ns();
    return span <= 0.0 ? 0.0 : controller_issue_busy_ns / span;
}

double ExternalBackingStats::media_utilization() const {
    // Demand media traffic (including write-miss read-for-ownership) plus the
    // device-internal flush and prefetch work the cache adds: the channels are
    // equally busy either way.
    const auto span = media_active_span_ns();
    return span <= 0.0 || media_channels == 0 ? 0.0 :
        (media_read_busy_ns + media_write_busy_ns +
         cache_flush_busy_ns + cache_prefetch_busy_ns) /
            (span * static_cast<double>(media_channels));
}

double ExternalBackingStats::m2s_utilization() const {
    const auto span = m2s_active_span_ns();
    return span <= 0.0 ? 0.0 : m2s_busy_ns / span;
}

double ExternalBackingStats::s2m_utilization() const {
    const auto span = s2m_active_span_ns();
    return span <= 0.0 ? 0.0 : s2m_busy_ns / span;
}

ExternalBackingDevice::Reservation
ExternalBackingDevice::SerialResourceTimeline::reserve(
    double ready_ns,
    double busy_ns) {
    validate_nonnegative_finite(ready_ns, "resource ready time");
    validate_nonnegative_finite(busy_ns, "resource busy time");
    if (busy_ns == 0.0) {
        return Reservation{
            .start_ns = ready_ns,
            .finish_ns = ready_ns,
            .queue_wait_ns = 0.0,
            .busy_ns = 0.0,
        };
    }

    double start_ns = ready_ns;
    // Causal homogeneous streams reach every serial stage in monotone order.
    // This is the overwhelmingly common path and can append without a search.
    if (!intervals_.empty() && ready_ns >= intervals_.back().first) {
        start_ns = std::max(start_ns, intervals_.back().second);
        const auto finish_ns = start_ns + busy_ns;
        if (!std::isfinite(finish_ns)) {
            throw std::runtime_error(
                "serial-resource reservation exceeds finite time");
        }
        if (start_ns == intervals_.back().second) {
            // The calendar represents occupied time, not request identity.
            // Adjacent reservations have no schedulable gap and are exactly
            // equivalent to one interval.  Coalescing them avoids walking a
            // credit-window worth of page records at every later query.
            intervals_.back().second = finish_ns;
        } else {
            intervals_.emplace_back(start_ns, finish_ns);
        }
        return Reservation{
            .start_ns = start_ns,
            .finish_ns = finish_ns,
            .queue_wait_ns = start_ns - ready_ns,
            .busy_ns = busy_ns,
        };
    }

    auto next = std::upper_bound(
        intervals_.begin(),
        intervals_.end(),
        start_ns,
        [](double value, const auto& interval) {
            return value < interval.first;
        });
    if (next != intervals_.begin()) {
        const auto& previous = *std::prev(next);
        if (previous.second > start_ns) {
            start_ns = previous.second;
        }
    }
    next = std::lower_bound(
        intervals_.begin(),
        intervals_.end(),
        start_ns,
        [](const auto& interval, double value) {
            return interval.first < value;
        });
    while (next != intervals_.end() &&
           start_ns + busy_ns > next->first) {
        start_ns = std::max(start_ns, next->second);
        next = std::lower_bound(
            next,
            intervals_.end(),
            start_ns,
            [](const auto& interval, double value) {
                return interval.first < value;
            });
    }
    const auto finish_ns = start_ns + busy_ns;
    if (!std::isfinite(finish_ns)) {
        throw std::runtime_error(
            "serial-resource reservation exceeds finite time");
    }
    const auto joins_previous =
        next != intervals_.begin() &&
        std::prev(next)->second == start_ns;
    const auto joins_next =
        next != intervals_.end() && finish_ns == next->first;
    if (joins_previous) {
        auto previous = std::prev(next);
        previous->second = finish_ns;
        if (joins_next) {
            previous->second = next->second;
            intervals_.erase(next);
        }
    } else if (joins_next) {
        next->first = start_ns;
    } else {
        intervals_.insert(next, {start_ns, finish_ns});
    }
    return Reservation{
        .start_ns = start_ns,
        .finish_ns = finish_ns,
        .queue_wait_ns = start_ns - ready_ns,
        .busy_ns = busy_ns,
    };
}

void ExternalBackingDevice::SerialResourceTimeline::
discard_finished_at_or_before(double frontier_ns) {
    validate_nonnegative_finite(frontier_ns, "resource discard frontier");
    while (!intervals_.empty() &&
           intervals_.front().second <= frontier_ns) {
        intervals_.pop_front();
    }
}

ExternalBackingDevice::ExternalBackingDevice(
    ExternalBackingConfig config,
    AddressHeatmap* address_heatmap)
    : config_(std::move(config)),
      address_heatmap_(address_heatmap) {
    if (config_.capacity_bytes == 0 || config_.page_size_bytes == 0 ||
        config_.request_segment_bytes == 0) {
        throw std::runtime_error(
            "external-backing capacity, page size, and request segment "
            "must be positive");
    }
    if (config_.capacity_bytes % config_.page_size_bytes != 0) {
        throw std::runtime_error(
            "external-backing capacity must be page aligned");
    }
    if (config_.request_segment_bytes < config_.page_size_bytes ||
        config_.request_segment_bytes % config_.page_size_bytes != 0) {
        throw std::runtime_error(
            "external-backing request segment must be a positive multiple "
            "of the address page size");
    }
    if (config_.media_read_queues == 0 ||
        config_.media_write_queues == 0 ||
        config_.media_channels == 0 ||
        config_.max_outstanding_requests == 0) {
        throw std::runtime_error(
            "external-backing media queues, channels, and outstanding "
            "limit must be positive");
    }
    if (config_.command_bytes == 0 || config_.completion_bytes == 0) {
        throw std::runtime_error(
            "external-backing protocol records must be positive");
    }
    validate_nonnegative_finite(
        config_.controller_issue_ns,
        "external-backing controller issue interval");
    validate_nonnegative_finite(
        config_.controller_processing_ns,
        "external-backing controller processing latency");
    validate_nonnegative_finite(
        config_.media_read_latency_ns,
        "external-backing media read latency");
    validate_nonnegative_finite(
        config_.media_write_latency_ns,
        "external-backing media write latency");
    validate_positive_finite(
        config_.media_read_bandwidth_GBps,
        "external-backing media read bandwidth");
    validate_positive_finite(
        config_.media_write_bandwidth_GBps,
        "external-backing media write bandwidth");
    validate_positive_finite(
        config_.m2s_bandwidth_GBps,
        "external-backing M2S bandwidth");
    validate_positive_finite(
        config_.s2m_bandwidth_GBps,
        "external-backing S2M bandwidth");
    validate_nonnegative_finite(
        config_.one_way_propagation_ns,
        "external-backing one-way propagation latency");

    const auto queue_count = std::max(
        config_.media_read_queues, config_.media_write_queues);
    if (static_cast<std::uint64_t>(queue_count) >
        std::numeric_limits<std::size_t>::max() /
            static_cast<std::uint64_t>(config_.media_channels)) {
        throw std::runtime_error(
            "external-backing media resource count exceeds size_t");
    }
    const auto media_resource_count =
        static_cast<std::size_t>(queue_count) * config_.media_channels;
    media_resources_.resize(media_resource_count);
    stats_.kind = config_.kind;
    stats_.media_channels = media_resource_count;
    stats_.media_channels_per_queue = config_.media_channels;
    stats_.media_read_queues = config_.media_read_queues;
    stats_.media_write_queues = config_.media_write_queues;

    const auto& cache_config = config_.device_cache;
    if (cache_config.enabled) {
        if (config_.kind != ExternalBackingKind::CxlSsd) {
            throw std::runtime_error(
                "external-backing device cache requires kind cxl-ssd");
        }
        if (cache_config.capacity_bytes < config_.request_segment_bytes ||
            cache_config.capacity_bytes %
                config_.request_segment_bytes != 0) {
            throw std::runtime_error(
                "external-backing cache capacity must be a positive "
                "multiple of the transport segment");
        }
        if (cache_config.capacity_bytes > config_.capacity_bytes) {
            throw std::runtime_error(
                "external-backing cache capacity cannot exceed device "
                "capacity");
        }
        // The buffer directory is keyed by transport segment, and a whole
        // line flushes through the flash program path. A segment larger
        // than one flash page would let a 4 KiB write-allocate flush a
        // multi-page line it never received (a 32x write amplification at
        // 128 KiB segments) and would hide the read-modify-write a partial
        // line needs. Lines must therefore be flash pages.
        if (config_.request_segment_bytes != config_.page_size_bytes) {
            throw std::runtime_error(
                "external-backing device cache lines are flash pages: set "
                "external-backing-request-segment-bytes equal to "
                "external-backing-page-size when the cache is enabled");
        }
        const auto cache_lines =
            cache_config.capacity_bytes / config_.request_segment_bytes;
        if (cache_config.ways != 0 &&
            cache_lines % cache_config.ways != 0) {
            throw std::runtime_error(
                "external-backing cache ways must divide the cache line "
                "count");
        }
        if (cache_config.prefetch_stride == 0) {
            throw std::runtime_error(
                "external-backing cache prefetch stride must be positive");
        }
        if (cache_config.prefetch_degree > 64) {
            throw std::runtime_error(
                "external-backing cache prefetch degree above 64 is "
                "unsupported");
        }
        validate_nonnegative_finite(
            cache_config.hit_latency_ns,
            "external-backing cache hit latency");
        validate_positive_finite(
            cache_config.hit_bandwidth_GBps,
            "external-backing cache hit bandwidth");
        device_cache_.emplace(
            cache_config,
            config_.request_segment_bytes,
            config_.capacity_bytes);
    }
}

std::size_t ExternalBackingDevice::select_media_queue(Op op) {
    auto* next_queue = op == Op::Read ?
        &next_read_queue_ : op == Op::Write ? &next_write_queue_ : nullptr;
    const auto queue_count = op == Op::Read ?
        config_.media_read_queues : config_.media_write_queues;
    if (next_queue == nullptr) {
        throw std::runtime_error(
            "external backing only supports read/write media queues");
    }
    const auto selected = *next_queue;
    *next_queue = selected + 1 == queue_count ? 0 : selected + 1;
    return selected;
}

std::size_t ExternalBackingDevice::striped_channel_for(
    std::uint64_t addr) const {
    return static_cast<std::size_t>(
        (addr / config_.request_segment_bytes) % config_.media_channels);
}

std::size_t ExternalBackingDevice::media_resource_for(
    std::size_t queue_index,
    std::size_t striped_channel_index) const {
    return queue_index * config_.media_channels + striped_channel_index;
}

std::string ExternalBackingDevice::media_resource_path(
    std::size_t queue_index,
    std::size_t striped_channel_index) const {
    return "external/" + std::string(to_string(config_.kind)) +
        "/media-queue" + std::to_string(queue_index) +
        "/channel" + std::to_string(striped_channel_index);
}

double ExternalBackingDevice::admit(double arrival_ns) {
    while (!inflight_finishes_.empty() &&
           inflight_finishes_.top() <= arrival_ns) {
        inflight_finishes_.pop();
    }
    // FIFO admission cannot move backwards after a waiting request has
    // advanced to a credit-release frontier.  This also handles several
    // credits completing at exactly the same instant: all queued requests may
    // enter at that instant, never retroactively at their old arrival time.
    double admitted_ns = std::max(arrival_ns, admission_frontier_ns_);
    while (!inflight_finishes_.empty() &&
           inflight_finishes_.top() <= admitted_ns) {
        inflight_finishes_.pop();
    }
    while (inflight_finishes_.size() >=
           config_.max_outstanding_requests) {
        admitted_ns = std::max(admitted_ns, inflight_finishes_.top());
        do {
            inflight_finishes_.pop();
        } while (!inflight_finishes_.empty() &&
                 inflight_finishes_.top() <= admitted_ns);
    }
    admission_frontier_ns_ = admitted_ns;
    return admitted_ns;
}

ExternalBackingDevice::RequestTiming ExternalBackingDevice::schedule(
    const PhysicalRequest& request,
    MediaResourceState& media_resource) {
    RequestTiming timing;
    timing.admitted_ns = admit(request.arrival_ns);
    // Every later request is admitted no earlier than this frontier, and no
    // downstream stage can become ready before admission.  Expired intervals
    // are therefore observationally dead for all future earliest-gap queries.
    m2s_.discard_finished_at_or_before(timing.admitted_ns);
    controller_.discard_finished_at_or_before(timing.admitted_ns);
    media_resource.media.discard_finished_at_or_before(timing.admitted_ns);
    s2m_.discard_finished_at_or_before(timing.admitted_ns);
    device_cache_port_.discard_finished_at_or_before(timing.admitted_ns);
    discard_completed_cache_events(timing.admitted_ns);
    const bool read = request.op == Op::Read;
    timing.m2s_payload_bytes = read ? 0 : request.bytes;
    timing.m2s_protocol_bytes = config_.command_bytes;
    timing.m2s_wire_bytes = checked_add(
        timing.m2s_payload_bytes,
        timing.m2s_protocol_bytes,
        "external-backing M2S wire bytes");
    timing.s2m_payload_bytes = read ? request.bytes : 0;
    timing.s2m_protocol_bytes = config_.completion_bytes;
    timing.s2m_wire_bytes = checked_add(
        timing.s2m_payload_bytes,
        timing.s2m_protocol_bytes,
        "external-backing S2M wire bytes");

    timing.m2s = m2s_.reserve(
        timing.admitted_ns,
        transfer_time_ns(
            timing.m2s_wire_bytes, config_.m2s_bandwidth_GBps));
    timing.m2s_arrival_ns =
        timing.m2s.finish_ns + config_.one_way_propagation_ns;
    timing.controller = controller_.reserve(
        timing.m2s_arrival_ns, config_.controller_issue_ns);
    timing.controller_finish_ns =
        timing.controller.finish_ns + config_.controller_processing_ns;
    double media_stage_ready_ns = timing.controller_finish_ns;
    bool served_from_cache = false;
    bool cache_read_for_ownership = false;
    std::vector<std::uint64_t> prefetch_segments;
    std::optional<std::uint64_t> cache_segment;
    if (device_cache_) {
        const auto segment =
            request.addr / config_.request_segment_bytes;
        cache_segment = segment;
        const auto decision = device_cache_->access(segment, request.op);
        // Dirty victims flush through the flash write path before the
        // evicting access proceeds: the freed buffer line is reusable only
        // once its old content is durable, which also back-pressures
        // sustained writes to the flash program rate once the buffer fills
        // with dirty lines. A victim whose own buffer write or
        // read-for-ownership is still landing cannot be flushed before that
        // content exists, so its fill state is consulted before it is
        // dropped.
        double gate_ns = schedule_cache_writebacks(
            decision.writeback_segments, media_stage_ready_ns);
        // Victims leave the directory; any fill state still tracked for
        // them is dead, whether or not they were dirty.
        for (const auto evicted : decision.evicted_segments) {
            pending_fill_ready_ns_.erase(evicted);
        }
        if (decision.hit) {
            // A hit on a line whose fill is still in flight, whether a
            // background prefetch or an earlier demand miss, waits for that
            // fill: the directory marks a line present at scheduling time,
            // but its data lands only when the flash read (or the buffer
            // write that installed it) completes. Every hit before that
            // instant waits, so the entry is dropped only once it is past.
            const auto pending = pending_fill_ready_ns_.find(segment);
            if (pending != pending_fill_ready_ns_.end()) {
                gate_ns = std::max(gate_ns, pending->second);
                if (pending->second <= media_stage_ready_ns) {
                    pending_fill_ready_ns_.erase(pending);
                }
            }
        } else {
            // A miss re-reads or re-installs a line whose last dirty flush
            // may still be programming: the flash contents are not the
            // new version before that flush lands, and two flushes of one
            // line must not reorder on the channel calendar.
            const auto flush = pending_flush_ready_ns_.find(segment);
            if (flush != pending_flush_ready_ns_.end()) {
                gate_ns = std::max(gate_ns, flush->second);
                pending_flush_ready_ns_.erase(flush);
            }
            cache_read_for_ownership = !read &&
                (request.addr % config_.request_segment_bytes != 0 ||
                 request.bytes != config_.request_segment_bytes);
        }
        if (gate_ns > media_stage_ready_ns) {
            timing.writeback_gate_wait_ns =
                gate_ns - media_stage_ready_ns;
            stats_.cache_writeback_gate_wait_ns +=
                timing.writeback_gate_wait_ns;
            media_stage_ready_ns = gate_ns;
        }
        prefetch_segments = decision.prefetch_segments;
        if (read) {
            auto& counter = decision.hit ?
                stats_.cache_read_hits : stats_.cache_read_misses;
            counter = checked_add(
                counter, 1, "external-backing cache read census");
        } else {
            auto& counter = decision.hit ?
                stats_.cache_write_hits : stats_.cache_write_misses;
            counter = checked_add(
                counter, 1, "external-backing cache write census");
        }
        // Write-back write-allocate: every write lands in the buffer DRAM;
        // a sub-page write miss first fetches the untouched bytes so a later
        // whole-page flash program cannot fabricate them.
        served_from_cache = !read || decision.hit;
    }
    timing.served_from_cache = served_from_cache;
    timing.cache_read_for_ownership = cache_read_for_ownership;
    if (cache_read_for_ownership) {
        const auto read_queue = select_media_queue(Op::Read);
        const auto resource_index = media_resource_for(
            read_queue, striped_channel_for(request.addr));
        auto& resource = media_resources_.at(resource_index);
        if (!resource.active) {
            resource.active = true;
            ++stats_.active_media_resources;
        }
        resource.media.discard_finished_at_or_before(timing.admitted_ns);
        const auto directional_media_resources =
            static_cast<double>(config_.media_read_queues) *
            static_cast<double>(config_.media_channels);
        timing.cache_read_for_ownership_media = resource.media.reserve(
            media_stage_ready_ns + config_.media_read_latency_ns,
            transfer_time_ns(
                config_.request_segment_bytes,
                config_.media_read_bandwidth_GBps /
                    directional_media_resources));
        media_stage_ready_ns =
            timing.cache_read_for_ownership_media.finish_ns;
        stats_.cache_read_for_ownership_segments = checked_add(
            stats_.cache_read_for_ownership_segments,
            1,
            "external-backing cache read-for-ownership segments");
        stats_.cache_read_for_ownership_bytes = checked_add(
            stats_.cache_read_for_ownership_bytes,
            config_.request_segment_bytes,
            "external-backing cache read-for-ownership bytes");
        observe_busy_interval(
            timing.cache_read_for_ownership_media.start_ns,
            timing.cache_read_for_ownership_media.finish_ns,
            stats_.media_first_busy_ns,
            stats_.media_last_busy_ns);
    }
    if (served_from_cache) {
        timing.media_latency_applied_ns =
            config_.device_cache.hit_latency_ns;
        timing.media_latency_finish_ns =
            media_stage_ready_ns + timing.media_latency_applied_ns;
        timing.media = device_cache_port_.reserve(
            timing.media_latency_finish_ns,
            transfer_time_ns(
                request.bytes, config_.device_cache.hit_bandwidth_GBps));
        if (cache_segment && !read) {
            // A write-allocated line holds the caller's payload only once
            // the buffer write has landed; a read hit before that waits.
            retain_pending_fill(*cache_segment, timing.media.finish_ns);
        }
    } else {
        timing.media_latency_applied_ns = read ?
            config_.media_read_latency_ns :
            config_.media_write_latency_ns;
        timing.media_latency_finish_ns =
            media_stage_ready_ns + timing.media_latency_applied_ns;
        const auto aggregate_media_bandwidth = read ?
            config_.media_read_bandwidth_GBps :
            config_.media_write_bandwidth_GBps;
        const auto directional_queue_count = read ?
            config_.media_read_queues : config_.media_write_queues;
        const auto directional_media_resources =
            static_cast<double>(directional_queue_count) *
            static_cast<double>(config_.media_channels);
        timing.media = media_resource.media.reserve(
            timing.media_latency_finish_ns,
            transfer_time_ns(
                request.bytes,
                aggregate_media_bandwidth / directional_media_resources));
        if (cache_segment) {
            // Demand-miss fill: the line is present in the directory from
            // now on, but readable only once the flash read has delivered
            // it to the buffer.
            retain_pending_fill(*cache_segment, timing.media.finish_ns);
        }
    }
    // Background fills reserve their channels only after the demand access,
    // so a prefetch never steals the demand read's earliest gap.
    for (const auto prefetch_segment : prefetch_segments) {
        schedule_cache_prefetch_fill(
            prefetch_segment, timing.controller_finish_ns);
    }
    timing.s2m = s2m_.reserve(
        timing.media.finish_ns,
        transfer_time_ns(
            timing.s2m_wire_bytes, config_.s2m_bandwidth_GBps));
    timing.finish_ns =
        timing.s2m.finish_ns + config_.one_way_propagation_ns;
    if (!std::isfinite(timing.finish_ns)) {
        throw std::runtime_error(
            "external-backing completion exceeds finite time");
    }
    return timing;
}

double ExternalBackingDevice::schedule_cache_writebacks(
    const std::vector<std::uint64_t>& writeback_segments,
    double ready_ns) {
    double gate_ns = ready_ns;
    for (const auto segment : writeback_segments) {
        const auto addr = segment * config_.request_segment_bytes;
        // Internal flush traffic always uses write queue 0 for the victim's
        // striped channel so it never perturbs the caller-range round-robin.
        const auto resource_index =
            media_resource_for(0, striped_channel_for(addr));
        auto& resource = media_resources_.at(resource_index);
        if (!resource.active) {
            resource.active = true;
            ++stats_.active_media_resources;
        }
        resource.media.discard_finished_at_or_before(admission_frontier_ns_);
        const auto directional_media_resources =
            static_cast<double>(config_.media_write_queues) *
            static_cast<double>(config_.media_channels);
        // The flush reads the line's content from the buffer, so it cannot
        // start before the fill that installed that content has landed.
        double flush_ready_ns = ready_ns;
        if (const auto fill = pending_fill_ready_ns_.find(segment);
            fill != pending_fill_ready_ns_.end()) {
            flush_ready_ns = std::max(flush_ready_ns, fill->second);
        }
        const auto reservation = resource.media.reserve(
            flush_ready_ns + config_.media_write_latency_ns,
            transfer_time_ns(
                config_.request_segment_bytes,
                config_.media_write_bandwidth_GBps /
                    directional_media_resources));
        gate_ns = std::max(gate_ns, reservation.finish_ns);
        auto& pending = pending_flush_ready_ns_[segment];
        pending = std::max(pending, reservation.finish_ns);
        pending_flush_expiries_.emplace(pending, segment);
        stats_.cache_writeback_segments = checked_add(
            stats_.cache_writeback_segments,
            1,
            "external-backing cache writeback segments");
        stats_.cache_writeback_bytes = checked_add(
            stats_.cache_writeback_bytes,
            config_.request_segment_bytes,
            "external-backing cache writeback bytes");
        stats_.cache_flush_busy_ns += reservation.busy_ns;
        observe_busy_interval(
            reservation.start_ns,
            reservation.finish_ns,
            stats_.media_first_busy_ns,
            stats_.media_last_busy_ns);
    }
    return gate_ns;
}

void ExternalBackingDevice::schedule_cache_prefetch_fill(
    std::uint64_t prefetch_segment,
    double ready_ns) {
    // The prefetched line's own dirty victims flush first; the background
    // fill then reads flash without ever gating the demand access.
    const auto install = device_cache_->install_prefetch(prefetch_segment);
    double victim_gate_ns = schedule_cache_writebacks(
        install.writeback_segments, ready_ns);
    for (const auto evicted : install.evicted_segments) {
        pending_fill_ready_ns_.erase(evicted);
    }
    if (const auto flush = pending_flush_ready_ns_.find(prefetch_segment);
        flush != pending_flush_ready_ns_.end()) {
        victim_gate_ns = std::max(victim_gate_ns, flush->second);
        pending_flush_ready_ns_.erase(flush);
    }
    const auto addr = prefetch_segment * config_.request_segment_bytes;
    const auto resource_index =
        media_resource_for(0, striped_channel_for(addr));
    auto& resource = media_resources_.at(resource_index);
    if (!resource.active) {
        resource.active = true;
        ++stats_.active_media_resources;
    }
    resource.media.discard_finished_at_or_before(admission_frontier_ns_);
    const auto directional_media_resources =
        static_cast<double>(config_.media_read_queues) *
        static_cast<double>(config_.media_channels);
    const auto reservation = resource.media.reserve(
        std::max(ready_ns, victim_gate_ns) +
            config_.media_read_latency_ns,
        transfer_time_ns(
            config_.request_segment_bytes,
            config_.media_read_bandwidth_GBps /
                directional_media_resources));
    stats_.cache_prefetch_segments = checked_add(
        stats_.cache_prefetch_segments,
        1,
        "external-backing cache prefetch segments");
    stats_.cache_prefetch_bytes = checked_add(
        stats_.cache_prefetch_bytes,
        config_.request_segment_bytes,
        "external-backing cache prefetch bytes");
    stats_.cache_prefetch_busy_ns += reservation.busy_ns;
    observe_busy_interval(
        reservation.start_ns,
        reservation.finish_ns,
        stats_.media_first_busy_ns,
        stats_.media_last_busy_ns);
    retain_pending_fill(prefetch_segment, reservation.finish_ns);
}

void ExternalBackingDevice::retain_pending_fill(
    std::uint64_t segment,
    double ready_ns) {
    auto& pending = pending_fill_ready_ns_[segment];
    pending = std::max(pending, ready_ns);
    pending_fill_expiries_.emplace(pending, segment);
}

void ExternalBackingDevice::discard_completed_cache_events(
    double causal_frontier_ns) {
    while (!pending_flush_expiries_.empty() &&
           pending_flush_expiries_.top().first <= causal_frontier_ns) {
        const auto [finish_ns, segment] = pending_flush_expiries_.top();
        pending_flush_expiries_.pop();
        const auto current = pending_flush_ready_ns_.find(segment);
        if (current != pending_flush_ready_ns_.end() &&
            current->second == finish_ns) {
            pending_flush_ready_ns_.erase(current);
        }
    }
    while (!pending_fill_expiries_.empty() &&
           pending_fill_expiries_.top().first <= causal_frontier_ns) {
        const auto [finish_ns, segment] = pending_fill_expiries_.top();
        pending_fill_expiries_.pop();
        const auto current = pending_fill_ready_ns_.find(segment);
        if (current != pending_fill_ready_ns_.end() &&
            current->second == finish_ns) {
            pending_fill_ready_ns_.erase(current);
        }
    }
}

void ExternalBackingDevice::validate_request(
    const PhysicalRequest& request) const {
    validate_range_request(request);
    const auto bytes_remaining_in_page =
        config_.page_size_bytes -
        request.addr % config_.page_size_bytes;
    if (request.bytes > bytes_remaining_in_page) {
        throw std::runtime_error(
            "external-backing physical request crosses a page boundary");
    }
}

void ExternalBackingDevice::validate_range_request(
    const PhysicalRequest& request) const {
    if (request.tier != Tier::External) {
        throw std::runtime_error(
            "ExternalBackingDevice received a non-external request");
    }
    if (request.op != Op::Read && request.op != Op::Write) {
        throw std::runtime_error(
            "external backing only supports read/write operations");
    }
    if (request.address_space != AddressSpace::Logical) {
        throw std::runtime_error(
            "external backing exposes one host-visible logical address space");
    }
    if (!std::isfinite(request.arrival_ns) || request.arrival_ns < 0.0) {
        throw std::runtime_error(
            "external-backing arrival must be finite and non-negative");
    }
    if (request.arrival_ns < last_issue_arrival_ns_) {
        throw std::runtime_error(
            "external-backing requests must be issued in causal time order");
    }
    if (request.bytes == 0) {
        throw std::runtime_error(
            "external-backing request bytes must be positive");
    }
    const auto end = checked_add(
        request.addr,
        request.bytes,
        "external-backing request end");
    if (end > config_.capacity_bytes) {
        throw std::runtime_error(
            "external-backing request exceeds configured capacity");
    }
}

Breakdown ExternalBackingDevice::account_timing(
    const PhysicalRequest& request,
    std::size_t media_resource_index,
    const RequestTiming& timing) {
    const bool read = request.op == Op::Read;
    if (!timing.served_from_cache) {
        auto& media_resource = media_resources_.at(media_resource_index);
        if (!media_resource.active) {
            media_resource.active = true;
            ++stats_.active_media_resources;
        }
    }

    auto& request_count = read ?
        stats_.read_requests : stats_.write_requests;
    auto& byte_count = read ? stats_.read_bytes : stats_.write_bytes;
    request_count = checked_add(
        request_count, 1, "external-backing request count");
    byte_count = checked_add(
        byte_count, request.bytes, "external-backing byte count");

    stats_.m2s_payload_bytes = checked_add(
        stats_.m2s_payload_bytes,
        timing.m2s_payload_bytes,
        "external-backing M2S payload bytes");
    stats_.m2s_protocol_bytes = checked_add(
        stats_.m2s_protocol_bytes,
        timing.m2s_protocol_bytes,
        "external-backing M2S protocol bytes");
    stats_.m2s_wire_bytes = checked_add(
        stats_.m2s_wire_bytes,
        timing.m2s_wire_bytes,
        "external-backing M2S wire bytes");
    stats_.s2m_payload_bytes = checked_add(
        stats_.s2m_payload_bytes,
        timing.s2m_payload_bytes,
        "external-backing S2M payload bytes");
    stats_.s2m_protocol_bytes = checked_add(
        stats_.s2m_protocol_bytes,
        timing.s2m_protocol_bytes,
        "external-backing S2M protocol bytes");
    stats_.s2m_wire_bytes = checked_add(
        stats_.s2m_wire_bytes,
        timing.s2m_wire_bytes,
        "external-backing S2M wire bytes");

    stats_.outstanding_wait_ns +=
        timing.admitted_ns - request.arrival_ns;
    stats_.m2s_queue_wait_ns += timing.m2s.queue_wait_ns;
    stats_.controller_queue_wait_ns += timing.controller.queue_wait_ns;
    stats_.s2m_queue_wait_ns += timing.s2m.queue_wait_ns;
    stats_.controller_issue_busy_ns += timing.controller.busy_ns;
    stats_.controller_processing_work_ns +=
        config_.controller_processing_ns;
    if (timing.served_from_cache) {
        stats_.cache_queue_wait_ns += timing.media.queue_wait_ns;
        stats_.cache_latency_work_ns += timing.media_latency_applied_ns;
        stats_.cache_busy_ns += timing.media.busy_ns;
    } else if (read) {
        stats_.media_queue_wait_ns += timing.media.queue_wait_ns;
        stats_.media_read_latency_work_ns +=
            timing.media_latency_applied_ns;
        stats_.media_read_busy_ns += timing.media.busy_ns;
    } else {
        stats_.media_queue_wait_ns += timing.media.queue_wait_ns;
        stats_.media_write_latency_work_ns +=
            timing.media_latency_applied_ns;
        stats_.media_write_busy_ns += timing.media.busy_ns;
    }
    if (timing.cache_read_for_ownership) {
        stats_.media_queue_wait_ns +=
            timing.cache_read_for_ownership_media.queue_wait_ns;
        stats_.media_read_latency_work_ns += config_.media_read_latency_ns;
        stats_.media_read_busy_ns +=
            timing.cache_read_for_ownership_media.busy_ns;
    }
    stats_.m2s_busy_ns += timing.m2s.busy_ns;
    stats_.s2m_busy_ns += timing.s2m.busy_ns;
    stats_.transport_propagation_work_ns +=
        2.0 * config_.one_way_propagation_ns;
    observe_busy_interval(
        timing.controller.start_ns,
        timing.controller.finish_ns,
        stats_.controller_first_busy_ns,
        stats_.controller_last_busy_ns);
    if (!timing.served_from_cache) {
        observe_busy_interval(
            timing.media.start_ns,
            timing.media.finish_ns,
            stats_.media_first_busy_ns,
            stats_.media_last_busy_ns);
    }
    observe_busy_interval(
        timing.m2s.start_ns,
        timing.m2s.finish_ns,
        stats_.m2s_first_busy_ns,
        stats_.m2s_last_busy_ns);
    observe_busy_interval(
        timing.s2m.start_ns,
        timing.s2m.finish_ns,
        stats_.s2m_first_busy_ns,
        stats_.s2m_last_busy_ns);
    stats_.first_arrival_ns = std::min(
        stats_.first_arrival_ns, request.arrival_ns);
    stats_.finish_ns = std::max(stats_.finish_ns, timing.finish_ns);

    Breakdown breakdown;
    breakdown.ingress_queue_wait_ns =
        timing.admitted_ns - request.arrival_ns;
    breakdown.scheduler_queue_wait_ns =
        timing.m2s.queue_wait_ns +
        timing.controller.queue_wait_ns +
        timing.media.queue_wait_ns +
        timing.cache_read_for_ownership_media.queue_wait_ns +
        timing.s2m.queue_wait_ns +
        timing.writeback_gate_wait_ns;
    breakdown.command_ns =
        timing.controller.busy_ns + config_.controller_processing_ns;
    if (timing.served_from_cache) {
        // Cache-served segments do buffer-DRAM work instead of flash array
        // and channel work.
        breakdown.write_buffer_dram_ns =
            timing.media_latency_applied_ns + timing.media.busy_ns;
    } else if (read) {
        breakdown.array_read_ns = timing.media_latency_applied_ns;
        breakdown.channel_transfer_ns = timing.media.busy_ns;
    } else {
        breakdown.array_program_ns = timing.media_latency_applied_ns;
        breakdown.channel_transfer_ns = timing.media.busy_ns;
    }
    if (timing.cache_read_for_ownership) {
        breakdown.array_read_ns += config_.media_read_latency_ns;
        breakdown.channel_transfer_ns +=
            timing.cache_read_for_ownership_media.busy_ns;
    }
    breakdown.hb_io_transfer_ns =
        timing.m2s.busy_ns + timing.s2m.busy_ns;
    breakdown.transport_latency_ns =
        2.0 * config_.one_way_propagation_ns;
    return breakdown;
}

void ExternalBackingDevice::account(
    const PhysicalRequest& request,
    std::size_t media_resource_index,
    const RequestTiming& timing,
    PhysicalCompletion& completion) {
    const bool read = request.op == Op::Read;
    completion.breakdown = account_timing(
        request, media_resource_index, timing);
    stats_.stage_work += completion.breakdown;

    auto* spans = trace_spans_enabled(request.trace) ?
        &completion.spans : nullptr;
    const auto prefix = read ? "external_read" : "external_write";
    add_trace_span(
        spans,
        prefix + std::string("_m2s_transfer"),
        "external_link",
        "external/m2s",
        timing.m2s.start_ns,
        timing.m2s.finish_ns,
        true,
        traffic_detail(
            timing.m2s_wire_bytes,
            timing.m2s_payload_bytes,
            timing.m2s_protocol_bytes));
    add_trace_span(
        spans,
        prefix + std::string("_m2s_propagation"),
        "external_link",
        "external/m2s",
        timing.m2s.finish_ns,
        timing.m2s_arrival_ns,
        true);
    add_trace_span(
        spans,
        prefix + std::string("_controller_issue"),
        "external_controller",
        "external/controller",
        timing.controller.start_ns,
        timing.controller.finish_ns,
        true);
    add_trace_span(
        spans,
        prefix + std::string("_controller_processing"),
        "external_controller",
        "external/controller",
        timing.controller.finish_ns,
        timing.controller_finish_ns,
        true);
    add_trace_span(
        spans,
        prefix + std::string("_media_latency"),
        "external_backing",
        completion.resource_path,
        timing.media_latency_finish_ns - timing.media_latency_applied_ns,
        timing.media_latency_finish_ns,
        true,
        std::to_string(request.bytes) + "B");
    add_trace_span(
        spans,
        prefix + std::string("_media_transfer"),
        "external_backing",
        completion.resource_path,
        timing.media.start_ns,
        timing.media.finish_ns,
        true,
        std::to_string(request.bytes) + "B");
    add_trace_span(
        spans,
        prefix + std::string("_s2m_transfer"),
        "external_link",
        "external/s2m",
        timing.s2m.start_ns,
        timing.s2m.finish_ns,
        true,
        traffic_detail(
            timing.s2m_wire_bytes,
            timing.s2m_payload_bytes,
            timing.s2m_protocol_bytes));
    add_trace_span(
        spans,
        prefix + std::string("_s2m_propagation"),
        "external_link",
        "external/s2m",
        timing.s2m.finish_ns,
        timing.finish_ns,
        true);

    if (address_heatmap_ != nullptr) {
        address_heatmap_->record(AddressTrafficRecord{
            .domain = AddressDomain::ExternalPhysical,
            .direction = read ?
                TrafficDirection::Read : TrafficDirection::Write,
            .source = request.heatmap_source,
            .address = request.addr,
            .bytes = request.bytes,
        });
    }
}

PhysicalCompletion ExternalBackingDevice::issue(
    const PhysicalRequest& request) {
    validate_request(request);
    last_issue_arrival_ns_ = request.arrival_ns;
    const auto queue_index = select_media_queue(request.op);
    const auto striped_channel_index = striped_channel_for(request.addr);
    const auto media_resource_index = media_resource_for(
        queue_index, striped_channel_index);
    auto& media_resource = media_resources_.at(media_resource_index);
    const auto timing = schedule(request, media_resource);
    inflight_finishes_.push(timing.finish_ns);
    stats_.max_device_outstanding = std::max<std::uint64_t>(
        stats_.max_device_outstanding,
        inflight_finishes_.size());

    PhysicalCompletion completion;
    completion.id = request.id;
    completion.tier = Tier::External;
    completion.op = request.op;
    completion.arrival_ns = request.arrival_ns;
    completion.start_ns = timing.admitted_ns;
    completion.finish_ns = timing.finish_ns;
    completion.logical_bytes = request.bytes;
    completion.physical_bytes = request.bytes;
    completion.resource_path = timing.served_from_cache ?
        "external/" + std::string(to_string(config_.kind)) +
            "/device-cache" :
        media_resource_path(queue_index, striped_channel_index);
    completion.note = request.op == Op::Read ?
        "external-backing-to-hbm" : "hbm-to-external-backing";
    account(request, media_resource_index, timing, completion);
    return completion;
}

PhysicalCompletion ExternalBackingDevice::issue_contiguous_range(
    const PhysicalRequest& request) {
    validate_range_request(request);
    last_issue_arrival_ns_ = request.arrival_ns;
    const auto end = checked_add(
        request.addr,
        request.bytes,
        "external-backing contiguous range end");
    const auto first_page = request.addr / config_.page_size_bytes;
    const auto last_page = (end - 1) / config_.page_size_bytes;
    const auto page_count = checked_add(
        last_page - first_page, 1, "external-backing page-run pages");

    PhysicalCompletion completion{
        .id = request.id,
        .tier = Tier::External,
        .op = request.op,
        .arrival_ns = request.arrival_ns,
        .start_ns = std::numeric_limits<double>::infinity(),
        .finish_ns = request.arrival_ns,
        .logical_bytes = request.bytes,
        .physical_bytes = request.bytes,
        .resource_path =
            "external/" + std::string(to_string(config_.kind)) +
            "/segmented-range",
        .note = "external-backing-segmented-range",
    };

    auto segment = request;
    auto cursor = request.addr;
    std::uint64_t segment_count = 0;
    const auto queue_index = select_media_queue(request.op);
    while (cursor < end) {
        const auto remaining_in_segment =
            config_.request_segment_bytes -
            cursor % config_.request_segment_bytes;
        const auto bytes = std::min(end - cursor, remaining_in_segment);
        segment.addr = cursor;
        segment.bytes = bytes;
        segment.id = request.id + "/segment" +
            std::to_string(segment_count);

        const auto striped_channel_index = striped_channel_for(cursor);
        const auto media_resource_index = media_resource_for(
            queue_index, striped_channel_index);
        auto& media_resource = media_resources_.at(media_resource_index);
        const auto timing = schedule(segment, media_resource);
        inflight_finishes_.push(timing.finish_ns);
        stats_.max_device_outstanding = std::max<std::uint64_t>(
            stats_.max_device_outstanding,
            inflight_finishes_.size());

        PhysicalCompletion segment_completion{
            .id = segment.id,
            .tier = Tier::External,
            .op = request.op,
            .arrival_ns = request.arrival_ns,
            .start_ns = timing.admitted_ns,
            .finish_ns = timing.finish_ns,
            .logical_bytes = bytes,
            .physical_bytes = bytes,
            .resource_path = timing.served_from_cache ?
                "external/" + std::string(to_string(config_.kind)) +
                    "/device-cache" :
                media_resource_path(queue_index, striped_channel_index),
            .note = "external-backing-transport-segment",
        };
        account(
            segment,
            media_resource_index,
            timing,
            segment_completion);
        completion.start_ns = std::min(
            completion.start_ns, segment_completion.start_ns);
        completion.finish_ns = std::max(
            completion.finish_ns, segment_completion.finish_ns);
        completion.breakdown += segment_completion.breakdown;
        completion.spans.insert(
            completion.spans.end(),
            std::make_move_iterator(segment_completion.spans.begin()),
            std::make_move_iterator(segment_completion.spans.end()));

        cursor += bytes;
        segment_count = checked_add(
            segment_count, 1, "external-backing page-run segments");
    }
    stats_.page_run_requests = checked_add(
        stats_.page_run_requests,
        1,
        "external-backing page-run requests");
    stats_.page_run_segments = checked_add(
        stats_.page_run_segments,
        segment_count,
        "external-backing page-run segments");
    stats_.page_run_pages = checked_add(
        stats_.page_run_pages,
        page_count,
        "external-backing page-run pages");
    return completion;
}

} // namespace hbfsim::physical::external
