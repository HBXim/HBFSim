#include "physical/external/cxl_ssd.hpp"

#include "physical/external/external_backing_device.hpp"

#include <algorithm>
#include <stdexcept>

namespace hbfsim::physical::external {

const char* to_string(DeviceCachePolicy policy) {
    switch (policy) {
    case DeviceCachePolicy::Fifo:
        return "fifo";
    case DeviceCachePolicy::Lifo:
        return "lifo";
    case DeviceCachePolicy::Clock:
        return "clock";
    case DeviceCachePolicy::S3Fifo:
        return "s3fifo";
    }
    throw std::invalid_argument("invalid device cache policy");
}

DeviceCachePolicy parse_device_cache_policy(const std::string& value) {
    if (value == "fifo") {
        return DeviceCachePolicy::Fifo;
    }
    if (value == "lifo") {
        return DeviceCachePolicy::Lifo;
    }
    if (value == "clock") {
        return DeviceCachePolicy::Clock;
    }
    if (value == "s3fifo") {
        return DeviceCachePolicy::S3Fifo;
    }
    throw std::runtime_error(
        "external-backing-cache-policy must be fifo, lifo, clock, or s3fifo");
}

ExternalBackingConfig cxl_ssd_profile() {
    // CXL-attached flash (CXL-SSD): the device class prototyped by Cylon
    // (FAST '26), a CXL.mem load/store front end over FEMU's black-box NAND
    // backend. Transport fields reuse the cxl-memory link envelope because
    // the host attachment is the same physical interface; only the
    // controller and media change. Controller timing keeps the NVMe-SSD
    // FTL-class envelope. Media follows the Cylon artifact's FEMU defaults:
    // a 4 KiB NAND page (512 B x 8 sectors), 8 channels x 8 LUNs, 40 us
    // page read, and 200 us page program. Aggregate directional media
    // bandwidth is the geometry envelope tt_luns x page / latency
    // (64 x 4 KiB / 40 us = 6.5536 GB/s; 64 x 4 KiB / 200 us =
    // 1.31072 GB/s), so one channel serializes exactly its 8 pipelined
    // LUNs. This is the uncached miss path: the device-side DRAM cache and
    // GC that Cylon also models stay disabled here; enable the cache with
    // the explicit device-cache overlay, and leave hit-rate calibration to a
    // measured overlay.
    auto config = ExternalBackingConfig{};
    config.kind = ExternalBackingKind::CxlSsd;
    config.capacity_bytes = 2ull << 40; // 2 TiB
    config.media_channels = 8;
    config.max_outstanding_requests = 512;
    config.controller_issue_ns = 20.0;
    config.controller_processing_ns = 500.0;
    config.media_read_latency_ns = 40'000.0;
    config.media_write_latency_ns = 200'000.0;
    config.media_read_bandwidth_GBps = 6.5536;
    config.media_write_bandwidth_GBps = 1.31072;
    config.m2s_bandwidth_GBps = 36.0;
    config.s2m_bandwidth_GBps = 36.0;
    config.one_way_propagation_ns = 75.0;
    return config;
}

CxlSsdDeviceCache::CxlSsdDeviceCache(
    const DeviceCacheConfig& config,
    std::uint64_t segment_bytes,
    std::uint64_t device_capacity_bytes)
    : config_(config),
      segment_bytes_(segment_bytes) {
    total_lines_ = config_.capacity_bytes / segment_bytes_;
    lines_per_set_ = config_.ways == 0 ? total_lines_ : config_.ways;
    set_count_ = config_.ways == 0 ? 1 : total_lines_ / config_.ways;
    // Cylon's S3-FIFO keeps a small probationary queue at one tenth of the
    // buffer, with at least one line so quick demotion always has a stage.
    small_target_lines_ = std::max<std::uint64_t>(1, lines_per_set_ / 10);
    device_segments_ = device_capacity_bytes / segment_bytes_;
}

std::uint64_t CxlSsdDeviceCache::set_index_for(std::uint64_t segment) const {
    return segment % set_count_;
}

std::size_t CxlSsdDeviceCache::occupied_lines(const Set& set) const {
    if (config_.policy == DeviceCachePolicy::S3Fifo) {
        return set.small.size() + set.main.size();
    }
    return set.queue.size();
}

void CxlSsdDeviceCache::remember_ghost(Set& set, std::uint64_t segment) {
    set.ghost.push_back(segment);
    set.ghost_members.insert(segment);
    // Drop stale front ids (readmitted lines erase membership only), then
    // bound the history to one set's worth of lines.
    while (!set.ghost.empty() &&
           !set.ghost_members.contains(set.ghost.front())) {
        set.ghost.pop_front();
    }
    while (set.ghost_members.size() > lines_per_set_) {
        while (!set.ghost.empty() &&
               !set.ghost_members.contains(set.ghost.front())) {
            set.ghost.pop_front();
        }
        if (set.ghost.empty()) {
            break;
        }
        set.ghost_members.erase(set.ghost.front());
        set.ghost.pop_front();
    }
}

void CxlSsdDeviceCache::evict_one(
    Set& set,
    std::vector<std::uint64_t>& writeback_segments,
    std::vector<std::uint64_t>& evicted_segments) {
    const auto retire = [&](std::uint64_t segment) {
        const auto it = present_.find(segment);
        if (it->second.dirty) {
            writeback_segments.push_back(segment);
        }
        evicted_segments.push_back(segment);
        present_.erase(it);
    };
    switch (config_.policy) {
    case DeviceCachePolicy::Fifo: {
        const auto segment = set.queue.front();
        set.queue.pop_front();
        retire(segment);
        return;
    }
    case DeviceCachePolicy::Lifo: {
        const auto segment = set.queue.back();
        set.queue.pop_back();
        retire(segment);
        return;
    }
    case DeviceCachePolicy::Clock: {
        // Second chance: a referenced front line is cleared and rotated;
        // the first pass clears every bit, so the walk is bounded.
        while (true) {
            const auto segment = set.queue.front();
            set.queue.pop_front();
            auto& state = present_.at(segment);
            if (state.counter > 0) {
                state.counter = 0;
                set.queue.push_back(segment);
                continue;
            }
            retire(segment);
            return;
        }
    }
    case DeviceCachePolicy::S3Fifo: {
        // Quick demotion from the small queue: an unreferenced probationary
        // line leaves to the ghost history, a referenced one promotes to
        // main (which does not free a line, so the walk continues). Main
        // reinserts referenced lines with a decremented counter, so the
        // walk terminates.
        while (true) {
            const bool from_small = !set.small.empty() &&
                (set.small.size() >= small_target_lines_ ||
                 set.main.empty());
            if (from_small) {
                const auto segment = set.small.front();
                set.small.pop_front();
                auto& state = present_.at(segment);
                if (state.counter > 0) {
                    state.counter = 0;
                    state.in_small = false;
                    set.main.push_back(segment);
                    continue;
                }
                retire(segment);
                remember_ghost(set, segment);
                return;
            }
            const auto segment = set.main.front();
            set.main.pop_front();
            auto& state = present_.at(segment);
            if (state.counter > 0) {
                --state.counter;
                set.main.push_back(segment);
                continue;
            }
            retire(segment);
            return;
        }
    }
    }
    throw std::invalid_argument("invalid device cache policy");
}

void CxlSsdDeviceCache::install(
    std::uint64_t segment,
    bool dirty,
    std::vector<std::uint64_t>& writeback_segments,
    std::vector<std::uint64_t>& evicted_segments) {
    auto& set = sets_[set_index_for(segment)];
    while (occupied_lines(set) >= lines_per_set_) {
        evict_one(set, writeback_segments, evicted_segments);
    }
    LineState state;
    state.dirty = dirty;
    if (config_.policy == DeviceCachePolicy::S3Fifo) {
        if (set.ghost_members.erase(segment) > 0) {
            // Ghost readmission installs directly into the protected queue.
            state.in_small = false;
            set.main.push_back(segment);
        } else {
            state.in_small = true;
            set.small.push_back(segment);
        }
    } else {
        set.queue.push_back(segment);
    }
    present_.emplace(segment, state);
}

DeviceCacheAccess CxlSsdDeviceCache::access(std::uint64_t segment, Op op) {
    DeviceCacheAccess result;
    const auto it = present_.find(segment);
    if (it != present_.end()) {
        result.hit = true;
        if (it->second.counter < 3) {
            ++it->second.counter;
        }
        if (op == Op::Write) {
            it->second.dirty = true;
        }
        return result;
    }
    install(
        segment,
        op == Op::Write,
        result.writeback_segments,
        result.evicted_segments);
    if (op == Op::Read && config_.prefetch_degree > 0) {
        for (std::uint32_t ahead = 1; ahead <= config_.prefetch_degree;
             ++ahead) {
            const auto stride = static_cast<std::uint64_t>(ahead) *
                config_.prefetch_stride;
            if (stride > device_segments_ ||
                segment >= device_segments_ - stride) {
                break;
            }
            const auto candidate = segment + stride;
            if (!present_.contains(candidate)) {
                result.prefetch_segments.push_back(candidate);
            }
        }
    }
    return result;
}

DeviceCacheAccess CxlSsdDeviceCache::install_prefetch(std::uint64_t segment) {
    DeviceCacheAccess result;
    if (!present_.contains(segment)) {
        install(
            segment,
            false,
            result.writeback_segments,
            result.evicted_segments);
    }
    return result;
}

} // namespace hbfsim::physical::external
