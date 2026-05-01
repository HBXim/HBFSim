#pragma once

#include "physical/physical_types.hpp"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace hbfsim::physical::external {

// CXL-SSD device module: everything specific to the CXL-attached-flash
// device class prototyped by Cylon (FAST '26) lives here, separate from the
// shared external-backing transport/controller/media pipeline.
//
// The distinctive mechanism is the device-side DRAM cache in front of the
// flash media stage: a set-associative, segment-granular directory with
// FIFO/LIFO/CLOCK/S3-FIFO eviction (the Cylon artifact's policy set),
// optional next-N prefetch, and write-back dirty flushing. This module owns
// the directory and every replacement decision; the shared pipeline owns all
// timing. The cache is disabled by default, and a disabled cache leaves the
// shared request/controller/media/response schedule bit-identical, so the
// registered cxl-ssd miss-path profile and every other backing kind are
// unaffected.
enum class DeviceCachePolicy {
    Fifo,
    Lifo,
    Clock,
    S3Fifo,
};

[[nodiscard]] const char* to_string(DeviceCachePolicy policy);
[[nodiscard]] DeviceCachePolicy parse_device_cache_policy(
    const std::string& value);

struct DeviceCacheConfig {
    bool enabled = false;
    std::uint64_t capacity_bytes = 0;
    // 0 selects one fully associative set; otherwise the way count of a
    // set-associative directory indexed by segment number (Cylon buffer_way).
    std::uint32_t ways = 0;
    DeviceCachePolicy policy = DeviceCachePolicy::Fifo;
    // Next-N prefetch issued on a demand read miss (Cylon prf_dg); stride is
    // in segments.
    std::uint32_t prefetch_degree = 0;
    std::uint32_t prefetch_stride = 1;
    // Buffer DRAM access latency and port bandwidth charged to cache-served
    // segments in place of the flash media stage.
    double hit_latency_ns = 90.0;
    double hit_bandwidth_GBps = 204.8;
};

struct DeviceCacheAccess {
    bool hit = false;
    // Dirty victims this access displaced; each is one whole line that
    // flushes through the flash write path.
    std::vector<std::uint64_t> writeback_segments;
    // Every victim this access displaced, dirty or clean, so the device can
    // drop any fill state it still tracks for them.
    std::vector<std::uint64_t> evicted_segments;
    // Absent next-N segments the prefetcher fetches after a demand read
    // miss.
    std::vector<std::uint64_t> prefetch_segments;
};

// Deterministic replacement directory. Line granularity is one transport
// segment; policy semantics mirror the Cylon artifact's bbssd buffer
// (fifo.c/lifo.c/clock.c/s3fifo.c): CLOCK gives one second chance on a set
// reference bit, and S3-FIFO runs per-set small/main FIFO queues with a
// ghost history whose readmissions install directly into main.
class CxlSsdDeviceCache {
public:
    CxlSsdDeviceCache(
        const DeviceCacheConfig& config,
        std::uint64_t segment_bytes,
        std::uint64_t device_capacity_bytes);

    // One demand access at segment granularity. Installs on a miss
    // (write-back write-allocate: a write miss installs dirty and never
    // touches flash for the caller's payload).
    [[nodiscard]] DeviceCacheAccess access(std::uint64_t segment, Op op);

    // Install one prefetched segment (clean); reports the victims it
    // displaced. Skips segments that are already present.
    [[nodiscard]] DeviceCacheAccess install_prefetch(std::uint64_t segment);

    [[nodiscard]] std::uint64_t total_lines() const { return total_lines_; }
    [[nodiscard]] std::uint64_t set_count() const { return set_count_; }

private:
    struct LineState {
        bool dirty = false;
        // CLOCK reference bit / S3-FIFO access frequency (saturating).
        std::uint8_t counter = 0;
        // S3-FIFO: whether the line currently sits in the small queue.
        bool in_small = false;
    };
    struct Set {
        // fifo/lifo/clock insertion order (front is the oldest line).
        std::deque<std::uint64_t> queue;
        // s3fifo probationary/protected queues plus the ghost id history.
        std::deque<std::uint64_t> small;
        std::deque<std::uint64_t> main;
        std::deque<std::uint64_t> ghost;
        std::unordered_set<std::uint64_t> ghost_members;
    };

    [[nodiscard]] std::uint64_t set_index_for(std::uint64_t segment) const;
    [[nodiscard]] std::size_t occupied_lines(const Set& set) const;
    void install(
        std::uint64_t segment,
        bool dirty,
        std::vector<std::uint64_t>& writeback_segments,
        std::vector<std::uint64_t>& evicted_segments);
    void evict_one(
        Set& set,
        std::vector<std::uint64_t>& writeback_segments,
        std::vector<std::uint64_t>& evicted_segments);
    void remember_ghost(Set& set, std::uint64_t segment);

    DeviceCacheConfig config_;
    std::uint64_t segment_bytes_ = 0;
    std::uint64_t total_lines_ = 0;
    std::uint64_t set_count_ = 0;
    std::uint64_t lines_per_set_ = 0;
    std::uint64_t small_target_lines_ = 0;
    std::uint64_t device_segments_ = 0;
    // Point lookups only; every ordered decision walks the per-set deques,
    // so behavior is independent of hash iteration order.
    std::unordered_map<std::uint64_t, LineState> present_;
    std::unordered_map<std::uint64_t, Set> sets_;
};

} // namespace hbfsim::physical::external
