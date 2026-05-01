#pragma once

#include "physical/address_heatmap.hpp"
#include "physical/external/cxl_ssd.hpp"
#include "physical/physical_types.hpp"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <optional>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

namespace hbfsim::physical::external {

// Non-HBM backing tiers share one explicit request/controller/media/response
// contract. The full-duplex transport represents either an on-package D2D
// path or a host attachment as selected by the profile. Profiles declare
// parameter envelopes; they do not select a different scheduling
// implementation.
enum class ExternalBackingKind {
    OnPackageLpddr,
    HostDram,
    CxlMemory,
    NvmeSsd,
    CxlSsd,
};

[[nodiscard]] const char* to_string(ExternalBackingKind kind);
[[nodiscard]] ExternalBackingKind parse_external_backing_kind(
    const std::string& value);

struct ExternalBackingConfig {
    ExternalBackingKind kind = ExternalBackingKind::CxlMemory;
    std::uint64_t capacity_bytes = 256ull << 30; // 256 GiB
    // Host-visible address/accounting page.  It is independent of the
    // transport command boundary below.
    std::uint64_t page_size_bytes = 4096;
    // Maximum transport command payload.  This is deliberately independent
    // of the host-visible address page: one application range may cover many
    // transport commands, and each command may cover many address pages.
    std::uint64_t request_segment_bytes = 4096;

    // Each caller-visible range is assigned round-robin to one directional
    // media queue.  Transport segments within that range remain pinned to the
    // selected queue and are address-striped across its channels. Read and
    // write queues with the same queue/channel index share one media timeline.
    std::uint32_t media_read_queues = 1;
    std::uint32_t media_write_queues = 1;
    // Channels per directional queue. Configured directional media bandwidth
    // is aggregate across all queues and channels for that direction.
    std::uint32_t media_channels = 8;
    // End-to-end device credits from admission through the returned response.
    std::uint32_t max_outstanding_requests = 512;

    // One pipelined controller issue resource followed by non-occupying
    // processing latency.
    double controller_issue_ns = 2.0;
    double controller_processing_ns = 20.0;

    double media_read_latency_ns = 90.0;
    double media_write_latency_ns = 90.0;
    double media_read_bandwidth_GBps = 204.8;
    double media_write_bandwidth_GBps = 204.8;

    // Full-duplex host transport. M2S means requester-to-device; S2M means
    // device-to-requester. Bandwidth is wire bandwidth. Propagation is paid
    // once after each directional transfer and is not resource occupancy.
    double m2s_bandwidth_GBps = 36.0;
    double s2m_bandwidth_GBps = 36.0;
    double one_way_propagation_ns = 75.0;

    // Per-request protocol bytes. A read sends only a command and returns
    // payload+completion. A write sends command+payload and returns completion.
    std::uint32_t command_bytes = 64;
    std::uint32_t completion_bytes = 16;

    // Optional CXL-SSD device-side DRAM cache in front of the media stage
    // (see cxl_ssd.hpp). Selected only by this explicit block, never by the
    // kind string; disabled (the default) leaves the shared schedule
    // untouched, and enabling it is valid only for kind cxl-ssd.
    DeviceCacheConfig device_cache;
};

// Reproducible, source-anchored sensitivity envelopes. Fields not explicitly
// supported by a public source remain exploratory assumptions in the
// parameter-provenance registry.
[[nodiscard]] ExternalBackingConfig cxl_memory_profile();
[[nodiscard]] ExternalBackingConfig cxl_ssd_profile();
[[nodiscard]] ExternalBackingConfig host_dram_profile();
[[nodiscard]] ExternalBackingConfig nvme_ssd_profile();
[[nodiscard]] ExternalBackingConfig on_package_lpddr_profile();

struct ExternalBackingStats {
    ExternalBackingKind kind = ExternalBackingKind::CxlMemory;
    std::uint64_t read_requests = 0;
    std::uint64_t write_requests = 0;
    std::uint64_t read_bytes = 0;
    std::uint64_t write_bytes = 0;
    // Three independent request layers: caller-visible ranges, transport
    // commands, and address pages covered by those ranges.
    std::uint64_t page_run_requests = 0;
    std::uint64_t page_run_segments = 0;
    std::uint64_t page_run_pages = 0;

    // Actual shared media timelines. The remaining fields make the resolved
    // directional queue geometry explicit.
    std::uint64_t media_channels = 0;
    std::uint64_t media_channels_per_queue = 0;
    std::uint64_t media_read_queues = 0;
    std::uint64_t media_write_queues = 0;
    std::uint64_t active_media_resources = 0;
    std::uint64_t max_device_outstanding = 0;
    double outstanding_wait_ns = 0.0;

    double controller_queue_wait_ns = 0.0;
    double controller_issue_busy_ns = 0.0;
    double controller_processing_work_ns = 0.0;

    double media_queue_wait_ns = 0.0;
    double media_read_latency_work_ns = 0.0;
    double media_write_latency_work_ns = 0.0;
    double media_read_busy_ns = 0.0;
    double media_write_busy_ns = 0.0;

    std::uint64_t m2s_payload_bytes = 0;
    std::uint64_t m2s_protocol_bytes = 0;
    std::uint64_t m2s_wire_bytes = 0;
    std::uint64_t s2m_payload_bytes = 0;
    std::uint64_t s2m_protocol_bytes = 0;
    std::uint64_t s2m_wire_bytes = 0;
    double m2s_queue_wait_ns = 0.0;
    double s2m_queue_wait_ns = 0.0;
    double m2s_busy_ns = 0.0;
    double s2m_busy_ns = 0.0;
    double transport_propagation_work_ns = 0.0;

    // CXL-SSD device-cache census. All zero while the cache is disabled.
    // Writeback/prefetch traffic is device-internal media work: it occupies
    // media timelines but never appears as caller requests or wire bytes.
    std::uint64_t cache_read_hits = 0;
    std::uint64_t cache_read_misses = 0;
    std::uint64_t cache_write_hits = 0;
    std::uint64_t cache_write_misses = 0;
    // A sub-page write miss must fetch the untouched bytes before the dirty
    // cache line can later be programmed as a complete flash page.
    std::uint64_t cache_read_for_ownership_segments = 0;
    std::uint64_t cache_read_for_ownership_bytes = 0;
    std::uint64_t cache_writeback_segments = 0;
    std::uint64_t cache_writeback_bytes = 0;
    std::uint64_t cache_prefetch_segments = 0;
    std::uint64_t cache_prefetch_bytes = 0;
    double cache_latency_work_ns = 0.0;
    double cache_busy_ns = 0.0;
    double cache_queue_wait_ns = 0.0;
    double cache_flush_busy_ns = 0.0;
    double cache_prefetch_busy_ns = 0.0;
    double cache_writeback_gate_wait_ns = 0.0;

    double first_arrival_ns = std::numeric_limits<double>::infinity();
    double finish_ns = 0.0;
    double controller_first_busy_ns =
        std::numeric_limits<double>::infinity();
    double controller_last_busy_ns = 0.0;
    double media_first_busy_ns = std::numeric_limits<double>::infinity();
    double media_last_busy_ns = 0.0;
    double m2s_first_busy_ns = std::numeric_limits<double>::infinity();
    double m2s_last_busy_ns = 0.0;
    double s2m_first_busy_ns = std::numeric_limits<double>::infinity();
    double s2m_last_busy_ns = 0.0;
    Breakdown stage_work;

    [[nodiscard]] double active_span_ns() const;
    [[nodiscard]] double controller_active_span_ns() const;
    [[nodiscard]] double media_active_span_ns() const;
    [[nodiscard]] double m2s_active_span_ns() const;
    [[nodiscard]] double s2m_active_span_ns() const;
    [[nodiscard]] double controller_utilization() const;
    [[nodiscard]] double media_utilization() const;
    [[nodiscard]] double m2s_utilization() const;
    [[nodiscard]] double s2m_utilization() const;
    [[nodiscard]] bool operator==(
        const ExternalBackingStats&) const = default;
};

// Cache data dependencies that still lie beyond the device's causal
// admission frontier. This is intentionally separate from architectural
// cache residency: completed fills remain resident but need no timestamp.
struct ExternalBackingCacheDependencyState {
    std::size_t pending_fill_segments = 0;
    std::size_t pending_flush_segments = 0;
};

class ExternalBackingDevice {
public:
    explicit ExternalBackingDevice(
        ExternalBackingConfig config,
        AddressHeatmap* address_heatmap = nullptr);

    [[nodiscard]] PhysicalCompletion issue(const PhysicalRequest& request);
    // Submit one caller-visible contiguous range, split only at configured
    // transport-command boundaries.  Address pages remain accounting units;
    // they are not silently promoted to commands.  The returned completion
    // joins all transport segments in the range.
    [[nodiscard]] PhysicalCompletion issue_contiguous_range(
        const PhysicalRequest& request);

    void attach_address_heatmap(AddressHeatmap& address_heatmap) {
        address_heatmap_ = &address_heatmap;
    }

    [[nodiscard]] const ExternalBackingConfig& config() const {
        return config_;
    }
    [[nodiscard]] const ExternalBackingStats& stats() const {
        return stats_;
    }
    [[nodiscard]] ExternalBackingCacheDependencyState
    cache_dependency_state() const {
        return {
            .pending_fill_segments = pending_fill_ready_ns_.size(),
            .pending_flush_segments = pending_flush_ready_ns_.size(),
        };
    }

private:
    struct Reservation {
        double start_ns = 0.0;
        double finish_ns = 0.0;
        double queue_wait_ns = 0.0;
        double busy_ns = 0.0;
    };

    // Future reservations are kept as disjoint intervals. reserve() chooses
    // the earliest legal gap at or after ready_ns, so a later API call whose
    // upstream work finishes earlier can backfill before an already-reserved
    // future interval.
    class SerialResourceTimeline {
    public:
        [[nodiscard]] Reservation reserve(
            double ready_ns,
            double busy_ns);
        // No future request can reach a resource before its nondecreasing
        // device admission time.  Retaining intervals that have already
        // finished by that frontier changes no earliest-gap decision and made
        // long page streams consume memory proportional to trace bytes.
        void discard_finished_at_or_before(double frontier_ns);

    private:
        // Intervals remain sorted and disjoint.  A deque makes the common
        // monotone append + expired-prefix removal path allocation-free while
        // retaining exact middle insertion for genuine backfill.
        std::deque<std::pair<double, double>> intervals_;
    };

    struct MediaResourceState {
        SerialResourceTimeline media;
        bool active = false;
    };

    struct RequestTiming {
        double admitted_ns = 0.0;
        Reservation m2s;
        double m2s_arrival_ns = 0.0;
        Reservation controller;
        double controller_finish_ns = 0.0;
        double media_latency_finish_ns = 0.0;
        Reservation media;
        Reservation s2m;
        double finish_ns = 0.0;
        std::uint64_t m2s_payload_bytes = 0;
        std::uint64_t m2s_protocol_bytes = 0;
        std::uint64_t m2s_wire_bytes = 0;
        std::uint64_t s2m_payload_bytes = 0;
        std::uint64_t s2m_protocol_bytes = 0;
        std::uint64_t s2m_wire_bytes = 0;
        // Device-cache stage outcome: a cache-served segment replaces the
        // flash media stage with the buffer-DRAM port, and an install that
        // displaced dirty victims first waits for their flush reservations.
        bool served_from_cache = false;
        bool cache_read_for_ownership = false;
        Reservation cache_read_for_ownership_media;
        double media_latency_applied_ns = 0.0;
        double writeback_gate_wait_ns = 0.0;
    };

    [[nodiscard]] std::size_t select_media_queue(Op op);
    [[nodiscard]] std::size_t striped_channel_for(
        std::uint64_t addr) const;
    [[nodiscard]] std::size_t media_resource_for(
        std::size_t queue_index,
        std::size_t striped_channel_index) const;
    [[nodiscard]] std::string media_resource_path(
        std::size_t queue_index,
        std::size_t striped_channel_index) const;
    [[nodiscard]] double admit(double arrival_ns);
    [[nodiscard]] RequestTiming schedule(
        const PhysicalRequest& request,
        MediaResourceState& media_resource);
    // Internal flush of dirty victims through the flash write path; returns
    // the latest flush finish so the evicting access can wait on it.
    [[nodiscard]] double schedule_cache_writebacks(
        const std::vector<std::uint64_t>& writeback_segments,
        double ready_ns);
    // Background next-N fill through the flash read path; never gates the
    // caller.
    void schedule_cache_prefetch_fill(
        std::uint64_t prefetch_segment,
        double ready_ns);
    void retain_pending_fill(
        std::uint64_t segment,
        double ready_ns);
    void discard_completed_cache_events(double causal_frontier_ns);
    void validate_request(const PhysicalRequest& request) const;
    void validate_range_request(const PhysicalRequest& request) const;
    [[nodiscard]] Breakdown account_timing(
        const PhysicalRequest& request,
        std::size_t media_resource_index,
        const RequestTiming& timing);
    void account(
        const PhysicalRequest& request,
        std::size_t media_resource_index,
        const RequestTiming& timing,
        PhysicalCompletion& completion);

    ExternalBackingConfig config_;
    ExternalBackingStats stats_;
    std::optional<CxlSsdDeviceCache> device_cache_;
    SerialResourceTimeline device_cache_port_;
    // Flash-side state per line that a later access must not run ahead of:
    // the flush that made a dirty victim durable. A miss that re-reads or
    // re-installs the same line waits for it, so earliest-gap placement can
    // never read stale flash contents or reorder two flushes of one line.
    std::unordered_map<std::uint64_t, double> pending_flush_ready_ns_;
    // Expiry index for the map above. Stale heap records are harmless when a
    // segment is reinstalled; the map's current timestamp is authoritative.
    using PendingCacheEvent = std::pair<double, std::uint64_t>;
    std::priority_queue<
        PendingCacheEvent,
        std::vector<PendingCacheEvent>,
        std::greater<PendingCacheEvent>> pending_flush_expiries_;
    // Background prefetch fills in flight: a cache-served access to one of
    // these segments waits for the fill and then clears the entry.
    std::unordered_map<std::uint64_t, double> pending_fill_ready_ns_;
    std::priority_queue<
        PendingCacheEvent,
        std::vector<PendingCacheEvent>,
        std::greater<PendingCacheEvent>> pending_fill_expiries_;
    std::vector<MediaResourceState> media_resources_;
    std::priority_queue<
        double,
        std::vector<double>,
        std::greater<double>> inflight_finishes_;
    SerialResourceTimeline controller_;
    SerialResourceTimeline m2s_;
    SerialResourceTimeline s2m_;
    AddressHeatmap* address_heatmap_ = nullptr;
    std::uint32_t next_read_queue_ = 0;
    std::uint32_t next_write_queue_ = 0;
    double last_issue_arrival_ns_ = 0.0;
    double admission_frontier_ns_ = 0.0;
};

} // namespace hbfsim::physical::external
