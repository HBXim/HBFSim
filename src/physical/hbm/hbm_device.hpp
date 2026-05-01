#pragma once

#include "physical/physical_types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace hbfsim::physical::hbm {

// Access commands the controller issues. Refresh is a bank-set reservation
// (REFab/REFsb) resolved by the scheduler, not an entry of this enum.
enum class HbmCommand {
    ACT,
    PRE,
    RD,
    WR,
};

inline constexpr std::size_t kHbmCommandCount = 4;

struct HbmConfig {
    // Versioned controller address map (see docs/reference/model.md). The
    // scheme id and the interleave size are recorded in every summary.
    static constexpr std::string_view address_mapping_scheme() {
        return "pch-interleave-bg-rotate-v2";
    }

    std::uint64_t capacity_bytes = 128ull * 1024ull * 1024ull * 1024ull;
    std::uint32_t stacks = 1;
    std::uint32_t channels_per_stack = 8;
    std::uint32_t pseudo_channels_per_channel = 2;
    std::uint32_t bank_groups_per_pseudo_channel = 4;
    std::uint32_t banks_per_group = 4;
    // Interface geometry and rate are the single source of truth for HBM
    // data movement. A channel is divided evenly into pseudo-channels. Each
    // pseudo-channel burst transfers burst_length beats over its share of the
    // DQ pins, so width/rate/BL derive both bytes and duration; callers cannot
    // configure an inconsistent aggregate GB/s or independent tBL.
    std::uint64_t channel_row_size_bytes = 2048;
    std::uint32_t channel_width_bits = 64;
    std::uint32_t burst_length = 8;
    double pin_rate_Gbps = 6.4;
    // HBM3/HBM4 use a command clock below the DQ transfer rate. The ratio is
    // explicit because tCCD is specified in command-clock cycles.
    std::uint32_t data_rate_per_command_clock = 4;
    // Contiguous bytes served by one pseudo-channel before the map rotates to
    // the next one. An explicit value must be a multiple of the derived burst
    // bytes and divide the per-pseudo-channel row bytes; 0 selects the
    // largest such value that does not exceed 256 B (256 B on every shipped
    // geometry). The resolved value is reported by effective_interleave_bytes()
    // and by HbmDevice::config().
    static constexpr std::uint64_t kDefaultInterleaveBytes = 256;
    std::uint64_t interleave_bytes = 0;
    double address_mapping_ns = 0.0;

    double tRCDRD_ns = 14.0;
    double tRCDWR_ns = 14.0;
    double tCL_ns = 14.0;
    double tCWL_ns = 10.0;
    double tRP_ns = 14.0;
    double tRAS_ns = 32.0;
    double tRC_ns = 46.0;
    double tWR_ns = 15.0;
    double tRTP_ns = 7.5;
    std::uint32_t tCCD_S_cycles = 2;
    std::uint32_t tCCD_L_cycles = 4;
    double tRRD_S_ns = 4.0;
    double tRRD_L_ns = 6.0;
    double tFAW_ns = 20.0;
    double tWTR_S_ns = 4.0;
    double tWTR_L_ns = 8.0;
    double tRTW_ns = 8.0;
    bool refresh_enabled = false;
    bool same_bank_refresh = false;
    double tREFI_ns = 3900.0;
    double tRFC_ns = 350.0;
    double tRFCsb_ns = 160.0;
    double tRREFD_ns = 10.0;
    // FR-FCFS scheduling: per-pseudo-channel pending-queue depth (a real
    // controller holds a few tens of requests per channel; a full queue
    // backpressures admission) and the anti-starvation time window. In
    // addition to this time gate, an entry may be bypassed at most queue_depth
    // times, so equal-timestamp row hits cannot starve an old conflict.
    std::uint32_t queue_depth = 32;
    double frfcfs_cap_ns = 5000.0;
    // Host-time fast path (config key hbm-replicate-symmetric-pseudo-channels):
    // when a stripe-aligned request presents the same burst sequence to
    // pseudo-channels in identical controller state, one representative is
    // simulated and its outcome is copied to the others. Results are
    // bit-identical to servicing every pseudo-channel individually (the
    // differential test runs both settings); only host time differs.
    bool replicate_symmetric_pseudo_channels = true;

    [[nodiscard]] std::uint64_t pseudo_channel_width_bits() const;
    [[nodiscard]] std::uint64_t row_size_bytes() const;
    [[nodiscard]] std::uint64_t burst_bytes() const;
    // The interleave the map actually uses: interleave_bytes when it is
    // explicit and legal (throws otherwise), else the auto default above.
    [[nodiscard]] std::uint64_t effective_interleave_bytes() const;
    [[nodiscard]] double channel_bandwidth_GBps() const;
    [[nodiscard]] double pseudo_channel_bandwidth_GBps() const;
    [[nodiscard]] double command_clock_period_ns() const;
    [[nodiscard]] double command_clock_MHz() const;
    [[nodiscard]] double burst_duration_ns() const;
    [[nodiscard]] double tCCD_S_ns() const;
    [[nodiscard]] double tCCD_L_ns() const;
    // Boundary conversion between caller nanoseconds and controller command
    // clocks. A time is rounded up to the next clock edge; a value within
    // 1/64 of a cycle below an edge is that edge, which absorbs the rounding
    // of ns<->cycle conversions (exact below 2^46 cycles) without letting a
    // genuinely later time issue early by more than 1/64 tCK.
    [[nodiscard]] std::uint64_t command_clock_cycles(double time_ns) const;
    [[nodiscard]] double command_clock_time_ns(std::uint64_t cycles) const;
};

struct HbmAddress {
    std::uint32_t stack = 0;
    std::uint32_t channel = 0;
    std::uint32_t pseudo_channel = 0;
    std::uint32_t bank_group = 0;
    std::uint32_t bank = 0;
    std::uint64_t row = 0;
    std::uint64_t offset = 0;

    [[nodiscard]] std::string path() const;
};

struct HbmStats {
    std::uint64_t read_bytes = 0;
    std::uint64_t write_bytes = 0;
    std::uint64_t row_hits = 0;
    std::uint64_t row_misses = 0;
    std::uint64_t row_conflicts = 0;
    std::uint64_t activations = 0;
    std::uint64_t precharges = 0;
    // Refresh commands issued (REFab: one per tREFI per pseudo-channel;
    // REFsb: banks_per_group per tREFI per pseudo-channel).
    std::uint64_t refresh_count = 0;
    // Exact: derived from an integer count of data-burst cycles.
    double bus_busy_ns = 0.0;
    double first_arrival_ns = std::numeric_limits<double>::infinity();
    double finish_ns = 0.0;
    std::uint64_t pseudo_channels = 0;
    std::uint64_t active_pseudo_channels = 0;
    std::uint64_t max_pseudo_channel_accesses = 0;
    std::uint64_t max_queue_occupancy = 0;
    double max_pseudo_channel_busy_ns = 0.0;
    double avg_active_pseudo_channel_busy_ns = 0.0;
    // Symmetric replication receipts: parent requests whose admission copied
    // one representative pseudo-channel to identical peers, and burst children
    // whose completion was produced by such a copy instead of being simulated.
    std::uint64_t replicated_requests = 0;
    std::uint64_t replicated_bursts = 0;
    // Sum of every physical burst child's stage work. Children can overlap
    // across pseudo-channels, so this is intentionally not wall time.
    Breakdown stage_work;

    [[nodiscard]] double row_hit_rate() const;
    [[nodiscard]] double active_span_ns() const;
    [[nodiscard]] double utilization() const;
    [[nodiscard]] double bus_parallelism() const;
    [[nodiscard]] double pseudo_channel_busy_skew() const;
};

class HbmDevice {
public:
    explicit HbmDevice(
        HbmConfig config,
        AddressHeatmap* address_heatmap = nullptr);

    [[nodiscard]] HbmAddress decode(std::uint64_t addr) const;
    [[nodiscard]] std::uint64_t encode(const HbmAddress& addr) const;
    // Synchronous request: enqueue + pump in one call.
    [[nodiscard]] PhysicalCompletion issue(const PhysicalRequest& request);
    // FR-FCFS front end: enqueue splits the logical range at DRAM burst
    // boundaries, hands each child to its mapped pseudo-channel queue (bounded
    // by hbm-queue-depth; a full queue services one entry before admitting the
    // next), and returns one parent ticket. pump services every routed
    // pseudo-channel one entry per sweep until the parent's last child has
    // finished, then returns the aggregate completion; drain_queues services
    // everything so remaining tickets become pump-able map lookups.
    [[nodiscard]] std::uint64_t enqueue(const PhysicalRequest& request);
    [[nodiscard]] PhysicalCompletion pump(std::uint64_t ticket);
    [[nodiscard]] bool service_before(double arrival_ns);
    [[nodiscard]] std::vector<std::pair<std::uint64_t, PhysicalCompletion>>
        take_completions();
    void drain_queues();
    [[nodiscard]] const HbmStats& stats() const {
        refresh_parallel_stats();
        return stats_;
    }
    // O(1) cumulative counters for persistent simulation-session batch deltas.
    // Geometry-wide parallelism fields are refreshed only by stats().
    [[nodiscard]] HbmStats execution_stats() const { return stats_; }
    [[nodiscard]] const HbmConfig& config() const { return config_; }
    void attach_address_heatmap(AddressHeatmap& address_heatmap) {
        if (last_enqueue_arrival_ns_) {
            throw std::runtime_error(
                "HBM address heatmap must be attached before the first request");
        }
        address_heatmap_ = &address_heatmap;
    }

private:
    // One gate per access command plus the refresh gate: the earliest cycle
    // at which a refresh may be issued to a bank that is already closed
    // (tRP after its PRE, or the end of its previous refresh window).
    static constexpr std::size_t kRefreshGate = kHbmCommandCount;
    using CommandGates = std::array<std::uint64_t, kHbmCommandCount + 1>;

    struct BankState {
        bool has_open_row = false;
        std::uint64_t open_row = 0;
        // Earliest command-clock cycle at which each command may issue.
        CommandGates ready{};
    };

    struct BankGroupState {
        CommandGates ready{};
    };

    struct QueuedRequest {
        std::uint64_t ticket = 0;
        Op op = Op::Read;
        double arrival_ns = 0.0;
        // First command-clock cycle at which a command may issue
        // (arrival plus address mapping, rounded to the clock grid).
        std::uint64_t ready_cycle = 0;
        std::uint64_t bytes = 0;
        HbmAddress addr;
        TraceConfig trace;
        std::uint32_t bypass_count = 0;
    };

    struct PendingRequest {
        PhysicalCompletion completion;
        std::size_t remaining_children = 0;
        std::size_t total_children = 0;
        std::size_t pseudo_channels = 0;
        // Pseudo-channel of the child currently on the critical path; ties on
        // finish time resolve to the lowest index so aggregation order never
        // changes the reported breakdown.
        std::size_t critical_pseudo_channel = 0;
        bool enqueue_complete = false;
        bool has_child_completion = false;
        bool retain_diagnostics = true;
    };

    struct PlacedColumn {
        std::uint64_t cycle = 0;
        bool write = false;
    };

    struct PseudoChannelState {
        std::vector<BankState> banks;
        std::vector<BankGroupState> bank_groups;
        std::uint32_t open_rows = 0;
        std::uint64_t service_floor_cycle = 0;
        // Pseudo-channel-level placement history, each sorted by cycle. A new
        // command is placed at the earliest cycle that keeps tCCD_S and the
        // RD/WR turnarounds against every placed column command on both
        // sides, tRRD_S and the tFAW four-ACT window against every placed
        // ACT on both sides, and a free data-bus slot; commands of other
        // banks may therefore backfill gaps left by a bank's PRE/ACT/tRCD.
        // Entries that can no longer constrain a future command are pruned.
        std::vector<PlacedColumn> columns;
        std::vector<std::uint64_t> activations;
        std::vector<std::uint64_t> bus_slots;
        // Position in the refresh schedule: the next unresolved refresh
        // command is command `refresh_command` of period `refresh_period`.
        std::uint64_t refresh_period = 0;
        std::uint32_t refresh_command = 0;
        bool has_refresh_issue = false;
        std::uint64_t last_refresh_issue = 0;
        // FR-FCFS pending queue, kept in arrival order.
        std::vector<QueuedRequest> queue;
    };

    struct LocalAddress {
        std::uint32_t bank_group = 0;
        std::uint32_t bank = 0;
        std::uint64_t row = 0;
        std::uint64_t column_unit = 0;
    };

    struct FirstCommand {
        std::uint64_t issue_cycle = 0;
        bool row_hit = false;
    };

    struct RefreshOutcome {
        // A resolved refresh closed the accessed bank's row; its window ends
        // at blocked_until.
        bool closed_bank = false;
        std::uint64_t blocked_until = 0;
    };

    // Per-ticket outcome of a representative pseudo-channel, replayed onto
    // its symmetric peers.
    struct ChildRecord {
        std::uint64_t ticket = 0;
        std::uint64_t count = 0;
        std::uint64_t physical_bytes = 0;
        double min_start_ns = std::numeric_limits<double>::infinity();
        double max_finish_ns = -std::numeric_limits<double>::infinity();
        Breakdown critical;
        Breakdown work;
    };

    // Counter deltas produced by a representative while it was recorded.
    struct CounterDelta {
        std::uint64_t read_bytes = 0;
        std::uint64_t write_bytes = 0;
        std::uint64_t row_hits = 0;
        std::uint64_t row_misses = 0;
        std::uint64_t row_conflicts = 0;
        std::uint64_t activations = 0;
        std::uint64_t precharges = 0;
        std::uint64_t refresh_count = 0;
        std::uint64_t bus_busy_cycles = 0;
        std::uint64_t accesses = 0;
        std::uint64_t busy_cycles = 0;
    };

    struct Recording {
        bool active = false;
        std::size_t representative = 0;
        std::uint64_t push_ticket = 0;
        std::uint64_t pushed = 0;
        // Deltas folded so far plus the counter snapshot taken when the
        // recording was last (re)started; a paused recording folds first so
        // work done for other classes in between is never attributed to it.
        CounterDelta delta;
        CounterDelta snapshot;
        std::vector<ChildRecord> records;
    };

    struct SymmetryClass {
        std::size_t representative = 0;
        std::vector<std::size_t> members;
    };

    HbmConfig config_;
    AddressHeatmap* address_heatmap_ = nullptr;
    // Immutable geometry derived and validated once at construction.
    std::uint64_t row_size_bytes_ = 0;
    std::uint64_t burst_bytes_ = 0;
    std::uint64_t bursts_per_unit_ = 0;
    std::uint64_t units_per_row_ = 0;
    std::uint64_t stripe_bytes_ = 0;
    std::uint64_t total_pseudo_channels_ = 0;
    std::uint64_t banks_per_pseudo_channel_ = 0;
    std::uint64_t pseudo_channels_per_stack_ = 0;
    double command_clock_period_ns_ = 0.0;
    // Every controller timing in command-clock cycles.
    std::uint64_t burst_cycles_ = 0;
    std::uint64_t tccd_s_ = 0;
    std::uint64_t tccd_l_ = 0;
    std::uint64_t trcdrd_ = 0;
    std::uint64_t trcdwr_ = 0;
    std::uint64_t tcl_ = 0;
    std::uint64_t tcwl_ = 0;
    std::uint64_t trp_ = 0;
    std::uint64_t tras_ = 0;
    std::uint64_t trc_ = 0;
    std::uint64_t twr_ = 0;
    std::uint64_t trtp_ = 0;
    std::uint64_t trrd_s_ = 0;
    std::uint64_t trrd_l_ = 0;
    std::uint64_t tfaw_ = 0;
    std::uint64_t twtr_s_ = 0;
    std::uint64_t twtr_l_ = 0;
    std::uint64_t trtw_ = 0;
    std::uint64_t trefi_ = 0;
    std::uint64_t trfc_ = 0;
    std::uint64_t trfcsb_ = 0;
    std::uint64_t trrefd_ = 0;
    std::uint32_t refresh_commands_per_period_ = 1;
    std::vector<PseudoChannelState> pseudo_channels_;
    // Cumulative accounting is deliberately separate from controller state:
    // it never affects scheduling.
    std::vector<std::uint64_t> pseudo_channel_accesses_;
    std::vector<std::uint64_t> pseudo_channel_bus_busy_cycles_;
    std::uint64_t bus_busy_cycles_ = 0;
    mutable HbmStats stats_;
    std::uint64_t next_ticket_ = 0;
    // Per-parent visitation marker used to build its unique pseudo-channel
    // route list in O(children).
    std::vector<std::uint64_t> route_ticket_markers_;
    std::unordered_map<std::uint64_t, std::vector<std::size_t>> ticket_pseudo_channels_;
    std::unordered_map<std::uint64_t, PendingRequest> pending_;
    std::unordered_map<std::uint64_t, PhysicalCompletion> completed_;
    std::optional<double> last_enqueue_arrival_ns_;
    // Command floor of the most recent enqueue: no later request can issue
    // before it, so expired timing state can be canonicalized against it.
    std::uint64_t floor_bound_cycle_ = 0;
    Recording recording_;

    [[nodiscard]] std::size_t pseudo_channel_index(const HbmAddress& addr) const;
    [[nodiscard]] std::size_t bank_index(const HbmAddress& addr) const;
    [[nodiscard]] LocalAddress local_address(std::uint64_t unit) const;
    void assign_pseudo_channel(HbmAddress& addr, std::size_t pseudo_channel) const;
    [[nodiscard]] std::uint64_t byte_address(
        std::size_t pseudo_channel,
        std::uint64_t unit,
        std::uint64_t unit_offset) const;
    [[nodiscard]] double ns(std::uint64_t cycles) const;
    void validate_and_begin_request(const PhysicalRequest& request);
    void create_parent(std::uint64_t ticket, const PhysicalRequest& request);
    void push_child(
        std::size_t pseudo_channel_index,
        std::uint64_t ticket,
        const PhysicalRequest& request,
        std::uint64_t ready_cycle,
        std::uint64_t bytes,
        HbmAddress addr);
    void enqueue_children(
        std::uint64_t ticket,
        const PhysicalRequest& request,
        std::uint64_t ready_cycle);
    void enqueue_replicated(
        std::uint64_t ticket,
        const PhysicalRequest& request,
        std::uint64_t ready_cycle);
    void service_until_complete(std::uint64_t ticket);
    void aggregate_child(
        std::uint64_t ticket,
        std::size_t pseudo_channel_index,
        PhysicalCompletion child);
    void complete_parent(std::uint64_t ticket, PendingRequest& parent);
    [[nodiscard]] std::size_t pick_next(const PseudoChannelState& pseudo_channel) const;
    [[nodiscard]] FirstCommand first_command(
        const PseudoChannelState& pseudo_channel,
        const QueuedRequest& queued) const;
    void service_one(std::size_t pseudo_channel_index);
    [[nodiscard]] PhysicalCompletion service_request(
        PseudoChannelState& pseudo_channel,
        std::size_t pseudo_channel_index,
        const QueuedRequest& queued);
    // Earliest legal cycle >= lower for a new ACT / column command given the
    // pseudo-channel's placement history (pure; used by previews too).
    [[nodiscard]] std::uint64_t place_activation(
        const PseudoChannelState& pseudo_channel,
        std::uint64_t lower) const;
    [[nodiscard]] std::uint64_t place_column(
        const PseudoChannelState& pseudo_channel,
        bool write,
        std::uint64_t lower) const;
    void prune_placements(PseudoChannelState& pseudo_channel, std::uint64_t floor) const;
    [[nodiscard]] std::uint64_t column_reach() const;
    [[nodiscard]] std::uint64_t refresh_nominal_cycle(
        std::uint64_t period,
        std::uint32_t command) const;
    [[nodiscard]] bool refresh_targets_bank(
        std::uint32_t command,
        std::size_t bank_index) const;
    void resolve_next_refresh(
        PseudoChannelState& pseudo_channel,
        std::size_t pseudo_channel_index,
        std::vector<TraceSpan>* spans);
    // Issue every refresh nominally due at or before issue_cycle (the first
    // command of an access) and report whether the accessed bank was closed.
    [[nodiscard]] RefreshOutcome apply_due_refreshes(
        PseudoChannelState& pseudo_channel,
        std::size_t pseudo_channel_index,
        std::size_t bank_index,
        std::uint64_t issue_cycle,
        std::vector<TraceSpan>* spans);
    void apply_command_state(
        PseudoChannelState& pseudo_channel,
        const HbmAddress& addr,
        HbmCommand command,
        std::uint64_t issue_cycle);
    // Symmetric replication.
    [[nodiscard]] bool replicable(const PhysicalRequest& request) const;
    [[nodiscard]] std::uint64_t normalization_floor(
        const PseudoChannelState& pseudo_channel) const;
    // A timing gate at or below the floor can no longer delay any command,
    // so it compares equal to an idle gate; anything later must match
    // exactly. Both the class hash and the exact comparison use this.
    [[nodiscard]] static std::uint64_t normalized_gate(
        std::uint64_t gate,
        std::uint64_t floor);
    [[nodiscard]] std::uint64_t symmetry_hash(std::size_t pseudo_channel_index) const;
    [[nodiscard]] bool symmetric(std::size_t lhs, std::size_t rhs) const;
    [[nodiscard]] std::vector<SymmetryClass> partition_symmetric(
        const std::vector<std::size_t>& candidates) const;
    [[nodiscard]] CounterDelta counter_snapshot(std::size_t representative) const;
    void begin_recording(std::size_t representative, std::uint64_t push_ticket);
    void fold_recording();
    [[nodiscard]] Recording pause_recording();
    void resume_recording(Recording recording);
    void replicate_recording(const SymmetryClass& symmetry_class);
    void refresh_parallel_stats() const;
};

} // namespace hbfsim::physical::hbm
