#include "physical/simulation_session.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <unordered_map>
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

std::uint64_t checked_mul(
    std::uint64_t lhs,
    std::uint64_t rhs,
    const char* description) {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
        throw std::runtime_error(
            std::string(description) + " overflows uint64_t");
    }
    return lhs * rhs;
}

std::size_t target_index(SimulationTarget target) {
    return static_cast<std::size_t>(target);
}

std::size_t operation_index(Op op) {
    if (op == Op::Read) return 0;
    if (op == Op::Write) return 1;
    throw std::runtime_error("simulation completion operation is not read/write");
}

void record_completion_stats(
    SimulationCompletionStats& stats,
    const PhysicalCompletion& completion) {
    if (!std::isfinite(completion.arrival_ns) ||
        !std::isfinite(completion.start_ns) ||
        !std::isfinite(completion.finish_ns) ||
        completion.start_ns < completion.arrival_ns ||
        completion.finish_ns < completion.start_ns) {
        throw std::runtime_error(
            "simulation completion has an invalid transaction latency");
    }
    stats.transactions = checked_add(
        stats.transactions, 1, "simulation completion transaction count");
    stats.logical_bytes = checked_add(
        stats.logical_bytes,
        completion.logical_bytes,
        "simulation completion logical bytes");
    stats.physical_bytes = checked_add(
        stats.physical_bytes,
        completion.physical_bytes,
        "simulation completion physical bytes");
    const auto queue_wait_ns = completion.start_ns - completion.arrival_ns;
    const auto service_ns = completion.finish_ns - completion.start_ns;
    const auto latency_ns = completion.finish_ns - completion.arrival_ns;
    stats.queue_wait_work_ns += queue_wait_ns;
    stats.service_work_ns += service_ns;
    stats.latency_work_ns += latency_ns;
    stats.min_latency_ns = std::min(stats.min_latency_ns, latency_ns);
    stats.max_latency_ns = std::max(stats.max_latency_ns, latency_ns);
    stats.first_arrival_ns = std::min(
        stats.first_arrival_ns, completion.arrival_ns);
    stats.finish_ns = std::max(stats.finish_ns, completion.finish_ns);
}

BaseDieLinkStats aggregate_base_die_links(
    const std::vector<BaseDieLink>& links) {
    BaseDieLinkStats total;
    total.links = links.size();
    for (const auto& link : links) {
        const auto& stats = link.stats();
        total.read_transfers = checked_add(
            total.read_transfers,
            stats.read_transfers,
            "simulation D2D read transfers");
        total.write_transfers = checked_add(
            total.write_transfers,
            stats.write_transfers,
            "simulation D2D write transfers");
        total.read_bytes = checked_add(
            total.read_bytes, stats.read_bytes, "simulation D2D read bytes");
        total.write_bytes = checked_add(
            total.write_bytes, stats.write_bytes, "simulation D2D write bytes");
        total.read_queue_wait_ns += stats.read_queue_wait_ns;
        total.write_queue_wait_ns += stats.write_queue_wait_ns;
        total.read_busy_ns += stats.read_busy_ns;
        total.write_busy_ns += stats.write_busy_ns;
        total.read_fixed_latency_work_ns +=
            stats.read_fixed_latency_work_ns;
        total.write_fixed_latency_work_ns +=
            stats.write_fixed_latency_work_ns;
        total.first_arrival_ns = std::min(
            total.first_arrival_ns, stats.first_arrival_ns);
        total.finish_ns = std::max(total.finish_ns, stats.finish_ns);
    }
    return total;
}

} // namespace

const char* to_string(SimulationTarget target) {
    switch (target) {
    case SimulationTarget::Hbm:
        return "HBM";
    case SimulationTarget::HbfLogical:
        return "HBF_LOGICAL";
    case SimulationTarget::HbfStatic:
        return "HBF_STATIC";
    case SimulationTarget::HbfPhysical:
        return "HBF_PHYSICAL";
    case SimulationTarget::D2dHbfToHbm:
        return "D2D_HBF_TO_HBM";
    case SimulationTarget::D2dHbmToHbf:
        return "D2D_HBM_TO_HBF";
    case SimulationTarget::External:
        return "EXTERNAL";
    case SimulationTarget::DirectHbfToExternal:
        return "DIRECT_HBF_TO_EXTERNAL";
    case SimulationTarget::DirectExternalToHbf:
        return "DIRECT_EXTERNAL_TO_HBF";
    case SimulationTarget::Barrier:
        return "BARRIER";
    }
    throw std::runtime_error("unknown simulation transaction target");
}

SimulationSession::SimulationSession(SimulationSessionConfig config)
    : config_(std::move(config)) {
    if (!config_.enable_hbm && !config_.enable_hbf &&
        !config_.enable_external) {
        throw std::runtime_error(
            "simulation session must enable at least one memory tier");
    }
    if (!config_.enable_hbf && config_.static_hbf_blocks_per_plane != 0) {
        throw std::runtime_error(
            "session static HBF placement requires the HBF tier");
    }
    if (!config_.enable_hbf && config_.published_hbf_blocks_per_plane != 0) {
        throw std::runtime_error(
            "session published HBF placement requires the HBF tier");
    }
    if (checked_add(
            static_cast<std::uint64_t>(config_.static_hbf_blocks_per_plane),
            config_.published_hbf_blocks_per_plane,
            "session static/published HBF block extent") >
        config_.hbf.blocks_per_plane) {
        throw std::runtime_error(
            "session static and published HBF extents exceed blocks per plane");
    }
    if (!config_.enable_hbf && config_.initial_hbf_logical_pages != 0) {
        throw std::runtime_error(
            "session initial HBF logical image requires the HBF tier");
    }
    if (config_.initial_hbf_logical_pages == 0 &&
        config_.initial_hbf_logical_first_lpn != 0) {
        throw std::runtime_error(
            "session initial HBF first LPN requires a non-empty image");
    }
    if (!config_.enable_hbf && config_.initial_hbf_persistent_image) {
        throw std::runtime_error(
            "session persistent HBF image requires the HBF tier");
    }
    if (config_.initial_hbf_persistent_image &&
        (config_.static_hbf_blocks_per_plane != 0 ||
         config_.initial_hbf_logical_first_lpn != 0 ||
         config_.initial_hbf_logical_pages != 0)) {
        throw std::runtime_error(
            "session persistent HBF image is mutually exclusive with static "
            "or dense initial placement");
    }
    if (config_.enable_hbm) {
        hbm_ = std::make_unique<hbm::HbmDevice>(config_.hbm);
    }
    if (config_.enable_hbf) {
        hbf_ = std::make_unique<hbf::HbfDevice>(config_.hbf);
        if (config_.hbf_physical_heatmap_bins != 0) {
            AddressHeatmapConfig heatmap_config;
            heatmap_config.bin_count = config_.hbf_physical_heatmap_bins;
            std::uint64_t physical_blocks = config_.hbf.stacks;
            for (const std::uint64_t factor : {
                     static_cast<std::uint64_t>(config_.hbf.channels_per_stack),
                     static_cast<std::uint64_t>(config_.hbf.dies_per_channel),
                     static_cast<std::uint64_t>(config_.hbf.planes_per_die),
                     static_cast<std::uint64_t>(config_.hbf.blocks_per_plane)}) {
                physical_blocks = checked_mul(
                    physical_blocks, factor, "HBF heatmap block count");
            }
            if (physical_blocks % config_.hbf_physical_heatmap_bins != 0) {
                throw std::runtime_error(
                    "HBF physical heatmap bins must divide the physical block count");
            }
            std::uint64_t capacity_bytes = config_.hbf.stacks;
            for (const std::uint64_t factor : {
                     static_cast<std::uint64_t>(config_.hbf.channels_per_stack),
                     static_cast<std::uint64_t>(config_.hbf.dies_per_channel),
                     static_cast<std::uint64_t>(config_.hbf.planes_per_die),
                     static_cast<std::uint64_t>(config_.hbf.blocks_per_plane),
                     static_cast<std::uint64_t>(config_.hbf.pages_per_block),
                     config_.hbf.page_size_bytes}) {
                capacity_bytes = checked_mul(
                    capacity_bytes, factor, "HBF heatmap capacity");
            }
            for (auto& domain : heatmap_config.domains) {
                // Only the physical HBF domain is binned meaningfully; the
                // other domains must still contain every record the device
                // emits (logical addresses span the user namespace).
                domain.size_bytes = domain.domain == AddressDomain::HbfPhysical ?
                    AddressBoundary{capacity_bytes} :
                    AddressBoundary::full_address_space_end();
            }
            hbf_heatmap_ = std::make_unique<AddressHeatmap>(heatmap_config);
            hbf_->attach_address_heatmap(*hbf_heatmap_);
        }
        if (config_.initial_hbf_persistent_image) {
            hbf_->restore_persistent_image(
                *config_.initial_hbf_persistent_image);
            const auto restored_stats = hbf_->stats();
            auto pages_per_block_extent = static_cast<std::uint64_t>(
                config_.hbf.stacks);
            pages_per_block_extent = checked_mul(
                pages_per_block_extent,
                config_.hbf.channels_per_stack,
                "restored HBF extent planes");
            pages_per_block_extent = checked_mul(
                pages_per_block_extent,
                config_.hbf.dies_per_channel,
                "restored HBF extent planes");
            pages_per_block_extent = checked_mul(
                pages_per_block_extent,
                config_.hbf.planes_per_die,
                "restored HBF extent planes");
            pages_per_block_extent = checked_mul(
                pages_per_block_extent,
                config_.hbf.pages_per_block,
                "restored HBF extent pages per block per plane");
            const auto restored_blocks_per_plane = [&pages_per_block_extent](
                    std::uint64_t pages,
                    std::string_view description) {
                if (pages % pages_per_block_extent != 0) {
                    throw std::runtime_error(
                        std::string(description) +
                        " is not a whole per-plane block extent");
                }
                const auto blocks = pages / pages_per_block_extent;
                if (blocks > std::numeric_limits<std::uint32_t>::max()) {
                    throw std::runtime_error(
                        std::string(description) + " exceeds uint32 range");
                }
                return static_cast<std::uint32_t>(blocks);
            };
            const auto restored_static_blocks = restored_blocks_per_plane(
                restored_stats.static_reserved_pages,
                "restored static HBF extent");
            const auto restored_published_blocks = restored_blocks_per_plane(
                restored_stats.raw_reserved_pages,
                "restored published HBF extent");
            // A persistent image owns its static-extent declaration.  The
            // caller cannot safely restate it before the image has been
            // validated, but restored static reads still need the same
            // address guard as a freshly declared extent.
            config_.static_hbf_blocks_per_plane = restored_static_blocks;
            if (config_.published_hbf_blocks_per_plane != 0 &&
                config_.published_hbf_blocks_per_plane !=
                    restored_published_blocks) {
                throw std::runtime_error(
                    "restored published HBF extent differs from its declaration");
            }
            config_.published_hbf_blocks_per_plane =
                restored_published_blocks;
            if (restored_published_blocks != 0) {
                hbf_->validate_raw_physical_block_extent(
                    config_.static_hbf_blocks_per_plane,
                    config_.published_hbf_blocks_per_plane);
            }
        } else {
            if (config_.static_hbf_blocks_per_plane != 0) {
                hbf_->reserve_static_physical_block_extent(
                    0, config_.static_hbf_blocks_per_plane);
            }
            if (config_.published_hbf_blocks_per_plane != 0) {
                hbf_->reserve_raw_physical_block_extent(
                    config_.static_hbf_blocks_per_plane,
                    config_.published_hbf_blocks_per_plane);
            }
            if (config_.initial_hbf_logical_pages != 0) {
                hbf_->prepopulate_mutable_logical_page_range(
                    config_.initial_hbf_logical_first_lpn,
                    config_.initial_hbf_logical_pages);
            }
        }
        hbf_logical_capacity_pages_ = hbf_->logical_capacity_pages();
        base_die_links_.reserve(config_.hbf.stacks);
        for (std::uint32_t stack = 0; stack < config_.hbf.stacks; ++stack) {
            base_die_links_.emplace_back(
                config_.base_die_link,
                "session/base_die_link/stack" + std::to_string(stack));
        }
    }
    if (config_.enable_external) {
        external_ = std::make_unique<external::ExternalBackingDevice>(
            config_.external);
    }
    if (config_.enable_hbf && config_.enable_external &&
        config_.hbf_external_direct_link) {
        hbf_external_direct_links_.reserve(config_.hbf.stacks);
        for (std::uint32_t stack = 0; stack < config_.hbf.stacks; ++stack) {
            hbf_external_direct_links_.emplace_back(
                *config_.hbf_external_direct_link,
                "session/hbf_external_direct_link/stack" +
                    std::to_string(stack));
        }
    }
}

std::size_t SimulationSession::resolvable_dependency_ids() const {
    std::size_t count = retained_finishes_.size();
    for (const auto& batch : recent_batch_finishes_) {
        count += batch.size();
    }
    return count;
}

const double* SimulationSession::find_completed_finish(
    const std::string& id) const {
    for (const auto& batch : recent_batch_finishes_) {
        if (const auto found = batch.find(id); found != batch.end()) {
            return &found->second;
        }
    }
    if (const auto found = retained_finishes_.find(id);
        found != retained_finishes_.end()) {
        return &found->second;
    }
    return nullptr;
}

SimulationDeviceSnapshot SimulationSession::device_snapshot() const {
    SimulationDeviceSnapshot snapshot;
    snapshot.has_hbm = static_cast<bool>(hbm_);
    snapshot.has_hbf = static_cast<bool>(hbf_);
    snapshot.has_external = static_cast<bool>(external_);
    if (hbm_) snapshot.hbm = hbm_->stats();
    if (hbf_) snapshot.hbf = hbf_->stats();
    if (external_) snapshot.external = external_->stats();
    snapshot.base_die_link = aggregate_base_die_links(base_die_links_);
    snapshot.hbf_external_direct_link =
        aggregate_base_die_links(hbf_external_direct_links_);
    return snapshot;
}

SimulationDeviceSnapshot SimulationSession::device_counters() const {
    SimulationDeviceSnapshot snapshot;
    snapshot.has_hbm = static_cast<bool>(hbm_);
    snapshot.has_hbf = static_cast<bool>(hbf_);
    snapshot.has_external = static_cast<bool>(external_);
    if (hbm_) snapshot.hbm = hbm_->execution_stats();
    if (hbf_) snapshot.hbf = hbf_->execution_stats();
    if (external_) snapshot.external = external_->stats();
    snapshot.base_die_link = aggregate_base_die_links(base_die_links_);
    snapshot.hbf_external_direct_link =
        aggregate_base_die_links(hbf_external_direct_links_);
    return snapshot;
}

SimulationDrainResult SimulationSession::persist_pending(
    std::string completion_id,
    bool end_session) {
    if (failed_) {
        throw std::runtime_error(
            "cannot persist a failed simulation session");
    }
    if (drained_) {
        throw std::runtime_error("simulation session was already drained");
    }
    if (completion_id.empty()) {
        throw std::runtime_error(
            "simulation persistence completion id must be non-empty");
    }
    if (end_session) drained_ = true;
    SimulationDrainResult result;
    result.has_hbf = static_cast<bool>(hbf_);
    // Persistence covers every issued transaction, including work the
    // caller left outside its blocking frontier, so it starts once all of
    // it has landed.
    result.serving_frontier_ns = issued_work_frontier_ns_;
    result.finish_ns = issued_work_frontier_ns_;
    result.device_before = device_counters();
    if (hbf_) {
        result.hbf_completion = hbf_->drain_pending(
            std::move(completion_id),
            issued_work_frontier_ns_,
            config_.trace);
        result.finish_ns = result.hbf_completion.finish_ns;
        result.quiescence = hbf_->quiescence_stats();
        if (!result.quiescence.quiescent()) {
            failed_ = true;
            throw std::runtime_error(
                "simulation HBF persistence operation did not reach quiescence");
        }
    }
    result.device_after = device_counters();
    return result;
}

SimulationCheckpointResult SimulationSession::checkpoint_pending(
    std::string checkpoint_id) {
    if (!hbf_) {
        throw std::runtime_error(
            "simulation checkpoint requires an enabled HBF tier");
    }
    if (checkpoint_id.empty() || checkpoint_ids_.contains(checkpoint_id)) {
        throw std::runtime_error(
            "simulation checkpoint id must be non-empty and globally unique");
    }
    const auto sequence = next_checkpoint_sequence_;
    auto persistence = persist_pending(
        "simulation-session/checkpoint/" + checkpoint_id,
        false);
    completed_frontier_ns_ = persistence.finish_ns;
    issued_work_frontier_ns_ = persistence.finish_ns;
    checkpoint_ids_.insert(checkpoint_id);
    ++next_checkpoint_sequence_;
    return SimulationCheckpointResult{
        .checkpoint_id = std::move(checkpoint_id),
        .sequence = sequence,
        .persistence = std::move(persistence),
    };
}

hbf::HbfPersistentImage SimulationSession::persistent_hbf_image() const {
    if (!hbf_) {
        throw std::runtime_error(
            "simulation persistent image requires an enabled HBF tier");
    }
    if (failed_ || drained_) {
        throw std::runtime_error(
            "simulation persistent image requires an active successful session");
    }
    return hbf_->persistent_image();
}

SimulationCrashResult SimulationSession::inject_crash(
    std::string crash_id) {
    if (failed_ || drained_) {
        throw std::runtime_error(
            "cannot inject a crash into a failed or drained simulation session");
    }
    if (crash_id.empty()) {
        throw std::runtime_error(
            "simulation crash id must be non-empty");
    }
    if (hbf_) {
        hbf_->materialize_committed_state_through(completed_frontier_ns_);
    }
    SimulationCrashResult result{
        .crash_id = std::move(crash_id),
        .completed_batches = next_sequence_,
        .completed_checkpoints = next_checkpoint_sequence_,
        .completed_frontier_ns = completed_frontier_ns_,
        .device_at_injection = device_counters(),
        .quiescence = hbf_ ?
            hbf_->quiescence_stats() : hbf::HbfQuiescenceStats{},
    };
    // Make every subsequent engine operation fail closed. No state transition
    // is allowed after the observation above because the process is modeling
    // an instantaneous power-loss boundary, not a graceful shutdown.
    failed_ = true;
    return result;
}

SimulationDrainResult SimulationSession::drain_pending() {
    return persist_pending(
        "simulation-session/end-of-session-drain",
        true);
}

void SimulationSession::validate_transaction(
    const SimulationTransaction& transaction) const {
    if (transaction.id.empty()) {
        throw std::runtime_error("simulation transaction id must be non-empty");
    }
    if (!std::isfinite(transaction.issue_ns) || transaction.issue_ns < 0.0) {
        throw std::runtime_error(
            "simulation transaction issue time must be finite and non-negative");
    }
    if (!std::isfinite(transaction.duration_ns) ||
        transaction.duration_ns < 0.0) {
        throw std::runtime_error(
            "simulation transaction duration must be finite and non-negative");
    }
    if (transaction.target == SimulationTarget::Barrier) {
        if (transaction.bytes != 0 || transaction.addr != 0 ||
            transaction.duration_ns < 0.0 || transaction.stack != 0) {
            throw std::runtime_error(
                "simulation barrier must have zero address/bytes/stack");
        }
        return;
    }
    if (transaction.op != Op::Read && transaction.op != Op::Write) {
        throw std::runtime_error(
            "simulation memory transactions support only reads and writes");
    }
    if (transaction.bytes == 0) {
        throw std::runtime_error(
            "simulation memory transaction bytes must be positive");
    }
    if (transaction.duration_ns != 0.0) {
        throw std::runtime_error(
            "simulation memory transaction duration must be zero");
    }
    const auto end = checked_add(
        transaction.addr,
        transaction.bytes,
        "simulation transaction address range");
    switch (transaction.target) {
    case SimulationTarget::Hbm:
        if (!hbm_) {
            throw std::runtime_error(
                "simulation transaction targets a disabled HBM tier");
        }
        if (transaction.stack != 0 || end > config_.hbm.capacity_bytes) {
            throw std::runtime_error(
                "simulation HBM transaction is outside physical capacity");
        }
        return;
    case SimulationTarget::HbfLogical:
        if (!hbf_) {
            throw std::runtime_error(
                "simulation transaction targets a disabled HBF tier");
        }
        if (transaction.stack != 0 || (end - 1) >> 63U != 0) {
            throw std::runtime_error(
                "simulation logical HBF transaction is outside the user LPN namespace");
        }
        if (const auto capacity_pages = hbf_logical_capacity_pages_;
            (end - 1) / config_.hbf.page_size_bytes >= capacity_pages) {
            throw std::runtime_error(
                "simulation logical HBF transaction is beyond the HBF "
                "logical capacity of " + std::to_string(capacity_pages) +
                " pages");
        }
        return;
    case SimulationTarget::HbfStatic: {
        if (!hbf_) {
            throw std::runtime_error(
                "simulation transaction targets a disabled HBF tier");
        }
        auto declared_capacity = static_cast<std::uint64_t>(
            config_.static_hbf_blocks_per_plane);
        declared_capacity = checked_mul(
            declared_capacity,
            config_.hbf.pages_per_block,
            "simulation static HBF capacity");
        declared_capacity = checked_mul(
            declared_capacity,
            config_.hbf.page_size_bytes,
            "simulation static HBF capacity");
        declared_capacity = checked_mul(
            declared_capacity,
            config_.hbf.stacks,
            "simulation static HBF capacity");
        declared_capacity = checked_mul(
            declared_capacity,
            config_.hbf.channels_per_stack,
            "simulation static HBF capacity");
        declared_capacity = checked_mul(
            declared_capacity,
            config_.hbf.dies_per_channel,
            "simulation static HBF capacity");
        declared_capacity = checked_mul(
            declared_capacity,
            config_.hbf.planes_per_die,
            "simulation static HBF capacity");
        if (transaction.stack != 0 || transaction.op != Op::Read ||
            config_.static_hbf_blocks_per_plane == 0 ||
            end > declared_capacity) {
            throw std::runtime_error(
                "simulation static HBF transactions must be reads inside "
                "the declared FTL-bypass extent");
        }
        return;
    }
    case SimulationTarget::HbfPhysical: {
        if (!hbf_) {
            throw std::runtime_error(
                "simulation transaction targets a disabled HBF tier");
        }
        auto capacity = static_cast<std::uint64_t>(config_.hbf.stacks);
        capacity = checked_mul(
            capacity,
            config_.hbf.channels_per_stack,
            "simulation HBF physical capacity");
        capacity = checked_mul(
            capacity,
            config_.hbf.dies_per_channel,
            "simulation HBF physical capacity");
        capacity = checked_mul(
            capacity,
            config_.hbf.planes_per_die,
            "simulation HBF physical capacity");
        capacity = checked_mul(
            capacity,
            config_.hbf.blocks_per_plane,
            "simulation HBF physical capacity");
        capacity = checked_mul(
            capacity,
            config_.hbf.pages_per_block,
            "simulation HBF physical capacity");
        capacity = checked_mul(
            capacity,
            config_.hbf.page_size_bytes,
            "simulation HBF physical capacity");
        if (transaction.stack != 0 || end > capacity) {
            throw std::runtime_error(
                "simulation physical HBF transaction is outside capacity");
        }
        const auto page_size = static_cast<std::uint64_t>(
            config_.hbf.page_size_bytes);
        const auto pages_per_block = static_cast<std::uint64_t>(
            config_.hbf.pages_per_block);
        const auto blocks_per_plane = static_cast<std::uint64_t>(
            config_.hbf.blocks_per_plane);
        const auto first_global_block =
            (transaction.addr / page_size) / pages_per_block;
        const auto last_global_block =
            ((end - 1) / page_size) / pages_per_block;
        const auto first_plane = first_global_block / blocks_per_plane;
        const auto last_plane = last_global_block / blocks_per_plane;
        const auto first_local_block = first_global_block % blocks_per_plane;
        const auto last_local_block = last_global_block % blocks_per_plane;
        const auto static_blocks = static_cast<std::uint64_t>(
            config_.static_hbf_blocks_per_plane);
        const auto published_blocks = static_cast<std::uint64_t>(
            config_.published_hbf_blocks_per_plane);
        const auto published_end = checked_add(
            static_blocks,
            published_blocks,
            "simulation published HBF block end");
        const auto within_extent = [&](std::uint64_t first_block,
                                       std::uint64_t block_count) {
            if (block_count == 0) {
                return false;
            }
            const auto extent_end = checked_add(
                first_block,
                block_count,
                "simulation physical HBF extent end");
            // Linear physical addresses cross the unreserved tail and prefix
            // at a plane boundary. Such a request is legal only when its
            // extent covers a complete plane; policies otherwise issue one
            // transaction per contiguous in-plane run.
            return first_local_block >= first_block &&
                last_local_block < extent_end &&
                (first_plane == last_plane ||
                 (first_block == 0 && extent_end == blocks_per_plane));
        };
        const bool readable = within_extent(0, published_end);
        const bool writable = within_extent(static_blocks, published_blocks);
        if (transaction.op == Op::Read && !readable) {
            throw std::runtime_error(
                "simulation physical HBF read escapes the declared static/published extents");
        }
        if (transaction.op == Op::Write) {
            if (!writable || config_.initial_hbf_persistent_image ||
                transaction.addr % page_size != 0 ||
                transaction.bytes % page_size != 0) {
                throw std::runtime_error(
                    "simulation physical HBF writes require an aligned fresh published extent");
            }
        } else if (transaction.op != Op::Read) {
            throw std::runtime_error(
                "simulation physical HBF transactions support only reads and writes");
        }
        return;
    }
    case SimulationTarget::D2dHbfToHbm:
        if (!hbm_ || !hbf_ || transaction.op != Op::Read ||
            transaction.stack >= base_die_links_.size()) {
            throw std::runtime_error(
                "simulation HBF-to-HBM D2D transaction is invalid");
        }
        return;
    case SimulationTarget::D2dHbmToHbf:
        if (!hbm_ || !hbf_ || transaction.op != Op::Write ||
            transaction.stack >= base_die_links_.size()) {
            throw std::runtime_error(
                "simulation HBM-to-HBF D2D transaction is invalid");
        }
        return;
    case SimulationTarget::DirectHbfToExternal:
        // The lane's read direction drains HBF pages toward the external
        // device without an HBM staging hop.
        if (!hbf_ || !external_ || transaction.op != Op::Read ||
            transaction.stack >= hbf_external_direct_links_.size()) {
            throw std::runtime_error(
                "simulation HBF-to-external direct transaction is invalid");
        }
        return;
    case SimulationTarget::DirectExternalToHbf:
        // The lane's write direction restores external pages into HBF.
        if (!hbf_ || !external_ || transaction.op != Op::Write ||
            transaction.stack >= hbf_external_direct_links_.size()) {
            throw std::runtime_error(
                "simulation external-to-HBF direct transaction is invalid");
        }
        return;
    case SimulationTarget::External:
        if (!external_) {
            throw std::runtime_error(
                "simulation transaction targets a disabled external tier");
        }
        if (transaction.stack != 0 ||
            end > config_.external.capacity_bytes) {
            throw std::runtime_error(
                "simulation external transaction is outside physical capacity");
        }
        return;
    case SimulationTarget::Barrier:
        return;
    }
}

PhysicalCompletion SimulationSession::issue_non_hbm_memory(
    const SimulationTransaction& transaction,
    double ready_ns) {
    if (transaction.target == SimulationTarget::HbfLogical ||
        transaction.target == SimulationTarget::HbfStatic ||
        transaction.target == SimulationTarget::HbfPhysical) {
        return hbf_->issue(PhysicalRequest{
            .id = transaction.id,
            .tier = Tier::HBF,
            .op = transaction.op,
            .address_space = transaction.target == SimulationTarget::HbfLogical ?
                AddressSpace::Logical :
                transaction.target == SimulationTarget::HbfStatic ?
                    AddressSpace::Static : AddressSpace::Physical,
            .trace = config_.trace,
            .arrival_ns = ready_ns,
            .addr = transaction.addr,
            .bytes = transaction.bytes,
            .stream_id = 0,
            .heatmap_source = HeatmapTrafficSource::Direct,
        });
    }
    if (transaction.target == SimulationTarget::D2dHbfToHbm ||
        transaction.target == SimulationTarget::D2dHbmToHbf) {
        return base_die_links_.at(transaction.stack).issue(
            transaction.id,
            transaction.op,
            ready_ns,
            transaction.bytes,
            config_.trace);
    }
    if (transaction.target == SimulationTarget::DirectHbfToExternal ||
        transaction.target == SimulationTarget::DirectExternalToHbf) {
        return hbf_external_direct_links_.at(transaction.stack).issue(
            transaction.id,
            transaction.op,
            ready_ns,
            transaction.bytes,
            config_.trace);
    }
    if (transaction.target == SimulationTarget::External) {
        return external_->issue_contiguous_range(PhysicalRequest{
            .id = transaction.id,
            .tier = Tier::External,
            .op = transaction.op,
            .address_space = AddressSpace::Logical,
            .trace = config_.trace,
            .arrival_ns = ready_ns,
            .addr = transaction.addr,
            .bytes = transaction.bytes,
            .stream_id = 0,
            .heatmap_source = HeatmapTrafficSource::Direct,
        });
    }
    throw std::runtime_error(
        "cannot synchronously issue an HBM or barrier session transaction");
}

SimulationBatchResult SimulationSession::run_batch(
    std::string batch_id,
    const std::vector<SimulationTransaction>& transactions,
    const SimulationBatchOptions& options) {
    if (failed_) {
        throw std::runtime_error(
            "simulation session is unusable after a failed batch");
    }
    if (drained_) {
        throw std::runtime_error(
            "simulation session cannot accept work after end-of-session drain");
    }
    if (batch_id.empty() || batch_ids_.contains(batch_id)) {
        throw SimulationInputError(
            "simulation batch id must be non-empty and unique: " + batch_id);
    }
    if (transactions.empty()) {
        throw SimulationInputError(
            "simulation batch must contain transactions");
    }

    // Everything up to the ready queue is validation of caller input and
    // touches no device state; a failure there leaves the session usable.
    struct Node {
        std::size_t indegree = 0;
        double dependency_ready_ns = 0.0;
        std::vector<std::size_t> successors;
    };
    std::vector<Node> graph(transactions.size());
    std::unordered_map<std::string, std::size_t> local_index;
    local_index.reserve(transactions.size());
    std::uint64_t dependency_edges = 0;
    std::vector<bool> in_frontier(transactions.size(), !options.frontier);
    std::uint64_t frontier_transactions = options.frontier ?
        0 : transactions.size();
    try {
        for (std::size_t index = 0; index < transactions.size(); ++index) {
            const auto& transaction = transactions[index];
            validate_transaction(transaction);
            if (find_completed_finish(transaction.id) != nullptr ||
                !local_index.emplace(transaction.id, index).second) {
                throw SimulationInputError(
                    "simulation transaction id repeats a resolvable "
                    "transaction id: " + transaction.id);
            }
        }
        for (std::size_t index = 0; index < transactions.size(); ++index) {
            std::unordered_set<std::string_view> unique_dependencies;
            for (const auto& dependency : transactions[index].dependencies) {
                if (dependency.empty() ||
                    dependency == transactions[index].id ||
                    !unique_dependencies.insert(dependency).second) {
                    throw SimulationInputError(
                        "simulation transaction has an invalid dependency: " +
                        transactions[index].id);
                }
                dependency_edges = checked_add(
                    dependency_edges, 1, "simulation dependency edge count");
                if (const auto local = local_index.find(dependency);
                    local != local_index.end()) {
                    ++graph[index].indegree;
                    graph[local->second].successors.push_back(index);
                    continue;
                }
                const auto* prior = find_completed_finish(dependency);
                if (prior == nullptr) {
                    throw SimulationInputError(
                        "simulation dependency does not name a transaction "
                        "of this batch, of the last " +
                        std::to_string(kDependencyWindowBatches) +
                        " completed batches, or of the retained set: " +
                        dependency);
                }
                graph[index].dependency_ready_ns = std::max(
                    graph[index].dependency_ready_ns, *prior);
            }
        }
        if (options.frontier) {
            for (const auto& id : *options.frontier) {
                const auto local = local_index.find(id);
                if (local == local_index.end()) {
                    throw SimulationInputError(
                        "simulation frontier names a transaction outside "
                        "the batch: " + id);
                }
                if (in_frontier[local->second]) {
                    throw SimulationInputError(
                        "simulation frontier repeats transaction id: " + id);
                }
                in_frontier[local->second] = true;
                ++frontier_transactions;
            }
        }
        if (options.retain) {
            std::unordered_set<std::string_view> unique_retained;
            for (const auto& id : *options.retain) {
                if (!unique_retained.insert(id).second) {
                    throw SimulationInputError(
                        "simulation retain list repeats transaction id: " + id);
                }
                if (!local_index.contains(id) &&
                    find_completed_finish(id) == nullptr) {
                    throw SimulationInputError(
                        "simulation retain list names an unresolvable "
                        "transaction id: " + id);
                }
            }
        }

        // Prove acyclicity before mutating any device state.
        std::queue<std::size_t> topology;
        auto indegrees = graph;
        for (std::size_t index = 0; index < graph.size(); ++index) {
            if (indegrees[index].indegree == 0) {
                topology.push(index);
            }
        }
        std::size_t topology_count = 0;
        while (!topology.empty()) {
            const auto index = topology.front();
            topology.pop();
            ++topology_count;
            for (const auto successor : indegrees[index].successors) {
                if (--indegrees[successor].indegree == 0) {
                    topology.push(successor);
                }
            }
        }
        if (topology_count != transactions.size()) {
            throw SimulationInputError(
                "simulation transaction dependency graph has a cycle");
        }
    } catch (const SimulationInputError&) {
        throw;
    } catch (const std::runtime_error& error) {
        throw SimulationInputError(error.what());
    }

    // Equal-time mutable/logical HBF traffic is latency-sensitive controller
    // work; immutable physical page runs consume the remaining fabric. This
    // low-level address-space priority prevents a large static DMA listed
    // first in the trace from monopolizing page credits ahead of ready KV
    // reads/writes. It depends on no model/layer/phase annotation.
    const auto ready_class = [&](std::size_t index) {
        return (transactions[index].target == SimulationTarget::HbfStatic ||
                transactions[index].target == SimulationTarget::HbfPhysical) ?
            std::uint8_t{1} : std::uint8_t{0};
    };
    using Ready = std::tuple<double, std::uint8_t, std::size_t>;
    std::priority_queue<Ready, std::vector<Ready>, std::greater<Ready>> ready;
    const auto batch_origin_ns = completed_frontier_ns_;
    for (std::size_t index = 0; index < graph.size(); ++index) {
        if (graph[index].indegree == 0) {
            const auto at = std::max(
                batch_origin_ns + transactions[index].issue_ns,
                graph[index].dependency_ready_ns);
            ready.emplace(at, ready_class(index), index);
        }
    }

    const auto device_before = device_counters();
    const auto hbf_read_engine_before = hbf_ ?
        hbf_->read_engine_stats() : hbf::HbfReadEngineStats{};
    SimulationBatchResult result;
    result.device_before = device_before;
    result.batch_id = std::move(batch_id);
    result.sequence = next_sequence_;
    result.transactions = transactions.size();
    result.dependency_edges = dependency_edges;
    result.frontier_transactions = frontier_transactions;
    result.batch_origin_ns = batch_origin_ns;
    result.first_issue_ns = std::numeric_limits<double>::infinity();
    if (options.record_completions) {
        result.completions.reserve(transactions.size());
    }
    std::vector<double> local_finish(
        transactions.size(), std::numeric_limits<double>::quiet_NaN());
    // Every transaction is issued within its batch even when its completion
    // is outside the frontier, so the next batch cannot start before the
    // last arrival of this one; that keeps device arrivals nondecreasing.
    double last_arrival_ns = batch_origin_ns;
    std::size_t completed = 0;
    std::unordered_map<std::uint64_t, std::size_t> deferred_hbm;
    deferred_hbm.reserve(transactions.size());

    const auto finish_node = [&](std::size_t index, double finish_ns) {
        const auto& transaction = transactions[index];
        if (!std::isfinite(finish_ns) ||
            finish_ns < std::max(
                batch_origin_ns + transaction.issue_ns,
                graph[index].dependency_ready_ns)) {
            throw std::runtime_error(
                "simulation transaction produced an invalid completion time");
        }
        local_finish[index] = finish_ns;
        ++completed;
        for (const auto successor : graph[index].successors) {
            graph[successor].dependency_ready_ns = std::max(
                graph[successor].dependency_ready_ns, finish_ns);
            if (--graph[successor].indegree == 0) {
                ready.emplace(
                    std::max(
                        batch_origin_ns + transactions[successor].issue_ns,
                        graph[successor].dependency_ready_ns),
                    ready_class(successor),
                    successor);
            }
        }
    };
    const auto record_memory_completion = [&](std::size_t index,
                                               PhysicalCompletion completion) {
        const auto& transaction = transactions[index];
        const auto target = target_index(transaction.target);
        const auto operation = operation_index(transaction.op);
        record_completion_stats(
            result.completion_by_target[target][operation], completion);
        record_completion_stats(
            completion_by_target_[target][operation], completion);
        const auto finish_ns = completion.finish_ns;
        if (options.record_completions) {
            result.completions.push_back(std::move(completion));
        }
        finish_node(index, finish_ns);
    };
    const auto collect_hbm_completions = [&]() {
        if (deferred_hbm.empty()) {
            return;
        }
        for (auto& [ticket, completion] : hbm_->take_completions()) {
            const auto found = deferred_hbm.find(ticket);
            if (found == deferred_hbm.end()) {
                throw std::runtime_error("simulation received an unknown HBM completion");
            }
            const auto index = found->second;
            deferred_hbm.erase(found);
            record_memory_completion(index, std::move(completion));
        }
    };
    try {
        while (!ready.empty() || !deferred_hbm.empty()) {
            collect_hbm_completions();
            const auto next_arrival_ns = ready.empty() ?
                std::numeric_limits<double>::infinity() : std::get<0>(ready.top());
            if (!deferred_hbm.empty() && hbm_->service_before(next_arrival_ns)) {
                continue;
            }
            if (ready.empty()) {
                if (!deferred_hbm.empty()) {
                    throw std::runtime_error("simulation lost a pending HBM request");
                }
                break;
            }
            const auto [ready_ns, ready_priority, index] = ready.top();
            (void)ready_priority;
            ready.pop();
            const auto& transaction = transactions[index];
            last_arrival_ns = std::max(last_arrival_ns, ready_ns);
            result.first_issue_ns = std::min(
                result.first_issue_ns,
                batch_origin_ns + transaction.issue_ns);
            auto& census = result.by_target[target_index(transaction.target)];
            census.transactions = checked_add(
                census.transactions, 1, "simulation target transaction count");
            census.bytes = checked_add(
                census.bytes, transaction.bytes, "simulation target byte count");
            result.transaction_bytes = checked_add(
                result.transaction_bytes,
                transaction.bytes,
                "simulation batch byte count");
            if (transaction.target == SimulationTarget::Hbm) {
                result.hbm_engine.requests = checked_add(
                    result.hbm_engine.requests,
                    1,
                    "simulation HBM request count");
                const auto burst_bytes = config_.hbm.burst_bytes();
                const auto covered = checked_add(
                    transaction.addr % burst_bytes,
                    transaction.bytes,
                    "simulation HBM covered bytes");
                const auto bursts = checked_add(
                    covered,
                    burst_bytes - 1,
                    "simulation HBM rounded burst bytes") / burst_bytes;
                result.hbm_engine.bursts = checked_add(
                    result.hbm_engine.bursts,
                    bursts,
                    "simulation HBM burst count");
            }

            double finish_ns = ready_ns;
            if (transaction.target == SimulationTarget::Barrier) {
                ++result.barriers;
                finish_ns += transaction.duration_ns;
                finish_node(index, finish_ns);
            } else if (transaction.target == SimulationTarget::Hbm) {
                ++result.memory_transactions;
                const auto ticket = hbm_->enqueue(PhysicalRequest{
                    .id = transaction.id,
                    .tier = Tier::HBM,
                    .op = transaction.op,
                    .address_space = AddressSpace::Physical,
                    .trace = config_.trace,
                    .arrival_ns = ready_ns,
                    .addr = transaction.addr,
                    .bytes = transaction.bytes,
                    .stream_id = 0,
                    .heatmap_source = HeatmapTrafficSource::Direct,
                });
                deferred_hbm.emplace(ticket, index);
            } else {
                ++result.memory_transactions;
                record_memory_completion(
                    index, issue_non_hbm_memory(transaction, ready_ns));
            }
        }
    } catch (...) {
        failed_ = true;
        throw;
    }
    if (completed != transactions.size()) {
        failed_ = true;
        throw std::runtime_error(
            "simulation transaction scheduler did not complete the graph");
    }
    FinishById batch_finishes;
    batch_finishes.reserve(transactions.size());
    result.blocking_finish_ns = last_arrival_ns;
    for (std::size_t index = 0; index < transactions.size(); ++index) {
        batch_finishes.emplace(transactions[index].id, local_finish[index]);
        result.finish_ns = std::max(result.finish_ns, local_finish[index]);
        if (in_frontier[index]) {
            result.blocking_finish_ns = std::max(
                result.blocking_finish_ns, local_finish[index]);
        }
    }
    if (!std::isfinite(result.first_issue_ns) ||
        result.finish_ns < result.first_issue_ns ||
        result.blocking_finish_ns > result.finish_ns) {
        failed_ = true;
        throw std::runtime_error("simulation batch completion frontier is invalid");
    }
    // A declared retain list replaces the retained set (a batch without one
    // leaves it as it is), then the batch window ages. Retained ids resolve
    // from this batch first, so a batch may retain its own transactions.
    if (options.retain) {
        FinishById retained;
        retained.reserve(options.retain->size());
        for (const auto& id : *options.retain) {
            const auto local = batch_finishes.find(id);
            retained.emplace(
                id, local != batch_finishes.end() ?
                    local->second : *find_completed_finish(id));
        }
        retained_finishes_ = std::move(retained);
    }
    recent_batch_finishes_.push_back(std::move(batch_finishes));
    while (recent_batch_finishes_.size() > kDependencyWindowBatches) {
        recent_batch_finishes_.pop_front();
    }
    completed_frontier_ns_ = result.blocking_finish_ns;
    issued_work_frontier_ns_ = std::max(
        issued_work_frontier_ns_, result.finish_ns);
    result.device_after = device_counters();
    if (hbf_) {
        const auto after = hbf_->read_engine_stats();
        const auto delta = [](std::uint64_t value,
                              std::uint64_t before,
                              const char* name) {
            if (value < before) {
                throw std::runtime_error(
                    std::string("simulation HBF read-engine counter regressed: ") +
                    name);
            }
            return value - before;
        };
        result.hbf_read_engine = hbf::HbfReadEngineStats{
            .page_run_requests = delta(
                after.page_run_requests,
                hbf_read_engine_before.page_run_requests,
                "page_run_requests"),
            .page_run_pages = delta(
                after.page_run_pages,
                hbf_read_engine_before.page_run_pages,
                "page_run_pages"),
            .page_run_physical_requests = delta(
                after.page_run_physical_requests,
                hbf_read_engine_before.page_run_physical_requests,
                "page_run_physical_requests"),
            .page_run_static_requests = delta(
                after.page_run_static_requests,
                hbf_read_engine_before.page_run_static_requests,
                "page_run_static_requests"),
            .page_run_logical_requests = delta(
                after.page_run_logical_requests,
                hbf_read_engine_before.page_run_logical_requests,
                "page_run_logical_requests"),
            .streaming_read_buffer_bypass_pages = delta(
                after.streaming_read_buffer_bypass_pages,
                hbf_read_engine_before.streaming_read_buffer_bypass_pages,
                "streaming_read_buffer_bypass_pages"),
            .scalar_read_requests = delta(
                after.scalar_read_requests,
                hbf_read_engine_before.scalar_read_requests,
                "scalar_read_requests"),
            .scalar_read_pages = delta(
                after.scalar_read_pages,
                hbf_read_engine_before.scalar_read_pages,
                "scalar_read_pages"),
        };
    }
    batch_ids_.insert(result.batch_id);
    ++next_sequence_;
    return result;
}

} // namespace hbfsim::physical
