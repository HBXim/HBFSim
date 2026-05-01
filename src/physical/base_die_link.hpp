#pragma once

#include "physical/physical_types.hpp"

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace hbfsim::physical {

// One independent link is instantiated for every HBF stack. The link is a
// physical transport mechanism; placement policies decide when to use it.
struct BaseDieLinkConfig {
    double read_bandwidth_GBps = 512.0;
    double write_bandwidth_GBps = 128.0;
    double latency_ns = 20.0;
};

struct BaseDieLinkStats {
    std::uint64_t read_transfers = 0;
    std::uint64_t write_transfers = 0;
    std::uint64_t read_bytes = 0;
    std::uint64_t write_bytes = 0;
    double read_queue_wait_ns = 0.0;
    double write_queue_wait_ns = 0.0;
    double read_busy_ns = 0.0;
    double write_busy_ns = 0.0;
    double read_fixed_latency_work_ns = 0.0;
    double write_fixed_latency_work_ns = 0.0;
    double first_arrival_ns = std::numeric_limits<double>::infinity();
    double finish_ns = 0.0;
    // Aggregate snapshots set this to the number of independent links.
    std::uint64_t links = 1;

    [[nodiscard]] double read_utilization() const;
    [[nodiscard]] double write_utilization() const;
    [[nodiscard]] double active_span_ns() const;
};

class BaseDieLink {
public:
    explicit BaseDieLink(
        BaseDieLinkConfig config,
        std::string entity = "physical/base_die_link");

    [[nodiscard]] PhysicalCompletion issue(
        std::string id,
        Op op,
        double arrival_ns,
        std::uint64_t bytes,
        TraceConfig trace);

    [[nodiscard]] const BaseDieLinkStats& stats() const { return stats_; }

private:
    BaseDieLinkConfig config_;
    std::string entity_;
    BaseDieLinkStats stats_;
    double read_ready_ns_ = 0.0;
    double write_ready_ns_ = 0.0;
};

[[nodiscard]] BaseDieLinkStats aggregate_base_die_link_stats(
    const std::vector<BaseDieLink>& links);

} // namespace hbfsim::physical
