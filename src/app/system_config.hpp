#pragma once

#include "physical/base_die_link.hpp"
#include "physical/external/external_backing_device.hpp"
#include "physical/hbf/hbf_device.hpp"
#include "physical/hbm/hbm_device.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace hbfsim::app {

// Configuration owned by the simulator engine. Workload identity, policy,
// and experiment parameters deliberately do not appear in this type.
struct SystemConfig {
    physical::hbm::HbmConfig hbm;
    physical::hbf::HbfConfig hbf;
    physical::external::ExternalBackingConfig external;
    physical::BaseDieLinkConfig base_die_link;
    // Direct per-stack HBF<->external lane (bypasses HBM). Present only when
    // `hbf-external-direct-link-enable=true` declared it together with its
    // bandwidth and latency; the session instantiates it only when both
    // tiers are enabled.
    std::optional<physical::BaseDieLinkConfig> hbf_external_direct_link;
    std::string hbf_processor_interconnect = "unspecified";
};

class SystemConfigBuilder {
public:
    SystemConfigBuilder();

    void apply_file(const std::string& path);
    void apply(std::string_view key, std::string_view value);
    [[nodiscard]] SystemConfig resolve() const;

    // Shared by the CLI parser and ownership-contract tests. Unknown and
    // policy/workload keys are rejected rather than filtered.
    [[nodiscard]] static bool owns_key(std::string_view key);
    // Every engine-owned key in sorted order, for writers that must prove
    // they cover the complete surface (the reference runner's resolved-config
    // export fails closed when a key is added here but not there).
    [[nodiscard]] static std::vector<std::string> owned_key_list();

private:
    physical::hbm::HbmConfig hbm_;
    physical::hbf::HbfConfig hbf_;
    physical::BaseDieLinkConfig base_die_link_;
    bool hbf_external_direct_link_enable_ = false;
    std::optional<double> hbf_external_direct_link_read_bandwidth_GBps_;
    std::optional<double> hbf_external_direct_link_write_bandwidth_GBps_;
    std::optional<double> hbf_external_direct_link_latency_ns_;
    std::string hbf_processor_interconnect_ = "unspecified";
    std::optional<std::uint64_t> hbf_capacity_bytes_;
    std::optional<double> hbf_capacity_ratio_;
    std::optional<std::uint64_t> hbf_ctrl_dram_capacity_denominator_;

    // The kind selects a base profile; every external-backing-* key after
    // it overrides that profile. Changing the kind after such overrides is
    // rejected rather than silently producing a hybrid device.
    physical::external::ExternalBackingKind external_kind_ =
        physical::external::ExternalBackingKind::CxlMemory;
    bool external_kind_explicit_ = false;
    bool external_overrides_present_ = false;
    std::optional<std::uint64_t> external_capacity_bytes_;
    std::optional<std::uint64_t> external_page_size_bytes_;
    std::optional<std::uint64_t> external_request_segment_bytes_;
    std::optional<std::uint32_t> external_media_channels_;
    std::optional<std::uint32_t> external_media_read_queues_;
    std::optional<std::uint32_t> external_media_write_queues_;
    std::optional<std::uint32_t> external_max_outstanding_requests_;
    std::optional<double> external_controller_issue_ns_;
    std::optional<double> external_controller_processing_ns_;
    std::optional<double> external_media_read_latency_ns_;
    std::optional<double> external_media_write_latency_ns_;
    std::optional<double> external_media_read_bandwidth_GBps_;
    std::optional<double> external_media_write_bandwidth_GBps_;
    std::optional<double> external_m2s_bandwidth_GBps_;
    std::optional<double> external_s2m_bandwidth_GBps_;
    std::optional<double> external_one_way_propagation_ns_;
    std::optional<std::uint32_t> external_command_bytes_;
    std::optional<std::uint32_t> external_completion_bytes_;
    std::optional<bool> external_cache_enabled_;
    std::optional<std::uint64_t> external_cache_capacity_bytes_;
    std::optional<std::uint32_t> external_cache_ways_;
    std::optional<std::string> external_cache_policy_;
    std::optional<std::uint32_t> external_cache_prefetch_degree_;
    std::optional<std::uint32_t> external_cache_prefetch_stride_;
    std::optional<double> external_cache_hit_latency_ns_;
    std::optional<double> external_cache_hit_bandwidth_GBps_;
};

} // namespace hbfsim::app
