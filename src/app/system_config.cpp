#include "app/system_config.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_set>

namespace hbfsim::app {
namespace {

std::string trim(std::string value) {
    const auto non_space = [](unsigned char ch) { return !std::isspace(ch); };
    value.erase(
        value.begin(),
        std::find_if(value.begin(), value.end(), non_space));
    value.erase(
        std::find_if(value.rbegin(), value.rend(), non_space).base(),
        value.end());
    return value;
}

// Decimal digits only: a leading zero never selects octal, and no sign,
// prefix, or exponent is accepted.
std::uint64_t parse_u64(std::string_view raw, std::string_view name) {
    std::uint64_t value = 0;
    const auto* const begin = raw.data();
    const auto* const end = raw.data() + raw.size();
    const auto [pointer, error] = std::from_chars(begin, end, value, 10);
    if (raw.empty() || error != std::errc{} || pointer != end) {
        throw std::runtime_error(
            std::string(name) + " must be an unsigned decimal integer");
    }
    return value;
}

std::uint32_t parse_u32(std::string_view raw, std::string_view name) {
    const auto value = parse_u64(raw, name);
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error(std::string(name) + " exceeds uint32_t");
    }
    return static_cast<std::uint32_t>(value);
}

double parse_double(std::string_view raw, std::string_view name) {
    std::size_t consumed = 0;
    double value = 0.0;
    try {
        value = std::stod(std::string(raw), &consumed);
    } catch (const std::exception&) {
        throw std::runtime_error(std::string(name) + " must be numeric");
    }
    if (consumed != raw.size() || !std::isfinite(value)) {
        throw std::runtime_error(std::string(name) + " must be finite numeric data");
    }
    return value;
}

bool parse_bool(std::string_view raw, std::string_view name) {
    if (raw == "true" || raw == "1") return true;
    if (raw == "false" || raw == "0") return false;
    throw std::runtime_error(std::string(name) + " must be true or false");
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

std::uint64_t ceil_div(std::uint64_t value, std::uint64_t divisor) {
    if (divisor == 0) {
        throw std::runtime_error("cannot divide HBF capacity by zero");
    }
    return value / divisor + (value % divisor != 0 ? 1 : 0);
}

const std::unordered_set<std::string>& owned_keys() {
    static const std::unordered_set<std::string> keys{
        "hbm-capacity-bytes", "hbf-capacity-bytes", "hbf-capacity-ratio",
        "base-die-link-read-bw", "base-die-link-write-bw",
        "base-die-link-latency-ns",
        "hbf-external-direct-link-enable",
        "hbf-external-direct-link-read-bw",
        "hbf-external-direct-link-write-bw",
        "hbf-external-direct-link-latency-ns",
        "external-backing-kind",
        "external-backing-capacity-bytes", "external-backing-page-size",
        "external-backing-request-segment-bytes",
        "external-backing-media-channels",
        "external-backing-media-read-queues",
        "external-backing-media-write-queues",
        "external-backing-max-outstanding-requests",
        "external-backing-controller-issue-ns",
        "external-backing-controller-processing-ns",
        "external-backing-media-read-latency-ns",
        "external-backing-media-write-latency-ns",
        "external-backing-media-read-bw", "external-backing-media-write-bw",
        "external-backing-m2s-bw", "external-backing-s2m-bw",
        "external-backing-one-way-propagation-ns",
        "external-backing-command-bytes",
        "external-backing-completion-bytes",
        "external-backing-cache-enabled",
        "external-backing-cache-capacity-bytes",
        "external-backing-cache-ways",
        "external-backing-cache-policy",
        "external-backing-cache-prefetch-degree",
        "external-backing-cache-prefetch-stride",
        "external-backing-cache-hit-latency-ns",
        "external-backing-cache-hit-bw",
        "hbm-stacks", "hbm-channels", "hbm-pseudo-channels",
        "hbm-bank-groups-per-pseudo-channel", "hbm-banks-per-group",
        "hbm-channel-row-size-bytes", "hbm-channel-width-bits",
        "hbm-burst-length", "hbm-pin-rate-gbps",
        "hbm-data-rate-per-command-clock", "hbm-address-mapping-ns",
        "hbm-trcdrd-ns", "hbm-trcdwr-ns", "hbm-tcl-ns", "hbm-tcwl-ns",
        "hbm-trp-ns", "hbm-tras-ns", "hbm-trc-ns", "hbm-twr-ns",
        "hbm-trtp-ns", "hbm-tccd-s-cycles", "hbm-tccd-l-cycles",
        "hbm-trrd-s-ns", "hbm-trrd-l-ns", "hbm-tfaw-ns",
        "hbm-twtr-s-ns", "hbm-twtr-l-ns", "hbm-trtw-ns",
        "hbm-refresh", "hbm-same-bank-refresh", "hbm-trefi-ns",
        "hbm-trfc-ns", "hbm-trfcsb-ns", "hbm-trrefd-ns", "hbm-queue-depth",
        "hbm-frfcfs-cap-ns", "hbm-interleave-bytes",
        "hbm-replicate-symmetric-pseudo-channels",
        "hbf-processor-interconnect", "hbf-stacks", "hbf-channels",
        "hbf-dies-per-channel", "hbf-planes-per-die",
        "hbf-blocks-per-plane", "hbf-pages-per-block", "hbf-page-size",
        "hbf-oob-bytes", "hbf-media-lanes-per-plane",
        "hbf-subarrays-per-plane", "hbf-page-buffer-banks-per-plane",
        "hbf-read-ns", "hbf-program-ns", "hbf-program-verify-ns",
        "hbf-erase-ns", "hbf-ecc-decode-latency-ns",
        "hbf-ecc-encode-latency-ns", "hbf-ecc-decode-raw-bw",
        "hbf-ecc-encode-raw-bw", "hbf-channel-bw", "hbf-hbio-bw",
        "hbf-tsv-bw", "hbf-media-lane-bw", "hbf-logic-sram-bw",
        "hbf-page-buffer-bw", "hbf-mapping-mode", "hbf-ctrl-dram-bytes",
        "hbf-ctrl-dram-capacity-denominator",
        "hbf-logic-scheduler-issue-ns", "hbf-flash-tsu-issue-ns",
        "hbf-batch-activation", "hbf-read-buffer-pages",
        "hbf-page-read-queue-depth-per-stack", "hbf-page-run-acceleration",
        "hbf-ctrl-dram-latency-ns", "hbf-ctrl-dram-issue-ns",
        "hbf-gc-low-watermark-pages", "hbf-gc-hard-watermark-pages",
        "hbf-gc-reserved-free-blocks-per-plane",
        "hbf-gc-wear-leveling-weight",
        "hbf-static-wear-leveling-erase-gap",
        "hbf-static-wear-leveling-interval-erases",
        "hbf-static-wear-leveling-start-erases", "hbf-write-coalescing",
        "hbf-write-buffer-completion-requires-flush",
        "hbf-write-buffer-pages", "hbf-write-buffer-flush-threshold-pages",
        "hbf-thermal-enable", "hbf-thermal-ambient-c",
        "hbf-thermal-resistance-c-per-w", "hbf-thermal-capacitance-j-per-c",
        "hbf-thermal-throttle-c", "hbf-thermal-release-c",
        "hbf-thermal-static-power-w", "hbf-thermal-read-energy-pj-per-bit",
        "hbf-thermal-program-energy-pj-per-bit",
        "hbf-thermal-erase-energy-uj-per-block",
        "hbf-thermal-throttle-power-w", "hbf-thermal-start-state",
        "hbf-thermal-neighbor-heat-c",
    };
    return keys;
}

} // namespace

SystemConfigBuilder::SystemConfigBuilder() {
    // Preserve the engine defaults that predate the separation from the
    // reference runner. Public device structs have intentionally smaller
    // library defaults for unit construction.
    hbm_.channels_per_stack = 4;
    hbf_.stacks = 4;
    hbf_.channels_per_stack = 4;
    hbf_.dies_per_channel = 2;
    hbf_.planes_per_die = 4;
    hbf_.blocks_per_plane = 2048;
    hbf_.pages_per_block = 256;
    hbf_.ecc_decode_raw_bandwidth_GBps_per_die = 136.0;
    hbf_.ecc_encode_raw_bandwidth_GBps_per_die = 136.0;
    hbf_.channel_bandwidth_GBps = 272.0;
}

bool SystemConfigBuilder::owns_key(std::string_view key) {
    return owned_keys().contains(std::string(key));
}

std::vector<std::string> SystemConfigBuilder::owned_key_list() {
    std::vector<std::string> keys(owned_keys().begin(), owned_keys().end());
    std::sort(keys.begin(), keys.end());
    return keys;
}

void SystemConfigBuilder::apply_file(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open system config: " + path);
    }
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        line = trim(line.substr(0, line.find('#')));
        if (line.empty()) continue;
        const auto separator = line.find('=');
        if (separator == std::string::npos) {
            throw std::runtime_error(
                path + ":" + std::to_string(line_number) +
                " is not a key=value system-config record");
        }
        const auto key = trim(line.substr(0, separator));
        const auto value = trim(line.substr(separator + 1));
        if (key.empty() || value.empty()) {
            throw std::runtime_error(
                path + ":" + std::to_string(line_number) +
                " has an empty system-config key or value");
        }
        try {
            apply(key, value);
        } catch (const std::exception& error) {
            throw std::runtime_error(
                path + ":" + std::to_string(line_number) + ": " +
                error.what());
        }
    }
}

void SystemConfigBuilder::apply(std::string_view key, std::string_view value) {
    if (!owns_key(key)) {
        throw std::runtime_error(
            "key is not owned by the simulator engine: " + std::string(key));
    }

#define SET_U32(name, field) \
    if (key == name) { field = parse_u32(value, name); return; }
#define SET_U64(name, field) \
    if (key == name) { field = parse_u64(value, name); return; }
#define SET_F64(name, field) \
    if (key == name) { field = parse_double(value, name); return; }
#define SET_BOOL(name, field) \
    if (key == name) { field = parse_bool(value, name); return; }

    SET_U64("hbm-capacity-bytes", hbm_.capacity_bytes)
    if (key == "hbf-capacity-bytes") {
        hbf_capacity_bytes_ = parse_u64(value, key);
        return;
    }
    if (key == "hbf-capacity-ratio") {
        hbf_capacity_ratio_ = parse_double(value, key);
        return;
    }
    SET_F64("base-die-link-read-bw", base_die_link_.read_bandwidth_GBps)
    SET_F64("base-die-link-write-bw", base_die_link_.write_bandwidth_GBps)
    SET_F64("base-die-link-latency-ns", base_die_link_.latency_ns)
    SET_BOOL(
        "hbf-external-direct-link-enable", hbf_external_direct_link_enable_)
    SET_F64(
        "hbf-external-direct-link-read-bw",
        hbf_external_direct_link_read_bandwidth_GBps_)
    SET_F64(
        "hbf-external-direct-link-write-bw",
        hbf_external_direct_link_write_bandwidth_GBps_)
    SET_F64(
        "hbf-external-direct-link-latency-ns",
        hbf_external_direct_link_latency_ns_)

    if (key == "external-backing-kind") {
        const auto kind = physical::external::parse_external_backing_kind(
            std::string(value));
        if (external_overrides_present_ && external_kind_explicit_ &&
            kind != external_kind_) {
            throw std::runtime_error(
                "external-backing-kind cannot change after external-backing-* "
                "overrides were applied; declare the kind before its overrides");
        }
        external_kind_ = kind;
        external_kind_explicit_ = true;
        return;
    }
    if (key.starts_with("external-backing-")) {
        external_overrides_present_ = true;
    }
    SET_U64("external-backing-capacity-bytes", external_capacity_bytes_)
    SET_U64("external-backing-page-size", external_page_size_bytes_)
    SET_U64(
        "external-backing-request-segment-bytes",
        external_request_segment_bytes_)
    SET_U32("external-backing-media-channels", external_media_channels_)
    SET_U32("external-backing-media-read-queues", external_media_read_queues_)
    SET_U32("external-backing-media-write-queues", external_media_write_queues_)
    SET_U32(
        "external-backing-max-outstanding-requests",
        external_max_outstanding_requests_)
    SET_F64("external-backing-controller-issue-ns", external_controller_issue_ns_)
    SET_F64(
        "external-backing-controller-processing-ns",
        external_controller_processing_ns_)
    SET_F64(
        "external-backing-media-read-latency-ns",
        external_media_read_latency_ns_)
    SET_F64(
        "external-backing-media-write-latency-ns",
        external_media_write_latency_ns_)
    SET_F64(
        "external-backing-media-read-bw",
        external_media_read_bandwidth_GBps_)
    SET_F64(
        "external-backing-media-write-bw",
        external_media_write_bandwidth_GBps_)
    SET_F64("external-backing-m2s-bw", external_m2s_bandwidth_GBps_)
    SET_F64("external-backing-s2m-bw", external_s2m_bandwidth_GBps_)
    SET_F64(
        "external-backing-one-way-propagation-ns",
        external_one_way_propagation_ns_)
    SET_U32("external-backing-command-bytes", external_command_bytes_)
    SET_U32("external-backing-completion-bytes", external_completion_bytes_)
    SET_BOOL("external-backing-cache-enabled", external_cache_enabled_)
    SET_U64(
        "external-backing-cache-capacity-bytes",
        external_cache_capacity_bytes_)
    SET_U32("external-backing-cache-ways", external_cache_ways_)
    if (key == "external-backing-cache-policy") {
        external_cache_policy_ = std::string(value);
        return;
    }
    SET_U32(
        "external-backing-cache-prefetch-degree",
        external_cache_prefetch_degree_)
    SET_U32(
        "external-backing-cache-prefetch-stride",
        external_cache_prefetch_stride_)
    SET_F64(
        "external-backing-cache-hit-latency-ns",
        external_cache_hit_latency_ns_)
    SET_F64(
        "external-backing-cache-hit-bw",
        external_cache_hit_bandwidth_GBps_)

    SET_U32("hbm-stacks", hbm_.stacks)
    SET_U32("hbm-channels", hbm_.channels_per_stack)
    SET_U32("hbm-pseudo-channels", hbm_.pseudo_channels_per_channel)
    SET_U32(
        "hbm-bank-groups-per-pseudo-channel",
        hbm_.bank_groups_per_pseudo_channel)
    SET_U32("hbm-banks-per-group", hbm_.banks_per_group)
    SET_U64("hbm-channel-row-size-bytes", hbm_.channel_row_size_bytes)
    SET_U32("hbm-channel-width-bits", hbm_.channel_width_bits)
    SET_U32("hbm-burst-length", hbm_.burst_length)
    SET_F64("hbm-pin-rate-gbps", hbm_.pin_rate_Gbps)
    SET_U32(
        "hbm-data-rate-per-command-clock",
        hbm_.data_rate_per_command_clock)
    SET_F64("hbm-address-mapping-ns", hbm_.address_mapping_ns)
    SET_F64("hbm-trcdrd-ns", hbm_.tRCDRD_ns)
    SET_F64("hbm-trcdwr-ns", hbm_.tRCDWR_ns)
    SET_F64("hbm-tcl-ns", hbm_.tCL_ns)
    SET_F64("hbm-tcwl-ns", hbm_.tCWL_ns)
    SET_F64("hbm-trp-ns", hbm_.tRP_ns)
    SET_F64("hbm-tras-ns", hbm_.tRAS_ns)
    SET_F64("hbm-trc-ns", hbm_.tRC_ns)
    SET_F64("hbm-twr-ns", hbm_.tWR_ns)
    SET_F64("hbm-trtp-ns", hbm_.tRTP_ns)
    SET_U32("hbm-tccd-s-cycles", hbm_.tCCD_S_cycles)
    SET_U32("hbm-tccd-l-cycles", hbm_.tCCD_L_cycles)
    SET_F64("hbm-trrd-s-ns", hbm_.tRRD_S_ns)
    SET_F64("hbm-trrd-l-ns", hbm_.tRRD_L_ns)
    SET_F64("hbm-tfaw-ns", hbm_.tFAW_ns)
    SET_F64("hbm-twtr-s-ns", hbm_.tWTR_S_ns)
    SET_F64("hbm-twtr-l-ns", hbm_.tWTR_L_ns)
    SET_F64("hbm-trtw-ns", hbm_.tRTW_ns)
    SET_BOOL("hbm-refresh", hbm_.refresh_enabled)
    SET_BOOL("hbm-same-bank-refresh", hbm_.same_bank_refresh)
    SET_F64("hbm-trefi-ns", hbm_.tREFI_ns)
    SET_F64("hbm-trfc-ns", hbm_.tRFC_ns)
    SET_F64("hbm-trfcsb-ns", hbm_.tRFCsb_ns)
    SET_F64("hbm-trrefd-ns", hbm_.tRREFD_ns)
    SET_U32("hbm-queue-depth", hbm_.queue_depth)
    SET_F64("hbm-frfcfs-cap-ns", hbm_.frfcfs_cap_ns)
    SET_U64("hbm-interleave-bytes", hbm_.interleave_bytes)
    SET_BOOL("hbm-replicate-symmetric-pseudo-channels",
             hbm_.replicate_symmetric_pseudo_channels)

    if (key == "hbf-processor-interconnect") {
        hbf_processor_interconnect_ = value;
        return;
    }
    SET_U32("hbf-stacks", hbf_.stacks)
    SET_U32("hbf-channels", hbf_.channels_per_stack)
    SET_U32("hbf-dies-per-channel", hbf_.dies_per_channel)
    SET_U32("hbf-planes-per-die", hbf_.planes_per_die)
    SET_U32("hbf-blocks-per-plane", hbf_.blocks_per_plane)
    SET_U32("hbf-pages-per-block", hbf_.pages_per_block)
    SET_U64("hbf-page-size", hbf_.page_size_bytes)
    SET_U64("hbf-oob-bytes", hbf_.oob_bytes_per_page)
    SET_U32("hbf-media-lanes-per-plane", hbf_.media_lanes_per_plane)
    SET_U32("hbf-subarrays-per-plane", hbf_.subarrays_per_plane)
    SET_U32(
        "hbf-page-buffer-banks-per-plane",
        hbf_.page_buffer_banks_per_plane)
    SET_F64("hbf-read-ns", hbf_.t_read_page_ns)
    SET_F64("hbf-program-ns", hbf_.t_program_page_ns)
    SET_F64("hbf-program-verify-ns", hbf_.t_program_verify_ns)
    SET_F64("hbf-erase-ns", hbf_.t_erase_block_ns)
    SET_F64("hbf-ecc-decode-latency-ns", hbf_.ecc_decode_latency_ns)
    SET_F64("hbf-ecc-encode-latency-ns", hbf_.ecc_encode_latency_ns)
    SET_F64(
        "hbf-ecc-decode-raw-bw",
        hbf_.ecc_decode_raw_bandwidth_GBps_per_die)
    SET_F64(
        "hbf-ecc-encode-raw-bw",
        hbf_.ecc_encode_raw_bandwidth_GBps_per_die)
    SET_F64("hbf-channel-bw", hbf_.channel_bandwidth_GBps)
    SET_F64("hbf-hbio-bw", hbf_.hb_io_bandwidth_GBps)
    SET_F64("hbf-tsv-bw", hbf_.tsv_bandwidth_GBps)
    SET_F64("hbf-media-lane-bw", hbf_.media_lane_bandwidth_GBps)
    SET_F64("hbf-logic-sram-bw", hbf_.logic_sram_bandwidth_GBps)
    SET_F64("hbf-page-buffer-bw", hbf_.page_buffer_bandwidth_GBps)
    if (key == "hbf-mapping-mode") {
        hbf_.mapping_mode = physical::hbf::parse_mapping_mode(value);
        return;
    }
    SET_U64("hbf-ctrl-dram-bytes", hbf_.ctrl_dram_bytes)
    if (key == "hbf-ctrl-dram-capacity-denominator") {
        hbf_ctrl_dram_capacity_denominator_ = parse_u64(value, key);
        return;
    }
    SET_F64("hbf-logic-scheduler-issue-ns", hbf_.logic_scheduler_issue_ns)
    SET_F64("hbf-flash-tsu-issue-ns", hbf_.flash_tsu_issue_ns)
    SET_BOOL("hbf-batch-activation", hbf_.batch_activation)
    SET_U64("hbf-read-buffer-pages", hbf_.read_buffer_pages)
    SET_U64(
        "hbf-page-read-queue-depth-per-stack",
        hbf_.page_read_queue_depth_per_stack)
    SET_BOOL("hbf-page-run-acceleration", hbf_.page_run_acceleration)
    SET_F64("hbf-ctrl-dram-latency-ns", hbf_.ctrl_dram_latency_ns)
    SET_F64("hbf-ctrl-dram-issue-ns", hbf_.ctrl_dram_issue_ns)
    SET_U64("hbf-gc-low-watermark-pages", hbf_.gc_low_watermark_pages)
    SET_U64("hbf-gc-hard-watermark-pages", hbf_.gc_hard_watermark_pages)
    SET_U64(
        "hbf-gc-reserved-free-blocks-per-plane",
        hbf_.gc_reserved_free_blocks_per_plane)
    SET_F64("hbf-gc-wear-leveling-weight", hbf_.gc_wear_leveling_weight)
    SET_U32("hbf-static-wear-leveling-erase-gap",
        hbf_.static_wear_leveling_erase_gap)
    SET_U32("hbf-static-wear-leveling-interval-erases",
        hbf_.static_wear_leveling_interval_erases)
    SET_U32("hbf-static-wear-leveling-start-erases",
        hbf_.static_wear_leveling_start_erases)
    SET_BOOL("hbf-write-coalescing", hbf_.write_coalescing_enabled)
    SET_BOOL(
        "hbf-write-buffer-completion-requires-flush",
        hbf_.write_buffer_completion_requires_flush)
    SET_U64("hbf-write-buffer-pages", hbf_.write_buffer_pages)
    SET_U64(
        "hbf-write-buffer-flush-threshold-pages",
        hbf_.write_buffer_flush_threshold_pages)
    SET_BOOL("hbf-thermal-enable", hbf_.thermal_enabled)
    SET_F64("hbf-thermal-ambient-c", hbf_.thermal_ambient_c)
    SET_F64(
        "hbf-thermal-resistance-c-per-w",
        hbf_.thermal_resistance_c_per_w)
    SET_F64(
        "hbf-thermal-capacitance-j-per-c",
        hbf_.thermal_capacitance_j_per_c)
    SET_F64("hbf-thermal-throttle-c", hbf_.thermal_throttle_c)
    SET_F64("hbf-thermal-release-c", hbf_.thermal_release_c)
    SET_F64("hbf-thermal-static-power-w", hbf_.thermal_static_power_w)
    SET_F64(
        "hbf-thermal-read-energy-pj-per-bit",
        hbf_.thermal_read_energy_pj_per_bit)
    SET_F64(
        "hbf-thermal-program-energy-pj-per-bit",
        hbf_.thermal_program_energy_pj_per_bit)
    SET_F64(
        "hbf-thermal-erase-energy-uj-per-block",
        hbf_.thermal_erase_energy_uj_per_block)
    SET_F64("hbf-thermal-throttle-power-w", hbf_.thermal_throttle_power_w)
    SET_F64("hbf-thermal-neighbor-heat-c", hbf_.thermal_neighbor_heat_c)
    if (key == "hbf-thermal-start-state") {
        if (value == "idle") {
            hbf_.thermal_start_at_ceiling = false;
        } else if (value == "throttle-ceiling") {
            hbf_.thermal_start_at_ceiling = true;
        } else {
            throw std::runtime_error(
                "hbf-thermal-start-state must be idle or throttle-ceiling");
        }
        return;
    }

#undef SET_BOOL
#undef SET_F64
#undef SET_U64
#undef SET_U32
    throw std::runtime_error("unhandled engine-owned config key: " + std::string(key));
}

SystemConfig SystemConfigBuilder::resolve() const {
    auto hbf = hbf_;
    if (hbf_capacity_bytes_ && hbf_capacity_ratio_) {
        throw std::runtime_error(
            "hbf-capacity-bytes and hbf-capacity-ratio are mutually exclusive");
    }
    auto capacity_target = hbf_capacity_bytes_;
    if (hbf_capacity_ratio_) {
        if (!(*hbf_capacity_ratio_ > 0.0)) {
            throw std::runtime_error("hbf-capacity-ratio must be positive");
        }
        const auto target = static_cast<long double>(hbm_.capacity_bytes) *
            static_cast<long double>(*hbf_capacity_ratio_);
        if (target > static_cast<long double>(
                std::numeric_limits<std::uint64_t>::max())) {
            throw std::runtime_error("HBF capacity target exceeds uint64_t");
        }
        capacity_target = static_cast<std::uint64_t>(std::ceil(target));
    }
    if (capacity_target) {
        auto unit = static_cast<std::uint64_t>(hbf.stacks);
        unit = checked_mul(unit, hbf.channels_per_stack, "HBF capacity unit");
        unit = checked_mul(unit, hbf.dies_per_channel, "HBF capacity unit");
        unit = checked_mul(unit, hbf.planes_per_die, "HBF capacity unit");
        unit = checked_mul(unit, hbf.pages_per_block, "HBF capacity unit");
        unit = checked_mul(unit, hbf.page_size_bytes, "HBF capacity unit");
        const auto blocks = std::max<std::uint64_t>(
            1, ceil_div(*capacity_target, unit));
        if (blocks > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("derived HBF blocks-per-plane exceeds uint32_t");
        }
        hbf.blocks_per_plane = static_cast<std::uint32_t>(blocks);
    }
    if (hbf_ctrl_dram_capacity_denominator_) {
        if (*hbf_ctrl_dram_capacity_denominator_ == 0) {
            throw std::runtime_error(
                "hbf-ctrl-dram-capacity-denominator must be positive");
        }
        if (hbf.ctrl_dram_bytes != 0) {
            throw std::runtime_error(
                "hbf-ctrl-dram-bytes and hbf-ctrl-dram-capacity-denominator "
                "are mutually exclusive");
        }
        hbf.ctrl_dram_capacity_denominator =
            *hbf_ctrl_dram_capacity_denominator_;
        hbf.ctrl_dram_bytes =
            physical::hbf::derive_controller_dram_budget(
                hbf, hbf.ctrl_dram_capacity_denominator).total_bytes;
    } else if (hbf.ctrl_dram_bytes == 0 &&
        hbf.mapping_mode == physical::hbf::MappingMode::FullResident) {
        const auto mapping =
            physical::hbf::derive_resident_mapping_capacity(hbf).total_bytes;
        const auto write_buffer =
            physical::hbf::derive_write_buffer_dram_capacity(hbf).total_bytes;
        if (mapping > std::numeric_limits<std::uint64_t>::max() - write_buffer) {
            throw std::overflow_error(
                "derived HBF controller-DRAM budget overflows uint64_t");
        }
        hbf.ctrl_dram_bytes = mapping + write_buffer;
    } else if (hbf.ctrl_dram_bytes == 0 &&
        hbf.mapping_mode == physical::hbf::MappingMode::Direct) {
        // The exposed-address-space mode has no L2P state; controller DRAM
        // carries only the configured write buffer (zero when coalescing is
        // disabled).
        hbf.ctrl_dram_bytes =
            physical::hbf::derive_write_buffer_dram_capacity(hbf).total_bytes;
    } else if (hbf.ctrl_dram_bytes == 0) {
        throw std::runtime_error(
            "cached HBF mapping requires explicit hbf-ctrl-dram-bytes");
    }

    physical::external::ExternalBackingConfig external;
    switch (external_kind_) {
    case physical::external::ExternalBackingKind::OnPackageLpddr:
        external = physical::external::on_package_lpddr_profile();
        break;
    case physical::external::ExternalBackingKind::HostDram:
        external = physical::external::host_dram_profile();
        break;
    case physical::external::ExternalBackingKind::CxlMemory:
        external = physical::external::cxl_memory_profile();
        break;
    case physical::external::ExternalBackingKind::NvmeSsd:
        external = physical::external::nvme_ssd_profile();
        break;
    case physical::external::ExternalBackingKind::CxlSsd:
        external = physical::external::cxl_ssd_profile();
        break;
    }
#define APPLY_OPTIONAL(source, field) if (source) external.field = *source
    APPLY_OPTIONAL(external_capacity_bytes_, capacity_bytes);
    APPLY_OPTIONAL(external_page_size_bytes_, page_size_bytes);
    APPLY_OPTIONAL(external_request_segment_bytes_, request_segment_bytes);
    APPLY_OPTIONAL(external_media_channels_, media_channels);
    APPLY_OPTIONAL(external_media_read_queues_, media_read_queues);
    APPLY_OPTIONAL(external_media_write_queues_, media_write_queues);
    APPLY_OPTIONAL(external_max_outstanding_requests_, max_outstanding_requests);
    APPLY_OPTIONAL(external_controller_issue_ns_, controller_issue_ns);
    APPLY_OPTIONAL(external_controller_processing_ns_, controller_processing_ns);
    APPLY_OPTIONAL(external_media_read_latency_ns_, media_read_latency_ns);
    APPLY_OPTIONAL(external_media_write_latency_ns_, media_write_latency_ns);
    APPLY_OPTIONAL(external_media_read_bandwidth_GBps_, media_read_bandwidth_GBps);
    APPLY_OPTIONAL(external_media_write_bandwidth_GBps_, media_write_bandwidth_GBps);
    APPLY_OPTIONAL(external_m2s_bandwidth_GBps_, m2s_bandwidth_GBps);
    APPLY_OPTIONAL(external_s2m_bandwidth_GBps_, s2m_bandwidth_GBps);
    APPLY_OPTIONAL(external_one_way_propagation_ns_, one_way_propagation_ns);
    APPLY_OPTIONAL(external_command_bytes_, command_bytes);
    APPLY_OPTIONAL(external_completion_bytes_, completion_bytes);
    APPLY_OPTIONAL(external_cache_enabled_, device_cache.enabled);
    APPLY_OPTIONAL(
        external_cache_capacity_bytes_, device_cache.capacity_bytes);
    APPLY_OPTIONAL(external_cache_ways_, device_cache.ways);
    if (external_cache_policy_) {
        external.device_cache.policy =
            physical::external::parse_device_cache_policy(
                *external_cache_policy_);
    }
    APPLY_OPTIONAL(
        external_cache_prefetch_degree_, device_cache.prefetch_degree);
    APPLY_OPTIONAL(
        external_cache_prefetch_stride_, device_cache.prefetch_stride);
    APPLY_OPTIONAL(
        external_cache_hit_latency_ns_, device_cache.hit_latency_ns);
    APPLY_OPTIONAL(
        external_cache_hit_bandwidth_GBps_, device_cache.hit_bandwidth_GBps);
#undef APPLY_OPTIONAL

    // The direct HBF<->external lane has no default envelope: it exists only
    // when explicitly enabled, and then every timing field must be declared.
    std::optional<physical::BaseDieLinkConfig> hbf_external_direct_link;
    const bool direct_link_fields_present =
        hbf_external_direct_link_read_bandwidth_GBps_ ||
        hbf_external_direct_link_write_bandwidth_GBps_ ||
        hbf_external_direct_link_latency_ns_;
    if (hbf_external_direct_link_enable_) {
        if (!hbf_external_direct_link_read_bandwidth_GBps_ ||
            !hbf_external_direct_link_write_bandwidth_GBps_ ||
            !hbf_external_direct_link_latency_ns_) {
            throw std::runtime_error(
                "hbf-external-direct-link-enable=true requires "
                "hbf-external-direct-link-read-bw, "
                "hbf-external-direct-link-write-bw, and "
                "hbf-external-direct-link-latency-ns");
        }
        hbf_external_direct_link = physical::BaseDieLinkConfig{
            .read_bandwidth_GBps =
                *hbf_external_direct_link_read_bandwidth_GBps_,
            .write_bandwidth_GBps =
                *hbf_external_direct_link_write_bandwidth_GBps_,
            .latency_ns = *hbf_external_direct_link_latency_ns_,
        };
    } else if (direct_link_fields_present) {
        throw std::runtime_error(
            "hbf-external-direct-link-* timing requires "
            "hbf-external-direct-link-enable=true");
    }

    if (hbf_processor_interconnect_ != "unspecified" &&
        hbf_processor_interconnect_ != "ucie") {
        throw std::runtime_error(
            "hbf-processor-interconnect must be unspecified or ucie");
    }
    return SystemConfig{
        .hbm = hbm_,
        .hbf = hbf,
        .external = external,
        .base_die_link = base_die_link_,
        .hbf_external_direct_link = hbf_external_direct_link,
        .hbf_processor_interconnect = hbf_processor_interconnect_,
    };
}

} // namespace hbfsim::app
