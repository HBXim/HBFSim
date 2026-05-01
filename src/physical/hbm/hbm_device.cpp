#include "physical/hbm/hbm_device.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace hbfsim::physical::hbm {
namespace {

void require_positive_count(std::uint64_t value, const char* name) {
    if (value == 0) {
        throw std::runtime_error(std::string(name) + " must be positive");
    }
}

void require_positive_timing(double value, const char* name) {
    if (!(value > 0.0) || !std::isfinite(value)) {
        throw std::runtime_error(std::string(name) + " must be positive and finite");
    }
}

void require_nonnegative_timing(double value, const char* name) {
    if (value < 0.0 || !std::isfinite(value)) {
        throw std::runtime_error(std::string(name) + " must be nonnegative and finite");
    }
}

constexpr double kMaximumCommandClockCycle = 70368744177664.0;  // 2^46

std::uint64_t checked_mul(std::uint64_t lhs, std::uint64_t rhs, const char* name) {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
        throw std::runtime_error(std::string(name) + " overflows uint64_t");
    }
    return lhs * rhs;
}

std::uint64_t checked_add(std::uint64_t lhs, std::uint64_t rhs, const char* name) {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
        throw std::runtime_error(std::string(name) + " overflows uint64_t");
    }
    return lhs + rhs;
}

std::uint64_t modular_add(
    std::uint64_t lhs,
    std::uint64_t rhs,
    std::uint64_t modulus) {
    // Both operands are reduced, so the sum cannot wrap even when modulus is
    // not a power of two.
    return lhs >= modulus - rhs ? lhs - (modulus - rhs) : lhs + rhs;
}

std::uint64_t modular_subtract(
    std::uint64_t lhs,
    std::uint64_t rhs,
    std::uint64_t modulus) {
    return lhs >= rhs ? lhs - rhs : modulus - (rhs - lhs);
}

std::size_t command_index(HbmCommand command) {
    return static_cast<std::size_t>(command);
}

const char* command_name(HbmCommand command) {
    switch (command) {
    case HbmCommand::ACT:
        return "ACT";
    case HbmCommand::PRE:
        return "PRE";
    case HbmCommand::RD:
        return "RD";
    case HbmCommand::WR:
        return "WR";
    }
    throw std::runtime_error("unknown HBM command");
}

enum class GateLevel {
    None,
    BankGroup,
    Bank,
    PseudoChannel,
};

// Trace-only labels for the timing rule that delayed a command.
std::string gate_reason(GateLevel level, HbmCommand command) {
    switch (level) {
    case GateLevel::None:
        return "no constraint";
    case GateLevel::Bank:
        switch (command) {
        case HbmCommand::ACT:
            return "bank constraint tRP/tRC (PRE/ACT->ACT)";
        case HbmCommand::PRE:
            return "bank constraint tRAS/tRTP/tWR (->PRE)";
        case HbmCommand::RD:
            return "bank constraint tRCDRD (ACT->RD)";
        case HbmCommand::WR:
            return "bank constraint tRCDWR (ACT->WR)";
        }
        break;
    case GateLevel::BankGroup:
        switch (command) {
        case HbmCommand::ACT:
            return "bankgroup constraint tRRD_L (ACT->ACT)";
        case HbmCommand::PRE:
            return "bankgroup constraint";
        case HbmCommand::RD:
            return "bankgroup constraint tCCD_L/tWTR_L (->RD)";
        case HbmCommand::WR:
            return "bankgroup constraint tCCD_L (WR->WR)";
        }
        break;
    case GateLevel::PseudoChannel:
        switch (command) {
        case HbmCommand::ACT:
            return "pseudochannel placement tRRD_S/tFAW (ACT->ACT)";
        case HbmCommand::PRE:
            return "pseudochannel constraint";
        case HbmCommand::RD:
        case HbmCommand::WR:
            return "pseudochannel placement tCCD_S/turnaround/data bus";
        }
        break;
    }
    throw std::runtime_error("unknown HBM gate");
}

// Golden-ratio hash of the per-pseudo-channel stripe index, taking the HIGH
// multiplier bits: it decorrelates power-of-two request strides before
// pseudo-channel selection (low multiplier bits would still alias for even
// strides).
std::uint64_t pseudo_channel_hash(std::uint64_t stripe) {
    return (stripe * 0x9E3779B97F4A7C15ull) >> 32;
}

std::uint64_t bank_group_hash(std::uint64_t bank_row) {
    return (bank_row * 0xD1B54A32D192ED03ull) >> 32;
}

void mix(std::uint64_t& seed, std::uint64_t value) {
    value += 0x9E3779B97F4A7C15ull;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
    value ^= value >> 31;
    seed ^= value + 0x9E3779B97F4A7C15ull + (seed << 6) + (seed >> 2);
}

template <class T, class Compare>
void insert_sorted(std::vector<T>& values, T value, Compare compare) {
    values.insert(
        std::upper_bound(values.begin(), values.end(), value, compare),
        std::move(value));
}

}  // namespace

std::uint64_t HbmConfig::pseudo_channel_width_bits() const {
    if (pseudo_channels_per_channel == 0 ||
        channel_width_bits % pseudo_channels_per_channel != 0) {
        throw std::runtime_error(
            "HBM channel width must divide evenly across pseudo-channels");
    }
    return channel_width_bits / pseudo_channels_per_channel;
}

std::uint64_t HbmConfig::row_size_bytes() const {
    if (pseudo_channels_per_channel == 0 ||
        channel_row_size_bytes % pseudo_channels_per_channel != 0) {
        throw std::runtime_error(
            "HBM channel row size must divide evenly across pseudo-channels");
    }
    return channel_row_size_bytes / pseudo_channels_per_channel;
}

std::uint64_t HbmConfig::burst_bytes() const {
    const auto pseudo_width = pseudo_channel_width_bits();
    if (pseudo_width % 8 != 0) {
        throw std::runtime_error("HBM pseudo-channel width must be byte-aligned");
    }
    return checked_mul(pseudo_width / 8, burst_length, "HBM burst bytes");
}

std::uint64_t HbmConfig::effective_interleave_bytes() const {
    const auto row = row_size_bytes();
    const auto burst = burst_bytes();
    if (burst > row || row % burst != 0) {
        throw std::runtime_error(
            "HBM derived burst bytes must divide the pseudo-channel row size");
    }
    if (interleave_bytes != 0) {
        if (interleave_bytes % burst != 0 || interleave_bytes > row ||
            row % interleave_bytes != 0) {
            throw std::runtime_error(
                "HBM interleave_bytes=" + std::to_string(interleave_bytes) +
                " must be a multiple of the derived burst bytes (" +
                std::to_string(burst) + ") and divide the pseudo-channel row (" +
                std::to_string(row) + " bytes); set hbm-interleave-bytes to such "
                "a value or to 0 for the automatic default");
        }
        return interleave_bytes;
    }
    // Largest burst multiple that divides the row and stays within the
    // default: walk the divisors of bursts-per-row.
    const auto bursts_per_row = row / burst;
    std::uint64_t best = burst;
    for (std::uint64_t bursts = 1; bursts <= bursts_per_row; ++bursts) {
        if (bursts_per_row % bursts != 0) {
            continue;
        }
        if (bursts > kDefaultInterleaveBytes / burst) {
            break;
        }
        best = bursts * burst;
    }
    return best;
}

double HbmConfig::channel_bandwidth_GBps() const {
    return pin_rate_Gbps * static_cast<double>(channel_width_bits) / 8.0;
}

double HbmConfig::pseudo_channel_bandwidth_GBps() const {
    return pin_rate_Gbps * static_cast<double>(pseudo_channel_width_bits()) / 8.0;
}

double HbmConfig::command_clock_period_ns() const {
    return data_rate_per_command_clock / pin_rate_Gbps;
}

double HbmConfig::command_clock_MHz() const {
    return 1000.0 / command_clock_period_ns();
}

double HbmConfig::burst_duration_ns() const {
    return static_cast<double>(burst_length) / pin_rate_Gbps;
}

double HbmConfig::tCCD_S_ns() const {
    return tCCD_S_cycles * command_clock_period_ns();
}

double HbmConfig::tCCD_L_ns() const {
    return tCCD_L_cycles * command_clock_period_ns();
}

std::uint64_t HbmConfig::command_clock_cycles(double time_ns) const {
    if (!(time_ns >= 0.0) || !std::isfinite(time_ns)) {
        throw std::runtime_error(
            "HBM command time must be nonnegative and finite");
    }
    const double tck_ns = command_clock_period_ns();
    require_positive_timing(tck_ns, "HBM derived command-clock period");
    const double cycles = time_ns / tck_ns;
    if (!(cycles < kMaximumCommandClockCycle)) {
        throw std::runtime_error(
            "HBM command time exceeds the representable clock horizon");
    }
    auto eligible = static_cast<std::uint64_t>(std::floor(cycles));
    while (command_clock_time_ns(eligible) < time_ns) {
        ++eligible;
    }
    while (eligible != 0 && command_clock_time_ns(eligible - 1) >= time_ns) {
        --eligible;
    }
    if (eligible >= static_cast<std::uint64_t>(kMaximumCommandClockCycle)) {
        throw std::runtime_error(
            "HBM command time exceeds the representable clock horizon");
    }
    return eligible;
}

double HbmConfig::command_clock_time_ns(std::uint64_t cycles) const {
    return static_cast<double>(cycles) * command_clock_period_ns();
}

std::string HbmAddress::path() const {
    std::ostringstream out;
    out << "stack" << stack << "/ch" << channel << "/pch" << pseudo_channel
        << "/bg" << bank_group << "/bank" << bank << "/row" << row
        << "/off" << offset;
    return out.str();
}

double HbmStats::row_hit_rate() const {
    const auto total = row_hits + row_misses + row_conflicts;
    return total == 0 ? 0.0 : static_cast<double>(row_hits) / static_cast<double>(total);
}

double HbmStats::active_span_ns() const {
    return std::isfinite(first_arrival_ns) && finish_ns > first_arrival_ns ?
        finish_ns - first_arrival_ns : 0.0;
}

double HbmStats::utilization() const {
    const auto span = active_span_ns();
    return span <= 0.0 || pseudo_channels == 0 ? 0.0 :
        bus_busy_ns / (span * static_cast<double>(pseudo_channels));
}

double HbmStats::bus_parallelism() const {
    const auto span = active_span_ns();
    return span <= 0.0 ? 0.0 : bus_busy_ns / span;
}

double HbmStats::pseudo_channel_busy_skew() const {
    return avg_active_pseudo_channel_busy_ns <= 0.0 ? 0.0 :
        max_pseudo_channel_busy_ns / avg_active_pseudo_channel_busy_ns;
}

HbmDevice::HbmDevice(HbmConfig config, AddressHeatmap* address_heatmap)
    : config_(config), address_heatmap_(address_heatmap) {
    require_positive_count(config_.capacity_bytes, "HBM capacity_bytes");
    require_positive_count(config_.stacks, "HBM stacks");
    require_positive_count(config_.channels_per_stack, "HBM channels_per_stack");
    require_positive_count(config_.pseudo_channels_per_channel, "HBM pseudo_channels_per_channel");
    require_positive_count(
        config_.bank_groups_per_pseudo_channel,
        "HBM bank_groups_per_pseudo_channel");
    require_positive_count(config_.banks_per_group, "HBM banks_per_group");
    require_positive_count(config_.channel_row_size_bytes, "HBM channel_row_size_bytes");
    require_positive_count(config_.channel_width_bits, "HBM channel_width_bits");
    require_positive_count(config_.burst_length, "HBM burst_length");
    require_positive_count(config_.queue_depth, "HBM queue_depth");
    require_positive_timing(config_.pin_rate_Gbps, "HBM pin_rate_Gbps");
    require_positive_count(
        config_.data_rate_per_command_clock,
        "HBM data_rate_per_command_clock");
    if (config_.channel_width_bits % 8 != 0) {
        throw std::runtime_error("HBM channel_width_bits must be byte-aligned");
    }
    const auto row_size_bytes = config_.row_size_bytes();
    const auto burst_bytes = config_.burst_bytes();
    if (burst_bytes > row_size_bytes || row_size_bytes % burst_bytes != 0) {
        throw std::runtime_error(
            "HBM derived burst bytes must divide the pseudo-channel row size");
    }
    // Resolve the automatic interleave once; config() then reports the value
    // the map actually uses.
    config_.interleave_bytes = config_.effective_interleave_bytes();
    if (config_.capacity_bytes % burst_bytes != 0) {
        throw std::runtime_error(
            "HBM capacity_bytes must be an integer number of physical bursts");
    }
    const double channel_bandwidth_GBps = config_.channel_bandwidth_GBps();
    const double pseudo_channel_bandwidth_GBps =
        config_.pseudo_channel_bandwidth_GBps();
    const double command_clock_period_ns = config_.command_clock_period_ns();
    const double command_clock_MHz = config_.command_clock_MHz();
    const double burst_duration_ns = config_.burst_duration_ns();
    require_positive_timing(channel_bandwidth_GBps, "HBM derived channel bandwidth");
    require_positive_timing(
        pseudo_channel_bandwidth_GBps,
        "HBM derived pseudo-channel bandwidth");
    require_positive_timing(command_clock_period_ns, "HBM derived command-clock period");
    require_positive_timing(command_clock_MHz, "HBM derived command-clock frequency");
    require_positive_timing(burst_duration_ns, "HBM derived burst duration");
    require_positive_timing(config_.tCCD_S_ns(), "HBM derived tCCD_S");
    require_positive_timing(config_.tCCD_L_ns(), "HBM derived tCCD_L");
    const double system_bandwidth_GBps = channel_bandwidth_GBps *
        static_cast<double>(config_.channels_per_stack) *
        static_cast<double>(config_.stacks);
    require_positive_timing(system_bandwidth_GBps, "HBM derived system bandwidth");
    if (config_.burst_length % config_.data_rate_per_command_clock != 0) {
        throw std::runtime_error(
            "HBM burst length must span a whole number of command-clock cycles");
    }
    const auto burst_command_cycles =
        config_.burst_length / config_.data_rate_per_command_clock;
    if (config_.tCCD_S_cycles < burst_command_cycles) {
        throw std::runtime_error(
            "HBM tCCD_S cycles cannot be shorter than one derived data burst");
    }
    if (config_.tCCD_L_cycles < config_.tCCD_S_cycles) {
        throw std::runtime_error("HBM tCCD_L cycles must be at least tCCD_S cycles");
    }
    require_nonnegative_timing(config_.address_mapping_ns, "HBM address_mapping_ns");
    require_positive_timing(config_.tRCDRD_ns, "HBM tRCDRD_ns");
    require_positive_timing(config_.tRCDWR_ns, "HBM tRCDWR_ns");
    require_positive_timing(config_.tCL_ns, "HBM tCL_ns");
    require_positive_timing(config_.tCWL_ns, "HBM tCWL_ns");
    require_positive_timing(config_.tRP_ns, "HBM tRP_ns");
    require_positive_timing(config_.tRAS_ns, "HBM tRAS_ns");
    require_positive_timing(config_.tRC_ns, "HBM tRC_ns");
    require_positive_timing(config_.tWR_ns, "HBM tWR_ns");
    require_positive_timing(config_.tRTP_ns, "HBM tRTP_ns");
    require_positive_count(config_.tCCD_S_cycles, "HBM tCCD_S_cycles");
    require_positive_count(config_.tCCD_L_cycles, "HBM tCCD_L_cycles");
    require_positive_timing(config_.tRRD_S_ns, "HBM tRRD_S_ns");
    require_positive_timing(config_.tRRD_L_ns, "HBM tRRD_L_ns");
    require_positive_timing(config_.tFAW_ns, "HBM tFAW_ns");
    require_positive_timing(config_.tWTR_S_ns, "HBM tWTR_S_ns");
    require_positive_timing(config_.tWTR_L_ns, "HBM tWTR_L_ns");
    require_positive_timing(config_.tRTW_ns, "HBM tRTW_ns");
    require_positive_timing(config_.tREFI_ns, "HBM tREFI_ns");
    require_positive_timing(config_.tRFC_ns, "HBM tRFC_ns");
    require_positive_timing(config_.tRFCsb_ns, "HBM tRFCsb_ns");
    require_positive_timing(config_.tRREFD_ns, "HBM tRREFD_ns");
    require_nonnegative_timing(config_.frfcfs_cap_ns, "HBM frfcfs_cap_ns");
    if (config_.tRRD_L_ns < config_.tRRD_S_ns) {
        throw std::runtime_error("HBM tRRD_L must be at least tRRD_S");
    }
    if (config_.tWTR_L_ns < config_.tWTR_S_ns) {
        throw std::runtime_error("HBM tWTR_L must be at least tWTR_S");
    }
    // Absolute-ns minima round up to whole command clocks once, here.
    const auto cycles = [this](double value_ns, const char* name) {
        const auto result = config_.command_clock_cycles(value_ns);
        if (result == 0) {
            throw std::runtime_error(
                std::string(name) + " must be at least one command clock");
        }
        return result;
    };
    burst_cycles_ = burst_command_cycles;
    tccd_s_ = config_.tCCD_S_cycles;
    tccd_l_ = config_.tCCD_L_cycles;
    trcdrd_ = cycles(config_.tRCDRD_ns, "HBM effective tRCDRD");
    trcdwr_ = cycles(config_.tRCDWR_ns, "HBM effective tRCDWR");
    tcl_ = cycles(config_.tCL_ns, "HBM effective tCL");
    tcwl_ = cycles(config_.tCWL_ns, "HBM effective tCWL");
    trp_ = cycles(config_.tRP_ns, "HBM effective tRP");
    tras_ = cycles(config_.tRAS_ns, "HBM effective tRAS");
    trc_ = cycles(config_.tRC_ns, "HBM effective tRC");
    twr_ = cycles(config_.tWR_ns, "HBM effective tWR");
    trtp_ = cycles(config_.tRTP_ns, "HBM effective tRTP");
    trrd_s_ = cycles(config_.tRRD_S_ns, "HBM effective tRRD_S");
    trrd_l_ = cycles(config_.tRRD_L_ns, "HBM effective tRRD_L");
    tfaw_ = cycles(config_.tFAW_ns, "HBM effective tFAW");
    twtr_s_ = cycles(config_.tWTR_S_ns, "HBM effective tWTR_S");
    twtr_l_ = cycles(config_.tWTR_L_ns, "HBM effective tWTR_L");
    trtw_ = cycles(config_.tRTW_ns, "HBM effective tRTW");
    trefi_ = cycles(config_.tREFI_ns, "HBM effective tREFI");
    trfc_ = cycles(config_.tRFC_ns, "HBM effective tRFC");
    trfcsb_ = cycles(config_.tRFCsb_ns, "HBM effective tRFCsb");
    trrefd_ = cycles(config_.tRREFD_ns, "HBM effective tRREFD");
    refresh_commands_per_period_ =
        config_.same_bank_refresh ? config_.banks_per_group : 1;
    if (config_.refresh_enabled) {
        // Every access must fit between the precharge for one refresh of its
        // bank and the next refresh of that bank; the same-bank command
        // spacing must fit the whole rotation into one tREFI.
        const auto refresh_cycles =
            config_.same_bank_refresh ? trfcsb_ : trfc_;
        const auto longest_access = std::max({
            trcdrd_, trcdwr_, tcl_ + burst_cycles_, tcwl_ + burst_cycles_, trp_});
        if (trp_ + refresh_cycles + longest_access >= trefi_) {
            throw std::runtime_error(
                "HBM tRP plus the refresh window must leave room for an "
                "access inside tREFI");
        }
        if (trefi_ / refresh_commands_per_period_ < trrefd_) {
            throw std::runtime_error(
                "HBM tRREFD must fit banks_per_group same-bank refreshes "
                "into one tREFI");
        }
    }
    auto total_pseudo_channels = checked_mul(
        config_.stacks,
        config_.channels_per_stack,
        "HBM pseudo-channel topology");
    total_pseudo_channels = checked_mul(
        total_pseudo_channels,
        config_.pseudo_channels_per_channel,
        "HBM pseudo-channel topology");
    const auto banks_per_pseudo_channel = checked_mul(
        config_.bank_groups_per_pseudo_channel,
        config_.banks_per_group,
        "HBM bank topology");
    if (total_pseudo_channels > std::numeric_limits<std::size_t>::max() ||
        banks_per_pseudo_channel > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("HBM topology cannot be represented by size_t");
    }
    row_size_bytes_ = row_size_bytes;
    burst_bytes_ = burst_bytes;
    bursts_per_unit_ = config_.interleave_bytes / burst_bytes;
    units_per_row_ = row_size_bytes / config_.interleave_bytes;
    total_pseudo_channels_ = total_pseudo_channels;
    stripe_bytes_ = checked_mul(
        total_pseudo_channels, config_.interleave_bytes, "HBM stripe bytes");
    banks_per_pseudo_channel_ = banks_per_pseudo_channel;
    pseudo_channels_per_stack_ = checked_mul(
        config_.channels_per_stack,
        config_.pseudo_channels_per_channel,
        "HBM pseudo-channels per stack");
    command_clock_period_ns_ = command_clock_period_ns;
    pseudo_channels_.resize(static_cast<std::size_t>(total_pseudo_channels));
    pseudo_channel_accesses_.assign(pseudo_channels_.size(), 0);
    pseudo_channel_bus_busy_cycles_.assign(pseudo_channels_.size(), 0);
    route_ticket_markers_.assign(
        pseudo_channels_.size(),
        std::numeric_limits<std::uint64_t>::max());
    for (auto& pseudo_channel : pseudo_channels_) {
        pseudo_channel.banks.resize(static_cast<std::size_t>(banks_per_pseudo_channel));
        pseudo_channel.bank_groups.resize(config_.bank_groups_per_pseudo_channel);
    }
    refresh_parallel_stats();
}

// ---------------------------------------------------------------------------
// Address map: pch-interleave-bg-rotate-v2
//
//   unit           = addr / interleave_bytes      (contiguous interleave unit)
//   lane, stripe   = unit % P, unit / P           (P = pseudo-channels)
//   pseudo-channel = (lane + hash(stripe)) % P    (reversible high-bit swizzle)
//   within a pseudo-channel, successive stripes rotate bank groups first,
//   then fill the row of the bank they land in (column units), then rotate
//   banks within the group, then rows:
//     lane_bg = stripe % BG; g1 = stripe / BG
//     column_unit = g1 % units_per_row; g2 = g1 / units_per_row
//     bank = g2 % banks_per_group; row = g2 / banks_per_group
//     bank_group = (lane_bg + hash(g2)) % BG
// A request that covers whole stripes presents the same local bank/row/column
// sequence to every pseudo-channel.
// ---------------------------------------------------------------------------

HbmDevice::LocalAddress HbmDevice::local_address(std::uint64_t unit) const {
    const auto bank_groups = config_.bank_groups_per_pseudo_channel;
    const auto lane_bg = unit % bank_groups;
    const auto g1 = unit / bank_groups;
    LocalAddress local;
    local.column_unit = g1 % units_per_row_;
    const auto g2 = g1 / units_per_row_;
    local.bank = static_cast<std::uint32_t>(g2 % config_.banks_per_group);
    local.row = g2 / config_.banks_per_group;
    local.bank_group = static_cast<std::uint32_t>(modular_add(
        lane_bg, bank_group_hash(g2) % bank_groups, bank_groups));
    return local;
}

void HbmDevice::assign_pseudo_channel(
    HbmAddress& addr,
    std::size_t pseudo_channel) const {
    addr.stack = static_cast<std::uint32_t>(
        pseudo_channel / pseudo_channels_per_stack_);
    const auto within_stack = pseudo_channel % pseudo_channels_per_stack_;
    addr.channel = static_cast<std::uint32_t>(
        within_stack / config_.pseudo_channels_per_channel);
    addr.pseudo_channel = static_cast<std::uint32_t>(
        within_stack % config_.pseudo_channels_per_channel);
}

std::uint64_t HbmDevice::byte_address(
    std::size_t pseudo_channel,
    std::uint64_t unit,
    std::uint64_t unit_offset) const {
    const auto lane = modular_subtract(
        pseudo_channel,
        pseudo_channel_hash(unit) % total_pseudo_channels_,
        total_pseudo_channels_);
    const auto global_unit = checked_add(
        checked_mul(unit, total_pseudo_channels_, "HBM encoded unit"),
        lane,
        "HBM encoded unit");
    return checked_add(
        checked_mul(global_unit, config_.interleave_bytes, "HBM encoded byte address"),
        unit_offset,
        "HBM encoded byte address");
}

HbmAddress HbmDevice::decode(std::uint64_t addr) const {
    if (addr >= config_.capacity_bytes) {
        throw std::runtime_error("HBM byte address is out of capacity");
    }
    const auto global_unit = addr / config_.interleave_bytes;
    const auto unit_offset = addr % config_.interleave_bytes;
    const auto lane = global_unit % total_pseudo_channels_;
    const auto stripe = global_unit / total_pseudo_channels_;
    const auto pseudo_channel_linear = modular_add(
        lane,
        pseudo_channel_hash(stripe) % total_pseudo_channels_,
        total_pseudo_channels_);
    const auto local = local_address(stripe);
    HbmAddress decoded;
    assign_pseudo_channel(decoded, static_cast<std::size_t>(pseudo_channel_linear));
    decoded.bank_group = local.bank_group;
    decoded.bank = local.bank;
    decoded.row = local.row;
    decoded.offset = checked_add(
        checked_mul(
            local.column_unit, config_.interleave_bytes, "HBM decoded column offset"),
        unit_offset,
        "HBM decoded burst offset");
    return decoded;
}

std::uint64_t HbmDevice::encode(const HbmAddress& addr) const {
    if (addr.stack >= config_.stacks || addr.channel >= config_.channels_per_stack ||
        addr.pseudo_channel >= config_.pseudo_channels_per_channel ||
        addr.bank_group >= config_.bank_groups_per_pseudo_channel ||
        addr.bank >= config_.banks_per_group || addr.offset >= row_size_bytes_) {
        throw std::runtime_error("HBM address field out of range");
    }
    const auto bank_groups = config_.bank_groups_per_pseudo_channel;
    const auto g2 = checked_add(
        checked_mul(addr.row, config_.banks_per_group, "HBM encoded bank row"),
        addr.bank,
        "HBM encoded bank row");
    const auto lane_bg = modular_subtract(
        addr.bank_group, bank_group_hash(g2) % bank_groups, bank_groups);
    const auto column_unit = addr.offset / config_.interleave_bytes;
    const auto unit_offset = addr.offset % config_.interleave_bytes;
    const auto g1 = checked_add(
        checked_mul(g2, units_per_row_, "HBM encoded row units"),
        column_unit,
        "HBM encoded row units");
    const auto stripe = checked_add(
        checked_mul(g1, bank_groups, "HBM encoded stripe"),
        lane_bg,
        "HBM encoded stripe");
    const auto encoded = byte_address(
        pseudo_channel_index(addr), stripe, unit_offset);
    if (encoded >= config_.capacity_bytes) {
        throw std::runtime_error("HBM encoded address is out of capacity");
    }
    return encoded;
}

std::size_t HbmDevice::pseudo_channel_index(const HbmAddress& addr) const {
    return (static_cast<std::size_t>(addr.stack) * config_.channels_per_stack + addr.channel) *
        config_.pseudo_channels_per_channel + addr.pseudo_channel;
}

std::size_t HbmDevice::bank_index(const HbmAddress& addr) const {
    return static_cast<std::size_t>(addr.bank_group) * config_.banks_per_group + addr.bank;
}

double HbmDevice::ns(std::uint64_t cycles) const {
    return static_cast<double>(cycles) * command_clock_period_ns_;
}

// ---------------------------------------------------------------------------
// Front end
// ---------------------------------------------------------------------------

PhysicalCompletion HbmDevice::issue(const PhysicalRequest& request) {
    return pump(enqueue(request));
}

void HbmDevice::validate_and_begin_request(const PhysicalRequest& request) {
    if (request.tier != Tier::HBM) {
        throw std::runtime_error("HbmDevice received non-HBM request");
    }
    if (request.op != Op::Read && request.op != Op::Write) {
        throw std::runtime_error("HbmDevice supports read/write requests only");
    }
    if (request.bytes == 0) {
        throw std::runtime_error("HBM request bytes must be positive");
    }
    if (!std::isfinite(request.arrival_ns) || request.arrival_ns < 0.0) {
        throw std::runtime_error("HBM request arrival must be finite and non-negative");
    }
    if (request.addr >= config_.capacity_bytes ||
        request.bytes > config_.capacity_bytes - request.addr) {
        throw std::runtime_error("HBM request range is out of capacity");
    }
    if (last_enqueue_arrival_ns_ &&
        request.arrival_ns < *last_enqueue_arrival_ns_) {
        throw std::runtime_error(
            "HBM requests must be enqueued in nondecreasing arrival order; "
            "sort or explicitly admit the event stream before enqueueing");
    }
    last_enqueue_arrival_ns_ = request.arrival_ns;
    floor_bound_cycle_ = config_.command_clock_cycles(
        request.arrival_ns + config_.address_mapping_ns);
    stats_.first_arrival_ns = std::min(stats_.first_arrival_ns, request.arrival_ns);
}

void HbmDevice::create_parent(std::uint64_t ticket, const PhysicalRequest& request) {
    PhysicalCompletion aggregate;
    aggregate.id = request.id;
    aggregate.tier = Tier::HBM;
    aggregate.op = request.op;
    aggregate.arrival_ns = request.arrival_ns;
    aggregate.start_ns = std::numeric_limits<double>::infinity();
    aggregate.logical_bytes = request.bytes;
    pending_.emplace(ticket, PendingRequest{
        .completion = std::move(aggregate),
        .remaining_children = 0,
        .total_children = 0,
        .pseudo_channels = 0,
        .critical_pseudo_channel = 0,
        .enqueue_complete = false,
        .has_child_completion = false,
        .retain_diagnostics = request.trace.retain_completion_diagnostics ||
            trace_spans_enabled(request.trace),
    });
    const auto [route_it, route_inserted] =
        ticket_pseudo_channels_.emplace(ticket, std::vector<std::size_t>{});
    if (!route_inserted) {
        throw std::runtime_error("HBM generated a duplicate parent ticket");
    }
}

std::uint64_t HbmDevice::enqueue(const PhysicalRequest& request) {
    validate_and_begin_request(request);
    if (next_ticket_ == std::numeric_limits<std::uint64_t>::max()) {
        throw std::runtime_error("HBM parent ticket space exhausted");
    }
    const auto ticket = next_ticket_++;
    create_parent(ticket, request);

    const auto first_burst_base = request.addr - request.addr % burst_bytes_;
    const auto request_last = checked_add(
        request.addr, request.bytes - 1, "HBM request end address");
    const auto last_burst_base = request_last - request_last % burst_bytes_;
    const auto burst_count = (last_burst_base - first_burst_base) / burst_bytes_ + 1;
    if (address_heatmap_ != nullptr) {
        address_heatmap_->record_contiguous_accesses(
            AddressTrafficRecord{
                .domain = AddressDomain::HbmPhysical,
                .direction = request.op == Op::Read ?
                    TrafficDirection::Read : TrafficDirection::Write,
                .source = request.heatmap_source,
                .address = first_burst_base,
                .bytes = burst_bytes_,
            },
            burst_count);
    }
    // Every child of one parent shares the arrival, so it shares the first
    // eligible command cycle.
    const auto ready_cycle = floor_bound_cycle_;
    if (replicable(request)) {
        enqueue_replicated(ticket, request, ready_cycle);
    } else {
        enqueue_children(ticket, request, ready_cycle);
    }
    auto& parent = pending_.at(ticket);
    parent.pseudo_channels = ticket_pseudo_channels_.at(ticket).size();
    parent.enqueue_complete = true;
    if (parent.remaining_children == 0) {
        complete_parent(ticket, parent);
    }
    return ticket;
}

void HbmDevice::push_child(
    std::size_t pseudo_channel_index,
    std::uint64_t ticket,
    const PhysicalRequest& request,
    std::uint64_t ready_cycle,
    std::uint64_t bytes,
    HbmAddress addr) {
    auto& queue = pseudo_channels_[pseudo_channel_index].queue;
    // Admission is strictly bounded: service an existing entry before a new
    // child is inserted, never after temporarily exceeding the cap.
    while (queue.size() >= config_.queue_depth) {
        service_one(pseudo_channel_index);
    }
    // The parent cannot complete (and be erased) while its enqueue is open.
    auto& parent = pending_.at(ticket);
    if (parent.remaining_children == std::numeric_limits<std::size_t>::max() ||
        parent.total_children == std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("HBM split request has too many children");
    }
    parent.remaining_children++;
    parent.total_children++;
    queue.push_back(QueuedRequest{
        .ticket = ticket,
        .op = request.op,
        .arrival_ns = request.arrival_ns,
        .ready_cycle = ready_cycle,
        .bytes = bytes,
        .addr = addr,
        .trace = request.trace,
        .bypass_count = 0,
    });
    stats_.max_queue_occupancy = std::max(
        stats_.max_queue_occupancy,
        static_cast<std::uint64_t>(queue.size()));
    if (recording_.active && recording_.representative == pseudo_channel_index &&
        recording_.push_ticket == ticket) {
        recording_.pushed++;
    }
}

void HbmDevice::enqueue_children(
    std::uint64_t ticket,
    const PhysicalRequest& request,
    std::uint64_t ready_cycle) {
    // A controller transaction may not cross a derived DRAM burst; each burst
    // resolves to exactly one pseudo-channel/bank/row/column tuple, and an
    // unaligned logical range is charged for every burst it touches. Children
    // are produced directly into their bounded controller queues.
    auto& routes = ticket_pseudo_channels_.at(ticket);
    std::uint64_t cursor = request.addr;
    std::uint64_t remaining = request.bytes;
    while (remaining != 0) {
        const auto burst_remaining = burst_bytes_ - cursor % burst_bytes_;
        const auto bytes = std::min(remaining, burst_remaining);
        const auto addr = decode(cursor);
        const auto pc_index = pseudo_channel_index(addr);
        if (route_ticket_markers_[pc_index] != ticket) {
            route_ticket_markers_[pc_index] = ticket;
            routes.push_back(pc_index);
        }
        push_child(pc_index, ticket, request, ready_cycle, bytes, addr);
        remaining -= bytes;
        if (remaining != 0) {
            cursor += bytes;
        }
    }
}

// ---------------------------------------------------------------------------
// Symmetric replication
//
// A request covering whole address-map stripes gives every pseudo-channel the
// same local burst sequence. Pseudo-channels whose controller state (bank
// rows, unexpired timing gates and placement history, refresh position) and
// pending queues are identical will therefore execute identical events. One
// representative per equivalence class is serviced by the ordinary scheduler;
// its state and per-parent outcome are then copied to the other class
// members. This is a host-time optimization only: results are bit-identical
// to servicing every pseudo-channel individually.
// ---------------------------------------------------------------------------

bool HbmDevice::replicable(const PhysicalRequest& request) const {
    return config_.replicate_symmetric_pseudo_channels &&
        total_pseudo_channels_ > 1 &&
        !trace_spans_enabled(request.trace) &&
        request.addr % stripe_bytes_ == 0 &&
        request.bytes % stripe_bytes_ == 0;
}

std::uint64_t HbmDevice::normalization_floor(
    const PseudoChannelState& pseudo_channel) const {
    // No queued or future command can issue before this cycle, so timing
    // state at or below it has no observable effect.
    return std::max(
        pseudo_channel.service_floor_cycle,
        pseudo_channel.queue.empty() ?
            floor_bound_cycle_ : pseudo_channel.queue.front().ready_cycle);
}

std::uint64_t HbmDevice::column_reach() const {
    // Farthest back a placed column command can constrain a new one.
    return std::max({
        tccd_s_, trtw_, tcwl_ + burst_cycles_ + twtr_s_});
}

void HbmDevice::prune_placements(
    PseudoChannelState& pseudo_channel,
    std::uint64_t floor) const {
    const auto reach = column_reach();
    const auto erase_prefix = [](auto& values, auto&& expired) {
        auto keep = values.begin();
        while (keep != values.end() && expired(*keep)) {
            ++keep;
        }
        values.erase(values.begin(), keep);
    };
    erase_prefix(pseudo_channel.columns, [&](const PlacedColumn& column) {
        return column.cycle + reach <= floor;
    });
    erase_prefix(pseudo_channel.activations, [&](std::uint64_t cycle) {
        return cycle + tfaw_ <= floor;
    });
    const auto bus_floor = floor + std::min(tcl_, tcwl_);
    erase_prefix(pseudo_channel.bus_slots, [&](std::uint64_t start) {
        return start + burst_cycles_ <= bus_floor;
    });
}

std::uint64_t HbmDevice::normalized_gate(std::uint64_t gate, std::uint64_t floor) {
    return gate <= floor ? 0 : gate;
}

std::uint64_t HbmDevice::symmetry_hash(std::size_t pseudo_channel_index) const {
    const auto& pc = pseudo_channels_[pseudo_channel_index];
    const auto floor = normalization_floor(pc);
    std::uint64_t seed = 0x9E3779B97F4A7C15ull;
    for (const auto& bank : pc.banks) {
        mix(seed, bank.has_open_row ? bank.open_row + 1 : 0);
        for (const auto value : bank.ready) {
            mix(seed, normalized_gate(value, floor));
        }
    }
    for (const auto& group : pc.bank_groups) {
        for (const auto value : group.ready) {
            mix(seed, normalized_gate(value, floor));
        }
    }
    const auto reach = column_reach();
    for (const auto& column : pc.columns) {
        if (column.cycle + reach > floor) {
            mix(seed, column.cycle * 2 + (column.write ? 1 : 0));
        }
    }
    for (const auto cycle : pc.activations) {
        if (cycle + tfaw_ > floor) {
            mix(seed, cycle);
        }
    }
    const auto bus_floor = floor + std::min(tcl_, tcwl_);
    for (const auto start : pc.bus_slots) {
        if (start + burst_cycles_ > bus_floor) {
            mix(seed, start);
        }
    }
    mix(seed, pc.refresh_period);
    mix(seed, pc.refresh_command);
    const auto next_refresh = refresh_nominal_cycle(pc.refresh_period, pc.refresh_command);
    mix(seed, pc.has_refresh_issue && pc.last_refresh_issue + trrefd_ > next_refresh ?
        pc.last_refresh_issue + 1 : 0);
    mix(seed, pc.queue.size());
    for (const auto& entry : pc.queue) {
        mix(seed, entry.ticket);
        mix(seed, entry.ready_cycle);
        mix(seed, entry.bytes);
        mix(seed, (static_cast<std::uint64_t>(entry.addr.bank_group) << 32) | entry.addr.bank);
        mix(seed, entry.addr.row);
        mix(seed, entry.addr.offset);
        mix(seed, entry.bypass_count);
    }
    return seed;
}

bool HbmDevice::symmetric(std::size_t lhs_index, std::size_t rhs_index) const {
    const auto& lhs = pseudo_channels_[lhs_index];
    const auto& rhs = pseudo_channels_[rhs_index];
    if (lhs.queue.size() != rhs.queue.size()) {
        return false;
    }
    for (std::size_t index = 0; index < lhs.queue.size(); ++index) {
        const auto& a = lhs.queue[index];
        const auto& b = rhs.queue[index];
        // Trace spans name the pseudo-channel, so traced entries are never
        // interchangeable.
        if (trace_spans_enabled(a.trace) || trace_spans_enabled(b.trace)) {
            return false;
        }
        if (a.ticket != b.ticket || a.op != b.op ||
            a.arrival_ns != b.arrival_ns || a.ready_cycle != b.ready_cycle ||
            a.bytes != b.bytes || a.addr.bank_group != b.addr.bank_group ||
            a.addr.bank != b.addr.bank || a.addr.row != b.addr.row ||
            a.addr.offset != b.addr.offset || a.bypass_count != b.bypass_count ||
            a.trace.retain_completion_diagnostics !=
                b.trace.retain_completion_diagnostics) {
            return false;
        }
    }
    const auto floor = normalization_floor(lhs);
    if (floor != normalization_floor(rhs)) {
        return false;
    }
    const auto same_gates = [floor](const CommandGates& a, const CommandGates& b) {
        for (std::size_t index = 0; index < a.size(); ++index) {
            if (normalized_gate(a[index], floor) != normalized_gate(b[index], floor)) {
                return false;
            }
        }
        return true;
    };
    for (std::size_t index = 0; index < lhs.banks.size(); ++index) {
        const auto& a = lhs.banks[index];
        const auto& b = rhs.banks[index];
        if (a.has_open_row != b.has_open_row ||
            (a.has_open_row && a.open_row != b.open_row) ||
            !same_gates(a.ready, b.ready)) {
            return false;
        }
    }
    for (std::size_t index = 0; index < lhs.bank_groups.size(); ++index) {
        if (!same_gates(lhs.bank_groups[index].ready, rhs.bank_groups[index].ready)) {
            return false;
        }
    }
    // Placement histories compare after dropping entries that can no longer
    // constrain a command at or after the floor.
    const auto same_history = [](const auto& a, const auto& b, auto&& live, auto&& equal) {
        auto ia = a.begin();
        auto ib = b.begin();
        for (;;) {
            while (ia != a.end() && !live(*ia)) ++ia;
            while (ib != b.end() && !live(*ib)) ++ib;
            if (ia == a.end() || ib == b.end()) {
                return ia == a.end() && ib == b.end();
            }
            if (!equal(*ia, *ib)) {
                return false;
            }
            ++ia;
            ++ib;
        }
    };
    const auto reach = column_reach();
    if (!same_history(
            lhs.columns, rhs.columns,
            [&](const PlacedColumn& column) { return column.cycle + reach > floor; },
            [](const PlacedColumn& a, const PlacedColumn& b) {
                return a.cycle == b.cycle && a.write == b.write;
            })) {
        return false;
    }
    if (!same_history(
            lhs.activations, rhs.activations,
            [&](std::uint64_t cycle) { return cycle + tfaw_ > floor; },
            [](std::uint64_t a, std::uint64_t b) { return a == b; })) {
        return false;
    }
    const auto bus_floor = floor + std::min(tcl_, tcwl_);
    if (!same_history(
            lhs.bus_slots, rhs.bus_slots,
            [&](std::uint64_t start) { return start + burst_cycles_ > bus_floor; },
            [](std::uint64_t a, std::uint64_t b) { return a == b; })) {
        return false;
    }
    if (lhs.refresh_period != rhs.refresh_period ||
        lhs.refresh_command != rhs.refresh_command) {
        return false;
    }
    const auto next_refresh = refresh_nominal_cycle(lhs.refresh_period, lhs.refresh_command);
    const auto spacing = [this, next_refresh](const PseudoChannelState& pc) {
        return pc.has_refresh_issue && pc.last_refresh_issue + trrefd_ > next_refresh ?
            pc.last_refresh_issue + 1 : 0;
    };
    return spacing(lhs) == spacing(rhs);
}

std::vector<HbmDevice::SymmetryClass> HbmDevice::partition_symmetric(
    const std::vector<std::size_t>& candidates) const {
    std::vector<SymmetryClass> classes;
    std::unordered_map<std::uint64_t, std::vector<std::size_t>> classes_by_hash;
    classes_by_hash.reserve(candidates.size());
    for (const auto pseudo_channel : candidates) {
        auto& bucket = classes_by_hash[symmetry_hash(pseudo_channel)];
        std::optional<std::size_t> matched;
        for (const auto class_index : bucket) {
            // The hash is only an index; membership requires the exact
            // comparison so a collision can never merge distinct states.
            if (symmetric(classes[class_index].representative, pseudo_channel)) {
                matched = class_index;
                break;
            }
        }
        if (matched) {
            classes[*matched].members.push_back(pseudo_channel);
        } else {
            bucket.push_back(classes.size());
            classes.push_back(SymmetryClass{
                .representative = pseudo_channel,
                .members = {},
            });
        }
    }
    return classes;
}

HbmDevice::CounterDelta HbmDevice::counter_snapshot(std::size_t representative) const {
    return CounterDelta{
        .read_bytes = stats_.read_bytes,
        .write_bytes = stats_.write_bytes,
        .row_hits = stats_.row_hits,
        .row_misses = stats_.row_misses,
        .row_conflicts = stats_.row_conflicts,
        .activations = stats_.activations,
        .precharges = stats_.precharges,
        .refresh_count = stats_.refresh_count,
        .bus_busy_cycles = bus_busy_cycles_,
        .accesses = pseudo_channel_accesses_[representative],
        .busy_cycles = pseudo_channel_bus_busy_cycles_[representative],
    };
}

void HbmDevice::begin_recording(std::size_t representative, std::uint64_t push_ticket) {
    recording_ = Recording{};
    recording_.active = true;
    recording_.representative = representative;
    recording_.push_ticket = push_ticket;
    recording_.snapshot = counter_snapshot(representative);
}

void HbmDevice::fold_recording() {
    if (!recording_.active) {
        throw std::runtime_error("HBM replication recording is not open");
    }
    const auto now = counter_snapshot(recording_.representative);
    const auto fold = [](std::uint64_t& total, std::uint64_t current, std::uint64_t then) {
        if (current < then) {
            throw std::runtime_error("HBM replication counter regressed");
        }
        total += current - then;
    };
    auto& delta = recording_.delta;
    const auto& then = recording_.snapshot;
    fold(delta.read_bytes, now.read_bytes, then.read_bytes);
    fold(delta.write_bytes, now.write_bytes, then.write_bytes);
    fold(delta.row_hits, now.row_hits, then.row_hits);
    fold(delta.row_misses, now.row_misses, then.row_misses);
    fold(delta.row_conflicts, now.row_conflicts, then.row_conflicts);
    fold(delta.activations, now.activations, then.activations);
    fold(delta.precharges, now.precharges, then.precharges);
    fold(delta.refresh_count, now.refresh_count, then.refresh_count);
    fold(delta.bus_busy_cycles, now.bus_busy_cycles, then.bus_busy_cycles);
    fold(delta.accesses, now.accesses, then.accesses);
    fold(delta.busy_cycles, now.busy_cycles, then.busy_cycles);
    recording_.snapshot = now;
}

HbmDevice::Recording HbmDevice::pause_recording() {
    fold_recording();
    Recording recording = std::move(recording_);
    recording_ = Recording{};
    return recording;
}

void HbmDevice::resume_recording(Recording recording) {
    recording_ = std::move(recording);
    recording_.active = true;
    recording_.snapshot = counter_snapshot(recording_.representative);
}

void HbmDevice::replicate_recording(const SymmetryClass& symmetry_class) {
    if (!recording_.active || recording_.representative != symmetry_class.representative) {
        throw std::runtime_error("HBM replication recording is not open");
    }
    const Recording recording = pause_recording();
    const auto representative = symmetry_class.representative;
    const auto& delta = recording.delta;
    const auto& source = pseudo_channels_[representative];
    for (const auto member : symmetry_class.members) {
        auto& target = pseudo_channels_[member];
        target = source;
        for (auto& entry : target.queue) {
            assign_pseudo_channel(entry.addr, member);
        }
        pseudo_channel_accesses_[member] = checked_add(
            pseudo_channel_accesses_[member], delta.accesses, "HBM pseudo-channel accesses");
        pseudo_channel_bus_busy_cycles_[member] += delta.busy_cycles;
        stats_.read_bytes += delta.read_bytes;
        stats_.write_bytes += delta.write_bytes;
        stats_.row_hits += delta.row_hits;
        stats_.row_misses += delta.row_misses;
        stats_.row_conflicts += delta.row_conflicts;
        stats_.activations += delta.activations;
        stats_.precharges += delta.precharges;
        stats_.refresh_count += delta.refresh_count;
        bus_busy_cycles_ += delta.bus_busy_cycles;
        if (recording.pushed != 0) {
            auto& parent = pending_.at(recording.push_ticket);
            if (parent.remaining_children >
                    std::numeric_limits<std::size_t>::max() - recording.pushed ||
                parent.total_children >
                    std::numeric_limits<std::size_t>::max() - recording.pushed) {
                throw std::runtime_error("HBM split request has too many children");
            }
            parent.remaining_children += recording.pushed;
            parent.total_children += recording.pushed;
        }
        for (const auto& record : recording.records) {
            const auto found = pending_.find(record.ticket);
            if (found == pending_.end() ||
                found->second.remaining_children < record.count) {
                throw std::runtime_error("HBM replicated children of an unknown parent");
            }
            auto& parent = found->second;
            auto& out = parent.completion;
            stats_.stage_work += record.work;
            stats_.replicated_bursts += record.count;
            parent.has_child_completion = true;
            out.start_ns = std::min(out.start_ns, record.min_start_ns);
            if (record.max_finish_ns > out.finish_ns ||
                (record.max_finish_ns == out.finish_ns &&
                 member < parent.critical_pseudo_channel)) {
                out.breakdown = record.critical;
                parent.critical_pseudo_channel = member;
            }
            out.finish_ns = std::max(out.finish_ns, record.max_finish_ns);
            out.physical_bytes = checked_add(
                out.physical_bytes, record.physical_bytes, "HBM aggregate physical bytes");
            parent.remaining_children -= record.count;
            if (parent.remaining_children == 0 && parent.enqueue_complete) {
                complete_parent(record.ticket, parent);
            }
        }
    }
    stats_.bus_busy_ns = ns(bus_busy_cycles_);
}

void HbmDevice::enqueue_replicated(
    std::uint64_t ticket,
    const PhysicalRequest& request,
    std::uint64_t ready_cycle) {
    auto& routes = ticket_pseudo_channels_.at(ticket);
    const auto first_stripe = request.addr / stripe_bytes_;
    const auto stripe_count = request.bytes / stripe_bytes_;
    std::vector<std::size_t> candidates(pseudo_channels_.size());
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        candidates[index] = index;
    }
    const auto classes = partition_symmetric(candidates);
    bool replicated = false;
    for (const auto& symmetry_class : classes) {
        const auto representative = symmetry_class.representative;
        if (!symmetry_class.members.empty()) {
            begin_recording(representative, ticket);
        }
        HbmAddress addr;
        assign_pseudo_channel(addr, representative);
        for (std::uint64_t stripe = first_stripe;
             stripe < first_stripe + stripe_count; ++stripe) {
            const auto local = local_address(stripe);
            addr.bank_group = local.bank_group;
            addr.bank = local.bank;
            addr.row = local.row;
            for (std::uint64_t burst = 0; burst < bursts_per_unit_; ++burst) {
                addr.offset = local.column_unit * config_.interleave_bytes +
                    burst * burst_bytes_;
                push_child(representative, ticket, request, ready_cycle, burst_bytes_, addr);
            }
        }
        routes.push_back(representative);
        if (!symmetry_class.members.empty()) {
            replicate_recording(symmetry_class);
            replicated = true;
            routes.insert(
                routes.end(),
                symmetry_class.members.begin(),
                symmetry_class.members.end());
        }
    }
    if (replicated) {
        stats_.replicated_requests++;
    }
}

PhysicalCompletion HbmDevice::pump(std::uint64_t ticket) {
    const auto routed = ticket_pseudo_channels_.find(ticket);
    if (routed == ticket_pseudo_channels_.end()) {
        throw std::runtime_error("HBM pump on unknown or already-pumped ticket");
    }
    if (completed_.find(ticket) == completed_.end()) {
        service_until_complete(ticket);
    }
    const auto done = completed_.find(ticket);
    if (done == completed_.end()) {
        throw std::runtime_error("HBM parent request lost an internal burst");
    }
    auto completion = std::move(done->second);
    completed_.erase(done);
    ticket_pseudo_channels_.erase(ticket);
    return completion;
}

bool HbmDevice::service_before(double arrival_ns) {
    if (std::isnan(arrival_ns) || arrival_ns < 0.0) {
        throw std::runtime_error("HBM service boundary must be non-negative");
    }
    double next_ns = arrival_ns;
    std::vector<std::size_t> candidates;
    for (std::size_t index = 0; index < pseudo_channels_.size(); ++index) {
        const auto& pseudo_channel = pseudo_channels_[index];
        if (pseudo_channel.queue.empty()) {
            continue;
        }
        const auto& queued = pseudo_channel.queue[pick_next(pseudo_channel)];
        const auto at_ns = std::max(
            queued.arrival_ns,
            ns(first_command(pseudo_channel, queued).issue_cycle));
        if (at_ns >= arrival_ns || at_ns > next_ns) {
            continue;
        }
        if (at_ns < next_ns) {
            next_ns = at_ns;
            candidates.clear();
        }
        candidates.push_back(index);
    }
    if (candidates.empty()) {
        return false;
    }
    if (!config_.replicate_symmetric_pseudo_channels || candidates.size() < 2) {
        for (const auto index : candidates) {
            service_one(index);
        }
    } else {
        for (const auto& symmetry_class : partition_symmetric(candidates)) {
            if (!symmetry_class.members.empty()) {
                begin_recording(symmetry_class.representative, 0);
            }
            service_one(symmetry_class.representative);
            if (!symmetry_class.members.empty()) {
                replicate_recording(symmetry_class);
            }
        }
    }
    return true;
}

std::vector<std::pair<std::uint64_t, PhysicalCompletion>>
HbmDevice::take_completions() {
    std::vector<std::pair<std::uint64_t, PhysicalCompletion>> completions;
    completions.reserve(completed_.size());
    for (auto& [ticket, completion] : completed_) {
        completions.emplace_back(ticket, std::move(completion));
        ticket_pseudo_channels_.erase(ticket);
    }
    completed_.clear();
    std::sort(completions.begin(), completions.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.first < rhs.first;
    });
    return completions;
}

void HbmDevice::service_until_complete(std::uint64_t ticket) {
    const auto& routes = ticket_pseudo_channels_.at(ticket);
    const auto holds_child = [ticket](const PseudoChannelState& pc) {
        return std::any_of(
            pc.queue.begin(), pc.queue.end(),
            [ticket](const QueuedRequest& entry) { return entry.ticket == ticket; });
    };
    if (!config_.replicate_symmetric_pseudo_channels || routes.size() < 2) {
        // One sweep services every routed pseudo-channel once; the parent's
        // completion is checked between sweeps.
        while (completed_.find(ticket) == completed_.end()) {
            bool progress = false;
            for (const auto pc_index : routes) {
                if (!pseudo_channels_[pc_index].queue.empty()) {
                    service_one(pc_index);
                    progress = true;
                }
            }
            if (!progress) {
                throw std::runtime_error("HBM parent request lost an internal burst");
            }
        }
        return;
    }
    // Same sweep semantics on symmetric classes: every pseudo-channel is
    // serviced once per sweep until the sweep in which the last child of the
    // ticket finishes, so each class receives the maximum service count.
    const auto classes = partition_symmetric(routes);
    std::vector<Recording> recordings(classes.size());
    std::vector<std::uint64_t> services(classes.size(), 0);
    std::uint64_t sweeps = 0;
    for (std::size_t index = 0; index < classes.size(); ++index) {
        const auto representative = classes[index].representative;
        const bool record = !classes[index].members.empty();
        if (record) {
            begin_recording(representative, ticket);
        }
        while (holds_child(pseudo_channels_[representative])) {
            service_one(representative);
            services[index]++;
        }
        sweeps = std::max(sweeps, services[index]);
        if (record) {
            recordings[index] = pause_recording();
        }
    }
    for (std::size_t index = 0; index < classes.size(); ++index) {
        const auto representative = classes[index].representative;
        const bool record = !classes[index].members.empty();
        if (record) {
            resume_recording(std::move(recordings[index]));
        }
        auto& queue = pseudo_channels_[representative].queue;
        while (services[index] < sweeps && !queue.empty()) {
            service_one(representative);
            services[index]++;
        }
        if (record) {
            replicate_recording(classes[index]);
        }
    }
}

void HbmDevice::drain_queues() {
    if (!config_.replicate_symmetric_pseudo_channels || pseudo_channels_.size() < 2) {
        for (std::size_t index = 0; index < pseudo_channels_.size(); ++index) {
            while (!pseudo_channels_[index].queue.empty()) {
                service_one(index);
            }
        }
        return;
    }
    std::vector<std::size_t> candidates;
    for (std::size_t index = 0; index < pseudo_channels_.size(); ++index) {
        if (!pseudo_channels_[index].queue.empty()) {
            candidates.push_back(index);
        }
    }
    const auto classes = partition_symmetric(candidates);
    for (const auto& symmetry_class : classes) {
        const auto representative = symmetry_class.representative;
        const bool record = !symmetry_class.members.empty();
        if (record) {
            begin_recording(representative, std::numeric_limits<std::uint64_t>::max());
        }
        while (!pseudo_channels_[representative].queue.empty()) {
            service_one(representative);
        }
        if (record) {
            replicate_recording(symmetry_class);
        }
    }
}

// ---------------------------------------------------------------------------
// Command placement
// ---------------------------------------------------------------------------

std::uint64_t HbmDevice::place_activation(
    const PseudoChannelState& pseudo_channel,
    std::uint64_t lower) const {
    const auto& acts = pseudo_channel.activations;
    auto cycle = lower;
    for (;;) {
        const auto next = std::lower_bound(acts.begin(), acts.end(), cycle);
        if (next != acts.begin() && cycle - *(next - 1) < trrd_s_) {
            cycle = *(next - 1) + trrd_s_;
            continue;
        }
        if (next != acts.end() && *next - cycle < trrd_s_) {
            cycle = *next + trrd_s_;
            continue;
        }
        // tFAW: no window of tFAW cycles may hold five activations. Check
        // every window that starts at a placed ACT or at the candidate and
        // contains the candidate.
        const auto window_begin = std::lower_bound(
            acts.begin(), acts.end(), cycle >= tfaw_ ? cycle - tfaw_ + 1 : 0);
        const auto window_end = std::lower_bound(acts.begin(), acts.end(), cycle + tfaw_);
        std::vector<std::uint64_t> nearby(window_begin, window_end);
        insert_sorted(nearby, cycle, std::less<std::uint64_t>{});
        std::optional<std::uint64_t> moved;
        for (std::size_t start = 0; start < nearby.size() && !moved; ++start) {
            if (nearby[start] > cycle) {
                break;
            }
            std::size_t count = 0;
            std::optional<std::uint64_t> first_placed;
            for (std::size_t index = start;
                 index < nearby.size() && nearby[index] < nearby[start] + tfaw_; ++index) {
                ++count;
                if (nearby[index] != cycle && !first_placed) {
                    first_placed = nearby[index];
                }
            }
            if (count > 4) {
                // The candidate must leave every window holding the four
                // placed activations that precede it there.
                moved = *first_placed + tfaw_;
            }
        }
        if (moved) {
            cycle = *moved;
            continue;
        }
        return cycle;
    }
}

std::uint64_t HbmDevice::place_column(
    const PseudoChannelState& pseudo_channel,
    bool write,
    std::uint64_t lower) const {
    const auto cas = write ? tcwl_ : tcl_;
    // Minimum distance from a placed column command of each type to this one
    // (when it follows) and from this one to a placed command it precedes.
    const auto after_read = write ? trtw_ : tccd_s_;
    const auto after_write = write ? tccd_s_ : tcwl_ + burst_cycles_ + twtr_s_;
    const auto before_read = write ? tcwl_ + burst_cycles_ + twtr_s_ : tccd_s_;
    const auto before_write = write ? tccd_s_ : trtw_;
    const auto reach = column_reach();
    const auto& columns = pseudo_channel.columns;
    const auto& slots = pseudo_channel.bus_slots;
    const auto by_cycle = [](const PlacedColumn& column, std::uint64_t value) {
        return column.cycle < value;
    };
    auto cycle = lower;
    for (;;) {
        const auto next = std::lower_bound(columns.begin(), columns.end(), cycle, by_cycle);
        bool moved = false;
        for (auto placed = next; placed != columns.begin();) {
            --placed;
            if (cycle - placed->cycle >= reach) {
                break;
            }
            const auto need = placed->write ? after_write : after_read;
            if (cycle - placed->cycle < need) {
                cycle = placed->cycle + need;
                moved = true;
                break;
            }
        }
        if (moved) {
            continue;
        }
        for (auto placed = next; placed != columns.end(); ++placed) {
            if (placed->cycle - cycle >= reach) {
                break;
            }
            const auto need = placed->write ? before_write : before_read;
            if (placed->cycle - cycle < need) {
                // No cycle between here and that command can satisfy the
                // gap, so continue past it.
                cycle = placed->cycle + (placed->write ? after_write : after_read);
                moved = true;
                break;
            }
        }
        if (moved) {
            continue;
        }
        const auto slot = cycle + cas;
        const auto busy = std::lower_bound(
            slots.begin(), slots.end(),
            slot + 1 > burst_cycles_ ? slot + 1 - burst_cycles_ : 0);
        if (busy != slots.end() && *busy < slot + burst_cycles_) {
            cycle = *busy + burst_cycles_ - cas;
            continue;
        }
        return cycle;
    }
}

// ---------------------------------------------------------------------------
// Scheduler
// ---------------------------------------------------------------------------

HbmDevice::FirstCommand HbmDevice::first_command(
    const PseudoChannelState& pseudo_channel,
    const QueuedRequest& queued) const {
    const auto& bank = pseudo_channel.banks[bank_index(queued.addr)];
    const auto& group = pseudo_channel.bank_groups[queued.addr.bank_group];
    const auto column = queued.op == Op::Read ? HbmCommand::RD : HbmCommand::WR;
    HbmCommand command = column;
    if (!bank.has_open_row) {
        command = HbmCommand::ACT;
    } else if (bank.open_row != queued.addr.row) {
        command = HbmCommand::PRE;
    }
    const auto index = command_index(command);
    const auto lower = std::max({
        queued.ready_cycle, pseudo_channel.service_floor_cycle,
        group.ready[index], bank.ready[index]});
    std::uint64_t issue = lower;
    switch (command) {
    case HbmCommand::ACT:
        issue = place_activation(pseudo_channel, lower);
        break;
    case HbmCommand::PRE:
        break;
    case HbmCommand::RD:
    case HbmCommand::WR:
        issue = place_column(pseudo_channel, command == HbmCommand::WR, lower);
        break;
    }
    return FirstCommand{.issue_cycle = issue, .row_hit = command == column};
}

std::size_t HbmDevice::pick_next(const PseudoChannelState& pseudo_channel) const {
    if (pseudo_channel.queue.size() == 1) {
        return 0;
    }
    const auto& oldest = pseudo_channel.queue.front();
    auto selected_preview = first_command(pseudo_channel, oldest);
    std::size_t selected = 0;
    for (std::size_t index = 1; index < pseudo_channel.queue.size(); ++index) {
        // Any candidate after a capped entry would bypass that entry once too
        // often. Tying the count cap to queue_depth gives each request at most
        // one full controller-window of younger priority.
        if (pseudo_channel.queue[index - 1].bypass_count >= config_.queue_depth) {
            break;
        }
        const auto& candidate = pseudo_channel.queue[index];
        if (candidate.arrival_ns - oldest.arrival_ns > config_.frfcfs_cap_ns) {
            break;
        }
        const auto preview = first_command(pseudo_channel, candidate);
        if (preview.issue_cycle < selected_preview.issue_cycle ||
            (preview.issue_cycle == selected_preview.issue_cycle &&
             preview.row_hit && !selected_preview.row_hit)) {
            selected = index;
            selected_preview = preview;
        }
    }
    return selected;
}

void HbmDevice::service_one(std::size_t pseudo_channel_index) {
    auto& pseudo_channel = pseudo_channels_[pseudo_channel_index];
    if (pseudo_channel.queue.empty()) {
        throw std::runtime_error("HBM scheduler serviced an empty queue");
    }
    prune_placements(pseudo_channel, normalization_floor(pseudo_channel));
    const auto picked = pick_next(pseudo_channel);
    for (std::size_t i = 0; i < picked; ++i) {
        if (pseudo_channel.queue[i].bypass_count >= config_.queue_depth) {
            throw std::runtime_error("HBM scheduler exceeded its bounded-bypass cap");
        }
        pseudo_channel.queue[i].bypass_count++;
    }
    const QueuedRequest entry = pseudo_channel.queue[picked];
    pseudo_channel.queue.erase(
        pseudo_channel.queue.begin() + static_cast<std::ptrdiff_t>(picked));
    auto child = service_request(pseudo_channel, pseudo_channel_index, entry);
    pseudo_channel_accesses_[pseudo_channel_index] = checked_add(
        pseudo_channel_accesses_[pseudo_channel_index], 1, "HBM pseudo-channel accesses");
    pseudo_channel_bus_busy_cycles_[pseudo_channel_index] += burst_cycles_;
    aggregate_child(entry.ticket, pseudo_channel_index, std::move(child));
}

void HbmDevice::aggregate_child(
    std::uint64_t ticket,
    std::size_t pseudo_channel_index,
    PhysicalCompletion child) {
    const auto found = pending_.find(ticket);
    if (found == pending_.end() || found->second.remaining_children == 0) {
        throw std::runtime_error("HBM completed an unknown internal burst");
    }
    auto& parent = found->second;
    auto& out = parent.completion;
    // Device totals represent physical work, so every burst child
    // contributes. The parent completion retains only its latency-critical
    // child's compact diagnostic; canonical additive work is accumulated
    // separately in stats_.stage_work so parallel children are neither lost
    // nor confused with elapsed wall time.
    stats_.stage_work += child.breakdown;
    const bool first = !parent.has_child_completion;
    parent.has_child_completion = true;
    if (first && parent.retain_diagnostics) {
        out.resource_path = child.resource_path;
        out.note = child.note;
    }
    out.start_ns = std::min(out.start_ns, child.start_ns);
    const bool critical = first || child.finish_ns > out.finish_ns ||
        (child.finish_ns == out.finish_ns &&
         pseudo_channel_index < parent.critical_pseudo_channel);
    out.finish_ns = std::max(out.finish_ns, child.finish_ns);
    out.physical_bytes = checked_add(
        out.physical_bytes, child.physical_bytes, "HBM aggregate physical bytes");
    if (critical) {
        out.breakdown = child.breakdown;
        parent.critical_pseudo_channel = pseudo_channel_index;
    }
    if (parent.retain_diagnostics) {
        out.spans.insert(out.spans.end(), child.spans.begin(), child.spans.end());
    }
    if (recording_.active && recording_.representative == pseudo_channel_index) {
        auto record = std::find_if(
            recording_.records.begin(), recording_.records.end(),
            [ticket](const ChildRecord& candidate) { return candidate.ticket == ticket; });
        if (record == recording_.records.end()) {
            recording_.records.push_back(ChildRecord{.ticket = ticket});
            record = recording_.records.end() - 1;
        }
        record->count++;
        record->physical_bytes = checked_add(
            record->physical_bytes, child.physical_bytes, "HBM replicated physical bytes");
        record->min_start_ns = std::min(record->min_start_ns, child.start_ns);
        if (child.finish_ns > record->max_finish_ns) {
            record->max_finish_ns = child.finish_ns;
            record->critical = child.breakdown;
        }
        record->work += child.breakdown;
    }
    parent.remaining_children--;
    if (parent.remaining_children != 0 || !parent.enqueue_complete) {
        return;
    }
    complete_parent(ticket, parent);
}

void HbmDevice::complete_parent(std::uint64_t ticket, PendingRequest& parent) {
    auto& out = parent.completion;
    if (!std::isfinite(out.start_ns)) {
        out.start_ns = out.arrival_ns;
    }
    if (parent.total_children > 1 && parent.retain_diagnostics) {
        out.resource_path = "hbm/split/" + std::to_string(parent.total_children) +
            "-bursts/" + std::to_string(parent.pseudo_channels) + "-pseudochannels";
        out.note = "split-burst-request";
    }
    completed_.emplace(ticket, std::move(out));
    pending_.erase(ticket);
}

PhysicalCompletion HbmDevice::service_request(
    PseudoChannelState& pseudo_channel,
    std::size_t pseudo_channel_index,
    const QueuedRequest& queued) {
    const auto flat_bank = bank_index(queued.addr);
    auto& bank = pseudo_channel.banks[flat_bank];
    auto& group = pseudo_channel.bank_groups[queued.addr.bank_group];
    const auto column = queued.op == Op::Read ? HbmCommand::RD : HbmCommand::WR;
    const auto cas = column == HbmCommand::RD ? tcl_ : tcwl_;

    PhysicalCompletion out;
    out.tier = Tier::HBM;
    out.op = queued.op;
    out.arrival_ns = queued.arrival_ns;
    out.logical_bytes = queued.bytes;
    out.physical_bytes = burst_bytes_;
    const bool retain_diagnostics =
        queued.trace.retain_completion_diagnostics || trace_spans_enabled(queued.trace);
    if (retain_diagnostics) {
        out.resource_path = queued.addr.path();
    }
    auto* spans = trace_spans_enabled(queued.trace) ? &out.spans : nullptr;
    static const std::string no_trace_entity;
    const auto& entity = retain_diagnostics ? out.resource_path : no_trace_entity;

    out.breakdown.address_mapping_ns = config_.address_mapping_ns;
    const double mapped_ns = queued.arrival_ns + config_.address_mapping_ns;
    add_trace_span(spans, "address_map", "mapping", entity, queued.arrival_ns, mapped_ns);
    // The request becomes command-eligible on the next command-clock edge.
    std::uint64_t ready = queued.ready_cycle;
    out.breakdown.scheduler_queue_wait_ns += std::max(0.0, ns(ready) - mapped_ns);
    add_trace_span(
        spans, "wait_command_clock", "queue", entity, mapped_ns, ns(ready), true,
        "request becomes command-eligible on the next CK edge");
    if (pseudo_channel.service_floor_cycle > ready) {
        out.breakdown.scheduler_queue_wait_ns += ns(pseudo_channel.service_floor_cycle - ready);
        add_trace_span(
            spans, "wait_controller", "queue", entity,
            ns(ready), ns(pseudo_channel.service_floor_cycle), true,
            "burst selection cannot precede an earlier controller decision");
        ready = pseudo_channel.service_floor_cycle;
    }

    std::optional<std::uint64_t> first_issue;
    bool issued_precharge = false;
    bool issued_activation = false;
    std::uint64_t bus_start = 0;
    std::uint64_t finish = 0;
    for (;;) {
        HbmCommand command = column;
        if (!bank.has_open_row) {
            command = HbmCommand::ACT;
        } else if (bank.open_row != queued.addr.row) {
            command = HbmCommand::PRE;
        }
        const auto index = command_index(command);
        std::uint64_t issue = ready;
        GateLevel level = GateLevel::None;
        const auto consider = [&issue, &level](std::uint64_t gate, GateLevel gate_level) {
            if (gate > issue) {
                issue = gate;
                level = gate_level;
            }
        };
        consider(group.ready[index], GateLevel::BankGroup);
        consider(bank.ready[index], GateLevel::Bank);
        std::uint64_t duration = 0;
        switch (command) {
        case HbmCommand::PRE:
            duration = trp_;
            break;
        case HbmCommand::ACT:
            consider(place_activation(pseudo_channel, issue), GateLevel::PseudoChannel);
            duration = column == HbmCommand::WR ? trcdwr_ : trcdrd_;
            break;
        case HbmCommand::RD:
        case HbmCommand::WR:
            consider(
                place_column(pseudo_channel, command == HbmCommand::WR, issue),
                GateLevel::PseudoChannel);
            duration = cas;
            break;
        }
        // Refreshes nominally due by the access's first command are issued
        // before it; a later command of the same access never has a refresh
        // inserted in front of it (the refresh is postponed behind the
        // access's precharge gate instead).
        if (!first_issue) {
            const auto refresh = apply_due_refreshes(
                pseudo_channel, pseudo_channel_index, flat_bank, issue, spans);
            if (refresh.closed_bank) {
                if (refresh.blocked_until > ready) {
                    out.breakdown.refresh_stall_ns += ns(refresh.blocked_until - ready);
                    add_trace_span(
                        spans, "wait_refresh", "queue", entity, ns(ready),
                        ns(refresh.blocked_until), true,
                        "bank row closed and blocked by refresh");
                    ready = refresh.blocked_until;
                }
                // Re-decide against the closed bank.
                continue;
            }
        }
        out.breakdown.scheduler_queue_wait_ns += ns(issue - ready);
        if (spans != nullptr) {
            add_trace_span(
                spans, std::string("wait_") + command_name(command), "queue", entity,
                ns(ready), ns(issue), true,
                std::string(command_name(command)) + " waits for " +
                    gate_reason(level, command));
        }
        if (!first_issue) {
            first_issue = issue;
        }
        switch (command) {
        case HbmCommand::PRE:
            out.breakdown.precharge_ns += ns(duration);
            stats_.precharges++;
            issued_precharge = true;
            break;
        case HbmCommand::ACT:
            out.breakdown.activation_ns += ns(duration);
            stats_.activations++;
            issued_activation = true;
            break;
        case HbmCommand::RD:
        case HbmCommand::WR:
            out.breakdown.command_ns += ns(duration);
            break;
        }
        if (spans != nullptr) {
            std::ostringstream detail;
            switch (command) {
            case HbmCommand::PRE:
                detail << "scope=bank closes_row";
                break;
            case HbmCommand::ACT:
                detail << "scope=row opens_row row=" << queued.addr.row
                       << " next_" << command_name(column) << "_wait="
                       << (column == HbmCommand::WR ? "tRCDWR" : "tRCDRD");
                break;
            case HbmCommand::RD:
            case HbmCommand::WR:
                detail << "scope=column accesses_column";
                break;
            }
            add_trace_span(
                spans, command_name(command), "hbm_command", entity,
                ns(issue), ns(issue + duration), true, detail.str());
        }
        apply_command_state(pseudo_channel, queued.addr, command, issue);
        ready = issue + duration;
        if (command == column) {
            bus_start = issue + cas;
            finish = bus_start + burst_cycles_;
            break;
        }
    }

    if (issued_precharge) {
        stats_.row_conflicts++;
        if (retain_diagnostics) out.note = "row-conflict";
    } else if (issued_activation) {
        stats_.row_misses++;
        if (retain_diagnostics) out.note = "row-miss";
    } else {
        stats_.row_hits++;
        if (retain_diagnostics) out.note = "row-hit";
    }
    out.breakdown.channel_transfer_ns = ns(burst_cycles_);
    out.start_ns = ns(*first_issue);
    out.finish_ns = ns(finish);
    pseudo_channel.service_floor_cycle = *first_issue;
    if (spans != nullptr) {
        add_trace_span(
            spans,
            queued.op == Op::Read ? "read_burst" : "write_burst",
            "hbm_bus",
            "stack" + std::to_string(queued.addr.stack) + "/ch" +
                std::to_string(queued.addr.channel) + "/pch" +
                std::to_string(queued.addr.pseudo_channel),
            ns(bus_start),
            ns(finish),
            true,
            std::to_string(out.physical_bytes) + "B");
    }
    if (queued.op == Op::Read) {
        stats_.read_bytes += out.physical_bytes;
    } else {
        stats_.write_bytes += out.physical_bytes;
    }
    bus_busy_cycles_ += burst_cycles_;
    stats_.bus_busy_ns = ns(bus_busy_cycles_);
    stats_.finish_ns = std::max(stats_.finish_ns, out.finish_ns);
    return out;
}

// ---------------------------------------------------------------------------
// Refresh
//
// Period k of every pseudo-channel starts at (k + 1) * tREFI. All-bank mode
// issues one REFab per period; same-bank mode issues banks_per_group REFsb
// commands per period, command j nominally at (k + 1) * tREFI + j * tREFI /
// banks_per_group and refreshing bank j of every bank group. A refresh
// command is issued at the latest of: its nominal cycle; for each open
// target bank, that bank's precharge gate plus tRP (the controller closes
// the row first); for each closed target bank, its refresh gate (tRP after
// its PRE, or the end of its previous refresh window); and tRREFD after the
// previous refresh command. It blocks its targets for tRFC/tRFCsb and
// leaves their rows closed. Refreshes are resolved lazily in service order,
// before the FIRST command of an access: every refresh nominally due at or
// before that command's issue cycle is issued first, and the access then
// re-decides against the closed bank. An access whose first command issued
// before the nominal cycle completes undisturbed and postpones the refresh
// behind its own precharge gate, which is what a controller does instead
// of refreshing between an ACT and its column command.
// ---------------------------------------------------------------------------

std::uint64_t HbmDevice::refresh_nominal_cycle(
    std::uint64_t period,
    std::uint32_t command) const {
    return (period + 1) * trefi_ + (command * trefi_) / refresh_commands_per_period_;
}

bool HbmDevice::refresh_targets_bank(
    std::uint32_t command,
    std::size_t bank_index) const {
    return !config_.same_bank_refresh ||
        bank_index % config_.banks_per_group == command;
}

void HbmDevice::resolve_next_refresh(
    PseudoChannelState& pseudo_channel,
    std::size_t pseudo_channel_index,
    std::vector<TraceSpan>* spans) {
    const auto command = pseudo_channel.refresh_command;
    const auto nominal = refresh_nominal_cycle(pseudo_channel.refresh_period, command);
    const auto for_each_target = [&](auto&& visit) {
        if (config_.same_bank_refresh) {
            for (std::uint32_t group = 0; group < config_.bank_groups_per_pseudo_channel; ++group) {
                visit(pseudo_channel.banks[
                    static_cast<std::size_t>(group) * config_.banks_per_group + command]);
            }
        } else {
            for (auto& bank : pseudo_channel.banks) {
                visit(bank);
            }
        }
    };
    std::uint64_t issue = nominal;
    for_each_target([&](const BankState& bank) {
        if (bank.has_open_row) {
            // The controller precharges the row at its PRE gate (never
            // before the nominal cycle) and refreshes tRP later.
            issue = std::max(
                issue,
                std::max(nominal, bank.ready[command_index(HbmCommand::PRE)]) + trp_);
        } else {
            issue = std::max(issue, bank.ready[kRefreshGate]);
        }
    });
    if (pseudo_channel.has_refresh_issue) {
        issue = std::max(issue, pseudo_channel.last_refresh_issue + trrefd_);
    }
    const auto end = issue + (config_.same_bank_refresh ? trfcsb_ : trfc_);
    for_each_target([&](BankState& bank) {
        if (bank.has_open_row) {
            bank.has_open_row = false;
            pseudo_channel.open_rows--;
        }
        bank.ready[command_index(HbmCommand::ACT)] = std::max(
            bank.ready[command_index(HbmCommand::ACT)], end);
        bank.ready[kRefreshGate] = std::max(bank.ready[kRefreshGate], end);
    });
    stats_.refresh_count++;
    if (spans != nullptr) {
        HbmAddress location;
        assign_pseudo_channel(location, pseudo_channel_index);
        add_trace_span(
            spans,
            config_.same_bank_refresh ? "REFsb" : "REFab",
            "refresh",
            "stack" + std::to_string(location.stack) + "/ch" +
                std::to_string(location.channel) + "/pch" +
                std::to_string(location.pseudo_channel),
            ns(issue),
            ns(end),
            true,
            config_.same_bank_refresh ?
                "same-bank refresh of bank " + std::to_string(command) +
                    " in every bank group" :
                std::string("all-bank refresh"));
    }
    pseudo_channel.has_refresh_issue = true;
    pseudo_channel.last_refresh_issue = issue;
    if (command + 1 == refresh_commands_per_period_) {
        pseudo_channel.refresh_command = 0;
        pseudo_channel.refresh_period++;
    } else {
        pseudo_channel.refresh_command = command + 1;
    }
}

HbmDevice::RefreshOutcome HbmDevice::apply_due_refreshes(
    PseudoChannelState& pseudo_channel,
    std::size_t pseudo_channel_index,
    std::size_t bank_index,
    std::uint64_t issue_cycle,
    std::vector<TraceSpan>* spans) {
    RefreshOutcome outcome;
    if (!config_.refresh_enabled) {
        return outcome;
    }
    const auto commands = refresh_commands_per_period_;
    const auto refresh_cycles = config_.same_bank_refresh ? trfcsb_ : trfc_;
    for (;;) {
        const auto nominal = refresh_nominal_cycle(
            pseudo_channel.refresh_period, pseudo_channel.refresh_command);
        if (nominal > issue_cycle) {
            break;
        }
        const bool targeted = refresh_targets_bank(pseudo_channel.refresh_command, bank_index);
        // Idle catch-up: with every row closed, every refresh gate at or
        // before the period's first nominal cycle, and the spacing rule
        // slack, each command of a whole elapsed period resolves exactly at
        // its nominal cycle. Skip such periods arithmetically; traced
        // requests keep the per-command path so every refresh event is
        // emitted.
        const auto gates_idle = [&]() {
            return std::all_of(
                pseudo_channel.banks.begin(), pseudo_channel.banks.end(),
                [nominal](const BankState& bank) {
                    return bank.ready[kRefreshGate] <= nominal;
                });
        };
        if (spans == nullptr && pseudo_channel.open_rows == 0 &&
            pseudo_channel.refresh_command == 0 &&
            (!pseudo_channel.has_refresh_issue ||
             pseudo_channel.last_refresh_issue + trrefd_ <= nominal) &&
            gates_idle()) {
            const auto last_command_offset =
                (static_cast<std::uint64_t>(commands - 1) * trefi_) / commands;
            if (issue_cycle >= last_command_offset + trefi_) {
                const auto elapsed_periods = (issue_cycle - last_command_offset) / trefi_;
                if (elapsed_periods > pseudo_channel.refresh_period) {
                    const auto skipped = elapsed_periods - pseudo_channel.refresh_period;
                    const auto last_period_start = elapsed_periods * trefi_;
                    for (std::size_t bank = 0; bank < pseudo_channel.banks.size(); ++bank) {
                        const auto command = config_.same_bank_refresh ?
                            static_cast<std::uint64_t>(bank % config_.banks_per_group) : 0;
                        const auto end = last_period_start +
                            (command * trefi_) / commands + refresh_cycles;
                        auto& gates = pseudo_channel.banks[bank].ready;
                        gates[command_index(HbmCommand::ACT)] =
                            std::max(gates[command_index(HbmCommand::ACT)], end);
                        gates[kRefreshGate] = std::max(gates[kRefreshGate], end);
                        if (bank == bank_index) {
                            outcome.closed_bank = true;
                            outcome.blocked_until = std::max(outcome.blocked_until, end);
                        }
                    }
                    stats_.refresh_count += skipped * commands;
                    pseudo_channel.refresh_period = elapsed_periods;
                    pseudo_channel.has_refresh_issue = true;
                    pseudo_channel.last_refresh_issue =
                        last_period_start + last_command_offset;
                    continue;
                }
            }
        }
        resolve_next_refresh(pseudo_channel, pseudo_channel_index, spans);
        if (targeted) {
            outcome.closed_bank = true;
            outcome.blocked_until = std::max(
                outcome.blocked_until,
                pseudo_channel.banks[bank_index].ready[command_index(HbmCommand::ACT)]);
        }
    }
    return outcome;
}

void HbmDevice::apply_command_state(
    PseudoChannelState& pseudo_channel,
    const HbmAddress& addr,
    HbmCommand command,
    std::uint64_t issue) {
    auto& bank = pseudo_channel.banks[bank_index(addr)];
    auto& group = pseudo_channel.bank_groups[addr.bank_group];
    const auto raise = [](std::uint64_t& gate, std::uint64_t value) {
        gate = std::max(gate, value);
    };
    constexpr auto kAct = static_cast<std::size_t>(HbmCommand::ACT);
    constexpr auto kPre = static_cast<std::size_t>(HbmCommand::PRE);
    constexpr auto kRd = static_cast<std::size_t>(HbmCommand::RD);
    constexpr auto kWr = static_cast<std::size_t>(HbmCommand::WR);
    const auto place_column_history = [&](bool write) {
        insert_sorted(
            pseudo_channel.columns,
            PlacedColumn{.cycle = issue, .write = write},
            [](const PlacedColumn& lhs, const PlacedColumn& rhs) {
                return lhs.cycle < rhs.cycle;
            });
        insert_sorted(
            pseudo_channel.bus_slots, issue + (write ? tcwl_ : tcl_),
            std::less<std::uint64_t>{});
    };
    switch (command) {
    case HbmCommand::PRE:
        if (bank.has_open_row) {
            bank.has_open_row = false;
            pseudo_channel.open_rows--;
        }
        raise(bank.ready[kAct], issue + trp_);
        // A refresh of a precharged bank waits tRP after its PRE.
        raise(bank.ready[kRefreshGate], issue + trp_);
        break;
    case HbmCommand::ACT:
        if (!bank.has_open_row) {
            bank.has_open_row = true;
            pseudo_channel.open_rows++;
        }
        bank.open_row = addr.row;
        raise(bank.ready[kPre], issue + tras_);
        raise(bank.ready[kAct], issue + std::max(trc_, tras_ + trp_));
        raise(bank.ready[kRd], issue + trcdrd_);
        raise(bank.ready[kWr], issue + trcdwr_);
        raise(group.ready[kAct], issue + trrd_l_);
        insert_sorted(pseudo_channel.activations, issue, std::less<std::uint64_t>{});
        break;
    case HbmCommand::RD:
        raise(bank.ready[kPre], issue + trtp_);
        // Column commands stay in order inside a bank and its bank group.
        raise(bank.ready[kRd], issue + tccd_l_);
        raise(bank.ready[kWr], issue + tccd_l_);
        raise(group.ready[kRd], issue + tccd_l_);
        raise(group.ready[kWr], issue + tccd_l_);
        place_column_history(false);
        break;
    case HbmCommand::WR:
        raise(bank.ready[kPre], issue + tcwl_ + burst_cycles_ + twr_);
        raise(bank.ready[kRd], issue + tcwl_ + burst_cycles_ + twtr_l_);
        raise(bank.ready[kWr], issue + tccd_l_);
        raise(group.ready[kWr], issue + tccd_l_);
        raise(group.ready[kRd], issue + tcwl_ + burst_cycles_ + twtr_l_);
        place_column_history(true);
        break;
    }
}

void HbmDevice::refresh_parallel_stats() const {
    stats_.pseudo_channels = pseudo_channels_.size();
    stats_.active_pseudo_channels = 0;
    stats_.max_pseudo_channel_accesses = 0;
    stats_.max_pseudo_channel_busy_ns = 0.0;
    stats_.avg_active_pseudo_channel_busy_ns = 0.0;
    double active_busy_ns = 0.0;
    for (std::size_t index = 0; index < pseudo_channels_.size(); ++index) {
        const auto accesses = pseudo_channel_accesses_[index];
        const auto busy_ns = ns(pseudo_channel_bus_busy_cycles_[index]);
        if (accesses == 0 && busy_ns <= 0.0) {
            continue;
        }
        ++stats_.active_pseudo_channels;
        active_busy_ns += busy_ns;
        stats_.max_pseudo_channel_busy_ns =
            std::max(stats_.max_pseudo_channel_busy_ns, busy_ns);
        stats_.max_pseudo_channel_accesses =
            std::max(stats_.max_pseudo_channel_accesses, accesses);
    }
    if (stats_.active_pseudo_channels != 0) {
        stats_.avg_active_pseudo_channel_busy_ns =
            active_busy_ns / static_cast<double>(stats_.active_pseudo_channels);
    }
}

} // namespace hbfsim::physical::hbm
