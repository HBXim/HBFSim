#include "physical/address_heatmap.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <utility>

namespace hbfsim::physical {
namespace {

[[nodiscard]] std::size_t domain_index(AddressDomain domain) {
    const auto index = static_cast<std::size_t>(domain);
    if (index >= kAddressDomainCount) {
        throw std::invalid_argument("invalid address heatmap domain");
    }
    return index;
}

[[nodiscard]] std::size_t source_index(HeatmapTrafficSource source) {
    const auto index = static_cast<std::size_t>(source);
    if (index >= kHeatmapTrafficSourceCount) {
        throw std::invalid_argument("invalid address heatmap traffic source");
    }
    return index;
}

[[nodiscard]] std::uint64_t checked_add(
    std::uint64_t lhs,
    std::uint64_t rhs,
    std::string_view label) {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
        throw std::overflow_error(std::string(label) + " overflows uint64_t");
    }
    return lhs + rhs;
}

[[nodiscard]] std::uint64_t checked_mul(
    std::uint64_t lhs,
    std::uint64_t rhs,
    std::string_view label) {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
        throw std::overflow_error(std::string(label) + " overflows uint64_t");
    }
    return lhs * rhs;
}

void require_addable(
    std::uint64_t current,
    std::uint64_t increment,
    std::string_view label) {
    (void)checked_add(current, increment, label);
}

[[nodiscard]] std::uint64_t& byte_counter(
    AddressTrafficCounters& counters,
    TrafficDirection direction) {
    switch (direction) {
    case TrafficDirection::Read:
        return counters.read_bytes;
    case TrafficDirection::Write:
        return counters.write_bytes;
    case TrafficDirection::Erase:
        return counters.erase_bytes;
    }
    throw std::invalid_argument("invalid address heatmap traffic direction");
}

[[nodiscard]] std::uint64_t& access_counter(
    AddressTrafficCounters& counters,
    TrafficDirection direction) {
    switch (direction) {
    case TrafficDirection::Read:
        return counters.read_accesses;
    case TrafficDirection::Write:
        return counters.write_accesses;
    case TrafficDirection::Erase:
        return counters.erase_accesses;
    }
    throw std::invalid_argument("invalid address heatmap traffic direction");
}

[[nodiscard]] AddressBoundary scaled_boundary(
    AddressBoundary size_bytes,
    std::size_t bin_count,
    std::size_t boundary) {
    if (boundary > bin_count || bin_count == 0) {
        throw std::invalid_argument(
            "address heatmap scaled-boundary arguments are invalid");
    }
    if (boundary == 0) {
        return AddressBoundary{0};
    }
    if (boundary == bin_count) {
        return size_bytes;
    }

    // floor(size * boundary / bin_count), rearranged as
    // floor(size / bin_count) * boundary +
    // floor((size % bin_count) * boundary / bin_count).  This stays in
    // portable uint64_t arithmetic even when size is the exclusive 2^64
    // boundary.  For an interior boundary, the result is strictly below size
    // and therefore always a finite address.
    std::uint64_t quotient = 0;
    std::uint64_t remainder = 0;
    if (size_bytes.is_full_address_space_end()) {
        const auto maximum = std::numeric_limits<std::uint64_t>::max();
        quotient = maximum / bin_count;
        const auto remainder_plus_one = maximum % bin_count + 1;
        quotient = checked_add(
            quotient,
            remainder_plus_one / bin_count,
            "address heatmap full-space quotient");
        remainder = remainder_plus_one % bin_count;
    } else {
        quotient = size_bytes.finite_value() / bin_count;
        remainder = size_bytes.finite_value() % bin_count;
    }
    const auto integral = checked_mul(
        quotient, boundary, "address heatmap scaled boundary");
    const auto fractional = checked_mul(
        remainder, boundary, "address heatmap scaled-boundary remainder") /
        bin_count;
    return AddressBoundary{checked_add(
        integral, fractional, "address heatmap scaled boundary")};
}

[[nodiscard]] std::size_t bin_for_address(
    const std::vector<AddressHeatmapBin>& bins,
    std::uint64_t address) {
    // Find the first half-open bin whose exclusive end is above address.  A
    // boundary search avoids multiplying a uint64 address by the bin count.
    std::size_t low = 0;
    std::size_t high = bins.size();
    const AddressBoundary target{address};
    while (low < high) {
        const auto middle = low + (high - low) / 2;
        if (target < bins[middle].end) {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    return low;
}

[[nodiscard]] std::uint64_t overlap_bytes(
    const AddressHeatmapBin& bin,
    std::uint64_t request_begin,
    AddressBoundary request_end) {
    const auto begin = std::max(bin.begin, request_begin);
    const auto end = std::min(bin.end, request_end);
    if (AddressBoundary{begin} >= end) {
        return 0;
    }
    if (end.is_full_address_space_end()) {
        // A traffic record is at most UINT64_MAX bytes, so any record ending
        // at 2^64 begins above zero and this distance remains representable.
        if (begin == 0) {
            throw std::overflow_error(
                "address heatmap overlap spans the entire uint64 address space");
        }
        return checked_add(
            std::numeric_limits<std::uint64_t>::max() - begin,
            1,
            "address heatmap request/bin overlap");
    }
    return end.finite_value() - begin;
}

void write_boundary(std::ostream& output, AddressBoundary boundary) {
    output << address_boundary_decimal(boundary);
}

void write_json_string(std::ostream& output, std::string_view value) {
    static constexpr char hex[] = "0123456789abcdef";
    output.put('"');
    for (const unsigned char byte : value) {
        switch (byte) {
        case '"':
            output << "\\\"";
            break;
        case '\\':
            output << "\\\\";
            break;
        case '\b':
            output << "\\b";
            break;
        case '\f':
            output << "\\f";
            break;
        case '\n':
            output << "\\n";
            break;
        case '\r':
            output << "\\r";
            break;
        case '\t':
            output << "\\t";
            break;
        default:
            if (byte < 0x20) {
                output << "\\u00" << hex[byte >> 4] << hex[byte & 0x0f];
            } else {
                output.put(static_cast<char>(byte));
            }
        }
    }
    output.put('"');
}

void write_counters(std::ostream& output, const AddressTrafficCounters& counters) {
    output << "{\"read_bytes\":" << counters.read_bytes
           << ",\"write_bytes\":" << counters.write_bytes
           << ",\"erase_bytes\":" << counters.erase_bytes
           << ",\"read_accesses\":" << counters.read_accesses
           << ",\"write_accesses\":" << counters.write_accesses
           << ",\"erase_accesses\":" << counters.erase_accesses << '}';
}

template <typename Selector>
void write_source_array(
    std::ostream& output,
    const AddressHeatmapBin& bin,
    Selector selector) {
    output.put('[');
    for (std::size_t source = 0; source < kHeatmapTrafficSourceCount; ++source) {
        if (source != 0) {
            output.put(',');
        }
        output << selector(bin.by_source[source]);
    }
    output.put(']');
}

} // namespace

AddressBoundary address_exclusive_end(
    std::uint64_t address,
    std::uint64_t bytes) {
    if (bytes == 0) {
        throw std::invalid_argument(
            "address heatmap range must contain at least one byte");
    }
    const auto remaining =
        std::numeric_limits<std::uint64_t>::max() - address;
    if (bytes - 1 > remaining) {
        throw std::out_of_range(
            "address heatmap range exceeds the uint64 address space");
    }
    if (bytes > remaining) {
        return AddressBoundary::full_address_space_end();
    }
    return AddressBoundary{address + bytes};
}

std::string address_boundary_decimal(AddressBoundary boundary) {
    if (boundary.is_full_address_space_end()) {
        return "18446744073709551616"; // 2^64
    }
    return std::to_string(boundary.finite_value());
}

const char* to_string(AddressDomain domain) {
    switch (domain) {
    case AddressDomain::WorkloadLogical:
        return "workload_logical";
    case AddressDomain::HbmPhysical:
        return "hbm_physical";
    case AddressDomain::HbfLogical:
        return "hbf_logical";
    case AddressDomain::HbfPhysical:
        return "hbf_physical";
    case AddressDomain::ExternalPhysical:
        return "external_physical";
    case AddressDomain::Count:
        break;
    }
    throw std::invalid_argument("invalid address heatmap domain");
}

const char* to_string(HeatmapTrafficSource source) {
    switch (source) {
    case HeatmapTrafficSource::Workload:
        return "workload";
    case HeatmapTrafficSource::Direct:
        return "direct";
    case HeatmapTrafficSource::Mapping:
        return "mapping";
    case HeatmapTrafficSource::Prepopulate:
        return "prepopulate";
    case HeatmapTrafficSource::CooperativeBuffer:
        return "cooperative_buffer";
    case HeatmapTrafficSource::DemandFill:
        return "demand_fill";
    case HeatmapTrafficSource::PrefetchFill:
        return "prefetch_fill";
    case HeatmapTrafficSource::StreamingInstall:
        return "streaming_install";
    case HeatmapTrafficSource::Destage:
        return "destage";
    case HeatmapTrafficSource::GarbageCollection:
        return "garbage_collection";
    case HeatmapTrafficSource::Maintenance:
        return "maintenance";
    case HeatmapTrafficSource::Count:
        break;
    }
    throw std::invalid_argument("invalid address heatmap traffic source");
}

const char* to_string(AddressRegionKind kind) {
    switch (kind) {
    case AddressRegionKind::Workload:
        return "workload";
    case AddressRegionKind::CooperativeBuffer:
        return "cooperative_buffer";
    case AddressRegionKind::LayerBuffer:
        return "layer_buffer";
    case AddressRegionKind::StaticData:
        return "static_data";
    case AddressRegionKind::MappedData:
        return "mapped_data";
    case AddressRegionKind::Reserved:
        return "reserved";
    case AddressRegionKind::Other:
        return "other";
    }
    throw std::invalid_argument("invalid address heatmap region kind");
}

AddressHeatmap::AddressHeatmap(AddressHeatmapConfig config)
    : bin_count_(config.bin_count) {
    if (bin_count_ == 0 || bin_count_ > kMaxAddressHeatmapBins) {
        throw std::invalid_argument(
            "address heatmap bin count must be in [1, 8192]");
    }

    std::array<bool, kAddressDomainCount> seen{};
    for (auto& domain_config : config.domains) {
        const auto index = domain_index(domain_config.domain);
        if (seen[index]) {
            throw std::invalid_argument(
                "address heatmap config repeats domain " +
                std::string(to_string(domain_config.domain)));
        }
        seen[index] = true;
        const auto domain_size = domain_config.size_bytes;
        if (domain_size < AddressBoundary{static_cast<std::uint64_t>(bin_count_)}) {
            throw std::invalid_argument(
                "address heatmap domain size must be at least the bin count");
        }

        std::sort(
            domain_config.regions.begin(),
            domain_config.regions.end(),
            [](const AddressRegion& lhs, const AddressRegion& rhs) {
                return std::tie(lhs.begin, lhs.end, lhs.kind, lhs.name) <
                    std::tie(rhs.begin, rhs.end, rhs.kind, rhs.name);
            });
        for (const auto& region : domain_config.regions) {
            if (region.name.empty()) {
                throw std::invalid_argument(
                    "address heatmap region name must not be empty");
            }
            if (AddressBoundary{region.begin} >= region.end ||
                region.end > domain_config.size_bytes) {
                throw std::invalid_argument(
                    "address heatmap region lies outside its domain");
            }
            (void)to_string(region.kind);
        }

        auto& state = domains_[index];
        state.domain = domain_config.domain;
        state.size_bytes = domain_config.size_bytes;
        state.regions = std::move(domain_config.regions);
        state.bins.resize(bin_count_);
        for (std::size_t bin = 0; bin < bin_count_; ++bin) {
            const auto begin = scaled_boundary(domain_size, bin_count_, bin);
            const auto end = scaled_boundary(domain_size, bin_count_, bin + 1);
            if (begin.is_full_address_space_end()) {
                throw std::runtime_error(
                    "address heatmap constructed a non-finite bin begin");
            }
            state.bins[bin].begin = begin.finite_value();
            state.bins[bin].end = end;
            if (begin >= end) {
                throw std::runtime_error(
                    "address heatmap constructed an empty bin");
            }
        }
    }
    if (std::find(seen.begin(), seen.end(), false) != seen.end()) {
        throw std::invalid_argument(
            "address heatmap config must define all five address domains");
    }
}

void AddressHeatmap::record(const AddressTrafficRecord& traffic) {
    const auto domain_slot = domain_index(traffic.domain);
    const auto source_slot = source_index(traffic.source);
    // Validate explicitly because an enum can be constructed from any byte.
    AddressTrafficCounters direction_check;
    (void)byte_counter(direction_check, traffic.direction);
    (void)access_counter(direction_check, traffic.direction);
    if (traffic.bytes == 0) {
        throw std::invalid_argument(
            "address heatmap traffic record must contain at least one byte");
    }

    auto& state = domains_[domain_slot];
    const auto request_end = address_exclusive_end(
        traffic.address, traffic.bytes);
    if (request_end > state.size_bytes) {
        throw std::out_of_range(
            "address heatmap traffic record lies outside its domain");
    }
    const auto first_bin = bin_for_address(
        state.bins, traffic.address);
    const auto last_address = request_end.is_full_address_space_end() ?
        std::numeric_limits<std::uint64_t>::max() :
        request_end.finite_value() - 1;
    const auto last_bin = bin_for_address(
        state.bins, last_address);
    if (first_bin >= bin_count_ || last_bin >= bin_count_ ||
        first_bin > last_bin) {
        throw std::runtime_error("address heatmap bin lookup is inconsistent");
    }

    // Preflight every counter before mutating any state. This second pass is
    // intentionally recomputed instead of allocating a per-record overlap
    // vector, so recording traffic never grows memory with trace length.
    require_addable(
        byte_counter(state.total, traffic.direction),
        traffic.bytes,
        "address heatmap domain bytes");
    require_addable(
        access_counter(state.total, traffic.direction),
        1,
        "address heatmap domain accesses");
    require_addable(
        byte_counter(state.by_source[source_slot], traffic.direction),
        traffic.bytes,
        "address heatmap source bytes");
    require_addable(
        access_counter(state.by_source[source_slot], traffic.direction),
        1,
        "address heatmap source accesses");

    std::uint64_t distributed_bytes = 0;
    for (std::size_t bin = first_bin; bin <= last_bin; ++bin) {
        auto& bucket = state.bins[bin];
        const auto overlap = overlap_bytes(
            bucket, traffic.address, request_end);
        if (overlap == 0) {
            throw std::runtime_error(
                "address heatmap selected a bin with zero request overlap");
        }
        distributed_bytes = checked_add(
            distributed_bytes,
            overlap,
            "address heatmap distributed bytes");
        require_addable(
            byte_counter(bucket.total, traffic.direction),
            overlap,
            "address heatmap bin bytes");
        require_addable(
            access_counter(bucket.total, traffic.direction),
            1,
            "address heatmap bin accesses");
        require_addable(
            byte_counter(bucket.by_source[source_slot], traffic.direction),
            overlap,
            "address heatmap source-bin bytes");
        require_addable(
            access_counter(bucket.by_source[source_slot], traffic.direction),
            1,
            "address heatmap source-bin accesses");
    }
    if (distributed_bytes != traffic.bytes) {
        throw std::runtime_error(
            "address heatmap bin overlap does not conserve request bytes");
    }

    byte_counter(state.total, traffic.direction) += traffic.bytes;
    ++access_counter(state.total, traffic.direction);
    byte_counter(state.by_source[source_slot], traffic.direction) += traffic.bytes;
    ++access_counter(state.by_source[source_slot], traffic.direction);
    for (std::size_t bin = first_bin; bin <= last_bin; ++bin) {
        auto& bucket = state.bins[bin];
        const auto overlap = overlap_bytes(
            bucket, traffic.address, request_end);
        byte_counter(bucket.total, traffic.direction) += overlap;
        ++access_counter(bucket.total, traffic.direction);
        byte_counter(bucket.by_source[source_slot], traffic.direction) += overlap;
        ++access_counter(bucket.by_source[source_slot], traffic.direction);
    }
}

void AddressHeatmap::record_contiguous_accesses(
    const AddressTrafficRecord& first,
    std::uint64_t access_count) {
    const auto domain_slot = domain_index(first.domain);
    const auto source_slot = source_index(first.source);
    AddressTrafficCounters direction_check;
    (void)byte_counter(direction_check, first.direction);
    (void)access_counter(direction_check, first.direction);
    if (first.bytes == 0 || access_count == 0) {
        throw std::invalid_argument(
            "contiguous address heatmap accesses require positive size/count");
    }
    const auto total_bytes = checked_mul(
        first.bytes,
        access_count,
        "contiguous address heatmap access bytes");

    auto& state = domains_[domain_slot];
    const auto request_end = address_exclusive_end(first.address, total_bytes);
    if (request_end > state.size_bytes) {
        throw std::out_of_range(
            "contiguous address heatmap accesses lie outside their domain");
    }
    const auto first_bin = bin_for_address(state.bins, first.address);
    const auto last_address = request_end.is_full_address_space_end() ?
        std::numeric_limits<std::uint64_t>::max() :
        request_end.finite_value() - 1;
    const auto last_bin = bin_for_address(state.bins, last_address);
    if (first_bin >= bin_count_ || last_bin >= bin_count_ ||
        first_bin > last_bin) {
        throw std::runtime_error(
            "contiguous address heatmap bin lookup is inconsistent");
    }

    require_addable(
        byte_counter(state.total, first.direction),
        total_bytes,
        "contiguous address heatmap domain bytes");
    require_addable(
        access_counter(state.total, first.direction),
        access_count,
        "contiguous address heatmap domain accesses");
    require_addable(
        byte_counter(state.by_source[source_slot], first.direction),
        total_bytes,
        "contiguous address heatmap source bytes");
    require_addable(
        access_counter(state.by_source[source_slot], first.direction),
        access_count,
        "contiguous address heatmap source accesses");

    const auto bin_contribution = [&] (const AddressHeatmapBin& bucket) {
        const auto overlap = overlap_bytes(
            bucket,
            first.address,
            request_end);
        if (overlap == 0) {
            throw std::runtime_error(
                "contiguous address heatmap selected a zero-overlap bin");
        }
        const auto overlap_begin = std::max(bucket.begin, first.address);
        const auto overlap_end = std::min(bucket.end, request_end);
        const auto overlap_last = overlap_end.is_full_address_space_end() ?
            std::numeric_limits<std::uint64_t>::max() :
            overlap_end.finite_value() - 1;
        const auto first_access =
            (overlap_begin - first.address) / first.bytes;
        const auto last_access =
            (overlap_last - first.address) / first.bytes;
        return std::pair{
            overlap,
            checked_add(
                last_access - first_access,
                1,
                "contiguous address heatmap bin accesses")};
    };

    std::uint64_t distributed_bytes = 0;
    for (std::size_t bin = first_bin; bin <= last_bin; ++bin) {
        auto& bucket = state.bins[bin];
        const auto [overlap, accesses] = bin_contribution(bucket);
        distributed_bytes = checked_add(
            distributed_bytes,
            overlap,
            "contiguous address heatmap distributed bytes");
        require_addable(
            byte_counter(bucket.total, first.direction),
            overlap,
            "contiguous address heatmap bin bytes");
        require_addable(
            access_counter(bucket.total, first.direction),
            accesses,
            "contiguous address heatmap bin accesses");
        require_addable(
            byte_counter(bucket.by_source[source_slot], first.direction),
            overlap,
            "contiguous address heatmap source-bin bytes");
        require_addable(
            access_counter(bucket.by_source[source_slot], first.direction),
            accesses,
            "contiguous address heatmap source-bin accesses");
    }
    if (distributed_bytes != total_bytes) {
        throw std::runtime_error(
            "contiguous address heatmap bin overlap does not conserve bytes");
    }

    byte_counter(state.total, first.direction) += total_bytes;
    access_counter(state.total, first.direction) += access_count;
    byte_counter(state.by_source[source_slot], first.direction) += total_bytes;
    access_counter(state.by_source[source_slot], first.direction) += access_count;
    for (std::size_t bin = first_bin; bin <= last_bin; ++bin) {
        auto& bucket = state.bins[bin];
        const auto [overlap, accesses] = bin_contribution(bucket);
        byte_counter(bucket.total, first.direction) += overlap;
        access_counter(bucket.total, first.direction) += accesses;
        byte_counter(bucket.by_source[source_slot], first.direction) += overlap;
        access_counter(bucket.by_source[source_slot], first.direction) += accesses;
    }
}

const AddressDomainHeatmap& AddressHeatmap::domain(AddressDomain domain) const {
    return domains_[domain_index(domain)];
}

AddressHeatmapSnapshot AddressHeatmap::snapshot() const {
    return AddressHeatmapSnapshot{
        .bin_count = bin_count_,
        .domains = domains_,
    };
}

namespace {

using CounterMember = std::uint64_t AddressTrafficCounters::*;

constexpr std::array<CounterMember, 6> kCounterMembers{{
    &AddressTrafficCounters::read_bytes,
    &AddressTrafficCounters::write_bytes,
    &AddressTrafficCounters::erase_bytes,
    &AddressTrafficCounters::read_accesses,
    &AddressTrafficCounters::write_accesses,
    &AddressTrafficCounters::erase_accesses,
}};

void validate_counter_presence(const AddressTrafficCounters& counters) {
    if ((counters.read_bytes == 0) != (counters.read_accesses == 0) ||
        (counters.write_bytes == 0) != (counters.write_accesses == 0) ||
        (counters.erase_bytes == 0) != (counters.erase_accesses == 0)) {
        throw std::invalid_argument(
            "address heatmap bytes and accesses disagree on traffic presence");
    }
}

template <typename Range, typename Selector>
[[nodiscard]] bool sums_exactly(
    const Range& values,
    std::uint64_t declared,
    Selector selector) {
    std::uint64_t sum = 0;
    for (const auto& value : values) {
        const auto increment = selector(value);
        if (increment > std::numeric_limits<std::uint64_t>::max() - sum) {
            return false;
        }
        sum += increment;
    }
    return sum == declared;
}

template <typename Range, typename Selector>
[[nodiscard]] bool has_valid_access_fanout(
    const Range& bins,
    std::uint64_t declared,
    std::size_t bin_count,
    Selector selector) {
    // Check declared <= sum <= declared * bin_count without requiring a
    // non-standard wide integer.  Track sum divided by bin_count as a
    // quotient/remainder pair; at most bin_count uint64 counters are added,
    // so the quotient itself remains representable.
    std::uint64_t missing_from_lower_bound = declared;
    std::uint64_t quotient = 0;
    std::size_t remainder = 0;
    for (const auto& bin : bins) {
        const auto increment = selector(bin);
        if (increment >= missing_from_lower_bound) {
            missing_from_lower_bound = 0;
        } else {
            missing_from_lower_bound -= increment;
        }

        const auto quotient_increment = increment / bin_count;
        if (quotient_increment >
            std::numeric_limits<std::uint64_t>::max() - quotient) {
            return false;
        }
        quotient += quotient_increment;
        const auto combined_remainder =
            remainder + static_cast<std::size_t>(increment % bin_count);
        const auto carry = combined_remainder / bin_count;
        if (carry > std::numeric_limits<std::uint64_t>::max() - quotient) {
            return false;
        }
        quotient += carry;
        remainder = combined_remainder % bin_count;
    }
    if (missing_from_lower_bound != 0) {
        return false;
    }
    return quotient < declared ||
        (quotient == declared && remainder == 0);
}

void validate_json_snapshot(const AddressHeatmapSnapshot& snapshot) {
    if (snapshot.bin_count == 0 ||
        snapshot.bin_count > kMaxAddressHeatmapBins) {
        throw std::invalid_argument(
            "address heatmap snapshot bin count must be in [1, 8192]");
    }
    for (std::size_t domain_slot = 0;
         domain_slot < kAddressDomainCount;
         ++domain_slot) {
        const auto& state = snapshot.domains[domain_slot];
        if (state.domain != static_cast<AddressDomain>(domain_slot)) {
            throw std::invalid_argument(
                "address heatmap snapshot domains are not in canonical order");
        }
        const auto domain_size = state.size_bytes;
        if (domain_size < AddressBoundary{
                static_cast<std::uint64_t>(snapshot.bin_count)} ||
            state.bins.size() != snapshot.bin_count) {
            throw std::invalid_argument(
                "address heatmap snapshot has an invalid domain extent or bin count");
        }
        validate_counter_presence(state.total);
        for (const auto& source : state.by_source) {
            validate_counter_presence(source);
        }
        for (std::size_t bin = 0; bin < state.bins.size(); ++bin) {
            const auto& bucket = state.bins[bin];
            const auto expected_begin = scaled_boundary(
                domain_size, snapshot.bin_count, bin);
            const auto expected_end = scaled_boundary(
                domain_size, snapshot.bin_count, bin + 1);
            if (expected_begin.is_full_address_space_end() ||
                bucket.begin != expected_begin.finite_value() ||
                bucket.end != expected_end) {
                throw std::invalid_argument(
                    "address heatmap snapshot bins do not use canonical boundaries");
            }
            validate_counter_presence(bucket.total);
            for (const auto& source : bucket.by_source) {
                validate_counter_presence(source);
            }
        }
        const auto region_less = [](const AddressRegion& lhs,
                                    const AddressRegion& rhs) {
            return std::tie(lhs.begin, lhs.end, lhs.kind, lhs.name) <
                std::tie(rhs.begin, rhs.end, rhs.kind, rhs.name);
        };
        if (!std::is_sorted(
                state.regions.begin(), state.regions.end(), region_less)) {
            throw std::invalid_argument(
                "address heatmap snapshot regions are not canonical");
        }
        for (const auto& region : state.regions) {
            if (region.name.empty() ||
                AddressBoundary{region.begin} >= region.end ||
                region.end > state.size_bytes) {
                throw std::invalid_argument(
                    "address heatmap snapshot contains an invalid region");
            }
            (void)to_string(region.kind);
        }

        for (std::size_t metric = 0; metric < kCounterMembers.size(); ++metric) {
            const auto member = kCounterMembers[metric];
            const auto declared = state.total.*member;
            for (const auto& bucket : state.bins) {
                if (!sums_exactly(
                        bucket.by_source,
                        bucket.total.*member,
                        [member](const AddressTrafficCounters& counters) {
                            return counters.*member;
                        })) {
                    throw std::invalid_argument(
                        "address heatmap snapshot bin source accounting diverges");
                }
            }
            if (!sums_exactly(
                    state.by_source,
                    declared,
                    [member](const AddressTrafficCounters& counters) {
                        return counters.*member;
                    })) {
                throw std::invalid_argument(
                    "address heatmap snapshot domain source accounting diverges");
            }
            if (metric < 3) {
                if (!sums_exactly(
                        state.bins,
                        declared,
                        [member](const AddressHeatmapBin& bucket) {
                            return bucket.total.*member;
                        })) {
                    throw std::invalid_argument(
                        "address heatmap snapshot does not conserve bytes across bins");
                }
            } else if (!has_valid_access_fanout(
                           state.bins,
                           declared,
                           snapshot.bin_count,
                           [member](const AddressHeatmapBin& bucket) {
                               return bucket.total.*member;
                           })) {
                throw std::invalid_argument(
                    "address heatmap snapshot access fan-out is inconsistent");
            }
            for (std::size_t source = 0;
                 source < kHeatmapTrafficSourceCount;
                ++source) {
                const auto source_declared = state.by_source[source].*member;
                if (metric < 3) {
                    if (!sums_exactly(
                            state.bins,
                            source_declared,
                            [source, member](const AddressHeatmapBin& bucket) {
                                return bucket.by_source[source].*member;
                            })) {
                        throw std::invalid_argument(
                            "address heatmap snapshot source bytes do not conserve");
                    }
                } else if (!has_valid_access_fanout(
                               state.bins,
                               source_declared,
                               snapshot.bin_count,
                               [source, member](
                                   const AddressHeatmapBin& bucket) {
                                   return bucket.by_source[source].*member;
                               })) {
                    throw std::invalid_argument(
                        "address heatmap snapshot source access fan-out is inconsistent");
                }
            }
        }
    }
}

void write_json_impl(
    std::ostream& output,
    std::size_t bin_count,
    const std::array<AddressDomainHeatmap, kAddressDomainCount>& domains) {
    output << "{\"schema\":\"hbfsim.address_heatmap.v1\",\"bin_count\":"
           << bin_count << ",\"traffic_sources\":[";
    for (std::size_t source = 0; source < kHeatmapTrafficSourceCount; ++source) {
        if (source != 0) {
            output.put(',');
        }
        write_json_string(
            output, to_string(static_cast<HeatmapTrafficSource>(source)));
    }
    output << "],\"domains\":[";
    for (std::size_t domain_slot = 0;
         domain_slot < kAddressDomainCount;
         ++domain_slot) {
        if (domain_slot != 0) {
            output.put(',');
        }
        const auto& state = domains[domain_slot];
        output << "{\"domain\":";
        write_json_string(output, to_string(state.domain));
        output << ",\"size_bytes\":";
        write_boundary(output, state.size_bytes);
        output << ",\"totals\":";
        write_counters(output, state.total);
        output << ",\"source_totals\":[";
        for (std::size_t source = 0;
             source < kHeatmapTrafficSourceCount;
             ++source) {
            if (source != 0) {
                output.put(',');
            }
            output << "{\"source\":";
            write_json_string(
                output, to_string(static_cast<HeatmapTrafficSource>(source)));
            const auto& counters = state.by_source[source];
            output << ",\"read_bytes\":" << counters.read_bytes
                   << ",\"write_bytes\":" << counters.write_bytes
                   << ",\"erase_bytes\":" << counters.erase_bytes
                   << ",\"read_accesses\":" << counters.read_accesses
                   << ",\"write_accesses\":" << counters.write_accesses
                   << ",\"erase_accesses\":" << counters.erase_accesses
                   << '}';
        }
        output << "],\"regions\":[";
        for (std::size_t region = 0; region < state.regions.size(); ++region) {
            if (region != 0) {
                output.put(',');
            }
            const auto& item = state.regions[region];
            output << "{\"name\":";
            write_json_string(output, item.name);
            output << ",\"kind\":";
            write_json_string(output, to_string(item.kind));
            output << ",\"begin\":" << item.begin
                   << ",\"end\":";
            write_boundary(output, item.end);
            output << '}';
        }
        output << "],\"bins\":[";
        for (std::size_t bin = 0; bin < state.bins.size(); ++bin) {
            if (bin != 0) {
                output.put(',');
            }
            const auto& bucket = state.bins[bin];
            output << "{\"begin\":" << bucket.begin
                   << ",\"end\":";
            write_boundary(output, bucket.end);
            output << ",\"read_bytes\":" << bucket.total.read_bytes
                   << ",\"write_bytes\":" << bucket.total.write_bytes
                   << ",\"erase_bytes\":" << bucket.total.erase_bytes
                   << ",\"read_accesses\":" << bucket.total.read_accesses
                   << ",\"write_accesses\":" << bucket.total.write_accesses
                   << ",\"erase_accesses\":" << bucket.total.erase_accesses;
            const std::array<std::pair<const char*, std::uint64_t AddressTrafficCounters::*>, 6>
                source_fields{{
                    {"source_read_bytes", &AddressTrafficCounters::read_bytes},
                    {"source_write_bytes", &AddressTrafficCounters::write_bytes},
                    {"source_erase_bytes", &AddressTrafficCounters::erase_bytes},
                    {"source_read_accesses", &AddressTrafficCounters::read_accesses},
                    {"source_write_accesses", &AddressTrafficCounters::write_accesses},
                    {"source_erase_accesses", &AddressTrafficCounters::erase_accesses},
                }};
            for (const auto& [name, member] : source_fields) {
                output << ",\"" << name << "\":";
                write_source_array(
                    output,
                    bucket,
                    [member](const AddressTrafficCounters& counters) {
                        return counters.*member;
                    });
            }
            output.put('}');
        }
        output << "]}";
    }
    output << "]}";
    if (!output) {
        throw std::runtime_error("failed to write address heatmap JSON");
    }
}

} // namespace

void write_address_heatmap_json(
    std::ostream& output,
    const AddressHeatmapSnapshot& snapshot) {
    // Validate before writing so a bad retained snapshot cannot leave a
    // syntactically valid-looking but semantically corrupt partial artifact.
    validate_json_snapshot(snapshot);
    write_json_impl(output, snapshot.bin_count, snapshot.domains);
}

void AddressHeatmap::write_json(std::ostream& output) const {
    write_json_impl(output, bin_count_, domains_);
}

} // namespace hbfsim::physical
