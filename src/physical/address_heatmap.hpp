#pragma once

#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace hbfsim::physical {

inline constexpr std::size_t kDefaultAddressHeatmapBins = 1024;
// A simulator run can retain six scenario snapshots simultaneously.
// Keep the public limit high enough for fine-grained inspection without
// allowing bin storage alone to approach a GiB at the supported policy count.
inline constexpr std::size_t kMaxAddressHeatmapBins = 8192;

enum class AddressDomain : std::uint8_t {
    WorkloadLogical,
    HbmPhysical,
    HbfLogical,
    HbfPhysical,
    ExternalPhysical,
    Count,
};

enum class TrafficDirection : std::uint8_t {
    Read,
    Write,
    Erase,
};

// Sources describe why traffic exists, independently of where it is observed.
// Keeping this vocabulary closed prevents output from fragmenting into
// spelling variants when device and composition hooks are added.
enum class HeatmapTrafficSource : std::uint8_t {
    Workload,
    Direct,
    Mapping,
    Prepopulate,
    CooperativeBuffer,
    DemandFill,
    PrefetchFill,
    StreamingInstall,
    Destage,
    GarbageCollection,
    Maintenance,
    Count,
};

enum class AddressRegionKind : std::uint8_t {
    Workload,
    CooperativeBuffer,
    LayerBuffer,
    StaticData,
    MappedData,
    Reserved,
    Other,
};

inline constexpr std::size_t kAddressDomainCount =
    static_cast<std::size_t>(AddressDomain::Count);
inline constexpr std::size_t kHeatmapTrafficSourceCount =
    static_cast<std::size_t>(HeatmapTrafficSource::Count);

[[nodiscard]] const char* to_string(AddressDomain domain);
[[nodiscard]] const char* to_string(HeatmapTrafficSource source);
[[nodiscard]] const char* to_string(AddressRegionKind kind);

// A half-open address interval over uint64_t addresses needs one boundary that
// is not itself an address: 2^64. Keeping that sentinel explicit lets the
// heatmap represent [UINT64_MAX, 2^64) without wrapping or silently dropping
// the highest byte. Every other boundary is an ordinary finite uint64_t value.
class AddressBoundary {
public:
    constexpr AddressBoundary() = default;
    constexpr AddressBoundary(std::uint64_t finite_value)
        : finite_value_(finite_value) {}

    [[nodiscard]] static constexpr AddressBoundary full_address_space_end() {
        AddressBoundary boundary;
        boundary.full_address_space_end_ = true;
        return boundary;
    }

    [[nodiscard]] constexpr bool is_full_address_space_end() const {
        return full_address_space_end_;
    }
    // Meaningful only when is_full_address_space_end() is false.
    [[nodiscard]] constexpr std::uint64_t finite_value() const {
        return finite_value_;
    }

    friend constexpr bool operator==(
        const AddressBoundary&,
        const AddressBoundary&) = default;
    friend constexpr std::strong_ordering operator<=>(
        const AddressBoundary& lhs,
        const AddressBoundary& rhs) {
        if (lhs.full_address_space_end_ != rhs.full_address_space_end_) {
            return lhs.full_address_space_end_ ?
                std::strong_ordering::greater : std::strong_ordering::less;
        }
        return lhs.finite_value_ <=> rhs.finite_value_;
    }

private:
    std::uint64_t finite_value_ = 0;
    bool full_address_space_end_ = false;
};

// Convert a positive, representable uint64_t byte range to its exact exclusive
// end. The one-byte range beginning at UINT64_MAX returns the 2^64 sentinel.
[[nodiscard]] AddressBoundary address_exclusive_end(
    std::uint64_t address,
    std::uint64_t bytes);
[[nodiscard]] std::string address_boundary_decimal(AddressBoundary boundary);

struct AddressRegion {
    std::string name;
    AddressRegionKind kind = AddressRegionKind::Other;
    std::uint64_t begin = 0;
    AddressBoundary end = 0; // exclusive; may be 2^64
};

struct AddressDomainConfig {
    AddressDomain domain = AddressDomain::WorkloadLogical;
    AddressBoundary size_bytes = 0;
    std::vector<AddressRegion> regions{};
};

struct AddressHeatmapConfig {
    std::size_t bin_count = kDefaultAddressHeatmapBins;
    std::array<AddressDomainConfig, kAddressDomainCount> domains{{
        {.domain = AddressDomain::WorkloadLogical},
        {.domain = AddressDomain::HbmPhysical},
        {.domain = AddressDomain::HbfLogical},
        {.domain = AddressDomain::HbfPhysical},
        {.domain = AddressDomain::ExternalPhysical},
    }};
};

struct AddressTrafficRecord {
    AddressDomain domain = AddressDomain::WorkloadLogical;
    TrafficDirection direction = TrafficDirection::Read;
    HeatmapTrafficSource source = HeatmapTrafficSource::Workload;
    std::uint64_t address = 0;
    std::uint64_t bytes = 0;
};

struct AddressTrafficCounters {
    std::uint64_t read_bytes = 0;
    std::uint64_t write_bytes = 0;
    std::uint64_t erase_bytes = 0;
    // At domain scope an access is one accepted record. At bin scope it is
    // one record touching that bin, regardless of overlap size. Therefore
    // bytes conserve across bins, while accesses intentionally describe
    // spatial fan-out and need not sum to the domain access count.
    std::uint64_t read_accesses = 0;
    std::uint64_t write_accesses = 0;
    std::uint64_t erase_accesses = 0;

};

struct AddressHeatmapBin {
    std::uint64_t begin = 0;
    AddressBoundary end = 0; // exclusive; may be 2^64
    AddressTrafficCounters total;
    std::array<AddressTrafficCounters, kHeatmapTrafficSourceCount> by_source{};
};

struct AddressDomainHeatmap {
    AddressDomain domain = AddressDomain::WorkloadLogical;
    AddressBoundary size_bytes = 0;
    AddressTrafficCounters total;
    std::array<AddressTrafficCounters, kHeatmapTrafficSourceCount> by_source{};
    std::vector<AddressRegion> regions;
    std::vector<AddressHeatmapBin> bins;
};

struct AddressHeatmapSnapshot {
    std::size_t bin_count = 0;
    std::array<AddressDomainHeatmap, kAddressDomainCount> domains;
};

// Serialize a retained snapshot without requiring the live accumulator. The
// writer validates the same structural and accounting invariants consumed by
// the offline renderer before emitting any JSON bytes.
void write_address_heatmap_json(
    std::ostream& output,
    const AddressHeatmapSnapshot& snapshot);

class AddressHeatmap {
public:
    explicit AddressHeatmap(AddressHeatmapConfig config);

    // A record must be wholly contained in its configured domain. Updates
    // have a strong exception guarantee: range or counter overflow errors do
    // not leave partially updated bins.
    void record(const AddressTrafficRecord& traffic);

    // Record access_count adjacent fixed-size accesses beginning with
    // `first`. This is exactly equivalent to calling record() once per access,
    // including per-bin access fan-out, but avoids repeating bin searches for
    // controller burst trains that cover one contiguous range.
    void record_contiguous_accesses(
        const AddressTrafficRecord& first,
        std::uint64_t access_count);

    [[nodiscard]] const AddressDomainHeatmap& domain(AddressDomain domain) const;
    [[nodiscard]] AddressHeatmapSnapshot snapshot() const;

    // Deterministic, dependency-free interchange used by the offline renderer.
    // Domain, source, region, and bin ordering are canonical.
    void write_json(std::ostream& output) const;

private:
    std::size_t bin_count_ = 0;
    std::array<AddressDomainHeatmap, kAddressDomainCount> domains_;
};

} // namespace hbfsim::physical
