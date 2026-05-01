#pragma once

#include "physical/address_heatmap.hpp"

#include <cstdint>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace hbfsim::physical {

enum class Tier {
    HBM,
    HBF,
    External,
};

enum class Op {
    Read,
    Write,
    Erase,
    Refresh,
};

enum class AddressSpace {
    Logical,
    Static,
    Physical,
};

enum class TraceMode {
    Off,
    Summary,
    Sampled,
    Full,
};

struct TraceConfig {
    TraceMode mode = TraceMode::Off;
    // Completion paths/notes are API diagnostics, not timing state. Large
    // summary-only runs can disable their construction while full/sampled
    // traces and the default library API retain them.
    bool retain_completion_diagnostics = true;
};

inline bool trace_spans_enabled(const TraceConfig& config) {
    return config.mode == TraceMode::Sampled || config.mode == TraceMode::Full;
}

inline std::string to_string(Tier tier) {
    switch (tier) {
    case Tier::HBM:
        return "HBM";
    case Tier::HBF:
        return "HBF";
    case Tier::External:
        return "External";
    }
    throw std::runtime_error("unknown tier");
}

inline std::string to_string(Op op) {
    switch (op) {
    case Op::Read:
        return "read";
    case Op::Write:
        return "write";
    case Op::Erase:
        return "erase";
    case Op::Refresh:
        return "refresh";
    }
    throw std::runtime_error("unknown op");
}

struct PhysicalRequest {
    std::string id{};
    Tier tier = Tier::HBM;
    Op op = Op::Read;
    AddressSpace address_space = AddressSpace::Logical;
    TraceConfig trace{};
    double arrival_ns = 0.0;
    std::uint64_t addr = 0;
    std::uint64_t bytes = 0;
    std::uint32_t stream_id = 0;
    HeatmapTrafficSource heatmap_source = HeatmapTrafficSource::Direct;
};

struct Breakdown {
    double ingress_queue_wait_ns = 0.0;
    double scheduler_queue_wait_ns = 0.0;
    double address_mapping_ns = 0.0;
    double translation_ns = 0.0;
    double mapping_dram_ns = 0.0;
    double write_buffer_dram_ns = 0.0;
    double refresh_stall_ns = 0.0;
    double precharge_ns = 0.0;
    double activation_ns = 0.0;
    double command_ns = 0.0;
    double array_read_ns = 0.0;
    double array_program_ns = 0.0;
    double array_erase_ns = 0.0;
    double program_verify_ns = 0.0;
    double media_lane_transfer_ns = 0.0;
    double page_buffer_ns = 0.0;
    double sram_staging_ns = 0.0;
    double channel_transfer_ns = 0.0;
    double tsv_transfer_ns = 0.0;
    double hb_io_transfer_ns = 0.0;
    // Non-occupying propagation/response delay on a transport. It is
    // separate from hb_io_transfer_ns so fixed latency cannot be mistaken for
    // serialization busy time.
    double transport_latency_ns = 0.0;
    // ECC is a pipelined resource: queueing for an initiation slot and
    // response latency are independent terms. Keeping them separate avoids
    // treating the full decode/encode latency as exclusive engine occupancy.
    double ecc_queue_wait_ns = 0.0;
    double ecc_latency_ns = 0.0;
    double maintenance_ns = 0.0;

    [[nodiscard]] bool operator==(const Breakdown&) const = default;

    // Additive WORK accounting. Individual fields may overlap in wall time
    // (and may aggregate parallel children), so neither this record nor its
    // total is a latency decomposition. A caller that needs elapsed time must
    // use explicit start/finish frontiers instead.
    void add(const Breakdown& other) {
        ingress_queue_wait_ns += other.ingress_queue_wait_ns;
        scheduler_queue_wait_ns += other.scheduler_queue_wait_ns;
        address_mapping_ns += other.address_mapping_ns;
        translation_ns += other.translation_ns;
        mapping_dram_ns += other.mapping_dram_ns;
        write_buffer_dram_ns += other.write_buffer_dram_ns;
        refresh_stall_ns += other.refresh_stall_ns;
        precharge_ns += other.precharge_ns;
        activation_ns += other.activation_ns;
        command_ns += other.command_ns;
        array_read_ns += other.array_read_ns;
        array_program_ns += other.array_program_ns;
        array_erase_ns += other.array_erase_ns;
        program_verify_ns += other.program_verify_ns;
        media_lane_transfer_ns += other.media_lane_transfer_ns;
        page_buffer_ns += other.page_buffer_ns;
        sram_staging_ns += other.sram_staging_ns;
        channel_transfer_ns += other.channel_transfer_ns;
        tsv_transfer_ns += other.tsv_transfer_ns;
        hb_io_transfer_ns += other.hb_io_transfer_ns;
        transport_latency_ns += other.transport_latency_ns;
        ecc_queue_wait_ns += other.ecc_queue_wait_ns;
        ecc_latency_ns += other.ecc_latency_ns;
        maintenance_ns += other.maintenance_ns;
    }

    Breakdown& operator+=(const Breakdown& other) {
        add(other);
        return *this;
    }

    [[nodiscard]] double total_work_ns() const {
        return ingress_queue_wait_ns + scheduler_queue_wait_ns + address_mapping_ns +
            translation_ns + mapping_dram_ns + write_buffer_dram_ns +
            refresh_stall_ns + precharge_ns +
            activation_ns + command_ns + array_read_ns + array_program_ns +
            array_erase_ns + program_verify_ns + media_lane_transfer_ns +
            page_buffer_ns + sram_staging_ns + channel_transfer_ns +
            tsv_transfer_ns + hb_io_transfer_ns + transport_latency_ns +
            ecc_queue_wait_ns +
            ecc_latency_ns + maintenance_ns;
    }
};

struct TraceSpan {
    std::string name;
    std::string category;
    std::string entity;
    double start_ns = 0.0;
    double end_ns = 0.0;
    bool critical = true;
    std::string detail;

    [[nodiscard]] double duration_ns() const {
        return end_ns - start_ns;
    }
};

struct PhysicalCompletion {
    std::string id;
    Tier tier = Tier::HBM;
    Op op = Op::Read;
    double arrival_ns = 0.0;
    double start_ns = 0.0;
    double finish_ns = 0.0;
    std::uint64_t logical_bytes = 0;
    std::uint64_t physical_bytes = 0;
    std::string resource_path;
    std::string note;
    Breakdown breakdown;
    std::vector<TraceSpan> spans;

    [[nodiscard]] double latency_ns() const {
        return finish_ns - arrival_ns;
    }

    [[nodiscard]] double service_ns() const {
        return finish_ns - start_ns;
    }
};

inline void add_trace_span(
    std::vector<TraceSpan>* spans,
    std::string name,
    std::string category,
    std::string entity,
    double start_ns,
    double end_ns,
    bool critical = true,
    std::string detail = {}) {
    if (spans == nullptr || end_ns <= start_ns) {
        return;
    }
    spans->push_back(TraceSpan{
        .name = std::move(name),
        .category = std::move(category),
        .entity = std::move(entity),
        .start_ns = start_ns,
        .end_ns = end_ns,
        .critical = critical,
        .detail = std::move(detail),
    });
}

inline void add_trace_span(
    std::vector<TraceSpan>& spans,
    std::string name,
    std::string category,
    std::string entity,
    double start_ns,
    double end_ns,
    bool critical = true,
    std::string detail = {}) {
    add_trace_span(
        &spans,
        std::move(name),
        std::move(category),
        std::move(entity),
        start_ns,
        end_ns,
        critical,
        std::move(detail));
}

inline double transfer_time_ns(std::uint64_t bytes, double bandwidth_gbps) {
    if (bytes == 0) {
        return 0.0;
    }
    if (!std::isfinite(bandwidth_gbps) || bandwidth_gbps <= 0.0) {
        throw std::runtime_error("bandwidth must be positive and finite");
    }
    // Decimal GB/s maps directly to bytes/ns.
    return static_cast<double>(bytes) / bandwidth_gbps;
}

inline std::uint64_t div_ceil(std::uint64_t value, std::uint64_t divisor) {
    if (divisor == 0) {
        throw std::runtime_error("division by zero");
    }
    return value / divisor + (value % divisor != 0 ? 1 : 0);
}

inline std::uint64_t round_up(std::uint64_t value, std::uint64_t granularity) {
    const auto units = div_ceil(value, granularity);
    if (units != 0 && granularity > std::numeric_limits<std::uint64_t>::max() / units) {
        throw std::runtime_error("rounded value overflows uint64_t");
    }
    return units * granularity;
}

inline std::string fixed(double value, int precision = 2) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << value;
    return out.str();
}

inline void print_completion_row(std::ostream& os, const PhysicalCompletion& c) {
    os << std::left << std::setw(28) << c.id
       << std::setw(7) << to_string(c.tier)
       << std::setw(8) << to_string(c.op)
       << std::right << std::setw(12) << fixed(c.arrival_ns)
       << std::setw(12) << fixed(c.start_ns)
       << std::setw(12) << fixed(c.finish_ns)
       << std::setw(12) << fixed(c.latency_ns())
       << std::setw(12) << c.logical_bytes
       << std::setw(12) << c.physical_bytes
       << "  " << c.resource_path;
    if (!c.note.empty()) {
        os << "  [" << c.note << "]";
    }
    os << '\n';
}

inline void print_completion_header(std::ostream& os) {
    os << std::left << std::setw(28) << "id"
       << std::setw(7) << "tier"
       << std::setw(8) << "op"
       << std::right << std::setw(12) << "arrival"
       << std::setw(12) << "start"
       << std::setw(12) << "finish"
       << std::setw(12) << "latency"
       << std::setw(12) << "logical"
       << std::setw(12) << "physical"
       << "  path\n";
}

inline void print_breakdown(std::ostream& os, const PhysicalCompletion& c) {
    os << "  breakdown[" << c.id << "]: "
       << "ingress_q=" << fixed(c.breakdown.ingress_queue_wait_ns)
       << " sched_q=" << fixed(c.breakdown.scheduler_queue_wait_ns)
       << " addr_map=" << fixed(c.breakdown.address_mapping_ns)
       << " xlate=" << fixed(c.breakdown.translation_ns)
       << " map_dram=" << fixed(c.breakdown.mapping_dram_ns)
       << " wb_dram=" << fixed(c.breakdown.write_buffer_dram_ns)
       << " refresh=" << fixed(c.breakdown.refresh_stall_ns)
       << " pre=" << fixed(c.breakdown.precharge_ns)
       << " act=" << fixed(c.breakdown.activation_ns)
       << " cmd=" << fixed(c.breakdown.command_ns)
       << " arr_rd=" << fixed(c.breakdown.array_read_ns)
       << " arr_prog=" << fixed(c.breakdown.array_program_ns)
       << " arr_erase=" << fixed(c.breakdown.array_erase_ns)
       << " verify=" << fixed(c.breakdown.program_verify_ns)
       << " lane=" << fixed(c.breakdown.media_lane_transfer_ns)
       << " page_buf=" << fixed(c.breakdown.page_buffer_ns)
       << " sram=" << fixed(c.breakdown.sram_staging_ns)
       << " channel=" << fixed(c.breakdown.channel_transfer_ns)
       << " tsv=" << fixed(c.breakdown.tsv_transfer_ns)
       << " hbio=" << fixed(c.breakdown.hb_io_transfer_ns)
       << " transport_lat=" << fixed(c.breakdown.transport_latency_ns)
       << " ecc_q=" << fixed(c.breakdown.ecc_queue_wait_ns)
       << " ecc_lat=" << fixed(c.breakdown.ecc_latency_ns)
       << " maint=" << fixed(c.breakdown.maintenance_ns)
       << '\n';
}

inline void print_trace_waterfall(
    std::ostream& os,
    const PhysicalCompletion& c,
    std::size_t max_spans = 120) {
    if (c.spans.empty()) {
        os << "  trace[" << c.id << "]: <no spans>\n";
        return;
    }
    auto spans = c.spans;
    std::stable_sort(spans.begin(), spans.end(), [](const TraceSpan& lhs, const TraceSpan& rhs) {
        if (lhs.start_ns == rhs.start_ns) {
            return lhs.end_ns < rhs.end_ns;
        }
        return lhs.start_ns < rhs.start_ns;
    });

    os << "  trace[" << c.id << "]:\n";
    os << "    " << std::left << std::setw(18) << "category"
       << std::setw(24) << "name"
       << std::right << std::setw(12) << "start"
       << std::setw(12) << "dur"
       << "  entity\n";
    const auto limit = std::min(max_spans, spans.size());
    for (std::size_t i = 0; i < limit; ++i) {
        const auto& span = spans[i];
        os << "    " << std::left << std::setw(18) << span.category
           << std::setw(24) << span.name
           << std::right << std::setw(12) << fixed(span.start_ns)
           << std::setw(12) << fixed(span.duration_ns())
           << "  " << span.entity;
        if (!span.detail.empty()) {
            os << "  [" << span.detail << "]";
        }
        if (!span.critical) {
            os << "  [resource]";
        }
        os << '\n';
    }
    if (spans.size() > limit) {
        os << "    ... " << (spans.size() - limit)
           << " more spans omitted; use --trace for full timeline\n";
    }
}

} // namespace hbfsim::physical
