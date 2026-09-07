#include "app/session_protocol.hpp"

#include "app/sha256.hpp"
#include "physical/hbf/hbf_persistent_image.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace hbfsim::app {
namespace {

using physical::Breakdown;
using physical::BaseDieLinkStats;
using physical::Op;
using physical::PhysicalCompletion;
using physical::SimulationBatchOptions;
using physical::SimulationBatchResult;
using physical::SimulationCompletionStats;
using physical::SimulationDeviceSnapshot;
using physical::SimulationInputError;
using physical::SimulationSession;
using physical::SimulationSessionConfig;
using physical::SimulationTarget;
using physical::SimulationTransaction;
using physical::external::ExternalBackingStats;

constexpr std::string_view kProtocolName = "simulation-transaction-text-v2";

bool parse_op_token(std::string_view token, Op& op) {
    if (token == "R" || token == "READ") {
        op = Op::Read;
        return true;
    }
    if (token == "W" || token == "WRITE") {
        op = Op::Write;
        return true;
    }
    return false;
}

// Decimal digits only: no sign, prefix, or exponent, and a leading zero
// never selects octal.
std::uint64_t parse_u64(std::string_view raw, const char* description) {
    std::uint64_t value = 0;
    const auto* const begin = raw.data();
    const auto* const end = raw.data() + raw.size();
    const auto [pointer, error] = std::from_chars(begin, end, value, 10);
    if (raw.empty() || error != std::errc{} || pointer != end) {
        throw std::runtime_error(
            std::string(description) + " must be an unsigned decimal integer");
    }
    return value;
}

std::uint32_t parse_u32(std::string_view raw, const char* description) {
    const auto value = parse_u64(raw, description);
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error(std::string(description) + " exceeds uint32_t");
    }
    return static_cast<std::uint32_t>(value);
}

// `raw` must be a view into a NUL-terminated buffer whose next character
// after the field is a separator (a space or the terminator); protocol
// lines satisfy that because fields are single-space separated.
double parse_double(std::string_view raw, const char* description) {
    if (raw.empty() || std::isspace(static_cast<unsigned char>(raw.front()))) {
        throw std::runtime_error(std::string(description) + " is malformed");
    }
    char* end = nullptr;
    const double value = std::strtod(raw.data(), &end);
    if (end != raw.data() + raw.size() || !std::isfinite(value)) {
        throw std::runtime_error(std::string(description) + " must be finite");
    }
    return value;
}

bool is_lower_hex_sha256(const std::string& value) {
    return value.size() == 64 && std::all_of(
        value.begin(), value.end(), [](unsigned char ch) {
            return std::isdigit(ch) || (ch >= 'a' && ch <= 'f');
        });
}

std::filesystem::path decode_path_hex(const std::string& value) {
    if (value.empty() || value.size() % 2 != 0 ||
        !std::all_of(value.begin(), value.end(), [](unsigned char ch) {
            return std::isdigit(ch) || (ch >= 'a' && ch <= 'f');
        })) {
        throw std::runtime_error(
            "simulation persistent-image path hex is malformed");
    }
    std::string decoded;
    decoded.reserve(value.size() / 2);
    const auto nibble = [](char ch) -> unsigned char {
        if (ch >= '0' && ch <= '9') {
            return static_cast<unsigned char>(ch - '0');
        }
        return static_cast<unsigned char>(10 + ch - 'a');
    };
    for (std::size_t index = 0; index < value.size(); index += 2) {
        const auto byte = static_cast<unsigned char>(
            (nibble(value[index]) << 4) | nibble(value[index + 1]));
        if (byte == 0) {
            throw std::runtime_error(
                "simulation persistent-image path contains NUL");
        }
        decoded.push_back(static_cast<char>(byte));
    }
    const auto path = std::filesystem::path(decoded);
    if (!path.is_absolute()) {
        throw std::runtime_error(
            "simulation persistent-image output path must be absolute");
    }
    return path;
}

std::string json_escape(const std::string& input) {
    std::ostringstream out;
    for (const char ch : input) {
        switch (ch) {
        case '\\':
            out << "\\\\";
            break;
        case '"':
            out << "\\\"";
            break;
        case '\n':
            out << "\\n";
            break;
        case '\r':
            out << "\\r";
            break;
        case '\t':
            out << "\\t";
            break;
        default:
            if (static_cast<unsigned char>(ch) < 0x20) {
                out << "\\u00" << std::hex << std::setw(2) << std::setfill('0')
                    << static_cast<unsigned int>(static_cast<unsigned char>(ch))
                    << std::dec;
            } else {
                out << ch;
            }
            break;
        }
    }
    return out.str();
}

bool transaction_identifier_valid(std::string_view value) {
    return !value.empty() && std::all_of(
        value.begin(), value.end(), [](unsigned char character) {
            return std::isalnum(character) || character == '_' ||
                character == '-' || character == '.' || character == ':' ||
                character == '/';
        });
}

SimulationTarget parse_simulation_target(std::string_view value) {
    if (value == "HBM") return SimulationTarget::Hbm;
    if (value == "HBF_LOGICAL") return SimulationTarget::HbfLogical;
    if (value == "HBF_STATIC") return SimulationTarget::HbfStatic;
    if (value == "HBF_PHYSICAL") return SimulationTarget::HbfPhysical;
    if (value == "D2D_HBF_TO_HBM") return SimulationTarget::D2dHbfToHbm;
    if (value == "D2D_HBM_TO_HBF") return SimulationTarget::D2dHbmToHbf;
    if (value == "DIRECT_HBF_TO_EXTERNAL") {
        return SimulationTarget::DirectHbfToExternal;
    }
    if (value == "DIRECT_EXTERNAL_TO_HBF") {
        return SimulationTarget::DirectExternalToHbf;
    }
    if (value == "EXTERNAL") return SimulationTarget::External;
    if (value == "BARRIER") return SimulationTarget::Barrier;
    throw std::runtime_error(
        "unknown simulation target: " + std::string(value));
}

// Splits a comma-separated identifier list ("-" is the empty list).
std::vector<std::string> parse_identifier_list(
    std::string_view raw,
    const char* description) {
    std::vector<std::string> identifiers;
    if (raw == "-") {
        return identifiers;
    }
    while (true) {
        const auto comma = raw.find(',');
        const auto item = raw.substr(0, comma);
        if (!transaction_identifier_valid(item)) {
            throw std::runtime_error(
                std::string(description) +
                " contains an empty or unsafe identifier");
        }
        identifiers.emplace_back(item);
        if (comma == std::string_view::npos) {
            return identifiers;
        }
        raw.remove_prefix(comma + 1);
    }
}

// Consumes `key=value ` (or `key=value` at end of line) from the front of
// `rest`; the value is never empty.
bool take_field(
    std::string_view& rest,
    std::string_view key,
    std::string_view& value) {
    if (rest.size() <= key.size() + 1 || !rest.starts_with(key) ||
        rest[key.size()] != '=') {
        return false;
    }
    rest.remove_prefix(key.size() + 1);
    const auto end = rest.find(' ');
    value = rest.substr(0, end);
    rest = end == std::string_view::npos ?
        std::string_view{} : rest.substr(end + 1);
    return !value.empty();
}

// One transaction line. The nine fields are positional (this exact order,
// single spaces, no trailing space); the Python client emits them so and
// the fixed layout is what lets the engine parse without per-line
// allocation beyond the transaction itself.
SimulationTransaction parse_simulation_transaction(
    const std::string& line,
    std::size_t line_no) {
    const auto malformed = [line_no]() {
        return std::runtime_error(
            "simulation protocol line " + std::to_string(line_no) +
            " must be `TX id= target= op= addr= bytes= issue_ns= "
            "duration_ns= deps= stack=` with single-space separators");
    };
    std::string_view rest(line);
    if (!rest.starts_with("TX ")) {
        throw malformed();
    }
    rest.remove_prefix(3);
    std::string_view id;
    std::string_view target_text;
    std::string_view op_text;
    std::string_view addr;
    std::string_view bytes;
    std::string_view issue_ns;
    std::string_view duration_ns;
    std::string_view deps;
    std::string_view stack_text;
    if (!take_field(rest, "id", id) ||
        !take_field(rest, "target", target_text) ||
        !take_field(rest, "op", op_text) ||
        !take_field(rest, "addr", addr) ||
        !take_field(rest, "bytes", bytes) ||
        !take_field(rest, "issue_ns", issue_ns) ||
        !take_field(rest, "duration_ns", duration_ns) ||
        !take_field(rest, "deps", deps) ||
        !take_field(rest, "stack", stack_text) || !rest.empty()) {
        throw malformed();
    }
    if (!transaction_identifier_valid(id)) {
        throw std::runtime_error(
            "simulation protocol transaction id contains an unsafe character");
    }
    const auto target = parse_simulation_target(target_text);
    Op op = Op::Read;
    if (target == SimulationTarget::Barrier) {
        if (op_text != "-") {
            throw std::runtime_error("simulation barrier op must be '-'");
        }
    } else if (!parse_op_token(op_text, op)) {
        throw std::runtime_error("simulation memory op must be R or W");
    }
    std::uint32_t stack = 0;
    if (stack_text != "-") {
        stack = parse_u32(stack_text, "simulation transaction stack");
    }
    return SimulationTransaction{
        .id = std::string(id),
        .target = target,
        .op = op,
        .addr = parse_u64(addr, "simulation transaction addr"),
        .bytes = parse_u64(bytes, "simulation transaction bytes"),
        .issue_ns = parse_double(
            issue_ns, "simulation transaction issue_ns"),
        .duration_ns = parse_double(
            duration_ns, "simulation transaction duration_ns"),
        .dependencies = parse_identifier_list(
            deps, "simulation transaction deps"),
        .stack = stack,
    };
}

// `BEGIN <batch_id> <logical_sha256> <transaction_sha256> [key=value ...]`
// with optional fields frontier=<ids|->, retain=<ids|->, completions=0|1.
struct BatchHeader {
    std::string batch_id;
    std::string logical_sha256;
    std::string transaction_sha256;
    SimulationBatchOptions options;
};

BatchHeader parse_batch_header(const std::string& line) {
    const auto malformed = []() {
        return std::runtime_error(
            "simulation protocol BEGIN line is malformed");
    };
    std::string_view rest(line);
    const auto next_token = [&rest]() {
        const auto end = rest.find(' ');
        const auto token = rest.substr(0, end);
        rest = end == std::string_view::npos ?
            std::string_view{} : rest.substr(end + 1);
        return token;
    };
    if (next_token() != "BEGIN") {
        throw malformed();
    }
    BatchHeader header;
    header.batch_id = std::string(next_token());
    header.logical_sha256 = std::string(next_token());
    header.transaction_sha256 = std::string(next_token());
    if (!transaction_identifier_valid(header.batch_id) ||
        !is_lower_hex_sha256(header.logical_sha256) ||
        !is_lower_hex_sha256(header.transaction_sha256)) {
        throw malformed();
    }
    bool saw_frontier = false;
    bool saw_retain = false;
    bool saw_completions = false;
    while (!rest.empty()) {
        std::string_view value;
        if (!saw_frontier && take_field(rest, "frontier", value)) {
            saw_frontier = true;
            header.options.frontier = parse_identifier_list(
                value, "simulation BEGIN frontier");
        } else if (!saw_retain && take_field(rest, "retain", value)) {
            saw_retain = true;
            header.options.retain = parse_identifier_list(
                value, "simulation BEGIN retain");
        } else if (!saw_completions &&
                   take_field(rest, "completions", value)) {
            saw_completions = true;
            if (value == "1") {
                header.options.record_completions = true;
            } else if (value == "0") {
                header.options.record_completions = false;
            } else {
                throw std::runtime_error(
                    "simulation BEGIN completions must be 0 or 1");
            }
        } else {
            throw std::runtime_error(
                "simulation protocol BEGIN accepts only frontier=, retain=, "
                "and completions= fields, each at most once");
        }
    }
    return header;
}

std::uint64_t session_counter_delta(
    std::uint64_t after,
    std::uint64_t before,
    std::string_view name) {
    if (after < before) {
        throw std::runtime_error(
            "session accounting counter regressed: " + std::string(name));
    }
    return after - before;
}

double session_work_delta(
    double after,
    double before,
    std::string_view name) {
    if (!std::isfinite(after) || !std::isfinite(before)) {
        throw std::runtime_error(
            "session accounting work is not finite: " + std::string(name));
    }
    const auto tolerance = std::max(
        1e-9, std::max(std::abs(after), std::abs(before)) * 1e-12);
    if (after + tolerance < before) {
        throw std::runtime_error(
            "session accounting work regressed: " + std::string(name));
    }
    return std::max(0.0, after - before);
}

Breakdown session_breakdown_delta(
    const Breakdown& after,
    const Breakdown& before) {
    Breakdown result;
#define HBFSIM_SESSION_BREAKDOWN_DELTA(field) \
    result.field = session_work_delta(after.field, before.field, #field)
    HBFSIM_SESSION_BREAKDOWN_DELTA(ingress_queue_wait_ns);
    HBFSIM_SESSION_BREAKDOWN_DELTA(scheduler_queue_wait_ns);
    HBFSIM_SESSION_BREAKDOWN_DELTA(address_mapping_ns);
    HBFSIM_SESSION_BREAKDOWN_DELTA(translation_ns);
    HBFSIM_SESSION_BREAKDOWN_DELTA(mapping_dram_ns);
    HBFSIM_SESSION_BREAKDOWN_DELTA(write_buffer_dram_ns);
    HBFSIM_SESSION_BREAKDOWN_DELTA(refresh_stall_ns);
    HBFSIM_SESSION_BREAKDOWN_DELTA(precharge_ns);
    HBFSIM_SESSION_BREAKDOWN_DELTA(activation_ns);
    HBFSIM_SESSION_BREAKDOWN_DELTA(command_ns);
    HBFSIM_SESSION_BREAKDOWN_DELTA(array_read_ns);
    HBFSIM_SESSION_BREAKDOWN_DELTA(array_program_ns);
    HBFSIM_SESSION_BREAKDOWN_DELTA(array_erase_ns);
    HBFSIM_SESSION_BREAKDOWN_DELTA(program_verify_ns);
    HBFSIM_SESSION_BREAKDOWN_DELTA(media_lane_transfer_ns);
    HBFSIM_SESSION_BREAKDOWN_DELTA(page_buffer_ns);
    HBFSIM_SESSION_BREAKDOWN_DELTA(sram_staging_ns);
    HBFSIM_SESSION_BREAKDOWN_DELTA(channel_transfer_ns);
    HBFSIM_SESSION_BREAKDOWN_DELTA(tsv_transfer_ns);
    HBFSIM_SESSION_BREAKDOWN_DELTA(hb_io_transfer_ns);
    HBFSIM_SESSION_BREAKDOWN_DELTA(transport_latency_ns);
    HBFSIM_SESSION_BREAKDOWN_DELTA(ecc_queue_wait_ns);
    HBFSIM_SESSION_BREAKDOWN_DELTA(ecc_latency_ns);
    HBFSIM_SESSION_BREAKDOWN_DELTA(maintenance_ns);
#undef HBFSIM_SESSION_BREAKDOWN_DELTA
    return result;
}

void write_session_breakdown(
    std::ostream& output,
    const Breakdown& value) {
    output << "{\"ingress_queue_wait_work_ns\":"
           << value.ingress_queue_wait_ns
           << ",\"scheduler_queue_wait_work_ns\":"
           << value.scheduler_queue_wait_ns
           << ",\"address_mapping_work_ns\":" << value.address_mapping_ns
           << ",\"translation_work_ns\":" << value.translation_ns
           << ",\"mapping_dram_work_ns\":" << value.mapping_dram_ns
           << ",\"write_buffer_dram_work_ns\":"
           << value.write_buffer_dram_ns
           << ",\"refresh_stall_work_ns\":" << value.refresh_stall_ns
           << ",\"precharge_work_ns\":" << value.precharge_ns
           << ",\"activation_work_ns\":" << value.activation_ns
           << ",\"command_work_ns\":" << value.command_ns
           << ",\"array_read_work_ns\":" << value.array_read_ns
           << ",\"array_program_work_ns\":" << value.array_program_ns
           << ",\"array_erase_work_ns\":" << value.array_erase_ns
           << ",\"program_verify_work_ns\":" << value.program_verify_ns
           << ",\"media_lane_transfer_work_ns\":"
           << value.media_lane_transfer_ns
           << ",\"page_buffer_work_ns\":" << value.page_buffer_ns
           << ",\"sram_staging_work_ns\":" << value.sram_staging_ns
           << ",\"channel_transfer_work_ns\":"
           << value.channel_transfer_ns
           << ",\"tsv_transfer_work_ns\":" << value.tsv_transfer_ns
           << ",\"hb_io_transfer_work_ns\":" << value.hb_io_transfer_ns
           << ",\"transport_latency_work_ns\":"
           << value.transport_latency_ns
           << ",\"ecc_queue_wait_work_ns\":" << value.ecc_queue_wait_ns
           << ",\"ecc_latency_work_ns\":" << value.ecc_latency_ns
           << ",\"maintenance_work_ns\":" << value.maintenance_ns
           << ",\"total_work_ns\":" << value.total_work_ns() << '}';
}

void write_session_completion_stats(
    std::ostream& output,
    const SimulationCompletionStats& value) {
    output << "{\"transactions\":" << value.transactions
           << ",\"logical_bytes\":" << value.logical_bytes
           << ",\"physical_bytes\":" << value.physical_bytes
           << ",\"queue_wait_work_ns\":" << value.queue_wait_work_ns
           << ",\"service_work_ns\":" << value.service_work_ns
           << ",\"latency_work_ns\":" << value.latency_work_ns;
    if (value.transactions == 0) {
        output << ",\"mean_queue_wait_ns\":null"
                  ",\"mean_service_ns\":null"
                  ",\"mean_latency_ns\":null"
                  ",\"min_latency_ns\":null"
                  ",\"max_latency_ns\":null"
                  ",\"first_arrival_ns\":null"
                  ",\"finish_ns\":null"
                  ",\"active_span_ns\":null"
                  ",\"effective_logical_GBps\":null"
                  ",\"effective_physical_GBps\":null}";
        return;
    }
    const auto count = static_cast<double>(value.transactions);
    const auto active_span_ns = value.finish_ns - value.first_arrival_ns;
    output << ",\"mean_queue_wait_ns\":" << value.queue_wait_work_ns / count
           << ",\"mean_service_ns\":" << value.service_work_ns / count
           << ",\"mean_latency_ns\":" << value.latency_work_ns / count
           << ",\"min_latency_ns\":" << value.min_latency_ns
           << ",\"max_latency_ns\":" << value.max_latency_ns
           << ",\"first_arrival_ns\":" << value.first_arrival_ns
           << ",\"finish_ns\":" << value.finish_ns
           << ",\"active_span_ns\":" << active_span_ns
           << ",\"effective_logical_GBps\":";
    if (active_span_ns > 0.0) {
        output << static_cast<double>(value.logical_bytes) / active_span_ns;
    } else {
        output << "null";
    }
    output << ",\"effective_physical_GBps\":";
    if (active_span_ns > 0.0) {
        output << static_cast<double>(value.physical_bytes) / active_span_ns;
    } else {
        output << "null";
    }
    output << '}';
}

void write_session_completion_matrix(
    std::ostream& output,
    const std::array<
        std::array<
            SimulationCompletionStats,
            hbfsim::physical::kSimulationOperationCount>,
        hbfsim::physical::kSimulationTargetCount>& values) {
    output << '{';
    for (std::size_t target_index = 0; target_index < values.size();
         ++target_index) {
        if (target_index != 0) output << ',';
        const auto target = static_cast<SimulationTarget>(target_index);
        output << '\"' << hbfsim::physical::to_string(target)
               << "\":{\"read\":";
        write_session_completion_stats(output, values[target_index][0]);
        output << ",\"write\":";
        write_session_completion_stats(output, values[target_index][1]);
        output << '}';
    }
    output << '}';
}

// Digest over the exported completion records, hashed incrementally: the
// material is `hbfsim.simulation_transaction_completions.v1`, the count,
// and per record the index, length-prefixed id, IEEE-754 bit patterns of
// the three timestamps, and both byte counts, each line-terminated.
std::string session_transaction_completion_digest(
    const std::vector<PhysicalCompletion>& completions) {
    Sha256 digest;
    const auto feed = [&digest](std::string_view text) {
        digest.update(
            reinterpret_cast<const std::uint8_t*>(text.data()),
            text.size());
    };
    char buffer[256];
    const auto feed_format = [&](const char* format, auto... values) {
        const auto count = std::snprintf(buffer, sizeof(buffer), format, values...);
        if (count < 0 || static_cast<std::size_t>(count) >= sizeof(buffer)) {
            throw std::runtime_error("completion digest formatting failed");
        }
        feed(std::string_view(buffer, static_cast<std::size_t>(count)));
    };
    feed("hbfsim.simulation_transaction_completions.v1\n");
    feed_format(
        "count=%llu\n",
        static_cast<unsigned long long>(completions.size()));
    for (std::size_t index = 0; index < completions.size(); ++index) {
        const auto& completion = completions[index];
        feed_format(
            "index=%llu\nid=%llu:",
            static_cast<unsigned long long>(index),
            static_cast<unsigned long long>(completion.id.size()));
        feed(completion.id);
        feed_format(
            "\narrival_bits=%016llx\nstart_bits=%016llx\nfinish_bits=%016llx\n"
            "logical_bytes=%llu\nphysical_bytes=%llu\n",
            static_cast<unsigned long long>(
                std::bit_cast<std::uint64_t>(completion.arrival_ns)),
            static_cast<unsigned long long>(
                std::bit_cast<std::uint64_t>(completion.start_ns)),
            static_cast<unsigned long long>(
                std::bit_cast<std::uint64_t>(completion.finish_ns)),
            static_cast<unsigned long long>(completion.logical_bytes),
            static_cast<unsigned long long>(completion.physical_bytes));
    }
    return digest.finish_hex();
}

void write_session_transaction_completions(
    std::ostream& output,
    const std::vector<PhysicalCompletion>& completions) {
    output << '[';
    for (std::size_t index = 0; index < completions.size(); ++index) {
        if (index != 0) output << ',';
        const auto& completion = completions[index];
        output << "{\"id\":\"" << json_escape(completion.id)
               << "\",\"arrival_ns\":" << completion.arrival_ns
               << ",\"start_ns\":" << completion.start_ns
               << ",\"finish_ns\":" << completion.finish_ns
               << ",\"logical_bytes\":" << completion.logical_bytes
               << ",\"physical_bytes\":" << completion.physical_bytes
               << '}';
    }
    output << ']';
}

void write_session_hbm_accounting(
    std::ostream& output,
    const hbfsim::physical::hbm::HbmStats& after,
    const hbfsim::physical::hbm::HbmStats& before,
    bool include_state) {
    const auto d = [&](std::uint64_t value, std::uint64_t baseline,
                       std::string_view name) {
        return session_counter_delta(value, baseline, name);
    };
    output << "{\"read_bytes\":"
           << d(after.read_bytes, before.read_bytes, "hbm.read_bytes")
           << ",\"write_bytes\":"
           << d(after.write_bytes, before.write_bytes, "hbm.write_bytes")
           << ",\"row_hits\":"
           << d(after.row_hits, before.row_hits, "hbm.row_hits")
           << ",\"row_misses\":"
           << d(after.row_misses, before.row_misses, "hbm.row_misses")
           << ",\"row_conflicts\":"
           << d(after.row_conflicts, before.row_conflicts, "hbm.row_conflicts")
           << ",\"activations\":"
           << d(after.activations, before.activations, "hbm.activations")
           << ",\"precharges\":"
           << d(after.precharges, before.precharges, "hbm.precharges")
           << ",\"refresh_count\":"
           << d(after.refresh_count, before.refresh_count, "hbm.refresh_count")
           << ",\"bus_busy_ns\":"
           << session_work_delta(
                  after.bus_busy_ns, before.bus_busy_ns, "hbm.bus_busy_ns")
           << ",\"replicated_requests\":"
           << d(after.replicated_requests, before.replicated_requests,
                "hbm.replicated_requests")
           << ",\"replicated_bursts\":"
           << d(after.replicated_bursts, before.replicated_bursts,
                "hbm.replicated_bursts")
           << ",\"stage_work\":";
    write_session_breakdown(
        output, session_breakdown_delta(after.stage_work, before.stage_work));
    if (include_state) {
        output << ",\"resource_busy\":{\"active_span_ns\":"
               << after.active_span_ns()
               << ",\"bus_busy_ns\":" << after.bus_busy_ns
               << ",\"utilization\":" << after.utilization()
               << ",\"bus_parallelism\":" << after.bus_parallelism()
               << ",\"pseudo_channels\":" << after.pseudo_channels
               << ",\"active_pseudo_channels\":"
               << after.active_pseudo_channels
               << ",\"max_pseudo_channel_busy_ns\":"
               << after.max_pseudo_channel_busy_ns
               << ",\"avg_active_pseudo_channel_busy_ns\":"
               << after.avg_active_pseudo_channel_busy_ns
               << '}';
    }
    output << '}';
}

void write_session_hbf_accounting(
    std::ostream& output,
    const hbfsim::physical::hbf::HbfStats& after,
    const hbfsim::physical::hbf::HbfStats& before,
    bool include_state) {
    const auto d = [&](std::uint64_t value, std::uint64_t baseline,
                       std::string_view name) {
        return session_counter_delta(value, baseline, name);
    };
    const auto read_requests = d(
        after.read_requests, before.read_requests, "hbf.read_requests");
    const auto program_requests = d(
        after.program_requests,
        before.program_requests,
        "hbf.program_requests");
    const auto erase_requests = d(
        after.erase_requests, before.erase_requests, "hbf.erase_requests");
    const auto logical_read_bytes = d(
        after.logical_read_bytes,
        before.logical_read_bytes,
        "hbf.logical_read_bytes");
    const auto logical_write_bytes = d(
        after.logical_write_bytes,
        before.logical_write_bytes,
        "hbf.logical_write_bytes");
    const auto physical_read_bytes = d(
        after.physical_read_bytes,
        before.physical_read_bytes,
        "hbf.physical_read_bytes");
    const auto physical_write_bytes = d(
        after.physical_write_bytes,
        before.physical_write_bytes,
        "hbf.physical_write_bytes");
    const auto data_program_payload_bytes = d(
        after.data_program_payload_bytes,
        before.data_program_payload_bytes,
        "hbf.data_program_payload_bytes");
    const auto raw_physical_program_payload_bytes = d(
        after.raw_physical_program_payload_bytes,
        before.raw_physical_program_payload_bytes,
        "hbf.raw_physical_program_payload_bytes");
    if (raw_physical_program_payload_bytes >
        std::numeric_limits<std::uint64_t>::max() - logical_write_bytes) {
        throw std::runtime_error(
            "simulation HBF WAF host-write denominator overflowed");
    }
    const auto host_write_bytes =
        logical_write_bytes + raw_physical_program_payload_bytes;
    const auto mapping_program_payload_bytes = d(
        after.mapping_program_payload_bytes,
        before.mapping_program_payload_bytes,
        "hbf.mapping_program_payload_bytes");
    const auto gc_relocation_payload_bytes = d(
        after.gc_relocation_payload_bytes,
        before.gc_relocation_payload_bytes,
        "hbf.gc_relocation_payload_bytes");
    const auto static_wear_leveling_relocation_payload_bytes = d(
        after.static_wear_leveling_relocation_payload_bytes,
        before.static_wear_leveling_relocation_payload_bytes,
        "hbf.static_wear_leveling_relocation_payload_bytes");
    if (physical_write_bytes != data_program_payload_bytes +
            mapping_program_payload_bytes + gc_relocation_payload_bytes +
            static_wear_leveling_relocation_payload_bytes) {
        throw std::runtime_error(
            "simulation HBF physical-write decomposition does not conserve");
    }
    output << "{\"read_requests\":" << read_requests
           << ",\"program_requests\":" << program_requests
           << ",\"erase_requests\":" << erase_requests
           << ",\"logical_read_bytes\":" << logical_read_bytes
           << ",\"logical_write_bytes\":" << logical_write_bytes
           << ",\"physical_read_bytes\":" << physical_read_bytes
           << ",\"physical_write_bytes\":" << physical_write_bytes
           << ",\"data_program_payload_bytes\":"
           << data_program_payload_bytes
           << ",\"raw_physical_program_payload_bytes\":"
           << raw_physical_program_payload_bytes
           << ",\"mapping_program_payload_bytes\":"
           << mapping_program_payload_bytes
           << ",\"gc_relocation_payload_bytes\":"
           << gc_relocation_payload_bytes
           << ",\"static_wear_leveling_relocation_payload_bytes\":"
           << static_wear_leveling_relocation_payload_bytes
           << ",\"waf\":";
    if (host_write_bytes == 0) {
        output << "null";
    } else {
        output << static_cast<double>(physical_write_bytes) /
            static_cast<double>(host_write_bytes);
    }
    output << ",\"waf_definition\":"
              "\"physical_write_bytes/(logical_write_bytes+"
              "raw_physical_program_payload_bytes)\""
           << ",\"page_reads\":"
           << d(after.page_reads, before.page_reads, "hbf.page_reads")
           << ",\"data_programs\":"
           << d(after.data_programs, before.data_programs, "hbf.data_programs")
           << ",\"raw_physical_programs\":"
           << d(after.raw_physical_programs,
                before.raw_physical_programs,
                "hbf.raw_physical_programs")
           << ",\"mapping_page_programs\":"
           << d(after.mapping_page_programs, before.mapping_page_programs,
                "hbf.mapping_page_programs")
           << ",\"page_programs\":"
           << d(after.page_programs, before.page_programs, "hbf.page_programs")
           << ",\"block_erases\":"
           << d(after.block_erases, before.block_erases, "hbf.block_erases")
           << ",\"invalidations\":"
           << d(after.invalidations, before.invalidations, "hbf.invalidations")
           << ",\"mapping_lookup_ops\":"
           << d(after.mapping_lookup_ops, before.mapping_lookup_ops,
                "hbf.mapping_lookup_ops")
           << ",\"mapping_user_lookup_ops\":"
           << d(after.mapping_user_lookup_ops, before.mapping_user_lookup_ops,
                "hbf.mapping_user_lookup_ops")
           << ",\"mapping_gc_lookup_ops\":"
           << d(after.mapping_gc_lookup_ops, before.mapping_gc_lookup_ops,
                "hbf.mapping_gc_lookup_ops")
           << ",\"mapping_update_ops\":"
           << d(after.mapping_update_ops, before.mapping_update_ops,
                "hbf.mapping_update_ops")
           << ",\"mapping_user_update_ops\":"
           << d(after.mapping_user_update_ops, before.mapping_user_update_ops,
                "hbf.mapping_user_update_ops")
           << ",\"mapping_gc_update_ops\":"
           << d(after.mapping_gc_update_ops, before.mapping_gc_update_ops,
                "hbf.mapping_gc_update_ops")
           << ",\"mapping_dram_wait_ops\":"
           << d(after.mapping_dram_wait_ops, before.mapping_dram_wait_ops,
                "hbf.mapping_dram_wait_ops")
           << ",\"mapping_dram_wait_work_ns\":"
           << session_work_delta(
                  after.mapping_dram_wait_ns,
                  before.mapping_dram_wait_ns,
                  "hbf.mapping_dram_wait_ns")
           << ",\"mapping_dram_issue_busy_ns\":"
           << session_work_delta(
                  after.mapping_dram_issue_busy_ns,
                  before.mapping_dram_issue_busy_ns,
                  "hbf.mapping_dram_issue_busy_ns")
           << ",\"mapping_cache_hits\":"
           << d(after.mapping_cache_hits, before.mapping_cache_hits,
                "hbf.mapping_cache_hits")
           << ",\"mapping_cache_misses\":"
           << d(after.mapping_cache_misses, before.mapping_cache_misses,
                "hbf.mapping_cache_misses")
           << ",\"mapping_cache_erased_misses\":"
           << d(after.mapping_cache_erased_misses,
                before.mapping_cache_erased_misses,
                "hbf.mapping_cache_erased_misses")
           << ",\"mapping_cache_coalesced_misses\":"
           << d(after.mapping_cache_coalesced_misses,
                before.mapping_cache_coalesced_misses,
                "hbf.mapping_cache_coalesced_misses")
           << ",\"mapping_cache_evictions\":"
           << d(after.mapping_cache_evictions,
                before.mapping_cache_evictions,
                "hbf.mapping_cache_evictions")
           << ",\"mapping_cache_dirty_evictions\":"
           << d(after.mapping_cache_dirty_evictions,
                before.mapping_cache_dirty_evictions,
                "hbf.mapping_cache_dirty_evictions")
           << ",\"mapping_media_reads\":"
           << d(after.mapping_media_reads, before.mapping_media_reads,
                "hbf.mapping_media_reads")
           << ",\"mapping_media_read_bytes\":"
           << d(after.mapping_media_read_bytes,
                before.mapping_media_read_bytes,
                "hbf.mapping_media_read_bytes")
           << ",\"read_buffer_hits\":"
           << d(after.read_buffer_hits, before.read_buffer_hits,
                "hbf.read_buffer_hits")
           << ",\"read_buffer_misses\":"
           << d(after.read_buffer_misses, before.read_buffer_misses,
                "hbf.read_buffer_misses")
           << ",\"read_buffer_read_bytes\":"
           << d(after.read_buffer_read_bytes, before.read_buffer_read_bytes,
                "hbf.read_buffer_read_bytes")
           << ",\"write_buffer_hits\":"
           << d(after.write_buffer_hits, before.write_buffer_hits,
                "hbf.write_buffer_hits")
           << ",\"write_buffer_misses\":"
           << d(after.write_buffer_misses, before.write_buffer_misses,
                "hbf.write_buffer_misses")
           << ",\"write_buffer_flushes\":"
           << d(after.write_buffer_flushes, before.write_buffer_flushes,
                "hbf.write_buffer_flushes")
           << ",\"write_buffer_merged_bytes\":"
           << d(after.write_buffer_merged_bytes,
                before.write_buffer_merged_bytes,
                "hbf.write_buffer_merged_bytes")
           << ",\"write_buffer_dram_read_ops\":"
           << d(after.write_buffer_dram_read_ops,
                before.write_buffer_dram_read_ops,
                "hbf.write_buffer_dram_read_ops")
           << ",\"write_buffer_dram_write_ops\":"
           << d(after.write_buffer_dram_write_ops,
                before.write_buffer_dram_write_ops,
                "hbf.write_buffer_dram_write_ops")
           << ",\"write_buffer_dram_read_bytes\":"
           << d(after.write_buffer_dram_read_bytes,
                before.write_buffer_dram_read_bytes,
                "hbf.write_buffer_dram_read_bytes")
           << ",\"write_buffer_dram_write_bytes\":"
           << d(after.write_buffer_dram_write_bytes,
                before.write_buffer_dram_write_bytes,
                "hbf.write_buffer_dram_write_bytes")
           << ",\"write_buffer_dram_wait_ops\":"
           << d(after.write_buffer_dram_wait_ops,
                before.write_buffer_dram_wait_ops,
                "hbf.write_buffer_dram_wait_ops")
           << ",\"write_buffer_dram_wait_work_ns\":"
           << session_work_delta(
                  after.write_buffer_dram_wait_ns,
                  before.write_buffer_dram_wait_ns,
                  "hbf.write_buffer_dram_wait_ns")
           << ",\"write_buffer_dram_issue_busy_ns\":"
           << session_work_delta(
                  after.write_buffer_dram_issue_busy_ns,
                  before.write_buffer_dram_issue_busy_ns,
                  "hbf.write_buffer_dram_issue_busy_ns")
           << ",\"write_buffer_slot_wait_ops\":"
           << d(after.write_buffer_slot_wait_ops,
                before.write_buffer_slot_wait_ops,
                "hbf.write_buffer_slot_wait_ops")
           << ",\"write_buffer_slot_wait_work_ns\":"
           << session_work_delta(
                  after.write_buffer_slot_wait_ns,
                  before.write_buffer_slot_wait_ns,
                  "hbf.write_buffer_slot_wait_ns")
           << ",\"read_splits\":"
           << d(after.read_splits, before.read_splits, "hbf.read_splits")
           << ",\"read_split_pages\":"
           << d(after.read_split_pages, before.read_split_pages,
                "hbf.read_split_pages")
           << ",\"page_read_admission_events\":"
           << d(after.page_read_admission_events,
                before.page_read_admission_events,
                "hbf.page_read_admission_events")
           << ",\"page_read_admission_waited_pages\":"
           << d(after.page_read_admission_waited_pages,
                before.page_read_admission_waited_pages,
                "hbf.page_read_admission_waited_pages")
           << ",\"page_read_admission_wait_work_ns\":"
           << session_work_delta(
                  after.page_read_admission_wait_ns,
                  before.page_read_admission_wait_ns,
                  "hbf.page_read_admission_wait_ns")
           << ",\"flash_scheduler_enqueues\":"
           << d(after.flash_scheduler_enqueues,
                before.flash_scheduler_enqueues,
                "hbf.flash_scheduler_enqueues")
           << ",\"flash_scheduler_issues\":"
           << d(after.flash_scheduler_issues,
                before.flash_scheduler_issues,
                "hbf.flash_scheduler_issues")
           << ",\"gc_runs\":"
           << d(after.gc_runs, before.gc_runs, "hbf.gc_runs")
           << ",\"gc_relocations\":"
           << d(after.gc_relocations, before.gc_relocations,
                "hbf.gc_relocations")
           << ",\"gc_data_relocations\":"
           << d(after.gc_data_relocations, before.gc_data_relocations,
                "hbf.gc_data_relocations")
           << ",\"gc_mapping_relocations\":"
           << d(after.gc_mapping_relocations, before.gc_mapping_relocations,
                "hbf.gc_mapping_relocations")
           << ",\"gc_reclaimed_invalid_pages\":"
           << d(after.gc_reclaimed_invalid_pages,
                before.gc_reclaimed_invalid_pages,
                "hbf.gc_reclaimed_invalid_pages")
           << ",\"gc_user_blocked_runs\":"
           << d(after.gc_user_blocked_runs, before.gc_user_blocked_runs,
                "hbf.gc_user_blocked_runs")
           << ",\"static_wear_leveling_checks\":"
           << d(after.static_wear_leveling_checks,
                before.static_wear_leveling_checks,
                "hbf.static_wear_leveling_checks")
           << ",\"static_wear_leveling_runs\":"
           << d(after.static_wear_leveling_runs,
                before.static_wear_leveling_runs,
                "hbf.static_wear_leveling_runs")
           << ",\"static_wear_leveling_relocations\":"
           << d(after.static_wear_leveling_relocations,
                before.static_wear_leveling_relocations,
                "hbf.static_wear_leveling_relocations")
           << ",\"static_wear_leveling_reclaimed_invalid_pages\":"
           << d(after.static_wear_leveling_reclaimed_invalid_pages,
                before.static_wear_leveling_reclaimed_invalid_pages,
                "hbf.static_wear_leveling_reclaimed_invalid_pages")
           << ",\"thermal_throttled_media_ops\":"
           << d(after.thermal_throttled_media_ops,
                before.thermal_throttled_media_ops,
                "hbf.thermal_throttled_media_ops")
           << ",\"thermal_throttle_engagements\":"
           << d(after.thermal_throttle_engagements,
                before.thermal_throttle_engagements,
                "hbf.thermal_throttle_engagements")
           << ",\"thermal_throttle_wait_work_ns\":"
           << session_work_delta(
                  after.thermal_throttle_wait_ns,
                  before.thermal_throttle_wait_ns,
                  "hbf.thermal_throttle_wait_ns")
           << ",\"thermal_pacing_busy_ns\":"
           << session_work_delta(
                  after.thermal_pacing_busy_ns,
                  before.thermal_pacing_busy_ns,
                  "hbf.thermal_pacing_busy_ns")
           << ",\"thermal_throttled_span_ns\":"
           << session_work_delta(
                  after.thermal_throttled_span_ns,
                  before.thermal_throttled_span_ns,
                  "hbf.thermal_throttled_span_ns")
           << ",\"thermal_media_energy_j\":"
           << session_work_delta(
                  after.thermal_media_energy_j,
                  before.thermal_media_energy_j,
                  "hbf.thermal_media_energy_j")
           << ",\"ecc_decode_ops\":"
           << d(after.ecc_decode_ops, before.ecc_decode_ops,
                "hbf.ecc_decode_ops")
           << ",\"ecc_encode_ops\":"
           << d(after.ecc_encode_ops, before.ecc_encode_ops,
                "hbf.ecc_encode_ops")
           << ",\"ecc_decode_codeword_bytes\":"
           << d(after.ecc_decode_codeword_bytes,
                before.ecc_decode_codeword_bytes,
                "hbf.ecc_decode_codeword_bytes")
           << ",\"ecc_encode_codeword_bytes\":"
           << d(after.ecc_encode_codeword_bytes,
                before.ecc_encode_codeword_bytes,
                "hbf.ecc_encode_codeword_bytes")
           << ",\"ecc_decode_queue_wait_work_ns\":"
           << session_work_delta(
                  after.ecc_decode_queue_wait_ns,
                  before.ecc_decode_queue_wait_ns,
                  "hbf.ecc_decode_queue_wait_ns")
           << ",\"ecc_encode_queue_wait_work_ns\":"
           << session_work_delta(
                  after.ecc_encode_queue_wait_ns,
                  before.ecc_encode_queue_wait_ns,
                  "hbf.ecc_encode_queue_wait_ns")
           << ",\"ecc_decode_latency_work_ns\":"
           << session_work_delta(
                  after.ecc_decode_latency_work_ns,
                  before.ecc_decode_latency_work_ns,
                  "hbf.ecc_decode_latency_work_ns")
           << ",\"ecc_encode_latency_work_ns\":"
           << session_work_delta(
                  after.ecc_encode_latency_work_ns,
                  before.ecc_encode_latency_work_ns,
                  "hbf.ecc_encode_latency_work_ns")
           << ",\"stage_work\":";
    write_session_breakdown(
        output, session_breakdown_delta(after.stage_work, before.stage_work));
    if (include_state) {
        const auto mean_erase_count = after.writable_blocks == 0 ? 0.0 :
            static_cast<double>(after.block_erase_count_sum) /
                static_cast<double>(after.writable_blocks);
        const auto mean_square = after.writable_blocks == 0 ? 0.0 :
            static_cast<double>(after.block_erase_count_sum_squares /
                static_cast<long double>(after.writable_blocks));
        const auto variance = std::max(
            0.0, mean_square - mean_erase_count * mean_erase_count);
        output << ",\"state\":{\"total_pages\":" << after.total_pages
               << ",\"free_pages\":" << after.free_pages
               << ",\"valid_pages\":" << after.valid_pages
               << ",\"invalid_pages\":" << after.invalid_pages
               << ",\"pending_program_pages\":"
               << after.pending_program_pages
               << ",\"pending_mapping_publications\":"
               << after.pending_mapping_publications
               << ",\"static_unmaterialized_pages\":"
               << after.static_unmaterialized_pages
               << ",\"accounting_verified\":"
               << (after.accounting_verified ? "true" : "false")
               << ",\"static_reserved_pages\":"
               << after.static_reserved_pages
               << ",\"raw_reserved_pages\":"
               << after.raw_reserved_pages
               << ",\"initial_logical_data_pages\":"
               << after.initial_logical_data_pages
               << ",\"initial_mapping_pages\":"
               << after.initial_mapping_pages
               << ",\"mapping_table_bytes\":"
               << after.mapping_table_bytes
               << ",\"resident_mapping_table_bytes\":"
               << after.resident_mapping_table_bytes
               << ",\"mapping_directory_entry_bytes\":"
               << after.mapping_directory_entry_bytes
               << ",\"mapping_directory_bytes\":"
               << after.mapping_directory_bytes
               << ",\"mapping_directory_bytes_per_stack\":"
               << after.mapping_directory_bytes_per_stack
               << ",\"mapping_cache_capacity_bytes\":"
               << after.mapping_cache_capacity_bytes
               << ",\"mapping_cache_capacity_bytes_per_stack\":"
               << after.mapping_cache_capacity_bytes_per_stack
               << ",\"mapping_cache_pages_per_stack\":"
               << after.mapping_cache_pages_per_stack
               << ",\"controller_dram_budget_bytes\":"
               << after.controller_dram_budget_bytes
               << ",\"controller_dram_budget_bytes_per_stack\":"
               << after.controller_dram_budget_bytes_per_stack
               << ",\"write_buffer_capacity_bytes\":"
               << after.write_buffer_capacity_bytes
               << ",\"write_buffer_capacity_bytes_per_stack\":"
               << after.write_buffer_capacity_bytes_per_stack
               << ",\"mapping_cache_entries\":"
               << after.mapping_cache_entries
               << ",\"mapping_cache_peak_entries\":"
               << after.mapping_cache_peak_entries
               << ",\"mapping_entries\":" << after.mapping_entries
               << ",\"writable_blocks\":" << after.writable_blocks
               << ",\"writable_pages\":" << after.writable_pages
               << ",\"writable_payload_bytes\":"
               << after.writable_payload_bytes
               << ",\"worn_blocks\":" << after.worn_blocks
               << ",\"block_erase_count_sum\":"
               << after.block_erase_count_sum
               << ",\"min_block_erase_count\":"
               << after.min_block_erase_count
               << ",\"max_block_erase_count\":"
               << after.max_block_erase_count
               << ",\"block_erase_count_histogram\":";
        {
            output << '{';
            bool first_bin = true;
            for (const auto& [erase_count, blocks] :
                 after.block_erase_count_histogram) {
                output << (first_bin ? "" : ",") << '"' << erase_count
                       << "\":" << blocks;
                first_bin = false;
            }
            output << '}';
        }
        output
               << ",\"mean_block_erase_count\":" << mean_erase_count
               << ",\"block_erase_count_stddev\":" << std::sqrt(variance)
               << ",\"thermal_enabled\":"
               << (after.thermal_enabled ? "true" : "false")
               << ",\"thermal_boundary_temperature_c\":"
               << after.thermal_boundary_temperature_c
               << ",\"thermal_boot_temperature_c\":"
               << after.thermal_boot_temperature_c
               << ",\"thermal_peak_temperature_c\":"
               << after.thermal_peak_temperature_c
               << ",\"thermal_final_temperature_c\":"
               << after.thermal_final_temperature_c
               << ",\"thermal_throttled_stacks\":"
               << after.thermal_throttled_stacks
               << "}";
        output << ",\"resource_busy\":{"
                  "\"active_span_ns\":" << after.active_span_ns()
               << ",\"media_busy_ns\":" << after.media_busy_ns
               << ",\"hb_io_command_busy_ns\":"
               << after.hb_io_command_busy_ns
               << ",\"hb_io_data_busy_ns\":" << after.hb_io_data_busy_ns
               << ",\"sequencer_busy_ns\":" << after.sequencer_busy_ns
               << ",\"page_buffer_bank_busy_ns\":"
               << after.page_buffer_bank_busy_ns
               << ",\"read_lane_busy_ns\":" << after.read_lane_busy_ns
               << ",\"subarray_read_busy_ns\":"
               << after.subarray_read_busy_ns
               << ",\"ecc_issue_busy_ns\":" << after.ecc_issue_busy_ns
               << ",\"logic_ingress_busy_ns\":"
               << after.logic_ingress_busy_ns
               << ",\"tsv_busy_ns\":" << after.tsv_busy_ns
               << ",\"sram_busy_ns\":" << after.sram_busy_ns
               << ",\"flash_source_queue_busy_ns\":"
               << after.flash_source_queue_busy_ns
               << ",\"channel_command_busy_ns\":"
               << after.channel_command_busy_ns
               << ",\"channel_data_busy_ns\":"
               << after.channel_data_busy_ns
               << ",\"stacks\":" << after.stacks
               << ",\"channels\":" << after.channels
               << ",\"dies\":" << after.dies
               << ",\"planes\":" << after.planes
               << ",\"media_lanes\":" << after.media_lanes
               << ",\"subarrays\":" << after.subarrays
               << ",\"page_buffer_banks\":" << after.page_buffer_banks
               << ",\"logic_ingress_resources\":"
               << after.logic_ingress_resources
               << ",\"tsv_resources\":" << after.tsv_resources
               << ",\"sram_resources\":" << after.sram_resources
               << ",\"flash_source_queue_resources\":"
               << after.flash_source_queue_resources
               << ",\"channel_command_resources\":"
               << after.channel_command_resources
               << ",\"channel_data_resources\":"
               << after.channel_data_resources
               << ",\"media_utilization\":"
               << after.media_utilization()
               << ",\"io_utilization\":" << after.io_utilization()
               << ",\"hbio_command_utilization\":"
               << after.hbio_command_utilization()
               << ",\"hbio_data_utilization\":"
               << after.hbio_data_utilization()
               << ",\"ecc_issue_utilization\":"
               << after.ecc_issue_utilization()
               << ",\"media_parallelism\":"
               << after.media_parallelism()
               << ",\"read_lane_parallelism\":"
               << after.read_lane_parallelism()
               << ",\"subarray_read_parallelism\":"
               << after.subarray_read_parallelism()
               << ",\"channel_parallelism\":"
               << after.channel_parallelism()
               << ",\"hbio_parallelism\":"
               << after.hbio_parallelism()
               << ",\"ecc_issue_parallelism\":"
               << after.ecc_issue_parallelism()
               << '}';
    }
    output << '}';
}

void write_session_link_accounting(
    std::ostream& output,
    const BaseDieLinkStats& after,
    const BaseDieLinkStats& before,
    bool include_state) {
    output << "{\"read_transfers\":"
           << session_counter_delta(
                  after.read_transfers,
                  before.read_transfers,
                  "d2d.read_transfers")
           << ",\"write_transfers\":"
           << session_counter_delta(
                  after.write_transfers,
                  before.write_transfers,
                  "d2d.write_transfers")
           << ",\"read_bytes\":"
           << session_counter_delta(
                  after.read_bytes, before.read_bytes, "d2d.read_bytes")
           << ",\"write_bytes\":"
           << session_counter_delta(
                  after.write_bytes, before.write_bytes, "d2d.write_bytes")
           << ",\"read_queue_wait_work_ns\":"
           << session_work_delta(
                  after.read_queue_wait_ns,
                  before.read_queue_wait_ns,
                  "d2d.read_queue_wait_ns")
           << ",\"write_queue_wait_work_ns\":"
           << session_work_delta(
                  after.write_queue_wait_ns,
                  before.write_queue_wait_ns,
                  "d2d.write_queue_wait_ns")
           << ",\"read_busy_ns\":"
           << session_work_delta(
                  after.read_busy_ns, before.read_busy_ns, "d2d.read_busy_ns")
           << ",\"write_busy_ns\":"
           << session_work_delta(
                  after.write_busy_ns,
                  before.write_busy_ns,
                  "d2d.write_busy_ns")
           << ",\"read_fixed_latency_work_ns\":"
           << session_work_delta(
                  after.read_fixed_latency_work_ns,
                  before.read_fixed_latency_work_ns,
                  "d2d.read_fixed_latency_work_ns")
           << ",\"write_fixed_latency_work_ns\":"
           << session_work_delta(
                  after.write_fixed_latency_work_ns,
                  before.write_fixed_latency_work_ns,
                  "d2d.write_fixed_latency_work_ns");
    if (include_state) {
        output << ",\"links\":" << after.links
               << ",\"active_span_ns\":" << after.active_span_ns()
               << ",\"read_utilization\":" << after.read_utilization()
               << ",\"write_utilization\":" << after.write_utilization();
    }
    output << '}';
}

void write_session_external_accounting(
    std::ostream& output,
    const ExternalBackingStats& after,
    const ExternalBackingStats& before,
    bool include_state) {
    const auto d = [&](std::uint64_t value, std::uint64_t baseline,
                       std::string_view name) {
        return session_counter_delta(value, baseline, name);
    };
    const auto w = [&](double value, double baseline, std::string_view name) {
        return session_work_delta(value, baseline, name);
    };
    output << "{\"kind\":\""
           << hbfsim::physical::external::to_string(after.kind)
           << "\",\"read_requests\":"
           << d(after.read_requests, before.read_requests,
                "external.read_requests")
           << ",\"write_requests\":"
           << d(after.write_requests, before.write_requests,
                "external.write_requests")
           << ",\"read_bytes\":"
           << d(after.read_bytes, before.read_bytes, "external.read_bytes")
           << ",\"write_bytes\":"
           << d(after.write_bytes, before.write_bytes, "external.write_bytes")
           << ",\"page_run_requests\":"
           << d(after.page_run_requests, before.page_run_requests,
                "external.page_run_requests")
           << ",\"page_run_segments\":"
           << d(after.page_run_segments, before.page_run_segments,
                "external.page_run_segments")
           << ",\"page_run_pages\":"
           << d(after.page_run_pages, before.page_run_pages,
                "external.page_run_pages")
           << ",\"outstanding_wait_work_ns\":"
           << w(after.outstanding_wait_ns, before.outstanding_wait_ns,
                "external.outstanding_wait_ns")
           << ",\"controller_queue_wait_work_ns\":"
           << w(after.controller_queue_wait_ns,
                before.controller_queue_wait_ns,
                "external.controller_queue_wait_ns")
           << ",\"controller_issue_busy_ns\":"
           << w(after.controller_issue_busy_ns,
                before.controller_issue_busy_ns,
                "external.controller_issue_busy_ns")
           << ",\"controller_processing_work_ns\":"
           << w(after.controller_processing_work_ns,
                before.controller_processing_work_ns,
                "external.controller_processing_work_ns")
           << ",\"media_queue_wait_work_ns\":"
           << w(after.media_queue_wait_ns, before.media_queue_wait_ns,
                "external.media_queue_wait_ns")
           << ",\"media_read_latency_work_ns\":"
           << w(after.media_read_latency_work_ns,
                before.media_read_latency_work_ns,
                "external.media_read_latency_work_ns")
           << ",\"media_write_latency_work_ns\":"
           << w(after.media_write_latency_work_ns,
                before.media_write_latency_work_ns,
                "external.media_write_latency_work_ns")
           << ",\"media_read_busy_ns\":"
           << w(after.media_read_busy_ns, before.media_read_busy_ns,
                "external.media_read_busy_ns")
           << ",\"media_write_busy_ns\":"
           << w(after.media_write_busy_ns, before.media_write_busy_ns,
                "external.media_write_busy_ns")
           << ",\"m2s_payload_bytes\":"
           << d(after.m2s_payload_bytes, before.m2s_payload_bytes,
                "external.m2s_payload_bytes")
           << ",\"m2s_protocol_bytes\":"
           << d(after.m2s_protocol_bytes, before.m2s_protocol_bytes,
                "external.m2s_protocol_bytes")
           << ",\"m2s_wire_bytes\":"
           << d(after.m2s_wire_bytes, before.m2s_wire_bytes,
                "external.m2s_wire_bytes")
           << ",\"s2m_payload_bytes\":"
           << d(after.s2m_payload_bytes, before.s2m_payload_bytes,
                "external.s2m_payload_bytes")
           << ",\"s2m_protocol_bytes\":"
           << d(after.s2m_protocol_bytes, before.s2m_protocol_bytes,
                "external.s2m_protocol_bytes")
           << ",\"s2m_wire_bytes\":"
           << d(after.s2m_wire_bytes, before.s2m_wire_bytes,
                "external.s2m_wire_bytes")
           << ",\"m2s_queue_wait_work_ns\":"
           << w(after.m2s_queue_wait_ns, before.m2s_queue_wait_ns,
                "external.m2s_queue_wait_ns")
           << ",\"s2m_queue_wait_work_ns\":"
           << w(after.s2m_queue_wait_ns, before.s2m_queue_wait_ns,
                "external.s2m_queue_wait_ns")
           << ",\"m2s_busy_ns\":"
           << w(after.m2s_busy_ns, before.m2s_busy_ns,
                "external.m2s_busy_ns")
           << ",\"s2m_busy_ns\":"
           << w(after.s2m_busy_ns, before.s2m_busy_ns,
                "external.s2m_busy_ns")
           << ",\"transport_propagation_work_ns\":"
           << w(after.transport_propagation_work_ns,
                before.transport_propagation_work_ns,
                "external.transport_propagation_work_ns")
           // CXL-SSD device-cache census. Read-for-ownership, flush, and
           // prefetch traffic never appears as caller requests or wire bytes;
           // the explicit cache census preserves that distinction.
           << ",\"device_cache\":{\"read_hits\":"
           << d(after.cache_read_hits, before.cache_read_hits,
                "external.cache_read_hits")
           << ",\"read_misses\":"
           << d(after.cache_read_misses, before.cache_read_misses,
                "external.cache_read_misses")
           << ",\"write_hits\":"
           << d(after.cache_write_hits, before.cache_write_hits,
                "external.cache_write_hits")
           << ",\"write_misses\":"
           << d(after.cache_write_misses, before.cache_write_misses,
                "external.cache_write_misses")
           << ",\"read_for_ownership_segments\":"
           << d(after.cache_read_for_ownership_segments,
                before.cache_read_for_ownership_segments,
                "external.cache_read_for_ownership_segments")
           << ",\"read_for_ownership_bytes\":"
           << d(after.cache_read_for_ownership_bytes,
                before.cache_read_for_ownership_bytes,
                "external.cache_read_for_ownership_bytes")
           << ",\"writeback_segments\":"
           << d(after.cache_writeback_segments,
                before.cache_writeback_segments,
                "external.cache_writeback_segments")
           << ",\"writeback_bytes\":"
           << d(after.cache_writeback_bytes, before.cache_writeback_bytes,
                "external.cache_writeback_bytes")
           << ",\"prefetch_segments\":"
           << d(after.cache_prefetch_segments,
                before.cache_prefetch_segments,
                "external.cache_prefetch_segments")
           << ",\"prefetch_bytes\":"
           << d(after.cache_prefetch_bytes, before.cache_prefetch_bytes,
                "external.cache_prefetch_bytes")
           << ",\"latency_work_ns\":"
           << w(after.cache_latency_work_ns, before.cache_latency_work_ns,
                "external.cache_latency_work_ns")
           << ",\"queue_wait_work_ns\":"
           << w(after.cache_queue_wait_ns, before.cache_queue_wait_ns,
                "external.cache_queue_wait_ns")
           << ",\"busy_ns\":"
           << w(after.cache_busy_ns, before.cache_busy_ns,
                "external.cache_busy_ns")
           << ",\"flush_busy_ns\":"
           << w(after.cache_flush_busy_ns, before.cache_flush_busy_ns,
                "external.cache_flush_busy_ns")
           << ",\"prefetch_busy_ns\":"
           << w(after.cache_prefetch_busy_ns, before.cache_prefetch_busy_ns,
                "external.cache_prefetch_busy_ns")
           << ",\"writeback_gate_wait_work_ns\":"
           << w(after.cache_writeback_gate_wait_ns,
                before.cache_writeback_gate_wait_ns,
                "external.cache_writeback_gate_wait_ns")
           << '}'
           << ",\"stage_work\":";
    write_session_breakdown(
        output,
        session_breakdown_delta(after.stage_work, before.stage_work));
    if (include_state) {
        output << ",\"resource_state\":{\"media_channels\":"
               << after.media_channels
               << ",\"media_channels_per_queue\":"
               << after.media_channels_per_queue
               << ",\"media_read_queues\":"
               << after.media_read_queues
               << ",\"media_write_queues\":"
               << after.media_write_queues
               << ",\"active_media_resources\":"
               << after.active_media_resources
               << ",\"max_device_outstanding\":"
               << after.max_device_outstanding
               << ",\"active_span_ns\":" << after.active_span_ns()
               << ",\"controller_active_span_ns\":"
               << after.controller_active_span_ns()
               << ",\"media_active_span_ns\":"
               << after.media_active_span_ns()
               << ",\"m2s_active_span_ns\":"
               << after.m2s_active_span_ns()
               << ",\"s2m_active_span_ns\":"
               << after.s2m_active_span_ns()
               << ",\"controller_utilization\":"
               << after.controller_utilization()
               << ",\"media_utilization\":"
               << after.media_utilization()
               << ",\"m2s_utilization\":"
               << after.m2s_utilization()
               << ",\"s2m_utilization\":"
               << after.s2m_utilization() << '}';
    }
    output << '}';
}

void write_session_device_accounting(
    std::ostream& output,
    const SimulationDeviceSnapshot& after,
    const SimulationDeviceSnapshot& before,
    bool include_state) {
    if (after.has_hbm != before.has_hbm || after.has_hbf != before.has_hbf ||
        after.has_external != before.has_external) {
        throw std::runtime_error("simulation device availability changed in-session");
    }
    output << "{\"hbm\":";
    if (after.has_hbm) {
        write_session_hbm_accounting(
            output, after.hbm, before.hbm, include_state);
    } else {
        output << "null";
    }
    output << ",\"hbf\":";
    if (after.has_hbf) {
        write_session_hbf_accounting(output, after.hbf, before.hbf, include_state);
    } else {
        output << "null";
    }
    output << ",\"external\":";
    if (after.has_external) {
        write_session_external_accounting(
            output, after.external, before.external, include_state);
    } else {
        output << "null";
    }
    output << ",\"base_die_link\":";
    write_session_link_accounting(
        output, after.base_die_link, before.base_die_link, include_state);
    output << ",\"hbf_external_direct_link\":";
    write_session_link_accounting(
        output,
        after.hbf_external_direct_link,
        before.hbf_external_direct_link,
        include_state);
    output << '}';
}

void write_session_quiescence(
    std::ostream& output,
    const physical::hbf::HbfQuiescenceStats& quiescence) {
    output << "{\"verified\":"
           << (quiescence.quiescent() ? "true" : "false")
           << ",\"dirty_mapping_pages\":"
           << quiescence.dirty_mapping_pages
           << ",\"pending_dirty_mapping_events\":"
           << quiescence.pending_dirty_mapping_events
           << ",\"pending_lpn_updates\":"
           << quiescence.pending_lpn_updates
           << ",\"pending_vpn_updates\":"
           << quiescence.pending_vpn_updates
           << ",\"pending_commits\":" << quiescence.pending_commits
           << ",\"write_buffer_entries\":"
           << quiescence.write_buffer_entries
           << ",\"inflight_buffered_generations\":"
           << quiescence.inflight_buffered_generations
           << ",\"pending_physical_programs\":"
           << quiescence.pending_physical_programs
           << ",\"pending_physical_erases\":"
           << quiescence.pending_physical_erases << '}';
}

// A pre-mutation rejection: the session is unchanged and stays usable.
void write_session_batch_error(
    std::ostream& output,
    const std::string& batch_id,
    const std::string& message) {
    output << "{\"schema\":{\"name\":\"hbfsim.simulation_batch_completion\","
              "\"version\":1},\"result\":\"error\",\"batch_id\":\""
           << json_escape(batch_id)
           << "\",\"message\":\"" << json_escape(message)
           << "\",\"session_state\":\"unchanged\"}\n";
    output.flush();
    if (!output) {
        throw std::runtime_error("failed to write simulation-session error");
    }
}

void write_session_batch_result(
    std::ostream& output,
    const SimulationBatchResult& result,
    const std::string& logical_sha256,
    const std::string& transaction_sha256,
    bool include_completions) {
    output << std::setprecision(17)
           << "{\"schema\":{\"name\":\"hbfsim.simulation_batch_completion\","
              "\"version\":1},\"result\":\"pass\",\"batch_id\":\""
           << json_escape(result.batch_id)
           << "\",\"sequence\":" << result.sequence
           << ",\"logical_trace_sha256\":\"" << logical_sha256
           << "\",\"transaction_trace_sha256\":\"" << transaction_sha256
           << "\",\"transactions\":" << result.transactions
           << ",\"memory_transactions\":" << result.memory_transactions
           << ",\"barriers\":" << result.barriers
           << ",\"dependency_edges\":" << result.dependency_edges
           << ",\"frontier_transactions\":" << result.frontier_transactions
           << ",\"transaction_bytes\":" << result.transaction_bytes
           << ",\"time_basis\":\"batch_relative_ns\""
           << ",\"batch_origin_ns\":" << result.batch_origin_ns
           << ",\"first_issue_ns\":" << result.first_issue_ns
           << ",\"finish_ns\":" << result.finish_ns
           << ",\"blocking_finish_ns\":" << result.blocking_finish_ns
           << ",\"elapsed_ns\":"
           << (result.blocking_finish_ns - result.batch_origin_ns)
           << ",\"total_elapsed_ns\":"
           << (result.finish_ns - result.batch_origin_ns)
           << ",\"transaction_latency_by_target\":";
    write_session_completion_matrix(output, result.completion_by_target);
    if (include_completions) {
        output << ",\"transaction_completions\":";
        write_session_transaction_completions(output, result.completions);
        output << ",\"transaction_completions_digest\":{"
                  "\"algorithm\":\"sha256_id_ieee754bits_bytes_v1\","
                  "\"sha256\":\""
               << session_transaction_completion_digest(result.completions)
               << "\"}";
    } else {
        output << ",\"transaction_completions\":null"
                  ",\"transaction_completions_digest\":null";
    }
    output << ",\"device_delta\":";
    write_session_device_accounting(
        output, result.device_after, result.device_before, false);
    output << ",\"hbm_engine\":{"
           << "\"requests\":"
           << result.hbm_engine.requests
           << ",\"bursts\":"
           << result.hbm_engine.bursts
           << "},\"hbf_read_engine\":{"
           << "\"page_run_requests\":"
           << result.hbf_read_engine.page_run_requests
           << ",\"page_run_pages\":"
           << result.hbf_read_engine.page_run_pages
           << ",\"page_run_physical_requests\":"
           << result.hbf_read_engine.page_run_physical_requests
           << ",\"page_run_static_requests\":"
           << result.hbf_read_engine.page_run_static_requests
           << ",\"page_run_logical_requests\":"
           << result.hbf_read_engine.page_run_logical_requests
           << ",\"streaming_read_buffer_bypass_pages\":"
           << result.hbf_read_engine.streaming_read_buffer_bypass_pages
           << ",\"scalar_read_requests\":"
           << result.hbf_read_engine.scalar_read_requests
           << ",\"scalar_read_pages\":"
           << result.hbf_read_engine.scalar_read_pages
           << "},\"by_target\":{";
    for (std::size_t index = 0; index < result.by_target.size(); ++index) {
        if (index != 0) output << ',';
        const auto target = static_cast<SimulationTarget>(index);
        output << '\"' << hbfsim::physical::to_string(target)
               << "\":{\"transactions\":"
               << result.by_target[index].transactions
               << ",\"bytes\":" << result.by_target[index].bytes << '}';
    }
    output << "}}\n";
    output.flush();
    if (!output) {
        throw std::runtime_error("failed to write simulation-session completion");
    }
}

double write_session_checkpoint(
    std::ostream& output,
    SimulationSession& replay,
    std::string checkpoint_id,
    std::optional<std::filesystem::path> persistent_image_path = std::nullopt) {
    const auto result = replay.checkpoint_pending(std::move(checkpoint_id));
    const auto& persistence = result.persistence;
    std::optional<SessionPersistentImageArtifact> image_artifact;
    if (persistent_image_path) {
        hbfsim::physical::hbf::write_persistent_image_file(
            *persistent_image_path,
            replay.persistent_hbf_image());
        image_artifact = SessionPersistentImageArtifact{
            .path = persistent_image_path->string(),
            .bytes = std::filesystem::file_size(*persistent_image_path),
            .sha256 = sha256_file(*persistent_image_path),
        };
    }
    output << std::setprecision(17)
           << "{\"schema\":{\"name\":"
              "\"hbfsim.simulation_checkpoint_completion\",\"version\":1},"
              "\"result\":\"pass\",\"checkpoint_id\":\""
           << json_escape(result.checkpoint_id)
           << "\",\"sequence\":" << result.sequence
           << ",\"has_hbf\":"
           << (persistence.has_hbf ? "true" : "false")
           << ",\"arrival_frontier_ns\":"
           << persistence.serving_frontier_ns
           << ",\"finish_ns\":" << persistence.finish_ns
           << ",\"elapsed_ns\":"
           << (persistence.finish_ns - persistence.serving_frontier_ns)
           << ",\"causal_for_subsequent_batches\":true"
           << ",\"completion_note\":\""
           << json_escape(persistence.hbf_completion.note) << '"'
           << ",\"physical_bytes\":"
           << persistence.hbf_completion.physical_bytes
           << ",\"device_delta\":";
    write_session_device_accounting(
        output,
        persistence.device_after,
        persistence.device_before,
        false);
    output << ",\"persistent_image\":";
    if (image_artifact) {
        output << "{\"schema\":{\"name\":"
                  "\"hbfsim.hbf_persistent_image\",\"version\":2},"
                  "\"path\":\""
               << json_escape(image_artifact->path)
               << "\",\"bytes\":" << image_artifact->bytes
               << ",\"sha256\":\"" << image_artifact->sha256 << "\"}";
    } else {
        output << "null";
    }
    output << ",\"quiescence\":";
    write_session_quiescence(output, persistence.quiescence);
    output << "}\n";
    output.flush();
    if (!output) {
        throw std::runtime_error(
            "failed to write simulation-session checkpoint completion");
    }
    return persistence.finish_ns;
}

void write_session_crash(
    std::ostream& output,
    SimulationSession& replay,
    std::string crash_id) {
    const auto result = replay.inject_crash(std::move(crash_id));
    auto zero_snapshot = result.device_at_injection;
    zero_snapshot.hbm = {};
    zero_snapshot.hbf = {};
    zero_snapshot.external = {};
    zero_snapshot.base_die_link = {};
    zero_snapshot.base_die_link.links =
        result.device_at_injection.base_die_link.links;
    zero_snapshot.hbf_external_direct_link = {};
    zero_snapshot.hbf_external_direct_link.links =
        result.device_at_injection.hbf_external_direct_link.links;
    output << std::setprecision(17)
           << "{\"schema\":{\"name\":"
              "\"hbfsim.simulation_crash_completion\",\"version\":1},"
              "\"result\":\"crashed\",\"crash_id\":\""
           << json_escape(result.crash_id)
           << "\",\"completed_batches\":" << result.completed_batches
           << ",\"completed_checkpoints\":"
           << result.completed_checkpoints
           << ",\"completed_frontier_ns\":"
           << result.completed_frontier_ns
           << ",\"injection_boundary\":"
              "\"after_previous_completed_protocol_command\""
           << ",\"terminal_drain_performed\":false"
           << ",\"device_workload_totals\":";
    write_session_device_accounting(
        output,
        result.device_at_injection,
        zero_snapshot,
        false);
    output << ",\"quiescence_at_injection\":";
    write_session_quiescence(output, result.quiescence);
    output << "}\n";
    output.flush();
    if (!output) {
        throw std::runtime_error(
            "failed to write simulation-session crash completion");
    }
}

void write_session_wear_snapshot(
    std::ostream& output,
    const SimulationSession& replay,
    const std::string& snapshot_id) {
    const auto* stats = replay.hbf_stats();
    if (stats == nullptr) {
        throw std::runtime_error(
            "simulation wear snapshot requires an enabled HBF tier");
    }
    const auto counts = replay.hbf_block_erase_counts();
    std::uint64_t erase_count_sum = 0;
    for (const auto count : counts) {
        erase_count_sum += count;
    }
    if (counts.size() != stats->writable_blocks ||
        erase_count_sum != stats->block_erase_count_sum) {
        throw std::runtime_error(
            "simulation wear snapshot diverged from HBF accounting");
    }
    output << std::setprecision(17)
           << "{\"schema\":{\"name\":\"hbfsim.hbf_wear_snapshot\","
              "\"version\":1},\"result\":\"pass\",\"snapshot_id\":\""
           << json_escape(snapshot_id)
           << "\",\"completed_frontier_ns\":"
           << replay.completed_frontier_ns()
           << ",\"writable_blocks\":" << counts.size()
           << ",\"block_erase_count_sum\":" << erase_count_sum
           << ",\"block_erase_counts\":[";
    for (std::size_t index = 0; index < counts.size(); ++index) {
        output << (index == 0 ? "" : ",") << counts[index];
    }
    output << "]}\n" << std::flush;
    if (!output) {
        throw std::runtime_error(
            "failed to write simulation-session wear snapshot");
    }
}

double write_session_stop(
    std::ostream& output,
    SimulationSession& replay) {
    const auto drain = replay.drain_pending();
    const auto final_snapshot = replay.device_snapshot();
    auto zero_snapshot = final_snapshot;
    zero_snapshot.hbm = {};
    zero_snapshot.hbf = {};
    zero_snapshot.external = {};
    zero_snapshot.base_die_link = {};
    zero_snapshot.base_die_link.links = final_snapshot.base_die_link.links;
    zero_snapshot.hbf_external_direct_link = {};
    zero_snapshot.hbf_external_direct_link.links =
        final_snapshot.hbf_external_direct_link.links;
    output << std::setprecision(17)
           << "{\"schema\":{\"name\":\"hbfsim.simulation_session\","
              "\"version\":1},\"result\":\"stopped\""
           << ",\"completed_batches\":" << replay.completed_batches()
           << ",\"completed_checkpoints\":"
           << replay.completed_checkpoints()
           << ",\"completed_frontier_ns\":"
           << replay.completed_frontier_ns()
           << ",\"drained_frontier_ns\":" << drain.finish_ns
           << ",\"transaction_latency_by_target\":";
    write_session_completion_matrix(output, replay.completion_stats());
    output << ",\"device_workload_totals\":";
    write_session_device_accounting(
        output, final_snapshot, zero_snapshot, true);
    output << ",\"end_of_session_drain\":{\"has_hbf\":"
           << (drain.has_hbf ? "true" : "false")
           << ",\"serving_frontier_ns\":" << drain.serving_frontier_ns
           << ",\"finish_ns\":" << drain.finish_ns
           << ",\"tail_ns\":"
           << (drain.finish_ns - drain.serving_frontier_ns)
           << ",\"serving_timing_excludes_drain\":true"
           << ",\"completion_note\":";
    if (drain.has_hbf) {
        output << '\"' << json_escape(drain.hbf_completion.note) << '\"';
    } else {
        output << "null";
    }
    output << ",\"drain_physical_bytes\":"
           << (drain.has_hbf ? drain.hbf_completion.physical_bytes : 0)
           << ",\"device_delta\":";
    write_session_device_accounting(
        output, drain.device_after, drain.device_before, false);
    output << ",\"quiescence\":";
    write_session_quiescence(output, drain.quiescence);
    output << '}';
    output << ",\"measurement_semantics\":{"
              "\"transaction_latency\":"
              "\"arrival-to-physical-completion latency of low-level memory "
              "or D2D transactions; barrier time is excluded\""
              ",\"transaction_completion_export\":"
              "\"ordered digest-bound memory/D2D records retain input ID, "
              "arrival, start, finish, logical bytes, and physical bytes; "
              "barriers are excluded; omitted when the batch was submitted "
              "with completions=0\""
              ",\"batch_frontier\":"
              "\"elapsed_ns and the next batch origin follow the completion "
              "of the BEGIN frontier transactions (all by default), never "
              "earlier than the batch's last arrival; finish_ns and "
              "total_elapsed_ns cover every transaction of the batch\""
              ",\"queue_service_relationship\":"
              "\"latency=queue_wait+service for each transaction\""
              ",\"work_fields\":"
              "\"additive work sums may overlap across resources and "
              "transactions and are not wall-clock decompositions\""
              ",\"busy_fields\":"
              "\"exclusive resource reservation time; normalize only by "
              "the stated resource count and active span\""
              ",\"effective_rate\":"
              "\"decimal GB/s, numerically bytes/ns over the operation "
              "class active span; it is an achieved workload traffic rate, "
              "not peak media bandwidth\""
              ",\"waf\":"
              "\"physical programmed payload bytes divided by host-written "
              "bytes (logical writes plus raw append-to-publish payload); "
              "includes page rounding, mapping programs, and GC relocation, "
              "excludes OOB/ECC/transport framing\""
              ",\"wear_scope\":"
              "\"observed replay-window writes and block erase distribution; "
              "lifetime requires an explicit P/E-cycle and sustained-write-rate "
              "sensitivity model\""
              ",\"drain_scope\":"
              "\"end-of-session persistence drain is excluded from serving "
              "latency and included in physical-write/WAF accounting\""
              ",\"checkpoint_scope\":"
              "\"nonterminal checkpoints persist pending HBF state, advance "
              "the causal frontier, and are included in physical-write/WAF "
              "accounting\"}"
           << "}\n";
    output.flush();
    if (!output) {
        throw std::runtime_error("failed to write simulation-session stop receipt");
    }
    return drain.finish_ns;
}

} // namespace

int run_simulation_session(
    SimulationSessionConfig config,
    std::istream& input,
    std::ostream& output,
    SessionProtocolOptions options) {
    const auto& initial_image = options.initial_image;
    if (static_cast<bool>(config.initial_hbf_persistent_image) !=
        static_cast<bool>(initial_image)) {
        throw std::runtime_error(
            "simulation persistent-image state and provenance diverged");
    }
    if (initial_image &&
        (initial_image->bytes == 0 ||
         !is_lower_hex_sha256(initial_image->sha256) ||
         !std::filesystem::path(initial_image->path).is_absolute())) {
        throw std::runtime_error(
            "simulation persistent-image provenance is malformed");
    }
    const auto session_config = config;
    SimulationSession replay(std::move(config));
    // Live physical heat stream: one JSON line after each batch, checkpoint,
    // and terminal drain with per-bin HBF media byte deltas and block state.
    std::ofstream heat_stream;
    std::vector<std::uint64_t> heat_prev_write;
    std::vector<std::uint64_t> heat_prev_gc_write;
    std::vector<std::uint64_t> heat_prev_mapping_write;
    std::vector<std::uint64_t> heat_prev_erase;
    std::vector<std::uint64_t> heat_prev_read;
    const char* heat_path = options.hbf_physical_heatmap_path ?
        options.hbf_physical_heatmap_path->c_str() : nullptr;
    const auto* heat_initial_stats = heat_path == nullptr ?
        nullptr : replay.hbf_execution_stats();
    std::uint64_t heat_prev_gc_runs = heat_initial_stats == nullptr ?
        0 : heat_initial_stats->gc_runs;
    std::uint64_t heat_prev_gc_relocations = heat_initial_stats == nullptr ?
        0 : heat_initial_stats->gc_relocations;
    std::uint64_t heat_prev_gc_data_relocations = heat_initial_stats == nullptr ?
        0 : heat_initial_stats->gc_data_relocations;
    std::uint64_t heat_prev_gc_mapping_relocations =
        heat_initial_stats == nullptr ?
            0 : heat_initial_stats->gc_mapping_relocations;
    std::uint64_t heat_prev_gc_reclaimed_invalid_pages =
        heat_initial_stats == nullptr ?
            0 : heat_initial_stats->gc_reclaimed_invalid_pages;
    std::uint64_t heat_prev_mapping_page_programs =
        heat_initial_stats == nullptr ?
            0 : heat_initial_stats->mapping_page_programs;
    if ((heat_path != nullptr) != (replay.hbf_heatmap() != nullptr)) {
        throw std::runtime_error(
            "--hbf-physical-heatmap and a positive --hbf-physical-heatmap-bins "
            "must be configured together");
    }
    if (heat_path != nullptr) {
        heat_stream.open(heat_path, std::ios::out | std::ios::app);
        if (!heat_stream) {
            throw std::runtime_error(
                "cannot open HBF physical heatmap stream: " +
                std::string(heat_path));
        }
        const auto& domain = replay.hbf_heatmap()->domain(
            physical::AddressDomain::HbfPhysical);
        heat_prev_write.assign(domain.bins.size(), 0);
        heat_prev_gc_write.assign(domain.bins.size(), 0);
        heat_prev_mapping_write.assign(domain.bins.size(), 0);
        heat_prev_erase.assign(domain.bins.size(), 0);
        heat_prev_read.assign(domain.bins.size(), 0);
        heat_stream << "{\"kind\":\"header\","
                    << "\"schema\":\"hbfsim.hbf_physical_live.v2\","
                    << "\"counter_scope\":\"delta_since_previous_event\","
                    << "\"event_kinds\":[\"batch\",\"checkpoint\",\"stop\"],"
                    << "\"bins\":" << domain.bins.size()
                    << ",\"capacity_bytes\":"
                    << physical::address_boundary_decimal(domain.size_bytes)
                    << ",\"stacks\":" << session_config.hbf.stacks
                    << ",\"channels_per_stack\":" << session_config.hbf.channels_per_stack
                    << ",\"dies_per_channel\":" << session_config.hbf.dies_per_channel
                    << ",\"planes_per_die\":" << session_config.hbf.planes_per_die
                    << ",\"blocks_per_plane\":" << session_config.hbf.blocks_per_plane
                    << ",\"pages_per_block\":" << session_config.hbf.pages_per_block
                    << ",\"page_size_bytes\":" << session_config.hbf.page_size_bytes
                    << "}\n" << std::flush;
    }
    const auto emit_heat_delta = [&](const char* event_kind,
                                     const std::string& event_id,
                                     double finish_ns) {
        if (!heat_stream.is_open()) {
            return;
        }
        const auto& domain = replay.hbf_heatmap()->domain(
            physical::AddressDomain::HbfPhysical);
        const auto gc_slot = static_cast<std::size_t>(
            physical::HeatmapTrafficSource::GarbageCollection);
        const auto mapping_slot = static_cast<std::size_t>(
            physical::HeatmapTrafficSource::Mapping);
        heat_stream << "{\"kind\":\"" << event_kind
                    << "\",\"event\":\"" << json_escape(event_id)
                    << "\",\"finish_ns\":" << std::setprecision(17) << finish_ns
                    << ",\"write\":[";
        for (std::size_t bin = 0; bin < domain.bins.size(); ++bin) {
            const auto value = domain.bins[bin].total.write_bytes;
            heat_stream << (bin ? "," : "") << (value - heat_prev_write[bin]);
            heat_prev_write[bin] = value;
        }
        heat_stream << "],\"gc_write\":[";
        for (std::size_t bin = 0; bin < domain.bins.size(); ++bin) {
            const auto value = domain.bins[bin].by_source[gc_slot].write_bytes;
            heat_stream << (bin ? "," : "") << (value - heat_prev_gc_write[bin]);
            heat_prev_gc_write[bin] = value;
        }
        heat_stream << "],\"mapping_write\":[";
        for (std::size_t bin = 0; bin < domain.bins.size(); ++bin) {
            const auto value =
                domain.bins[bin].by_source[mapping_slot].write_bytes;
            heat_stream << (bin ? "," : "")
                        << (value - heat_prev_mapping_write[bin]);
            heat_prev_mapping_write[bin] = value;
        }
        heat_stream << "],\"erase\":[";
        for (std::size_t bin = 0; bin < domain.bins.size(); ++bin) {
            const auto value = domain.bins[bin].total.erase_bytes;
            heat_stream << (bin ? "," : "") << (value - heat_prev_erase[bin]);
            heat_prev_erase[bin] = value;
        }
        heat_stream << "],\"read\":[";
        for (std::size_t bin = 0; bin < domain.bins.size(); ++bin) {
            const auto value = domain.bins[bin].total.read_bytes;
            heat_stream << (bin ? "," : "") << (value - heat_prev_read[bin]);
            heat_prev_read[bin] = value;
        }
        // Absolute block-state profile (pages per bin) after this batch.
        const auto profile = replay.hbf_block_profile(domain.bins.size());
        const auto emit_profile = [&](const char* name, auto field) {
            heat_stream << "],\"" << name << "\":[";
            for (std::size_t bin = 0; bin < profile.size(); ++bin) {
                heat_stream << (bin ? "," : "") << field(profile[bin]);
            }
        };
        emit_profile("valid", [](const auto& b) { return b.valid_pages; });
        emit_profile("invalid", [](const auto& b) { return b.invalid_pages; });
        emit_profile("free", [](const auto& b) { return b.free_pages; });
        emit_profile("pending", [](const auto& b) { return b.pending_pages; });
        emit_profile("erase_count", [](const auto& b) { return b.erase_count_sum; });
        emit_profile("data_blocks", [](const auto& b) { return b.data_blocks; });
        emit_profile("mapping_blocks", [](const auto& b) { return b.mapping_blocks; });
        emit_profile("gc_blocks", [](const auto& b) { return b.gc_blocks; });
        emit_profile("free_blocks", [](const auto& b) { return b.free_blocks; });
        emit_profile("static_blocks", [](const auto& b) { return b.static_blocks; });
        const auto* stats = replay.hbf_execution_stats();
        if (stats == nullptr || stats->gc_runs < heat_prev_gc_runs ||
            stats->gc_relocations < heat_prev_gc_relocations ||
            stats->gc_data_relocations < heat_prev_gc_data_relocations ||
            stats->gc_mapping_relocations <
                heat_prev_gc_mapping_relocations ||
            stats->gc_reclaimed_invalid_pages <
                heat_prev_gc_reclaimed_invalid_pages ||
            stats->mapping_page_programs <
                heat_prev_mapping_page_programs) {
            throw std::runtime_error(
                "HBF live-heat cumulative counters regressed");
        }
        heat_stream << "],\"gc\":{\"runs\":"
                    << stats->gc_runs - heat_prev_gc_runs
                    << ",\"relocations\":"
                    << stats->gc_relocations - heat_prev_gc_relocations
                    << ",\"data_relocations\":"
                    << stats->gc_data_relocations -
                        heat_prev_gc_data_relocations
                    << ",\"mapping_relocations\":"
                    << stats->gc_mapping_relocations -
                        heat_prev_gc_mapping_relocations
                    << ",\"reclaimed_invalid_pages\":"
                    << stats->gc_reclaimed_invalid_pages -
                        heat_prev_gc_reclaimed_invalid_pages
                    << ",\"mapping_page_programs\":"
                    << stats->mapping_page_programs -
                        heat_prev_mapping_page_programs
                    << "}}\n" << std::flush;
        heat_prev_gc_runs = stats->gc_runs;
        heat_prev_gc_relocations = stats->gc_relocations;
        heat_prev_gc_data_relocations = stats->gc_data_relocations;
        heat_prev_gc_mapping_relocations = stats->gc_mapping_relocations;
        heat_prev_gc_reclaimed_invalid_pages =
            stats->gc_reclaimed_invalid_pages;
        heat_prev_mapping_page_programs = stats->mapping_page_programs;
    };
    const auto* initial_hbf_stats = replay.hbf_stats();
    const auto initial_hbf_data_pages = initial_hbf_stats == nullptr ?
        0 : initial_hbf_stats->initial_logical_data_pages;
    const auto initial_hbf_mapping_pages = initial_hbf_stats == nullptr ?
        0 : initial_hbf_stats->initial_mapping_pages;
    const auto initial_hbf_free_pages = initial_hbf_stats == nullptr ?
        0 : initial_hbf_stats->free_pages;
    const auto hbf_capacity_pages = session_config.enable_hbf ?
        static_cast<std::uint64_t>(session_config.hbf.stacks) *
            session_config.hbf.channels_per_stack *
            session_config.hbf.dies_per_channel *
            session_config.hbf.planes_per_die *
            session_config.hbf.blocks_per_plane *
            session_config.hbf.pages_per_block :
        0;
    const bool restored_image = static_cast<bool>(initial_image);
    const auto hbf_static_pages = initial_hbf_stats == nullptr ?
        0 : initial_hbf_stats->static_reserved_pages;
    const auto hbf_raw_pages = initial_hbf_stats == nullptr ?
        0 : initial_hbf_stats->raw_reserved_pages;
    const auto explicit_initial_data_pages = restored_image ?
        0 : initial_hbf_data_pages;
    const auto explicit_initial_mapping_pages = restored_image ?
        0 : initial_hbf_mapping_pages;
    const auto explicit_static_pages = restored_image ? 0 : hbf_static_pages;
    const auto explicit_initial_free_pages = restored_image ?
        hbf_capacity_pages : initial_hbf_free_pages;
    output << std::setprecision(17)
        << "{\"schema\":{\"name\":\"hbfsim.simulation_session\","
           "\"version\":1},\"result\":\"ready\","
           "\"protocol\":\"" << kProtocolName << "\","
           "\"time_basis\":\"batch_relative_ns\","
           "\"dependency_window_batches\":"
        << physical::kDependencyWindowBatches
        << ",\"nonterminal_hbf_checkpoint\":true,"
           "\"hbf_wear_snapshot\":true,"
           "\"source\":{\"git_commit\":\""
        << json_escape(options.source.git_commit)
        << "\",\"git_dirty\":"
        << (options.source.git_dirty ? "true" : "false")
        << ",\"tree_hash\":\"" << json_escape(options.source.tree_hash)
        << "\",\"source_sha256\":\"" << options.source.source_sha256
        << "\",\"provenance_source\":\""
        << json_escape(options.source.provenance_source)
        << "\"},\"enable_hbm\":"
        << (session_config.enable_hbm ? "true" : "false")
        << ",\"enable_hbf\":"
        << (session_config.enable_hbf ? "true" : "false")
        << ",\"enable_external\":"
        << (session_config.enable_external ? "true" : "false")
        << ",\"hbf_mapping_mode\":\""
        << hbfsim::physical::hbf::to_string(session_config.hbf.mapping_mode)
        << "\",\"hbf_ctrl_dram_bytes\":"
        << session_config.hbf.ctrl_dram_bytes
        << ",\"hbf_ctrl_dram_capacity_denominator\":"
        << session_config.hbf.ctrl_dram_capacity_denominator
        << ",\"static_hbf_blocks_per_plane\":"
        << session_config.static_hbf_blocks_per_plane
        << ",\"published_hbf_blocks_per_plane\":"
        << session_config.published_hbf_blocks_per_plane
        << ",\"initial_hbf_logical_image\":{"
           "\"mode\":\""
        << (session_config.initial_hbf_logical_pages == 0 ?
                "none" : "preloaded_mutable_dense")
        << "\",\"first_lpn\":"
        << session_config.initial_hbf_logical_first_lpn
        << ",\"pages\":"
        << session_config.initial_hbf_logical_pages
        << ",\"physical_data_pages\":" << explicit_initial_data_pages
        << ",\"physical_mapping_pages\":" << explicit_initial_mapping_pages
        << ",\"static_pages\":" << explicit_static_pages
        << ",\"free_pages_after_setup\":" << explicit_initial_free_pages
        << ",\"raw_capacity_pages\":" << hbf_capacity_pages
        << "},\"initial_hbf_persistent_image\":";
    if (initial_image) {
        output << "{\"schema\":{\"name\":"
                  "\"hbfsim.hbf_persistent_image\",\"version\":2},"
                  "\"path\":\""
               << json_escape(initial_image->path)
               << "\",\"bytes\":" << initial_image->bytes
               << ",\"sha256\":\"" << initial_image->sha256
               << "\",\"logical_data_pages\":"
               << initial_hbf_data_pages
               << ",\"mapping_pages\":" << initial_hbf_mapping_pages
               << ",\"compact_logical_data_pages\":"
               << initial_hbf_stats->compact_initial_logical_data_pages
               << ",\"compact_mapping_pages\":"
               << initial_hbf_stats->compact_initial_mapping_pages
               << ",\"encoding\":\""
               << (initial_hbf_stats->compact_initial_logical_data_pages == 0 ?
                       "materialized_v2" : "compact_v2")
               << '"'
               << ",\"static_pages\":" << hbf_static_pages
               << ",\"raw_pages\":" << hbf_raw_pages
               << ",\"free_pages_after_restore\":"
               << initial_hbf_free_pages
               << ",\"raw_capacity_pages\":" << hbf_capacity_pages
               << ",\"block_erase_count_sum\":"
               << initial_hbf_stats->block_erase_count_sum << '}';
    } else {
        output << "null";
    }
    output << ",\"external_backing\":";
    if (session_config.enable_external) {
        const auto& resolved = session_config.external;
        output << "{\"kind\":\""
                  << hbfsim::physical::external::to_string(resolved.kind)
                  << "\",\"capacity_bytes\":" << resolved.capacity_bytes
                  << ",\"page_size_bytes\":" << resolved.page_size_bytes
                  << ",\"request_segment_bytes\":"
                  << resolved.request_segment_bytes
                  << ",\"media_channels\":" << resolved.media_channels
                  << ",\"media_read_queues\":"
                  << resolved.media_read_queues
                  << ",\"media_write_queues\":"
                  << resolved.media_write_queues
                  << ",\"max_outstanding_requests\":"
                  << resolved.max_outstanding_requests;
        if (resolved.device_cache.enabled) {
            output << ",\"device_cache\":{\"enabled\":true"
                      ",\"capacity_bytes\":"
                   << resolved.device_cache.capacity_bytes
                   << ",\"ways\":" << resolved.device_cache.ways
                   << ",\"policy\":\""
                   << hbfsim::physical::external::to_string(
                          resolved.device_cache.policy)
                   << "\",\"prefetch_degree\":"
                   << resolved.device_cache.prefetch_degree
                   << ",\"prefetch_stride\":"
                   << resolved.device_cache.prefetch_stride << '}';
        }
        output << '}';
    } else {
        output << "null";
    }
    output << ",\"hbf_external_direct_link\":";
    if (session_config.enable_hbf && session_config.enable_external &&
        session_config.hbf_external_direct_link) {
        const auto& lane = *session_config.hbf_external_direct_link;
        output << "{\"links\":" << session_config.hbf.stacks
               << ",\"read_bandwidth_GBps\":" << lane.read_bandwidth_GBps
               << ",\"write_bandwidth_GBps\":" << lane.write_bandwidth_GBps
               << ",\"latency_ns\":" << lane.latency_ns << '}';
    } else {
        output << "null";
    }
    output << "}\n";
    output.flush();

    std::string line;
    std::size_t line_no = 0;
    std::set<std::string> wear_snapshot_ids;
    while (std::getline(input, line)) {
        ++line_no;
        if (line == "QUIT") {
            const auto finish_ns = write_session_stop(output, replay);
            emit_heat_delta("stop", "terminal-drain", finish_ns);
            return 0;
        }
        if (line.starts_with("CRASH ")) {
            std::istringstream crash_header(line);
            std::string command;
            std::string crash_id;
            std::string crash_extra;
            if (!(crash_header >> command >> crash_id) ||
                crash_header >> crash_extra || command != "CRASH" ||
                !transaction_identifier_valid(crash_id)) {
                throw std::runtime_error(
                    "simulation protocol CRASH line is malformed");
            }
            write_session_crash(output, replay, std::move(crash_id));
            return 0;
        }
        if (line.starts_with("WEAR_SNAPSHOT ")) {
            std::istringstream snapshot_header(line);
            std::string command;
            std::string snapshot_id;
            std::string snapshot_extra;
            if (!(snapshot_header >> command >> snapshot_id) ||
                snapshot_header >> snapshot_extra ||
                command != "WEAR_SNAPSHOT" ||
                !transaction_identifier_valid(snapshot_id)) {
                throw std::runtime_error(
                    "simulation protocol WEAR_SNAPSHOT line is malformed");
            }
            if (!wear_snapshot_ids.insert(snapshot_id).second) {
                throw std::runtime_error(
                    "simulation protocol wear snapshot id is duplicated: " +
                    snapshot_id);
            }
            write_session_wear_snapshot(output, replay, snapshot_id);
            continue;
        }
        if (line.starts_with("CHECKPOINT_IMAGE ")) {
            std::istringstream checkpoint_header(line);
            std::string command;
            std::string checkpoint_id;
            std::string path_hex;
            std::string checkpoint_extra;
            if (!(checkpoint_header >> command >> checkpoint_id >> path_hex) ||
                checkpoint_header >> checkpoint_extra ||
                command != "CHECKPOINT_IMAGE" ||
                !transaction_identifier_valid(checkpoint_id)) {
                throw std::runtime_error(
                    "simulation protocol CHECKPOINT_IMAGE line is malformed");
            }
            const auto finish_ns = write_session_checkpoint(
                output,
                replay,
                checkpoint_id,
                decode_path_hex(path_hex));
            emit_heat_delta("checkpoint", checkpoint_id, finish_ns);
            continue;
        }
        if (line.starts_with("CHECKPOINT")) {
            std::istringstream checkpoint_header(line);
            std::string command;
            std::string checkpoint_id;
            std::string checkpoint_extra;
            if (!(checkpoint_header >> command >> checkpoint_id) ||
                checkpoint_header >> checkpoint_extra ||
                command != "CHECKPOINT" ||
                !transaction_identifier_valid(checkpoint_id)) {
                throw std::runtime_error(
                    "simulation protocol CHECKPOINT line is malformed");
            }
            const auto finish_ns = write_session_checkpoint(
                output, replay, checkpoint_id);
            emit_heat_delta("checkpoint", checkpoint_id, finish_ns);
            continue;
        }
        if (line.empty()) {
            throw std::runtime_error(
                "simulation protocol does not permit blank lines");
        }
        const auto header = parse_batch_header(line);
        const auto& batch_id = header.batch_id;
        // The batch body is consumed to its END line even after an input
        // error so the stream stays in sync; the first error is reported in
        // an error record and the session continues unchanged. Structural
        // faults (EOF, a nested BEGIN/QUIT/END) leave the stream position
        // undefined and remain fatal.
        std::vector<SimulationTransaction> transactions;
        Sha256 payload_digest;
        std::optional<std::string> input_error;
        bool found_end = false;
        const std::string end_marker = "END " + batch_id;
        while (std::getline(input, line)) {
            ++line_no;
            if (line == end_marker) {
                found_end = true;
                break;
            }
            if (line.empty() || line.starts_with("BEGIN ") ||
                line == "QUIT" || line.starts_with("END ")) {
                throw std::runtime_error(
                    "simulation protocol batch body is malformed");
            }
            payload_digest.update(
                reinterpret_cast<const std::uint8_t*>(line.data()),
                line.size());
            payload_digest.update(
                reinterpret_cast<const std::uint8_t*>("\n"), 1);
            if (input_error) {
                continue;
            }
            try {
                transactions.push_back(
                    parse_simulation_transaction(line, line_no));
            } catch (const std::runtime_error& error) {
                input_error = error.what();
                transactions.clear();
            }
        }
        if (!found_end) {
            throw std::runtime_error(
                "simulation protocol reached EOF before END " + batch_id);
        }
        if (!input_error &&
            payload_digest.finish_hex() != header.transaction_sha256) {
            input_error =
                "simulation transaction payload digest mismatch for batch " +
                batch_id;
        }
        if (!input_error) {
            try {
                auto result = replay.run_batch(
                    batch_id, transactions, header.options);
                write_session_batch_result(
                    output,
                    result,
                    header.logical_sha256,
                    header.transaction_sha256,
                    header.options.record_completions);
                emit_heat_delta("batch", batch_id, result.finish_ns);
                continue;
            } catch (const SimulationInputError& error) {
                input_error = error.what();
            }
        }
        write_session_batch_error(output, batch_id, *input_error);
    }
    return 0;
}

} // namespace hbfsim::app
