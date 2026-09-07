#include "app/session_protocol.hpp"
#include "app/sha256.hpp"
#include "app/source_provenance.hpp"
#include "app/system_config.hpp"
#include "physical/hbf/hbf_persistent_image.hpp"

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <unistd.h>

#ifndef HBFSIM_VERSION
#define HBFSIM_VERSION "unknown"
#endif

namespace {

// Block-buffered stream buffer over a POSIX descriptor. libc++'s std::cin
// and std::cout are always routed through C stdio one character at a time
// (with a lock round trip per character, whatever sync_with_stdio says),
// which made reading a batch cost more than simulating it. The protocol is
// line-oriented and flushes explicitly after every receipt, so a plain
// read(2)/write(2) buffer is all the engine needs.
class PosixStreambuf : public std::streambuf {
public:
    PosixStreambuf(int fd, bool writable)
        : fd_(fd), writable_(writable), buffer_(1 << 20) {
        if (writable_) {
            setp(buffer_.data(), buffer_.data() + buffer_.size());
        } else {
            setg(buffer_.data(), buffer_.data(), buffer_.data());
        }
    }
    ~PosixStreambuf() override {
        if (writable_) {
            (void)flush_buffer();
        }
    }
    PosixStreambuf(const PosixStreambuf&) = delete;
    PosixStreambuf& operator=(const PosixStreambuf&) = delete;

protected:
    int_type underflow() override {
        if (writable_) {
            return traits_type::eof();
        }
        for (;;) {
            const auto count = ::read(fd_, buffer_.data(), buffer_.size());
            if (count < 0 && errno == EINTR) {
                continue;
            }
            if (count <= 0) {
                return traits_type::eof();
            }
            setg(buffer_.data(), buffer_.data(), buffer_.data() + count);
            return traits_type::to_int_type(*gptr());
        }
    }
    int_type overflow(int_type ch) override {
        if (!writable_ || flush_buffer() != 0) {
            return traits_type::eof();
        }
        if (!traits_type::eq_int_type(ch, traits_type::eof())) {
            *pptr() = traits_type::to_char_type(ch);
            pbump(1);
        }
        return traits_type::not_eof(ch);
    }
    std::streamsize xsputn(const char* data, std::streamsize count) override {
        std::streamsize written = 0;
        while (written < count) {
            if (pptr() == epptr() && flush_buffer() != 0) {
                return written;
            }
            const auto take = std::min<std::streamsize>(
                count - written, epptr() - pptr());
            std::memcpy(pptr(), data + written, static_cast<std::size_t>(take));
            pbump(static_cast<int>(take));
            written += take;
        }
        return written;
    }
    int sync() override {
        return writable_ ? flush_buffer() : 0;
    }

private:
    int flush_buffer() {
        const char* cursor = pbase();
        while (cursor < pptr()) {
            const auto count = ::write(
                fd_, cursor, static_cast<std::size_t>(pptr() - cursor));
            if (count < 0 && errno == EINTR) {
                continue;
            }
            if (count <= 0) {
                return -1;
            }
            cursor += count;
        }
        setp(buffer_.data(), buffer_.data() + buffer_.size());
        return 0;
    }

    int fd_;
    bool writable_;
    std::vector<char> buffer_;
};

struct SessionOptions {
    bool describe_system = false;
    std::vector<std::string> system_configs;
    std::vector<std::pair<std::string, std::string>> system_overrides;
    bool enable_hbm = true;
    bool enable_hbf = true;
    bool enable_external = false;
    std::uint32_t static_hbf_blocks_per_plane = 0;
    std::uint32_t published_hbf_blocks_per_plane = 0;
    std::uint64_t initial_hbf_logical_first_lpn = 0;
    std::uint64_t initial_hbf_logical_pages = 0;
    std::optional<std::string> initial_hbf_persistent_image;
    std::optional<std::string> hbf_physical_heatmap;
    std::uint64_t hbf_physical_heatmap_bins = 0;
};

bool parse_bool(std::string_view value, std::string_view option) {
    if (value == "true" || value == "1") return true;
    if (value == "false" || value == "0") return false;
    throw std::runtime_error(std::string(option) + " must be true or false");
}

// Decimal digits only; a leading zero never selects octal.
std::uint64_t parse_u64(std::string_view value, std::string_view option) {
    std::uint64_t result = 0;
    const auto* const begin = value.data();
    const auto* const end = value.data() + value.size();
    const auto [pointer, error] = std::from_chars(begin, end, result, 10);
    if (value.empty() || error != std::errc{} || pointer != end) {
        throw std::runtime_error(
            std::string(option) + " must be an unsigned decimal integer");
    }
    return result;
}

std::uint32_t parse_u32(std::string_view value, std::string_view option) {
    const auto result = parse_u64(value, option);
    if (result > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error(std::string(option) + " exceeds uint32_t");
    }
    return static_cast<std::uint32_t>(result);
}

void usage(const char* executable) {
    std::cout
        << "Usage: " << executable << " --system-config FILE [options]\n\n"
        << "HBFSim is a semantic-free simulation engine. It accepts transaction\n"
        << "DAG batches on stdin and emits causal completion receipts on stdout.\n"
        << "Workload semantics, placement policies, and experiment matrices live\n"
        << "outside this executable.\n\n"
        << "Session options:\n"
        << "  --describe-system                 print resolved geometry/capacity and exit\n"
        << "  --system-config FILE              repeatable system/physical config\n"
        << "  --enable-hbm BOOL                 instantiate HBM\n"
        << "  --enable-hbf BOOL                 instantiate HBF\n"
        << "  --enable-external BOOL            instantiate external backing\n"
        << "  --static-hbf-blocks-per-plane N   immutable physical HBF extent\n"
        << "  --published-hbf-blocks-per-plane N raw append-to-publish HBF extent\n"
        << "  --initial-hbf-logical-first-lpn N initial mutable image base\n"
        << "  --initial-hbf-logical-pages N     initial mutable image size\n"
        << "  --initial-hbf-persistent-image FILE restore exact quiescent FTL image\n"
        << "  --hbf-physical-heatmap PATH       append a live HBF physical heat\n"
        << "                                    stream (JSON lines) to PATH\n"
        << "  --hbf-physical-heatmap-bins N     bins of that stream (must divide\n"
        << "                                    the physical block count)\n"
        << "  --<system-key> VALUE              override an engine-owned config key\n"
        << "  --version                         print engine version and source\n"
        << "                                    provenance\n";
}

SessionOptions parse_args(int argc, char** argv) {
    SessionOptions options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help" || argument == "-h") {
            usage(argv[0]);
            std::exit(0);
        }
        if (argument == "--version") {
            const hbfsim::app::SourceProvenance source;
            std::cout << "HBFSim engine " << HBFSIM_VERSION
                      << " (git " << source.git_commit
                      << (source.git_dirty ? ", dirty" : "")
                      << ", tree " << source.tree_hash
                      << ", source-sha256 " << source.source_sha256
                      << ", " << source.provenance_source << ")\n";
            std::exit(0);
        }
        if (argument == "--describe-system") {
            options.describe_system = true;
            continue;
        }
        if (!argument.starts_with("--")) {
            throw std::runtime_error("unknown argument: " + argument);
        }
        if (index + 1 >= argc) {
            throw std::runtime_error(argument + " requires a value");
        }
        const std::string value = argv[++index];
        const auto key = argument.substr(2);
        if (key == "system-config") {
            options.system_configs.push_back(value);
        } else if (key == "enable-hbm") {
            options.enable_hbm = parse_bool(value, argument);
        } else if (key == "enable-hbf") {
            options.enable_hbf = parse_bool(value, argument);
        } else if (key == "enable-external") {
            options.enable_external = parse_bool(value, argument);
        } else if (key == "static-hbf-blocks-per-plane") {
            options.static_hbf_blocks_per_plane = parse_u32(value, argument);
        } else if (key == "published-hbf-blocks-per-plane") {
            options.published_hbf_blocks_per_plane = parse_u32(value, argument);
        } else if (key == "initial-hbf-logical-first-lpn") {
            options.initial_hbf_logical_first_lpn = parse_u64(value, argument);
        } else if (key == "initial-hbf-logical-pages") {
            options.initial_hbf_logical_pages = parse_u64(value, argument);
        } else if (key == "initial-hbf-persistent-image") {
            options.initial_hbf_persistent_image = value;
        } else if (key == "hbf-physical-heatmap") {
            if (value.empty()) {
                throw std::runtime_error(argument + " requires a path");
            }
            options.hbf_physical_heatmap = value;
        } else if (key == "hbf-physical-heatmap-bins") {
            options.hbf_physical_heatmap_bins = parse_u64(value, argument);
        } else if (hbfsim::app::SystemConfigBuilder::owns_key(key)) {
            options.system_overrides.emplace_back(key, value);
        } else {
            throw std::runtime_error(
                "option is not owned by the simulator engine: " + argument);
        }
    }
    if (options.system_configs.empty()) {
        throw std::runtime_error("at least one --system-config is required");
    }
    if (!options.enable_hbm && !options.enable_hbf &&
        !options.enable_external) {
        throw std::runtime_error("a simulation session must enable a memory tier");
    }
    if (options.hbf_physical_heatmap.has_value() !=
        (options.hbf_physical_heatmap_bins != 0)) {
        throw std::runtime_error(
            "--hbf-physical-heatmap and a positive --hbf-physical-heatmap-bins "
            "must be given together");
    }
    if (options.hbf_physical_heatmap && !options.enable_hbf) {
        throw std::runtime_error(
            "--hbf-physical-heatmap requires the HBF tier");
    }
    return options;
}

} // namespace

int main(int argc, char** argv) {
    PosixStreambuf input_buffer(STDIN_FILENO, false);
    PosixStreambuf output_buffer(STDOUT_FILENO, true);
    std::istream input(&input_buffer);
    std::ostream output(&output_buffer);
    try {
        const auto options = parse_args(argc, argv);
        hbfsim::app::SystemConfigBuilder builder;
        for (const auto& path : options.system_configs) {
            builder.apply_file(path);
        }
        for (const auto& [key, value] : options.system_overrides) {
            builder.apply(key, value);
        }
        const auto system = builder.resolve();
        if (options.describe_system) {
            if (options.static_hbf_blocks_per_plane != 0 ||
                options.published_hbf_blocks_per_plane != 0 ||
                options.initial_hbf_logical_first_lpn != 0 ||
                options.initial_hbf_logical_pages != 0 ||
                options.initial_hbf_persistent_image || options.hbf_physical_heatmap) {
                throw std::runtime_error(
                    "--describe-system describes unreserved hardware; session image, "
                    "extent and heatmap options are not accepted");
            }
            std::optional<hbfsim::physical::hbf::HbfDevice> hbf;
            if (options.enable_hbf) {
                hbf.emplace(system.hbf);
            }
            const auto& geometry = system.hbf;
            output << "{\"schema\":{\"name\":\"hbfsim.resolved_system\",\"version\":1},"
                   << "\"hbf_logical_capacity_bytes\":"
                   << (hbf ? hbf->logical_capacity_pages() * geometry.page_size_bytes : 0)
                   << ",\"values\":{"
                   << "\"hbm-capacity-bytes\":\"" << system.hbm.capacity_bytes
                   << "\",\"hbf-stacks\":\"" << geometry.stacks
                   << "\",\"hbf-channels\":\"" << geometry.channels_per_stack
                   << "\",\"hbf-dies-per-channel\":\"" << geometry.dies_per_channel
                   << "\",\"hbf-planes-per-die\":\"" << geometry.planes_per_die
                   << "\",\"hbf-blocks-per-plane\":\"" << geometry.blocks_per_plane
                   << "\",\"hbf-pages-per-block\":\"" << geometry.pages_per_block
                   << "\",\"hbf-page-size\":\"" << geometry.page_size_bytes
                   << "\",\"hbf-mapping-entries-per-page\":\"" << geometry.mapping_entries_per_page
                   << "\",\"hbf-mapping-mode\":\"" << hbfsim::physical::hbf::to_string(geometry.mapping_mode)
                   << "\",\"hbf-ctrl-dram-bytes\":\""
                   << (hbf ? hbf->config().ctrl_dram_bytes : geometry.ctrl_dram_bytes)
                   << "\"}}\n" << std::flush;
            return 0;
        }
        std::shared_ptr<const hbfsim::physical::hbf::HbfPersistentImage>
            persistent_image;
        hbfsim::app::SessionProtocolOptions protocol_options{
            .initial_image = std::nullopt,
            .hbf_physical_heatmap_path = options.hbf_physical_heatmap,
            .source = {},
        };
        if (options.initial_hbf_persistent_image) {
            const auto path = std::filesystem::absolute(
                *options.initial_hbf_persistent_image);
            auto image =
                hbfsim::physical::hbf::read_persistent_image_file(path);
            persistent_image = std::make_shared<
                const hbfsim::physical::hbf::HbfPersistentImage>(
                    std::move(image));
            protocol_options.initial_image =
                hbfsim::app::SessionPersistentImageArtifact{
                    .path = path.string(),
                    .bytes = std::filesystem::file_size(path),
                    .sha256 = hbfsim::app::sha256_file(path),
                };
        }
        hbfsim::physical::SimulationSessionConfig config{
            .enable_hbm = options.enable_hbm,
            .enable_hbf = options.enable_hbf,
            .enable_external = options.enable_external,
            .hbm = system.hbm,
            .hbf = system.hbf,
            .external = system.external,
            .base_die_link = system.base_die_link,
            .hbf_external_direct_link = system.hbf_external_direct_link,
            .static_hbf_blocks_per_plane =
                options.static_hbf_blocks_per_plane,
            .published_hbf_blocks_per_plane =
                options.published_hbf_blocks_per_plane,
            .initial_hbf_logical_first_lpn =
                options.initial_hbf_logical_first_lpn,
            .initial_hbf_logical_pages = options.initial_hbf_logical_pages,
            .initial_hbf_persistent_image = std::move(persistent_image),
            .trace = {
                .mode = hbfsim::physical::TraceMode::Off,
                .retain_completion_diagnostics = false,
            },
            .hbf_physical_heatmap_bins = static_cast<std::size_t>(
                options.hbf_physical_heatmap_bins),
        };
        const auto status = hbfsim::app::run_simulation_session(
            std::move(config),
            input,
            output,
            std::move(protocol_options));
        output.flush();
        return status;
    } catch (const std::exception& error) {
        output.flush();
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
