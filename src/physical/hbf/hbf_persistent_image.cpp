#include "physical/hbf/hbf_persistent_image.hpp"

#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>

namespace hbfsim::physical::hbf {
namespace {

constexpr std::uint64_t kImageVersion = 2;

void require_regular_non_symlink(
    const std::filesystem::path& path,
    const char* description) {
    std::error_code error;
    const auto status = std::filesystem::symlink_status(path, error);
    if (error || status.type() == std::filesystem::file_type::symlink ||
        status.type() != std::filesystem::file_type::regular) {
        throw std::runtime_error(
            std::string(description) +
            " must be a regular non-symlink file: " + path.string());
    }
}

template <typename Value>
Value read_value(std::istream& input, const char* description) {
    Value value{};
    if (!(input >> value)) {
        throw std::runtime_error(
            std::string("cannot read HBF persistent image ") + description);
    }
    return value;
}

void expect(std::istream& input, const char* token) {
    const auto actual = read_value<std::string>(input, token);
    if (actual != token) {
        throw std::runtime_error(
            "HBF persistent image expected " + std::string(token) +
            ", got " + actual);
    }
}

std::size_t count_as_size(
    std::uint64_t count,
    std::uint64_t maximum,
    const char* description) {
    if (count > maximum || count > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error(
            std::string("HBF persistent image ") + description +
            " exceeds its declared geometry");
    }
    return static_cast<std::size_t>(count);
}

void write_optional(
    std::ostream& output,
    const std::optional<std::uint64_t>& value) {
    if (value) output << *value;
    else output << '-';
}

std::optional<std::uint64_t> read_optional(
    std::istream& input,
    const char* description) {
    const auto token = read_value<std::string>(input, description);
    if (token == "-") return std::nullopt;
    if (token.empty() || token.front() == '-') {
        throw std::runtime_error(
            std::string("invalid HBF persistent image ") + description);
    }
    std::size_t consumed = 0;
    std::uint64_t value = 0;
    try {
        value = std::stoull(token, &consumed, 10);
    } catch (const std::exception&) {
        throw std::runtime_error(
            std::string("invalid HBF persistent image ") + description);
    }
    if (consumed != token.size()) {
        throw std::runtime_error(
            std::string("invalid HBF persistent image ") + description);
    }
    return value;
}

} // namespace

void write_persistent_image_file(
    const std::filesystem::path& path,
    const HbfPersistentImage& image) {
    if (image.version != kImageVersion) {
        throw std::runtime_error(
            "cannot write an obsolete HBF persistent image version");
    }
    if (path.empty() || path.has_filename() == false) {
        throw std::runtime_error("HBF persistent image output path is empty");
    }
    std::error_code error;
    if (std::filesystem::exists(path, error) || error) {
        throw std::runtime_error(
            "HBF persistent image output already exists: " + path.string());
    }
    const auto parent = path.parent_path();
    if (parent.empty() || !std::filesystem::is_directory(parent, error) || error) {
        throw std::runtime_error(
            "HBF persistent image output parent is not a directory: " +
            parent.string());
    }
    auto temporary = path;
    temporary += ".tmp";
    if (std::filesystem::exists(temporary, error) || error) {
        throw std::runtime_error(
            "HBF persistent image temporary output already exists: " +
            temporary.string());
    }

    try {
        std::ofstream output(temporary, std::ios::binary | std::ios::out);
        if (!output) {
            throw std::runtime_error(
                "cannot create HBF persistent image: " + temporary.string());
        }
        output << "HBFSIM_HBF_PERSISTENT_IMAGE " << image.version << '\n';
        output << "GEOMETRY " << image.stacks << ' ' << image.planes << ' '
               << image.blocks_per_plane << ' ' << image.pages_per_block << ' '
               << image.page_size_bytes << ' '
               << image.mapping_entries_per_page << '\n';
        output << "FREE " << image.state.free_pages << ' '
               << image.state.free_pages_per_stack.size();
        for (const auto value : image.state.free_pages_per_stack) {
            output << ' ' << value;
        }
        output << '\n';
        const auto write_cursors = [&output](
                                       const char* name,
                                       const std::vector<std::uint64_t>& values) {
            output << name << ' ' << values.size();
            for (const auto value : values) output << ' ' << value;
            output << '\n';
        };
        write_cursors("DATA_CURSORS", image.state.data_allocation_cursors);
        write_cursors("MAPPING_CURSORS", image.state.mapping_allocation_cursors);
        write_cursors("GC_CURSORS", image.state.gc_allocation_cursors);

        output << "LPN_COUNT " << image.state.logical_mappings.size() << '\n';
        for (const auto& mapping : image.state.logical_mappings) {
            output << "LPN " << mapping.key << ' ' << mapping.ppn << '\n';
        }
        output << "VPN_COUNT " << image.state.mapping_pages.size() << '\n';
        for (const auto& mapping : image.state.mapping_pages) {
            output << "VPN " << mapping.key << ' ' << mapping.ppn << '\n';
        }
        output << "PAGE_COUNT " << image.state.materialized_pages.size() << '\n';
        for (const auto& page : image.state.materialized_pages) {
            output << "PAGE " << page.ppn << ' ' << page.status << ' '
                   << page.owner << ' ' << page.logical_key << ' '
                   << page.block_epoch << '\n';
        }
        output << "BLOCK_COUNT " << image.state.blocks.size() << '\n';
        for (const auto& block : image.state.blocks) {
            output << "BLOCK " << block.block << ' ' << block.role << ' '
                   << block.valid_pages << ' ' << block.invalid_pages << ' '
                   << block.free_pages << ' ' << block.next_page << ' '
                   << block.erase_count << ' ' << block.epoch << '\n';
        }
        output << "PLANE_COUNT " << image.plane_state.size() << '\n';
        for (std::size_t index = 0; index < image.plane_state.size(); ++index) {
            const auto& plane = image.plane_state[index];
            output << "PLANE " << index << ' ';
            write_optional(output, plane.active_data_block);
            output << ' ';
            write_optional(output, plane.active_mapping_block);
            output << ' ';
            write_optional(output, plane.active_gc_block);
            output << ' ' << plane.free_blocks.size();
            for (const auto block : plane.free_blocks) output << ' ' << block;
            output << '\n';
        }
        output << "COMPACT " << (image.compact_image ? 1 : 0);
        if (image.compact_image) {
            const auto& compact = *image.compact_image;
            output << ' ' << (compact.mutable_image ? 1 : 0)
                   << ' ' << compact.first_lpn
                   << ' ' << compact.page_count
                   << ' ' << compact.first_vpn
                   << ' ' << compact.vpn_slot_count
                   << ' ' << compact.mapping_page_count;
        }
        output << '\n';
        if (image.compact_image) {
            const auto& compact = *image.compact_image;
            output << "COMPACT_DATA_PLANE_COUNT "
                   << compact.data_blocks_by_plane.size() << '\n';
            for (std::size_t plane = 0;
                 plane < compact.data_blocks_by_plane.size();
                 ++plane) {
                const auto& blocks = compact.data_blocks_by_plane[plane];
                output << "COMPACT_DATA_PLANE " << plane << ' '
                       << blocks.size();
                for (const auto block : blocks) output << ' ' << block;
                output << '\n';
            }
            output << "COMPACT_MAPPING_SLOT_COUNT "
                   << compact.mapping_ppns.size() << '\n';
            for (std::size_t index = 0;
                 index < compact.mapping_ppns.size();
                 ++index) {
                output << "COMPACT_MAPPING_SLOT " << index << ' ';
                write_optional(output, compact.mapping_ppns[index]);
                output << '\n';
            }
            output << "COMPACT_VPN_RANGE_COUNT "
                   << compact.vpn_ranges.size() << '\n';
            for (std::size_t index = 0;
                 index < compact.vpn_ranges.size();
                 ++index) {
                const auto& range = compact.vpn_ranges[index];
                output << "COMPACT_VPN_RANGE " << index << ' '
                       << range.first_entry << ' ' << range.page_count << ' '
                       << range.stack_page_offset << '\n';
            }
            const auto write_live_blocks = [&output](
                const char* count_name,
                const char* row_name,
                const auto& entries) {
                output << count_name << ' ' << entries.size() << '\n';
                for (const auto& entry : entries) {
                    output << row_name << ' ' << entry.block << ' '
                           << entry.live_pages << '\n';
                }
            };
            write_live_blocks(
                "COMPACT_LIVE_DATA_COUNT",
                "COMPACT_LIVE_DATA",
                compact.live_data_pages_by_block);
            write_live_blocks(
                "COMPACT_LIVE_MAPPING_COUNT",
                "COMPACT_LIVE_MAPPING",
                compact.live_mapping_pages_by_block);
            const auto write_retired = [&output](
                const char* count_name,
                const char* row_name,
                const auto& entries) {
                output << count_name << ' ' << entries.size() << '\n';
                for (const auto value : entries) {
                    output << row_name << ' ' << value << '\n';
                }
            };
            write_retired(
                "COMPACT_RETIRED_LPN_COUNT",
                "COMPACT_RETIRED_LPN",
                compact.retired_lpns);
            write_retired(
                "COMPACT_RETIRED_VPN_COUNT",
                "COMPACT_RETIRED_VPN",
                compact.retired_mapping_vpns);
        }
        output << "END\n";
        output.flush();
        if (!output) {
            throw std::runtime_error(
                "failed to write HBF persistent image: " + temporary.string());
        }
        output.close();
        if (!output) {
            throw std::runtime_error(
                "failed to close HBF persistent image: " + temporary.string());
        }
        std::filesystem::rename(temporary, path);
    } catch (...) {
        std::filesystem::remove(temporary, error);
        throw;
    }
}

HbfPersistentImage read_persistent_image_file(
    const std::filesystem::path& path) {
    require_regular_non_symlink(path, "HBF persistent image input");
    std::ifstream input(path, std::ios::binary | std::ios::in);
    if (!input) {
        throw std::runtime_error(
            "cannot open HBF persistent image: " + path.string());
    }
    expect(input, "HBFSIM_HBF_PERSISTENT_IMAGE");
    HbfPersistentImage image;
    image.version = read_value<std::uint32_t>(input, "version");
    if (image.version != kImageVersion) {
        throw std::runtime_error("unsupported HBF persistent image version");
    }
    expect(input, "GEOMETRY");
    image.stacks = read_value<std::uint32_t>(input, "stacks");
    image.planes = read_value<std::uint64_t>(input, "planes");
    image.blocks_per_plane =
        read_value<std::uint64_t>(input, "blocks per plane");
    image.pages_per_block =
        read_value<std::uint64_t>(input, "pages per block");
    image.page_size_bytes = read_value<std::uint64_t>(input, "page size");
    image.mapping_entries_per_page =
        read_value<std::uint64_t>(input, "mapping entries per page");
    if (image.stacks == 0 || image.planes == 0 ||
        image.blocks_per_plane == 0 || image.pages_per_block == 0 ||
        image.page_size_bytes == 0 || image.mapping_entries_per_page == 0) {
        throw std::runtime_error("HBF persistent image geometry contains zero");
    }
    const auto total_blocks = image.planes * image.blocks_per_plane;
    if (image.blocks_per_plane != 0 &&
        total_blocks / image.blocks_per_plane != image.planes) {
        throw std::runtime_error("HBF persistent image block geometry overflows");
    }
    const auto total_pages = total_blocks * image.pages_per_block;
    if (image.pages_per_block != 0 &&
        total_pages / image.pages_per_block != total_blocks) {
        throw std::runtime_error("HBF persistent image page geometry overflows");
    }

    expect(input, "FREE");
    image.state.free_pages = read_value<std::uint64_t>(input, "free pages");
    const auto stack_count = count_as_size(
        read_value<std::uint64_t>(input, "free-page stack count"),
        image.stacks,
        "free-page stack count");
    if (stack_count != image.stacks) {
        throw std::runtime_error(
            "HBF persistent image free-page stack count diverged");
    }
    image.state.free_pages_per_stack.resize(stack_count);
    for (auto& value : image.state.free_pages_per_stack) {
        value = read_value<std::uint64_t>(input, "per-stack free pages");
    }
    const auto read_cursors = [&, stack_count](
                                  const char* name,
                                  std::vector<std::uint64_t>& values) {
        expect(input, name);
        const auto count = count_as_size(
            read_value<std::uint64_t>(input, "cursor count"),
            stack_count,
            "cursor count");
        if (count != stack_count) {
            throw std::runtime_error(
                "HBF persistent image cursor stack count diverged");
        }
        values.resize(count);
        for (auto& value : values) {
            value = read_value<std::uint64_t>(input, "allocation cursor");
        }
    };
    read_cursors("DATA_CURSORS", image.state.data_allocation_cursors);
    read_cursors("MAPPING_CURSORS", image.state.mapping_allocation_cursors);
    read_cursors("GC_CURSORS", image.state.gc_allocation_cursors);

    expect(input, "LPN_COUNT");
    const auto lpn_count = count_as_size(
        read_value<std::uint64_t>(input, "LPN count"),
        total_pages,
        "LPN count");
    image.state.logical_mappings.reserve(lpn_count);
    for (std::size_t index = 0; index < lpn_count; ++index) {
        expect(input, "LPN");
        image.state.logical_mappings.push_back({
            .key = read_value<std::uint64_t>(input, "LPN key"),
            .ppn = read_value<std::uint64_t>(input, "LPN PPN"),
        });
    }
    expect(input, "VPN_COUNT");
    const auto vpn_count = count_as_size(
        read_value<std::uint64_t>(input, "VPN count"),
        total_pages,
        "VPN count");
    image.state.mapping_pages.reserve(vpn_count);
    for (std::size_t index = 0; index < vpn_count; ++index) {
        expect(input, "VPN");
        image.state.mapping_pages.push_back({
            .key = read_value<std::uint64_t>(input, "VPN key"),
            .ppn = read_value<std::uint64_t>(input, "VPN PPN"),
        });
    }
    expect(input, "PAGE_COUNT");
    const auto page_count = count_as_size(
        read_value<std::uint64_t>(input, "page count"),
        total_pages,
        "page count");
    image.state.materialized_pages.reserve(page_count);
    for (std::size_t index = 0; index < page_count; ++index) {
        expect(input, "PAGE");
        image.state.materialized_pages.push_back({
            .ppn = read_value<std::uint64_t>(input, "page PPN"),
            .status = read_value<std::string>(input, "page status"),
            .owner = read_value<std::string>(input, "page owner"),
            .logical_key = read_value<std::uint64_t>(input, "page key"),
            .block_epoch = read_value<std::uint64_t>(input, "page epoch"),
        });
    }
    expect(input, "BLOCK_COUNT");
    const auto block_count = count_as_size(
        read_value<std::uint64_t>(input, "block count"),
        total_blocks,
        "block count");
    if (block_count != total_blocks) {
        throw std::runtime_error("HBF persistent image omits block state");
    }
    image.state.blocks.reserve(block_count);
    for (std::size_t index = 0; index < block_count; ++index) {
        expect(input, "BLOCK");
        image.state.blocks.push_back({
            .block = read_value<std::uint64_t>(input, "block index"),
            .role = read_value<std::string>(input, "block role"),
            .valid_pages = read_value<std::uint32_t>(input, "valid pages"),
            .invalid_pages = read_value<std::uint32_t>(input, "invalid pages"),
            .free_pages = read_value<std::uint32_t>(input, "free pages"),
            .next_page = read_value<std::uint32_t>(input, "next page"),
            .erase_count = read_value<std::uint32_t>(input, "erase count"),
            .pending_program_pages = 0,
            .pending_mapping_publications = 0,
            .epoch = read_value<std::uint64_t>(input, "block epoch"),
            .erase_pending = false,
        });
    }
    expect(input, "PLANE_COUNT");
    const auto plane_count = count_as_size(
        read_value<std::uint64_t>(input, "plane count"),
        image.planes,
        "plane count");
    if (plane_count != image.planes) {
        throw std::runtime_error("HBF persistent image omits plane state");
    }
    image.plane_state.reserve(plane_count);
    for (std::size_t index = 0; index < plane_count; ++index) {
        expect(input, "PLANE");
        const auto declared = read_value<std::uint64_t>(input, "plane index");
        if (declared != index) {
            throw std::runtime_error("HBF persistent image plane order diverged");
        }
        HbfPersistentPlane plane;
        plane.active_data_block = read_optional(input, "active data block");
        plane.active_mapping_block =
            read_optional(input, "active mapping block");
        plane.active_gc_block = read_optional(input, "active GC block");
        const auto free_count = count_as_size(
            read_value<std::uint64_t>(input, "free-block count"),
            image.blocks_per_plane,
            "free-block count");
        plane.free_blocks.reserve(free_count);
        for (std::size_t offset = 0; offset < free_count; ++offset) {
            plane.free_blocks.push_back(
                read_value<std::uint64_t>(input, "free block"));
        }
        image.plane_state.push_back(std::move(plane));
    }
    expect(input, "COMPACT");
    const auto compact_present =
        read_value<std::uint32_t>(input, "compact presence");
    if (compact_present > 1) {
        throw std::runtime_error(
            "HBF persistent image compact presence is not boolean");
    }
    if (compact_present == 1) {
        HbfPersistentCompactImage compact;
        const auto mutable_image =
            read_value<std::uint32_t>(input, "compact mutability");
        if (mutable_image > 1) {
            throw std::runtime_error(
                "HBF persistent image compact mutability is not boolean");
        }
        compact.mutable_image = mutable_image == 1;
        compact.first_lpn = read_value<std::uint64_t>(input, "compact first LPN");
        compact.page_count = read_value<std::uint64_t>(input, "compact page count");
        compact.first_vpn = read_value<std::uint64_t>(input, "compact first VPN");
        compact.vpn_slot_count =
            read_value<std::uint64_t>(input, "compact VPN slot count");
        compact.mapping_page_count =
            read_value<std::uint64_t>(input, "compact mapping page count");
        if (compact.page_count == 0 || compact.page_count > total_pages ||
            compact.vpn_slot_count > total_pages ||
            compact.mapping_page_count > compact.vpn_slot_count) {
            throw std::runtime_error(
                "HBF persistent image compact geometry exceeds capacity");
        }

        expect(input, "COMPACT_DATA_PLANE_COUNT");
        const auto compact_plane_count = count_as_size(
            read_value<std::uint64_t>(input, "compact data plane count"),
            image.planes,
            "compact data plane count");
        if (compact_plane_count != image.planes) {
            throw std::runtime_error(
                "HBF persistent image compact data planes diverged");
        }
        compact.data_blocks_by_plane.resize(compact_plane_count);
        for (std::size_t plane = 0; plane < compact_plane_count; ++plane) {
            expect(input, "COMPACT_DATA_PLANE");
            if (read_value<std::uint64_t>(input, "compact plane index") !=
                plane) {
                throw std::runtime_error(
                    "HBF persistent image compact plane order diverged");
            }
            const auto count = count_as_size(
                read_value<std::uint64_t>(input, "compact data block count"),
                image.blocks_per_plane,
                "compact data block count");
            auto& blocks = compact.data_blocks_by_plane[plane];
            blocks.resize(count);
            for (auto& block : blocks) {
                block = read_value<std::uint64_t>(
                    input, "compact data block");
                if (block >= total_blocks) {
                    throw std::runtime_error(
                        "HBF persistent image compact data block is out of range");
                }
            }
        }

        expect(input, "COMPACT_MAPPING_SLOT_COUNT");
        const auto mapping_slots = count_as_size(
            read_value<std::uint64_t>(input, "compact mapping slot count"),
            total_pages,
            "compact mapping slot count");
        if (mapping_slots != compact.vpn_slot_count) {
            throw std::runtime_error(
                "HBF persistent image compact mapping slots diverged");
        }
        compact.mapping_ppns.resize(mapping_slots);
        for (std::size_t index = 0; index < mapping_slots; ++index) {
            expect(input, "COMPACT_MAPPING_SLOT");
            if (read_value<std::uint64_t>(input, "compact mapping index") !=
                index) {
                throw std::runtime_error(
                    "HBF persistent image compact mapping order diverged");
            }
            compact.mapping_ppns[index] =
                read_optional(input, "compact mapping PPN");
            if (compact.mapping_ppns[index] &&
                *compact.mapping_ppns[index] >= total_pages) {
                throw std::runtime_error(
                    "HBF persistent image compact mapping PPN is out of range");
            }
        }

        expect(input, "COMPACT_VPN_RANGE_COUNT");
        const auto range_count = count_as_size(
            read_value<std::uint64_t>(input, "compact VPN range count"),
            total_pages,
            "compact VPN range count");
        if (range_count != compact.vpn_slot_count) {
            throw std::runtime_error(
                "HBF persistent image compact VPN ranges diverged");
        }
        compact.vpn_ranges.resize(range_count);
        for (std::size_t index = 0; index < range_count; ++index) {
            expect(input, "COMPACT_VPN_RANGE");
            if (read_value<std::uint64_t>(input, "compact range index") !=
                index) {
                throw std::runtime_error(
                    "HBF persistent image compact range order diverged");
            }
            compact.vpn_ranges[index] = {
                .first_entry = read_value<std::uint64_t>(
                    input, "compact range first entry"),
                .page_count = read_value<std::uint64_t>(
                    input, "compact range page count"),
                .stack_page_offset = read_value<std::uint64_t>(
                    input, "compact range stack offset"),
            };
        }

        const auto read_live_blocks = [&, total_blocks](
            const char* count_name,
            const char* row_name,
            std::vector<HbfPersistentCompactLiveBlock>& entries) {
            expect(input, count_name);
            const auto count = count_as_size(
                read_value<std::uint64_t>(input, "compact live-block count"),
                total_blocks,
                "compact live-block count");
            entries.reserve(count);
            std::optional<std::uint64_t> previous;
            for (std::size_t index = 0; index < count; ++index) {
                expect(input, row_name);
                HbfPersistentCompactLiveBlock entry{
                    .block = read_value<std::uint64_t>(
                        input, "compact live block"),
                    .live_pages = read_value<std::uint32_t>(
                        input, "compact live pages"),
                };
                if (entry.block >= total_blocks || entry.live_pages == 0 ||
                    (previous && entry.block <= *previous)) {
                    throw std::runtime_error(
                        "HBF persistent image compact live-block order diverged");
                }
                previous = entry.block;
                entries.push_back(entry);
            }
        };
        read_live_blocks(
            "COMPACT_LIVE_DATA_COUNT",
            "COMPACT_LIVE_DATA",
            compact.live_data_pages_by_block);
        read_live_blocks(
            "COMPACT_LIVE_MAPPING_COUNT",
            "COMPACT_LIVE_MAPPING",
            compact.live_mapping_pages_by_block);

        const auto read_retired = [&](
            const char* count_name,
            const char* row_name,
            std::uint64_t maximum,
            std::vector<std::uint64_t>& entries) {
            expect(input, count_name);
            const auto count = count_as_size(
                read_value<std::uint64_t>(input, "compact retired count"),
                maximum,
                "compact retired count");
            entries.reserve(count);
            std::optional<std::uint64_t> previous;
            for (std::size_t index = 0; index < count; ++index) {
                expect(input, row_name);
                const auto value = read_value<std::uint64_t>(
                    input, "compact retired key");
                if (previous && value <= *previous) {
                    throw std::runtime_error(
                        "HBF persistent image compact retired order diverged");
                }
                previous = value;
                entries.push_back(value);
            }
        };
        read_retired(
            "COMPACT_RETIRED_LPN_COUNT",
            "COMPACT_RETIRED_LPN",
            compact.page_count,
            compact.retired_lpns);
        read_retired(
            "COMPACT_RETIRED_VPN_COUNT",
            "COMPACT_RETIRED_VPN",
            compact.mapping_page_count,
            compact.retired_mapping_vpns);
        image.compact_image = std::move(compact);
    }
    expect(input, "END");
    std::string extra;
    if (input >> extra) {
        throw std::runtime_error(
            "HBF persistent image contains trailing records");
    }
    return image;
}

} // namespace hbfsim::physical::hbf
