#pragma once

#include "physical/hbf/hbf_device.hpp"

#include <filesystem>

namespace hbfsim::physical::hbf {

// Canonical, strict, semantic-free representation of a quiescent HBF media
// image. The file contains no workload labels or payload values.
void write_persistent_image_file(
    const std::filesystem::path& path,
    const HbfPersistentImage& image);

[[nodiscard]] HbfPersistentImage read_persistent_image_file(
    const std::filesystem::path& path);

} // namespace hbfsim::physical::hbf
