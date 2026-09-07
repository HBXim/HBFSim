#pragma once

#include "hbfsim_build_provenance.hpp"

#include <string>

namespace hbfsim::app {

struct SourceProvenance {
    std::string git_commit = HBFSIM_GIT_COMMIT;
    bool git_dirty = HBFSIM_GIT_DIRTY != 0;
    std::string tree_hash = HBFSIM_GIT_TREE;
    std::string source_sha256 = HBFSIM_SOURCE_SHA256;
    std::string provenance_source = "build-time";
};

}
