#pragma once

#include "app/source_provenance.hpp"
#include "physical/simulation_session.hpp"

#include <cstdint>
#include <iosfwd>
#include <optional>
#include <string>

namespace hbfsim::app {

struct SessionPersistentImageArtifact {
    std::string path;
    std::uint64_t bytes = 0;
    std::string sha256;
};

struct SessionProtocolOptions {
    // Provenance of the restored HBF image, when the session restores one.
    std::optional<SessionPersistentImageArtifact> initial_image;
    // Live HBF physical heat stream (JSON lines appended per batch,
    // checkpoint, and terminal drain). Requires a positive
    // SimulationSessionConfig::hbf_physical_heatmap_bins.
    std::optional<std::string> hbf_physical_heatmap_path;
    // Source-tree provenance echoed in the ready receipt.
    SourceProvenance source;
};

// Runs the sole simulator-engine protocol over caller-owned streams. Policy
// controllers submit semantic-free transaction DAGs and consume causal
// completion receipts; the engine never selects an experiment scenario.
int run_simulation_session(
    physical::SimulationSessionConfig config,
    std::istream& input,
    std::ostream& output,
    SessionProtocolOptions options = {});

} // namespace hbfsim::app
