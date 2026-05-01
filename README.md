# HBFSim Core

HBFSim is a C++20 event-driven simulator for heterogeneous memory systems
composed of high-bandwidth memory (HBM), high-bandwidth flash (HBF), and an
optional external backing tier.

This repository is a reviewer-facing snapshot of the semantic-free simulator
core. It intentionally contains no workloads, experiment drivers, study
matrices, result data, plotting/reporting code, calibration artifacts, or
paper-specific evaluation scripts.

## Included implementation

- `src/physical/hbm/`: HBM address mapping, scheduling, timing, refresh, and
  accounting.
- `src/physical/hbf/`: HBF media timing, page mapping, mapping cache, write
  buffering, garbage collection, wear/thermal behavior, and persistent images.
- `src/physical/external/`: external backing controller/media timing and the
  CXL-SSD device model.
- `src/physical/base_die_link.*`: directional per-stack data-movement links.
- `src/physical/simulation_session.*`: the transaction-DAG execution boundary,
  including checkpoints and explicit crash injection.
- `src/physical/physical_types.hpp` and `address_heatmap.*`: shared physical
  request, completion, timing-span, and accounting types.

Placement policies and workload semantics are deliberately outside the core.
Callers select explicit physical targets and dependency edges through
`SimulationSession`; the engine itself does not select a workload or experiment.

## Build

Requirements: CMake 3.20+ and a C++20 compiler.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

The build produces the `hbfsim_core` library and the `HBFSim::core` CMake alias.

## License

HBFSim Core is released under the MIT License. See `LICENSE`.
