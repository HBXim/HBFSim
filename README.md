# HBFSim

HBFSim is a C++20 event-driven simulator for heterogeneous memory systems
composed of high-bandwidth memory (HBM), high-bandwidth flash (HBF), and an
optional external backing tier.

The engine models timing, capacity, data movement, flash translation, garbage
collection, persistence, thermal pacing, and accounting in one causal event
system. Workload and placement logic stay above the engine boundary: callers
submit explicit physical transactions and dependency edges through
`SimulationSession`.

## Components

- `src/physical/hbm/`: HBM address mapping, command scheduling, timing,
  refresh, and accounting.
- `src/physical/hbf/`: HBF media timing, page mapping, mapping cache, write
  buffering, garbage collection, wear and thermal behavior, and persistent
  images.
- `src/physical/external/`: external backing controller/media timing and the
  CXL-SSD device model.
- `src/physical/base_die_link.*`: directional per-stack data-movement links.
- `src/physical/simulation_session.*`: persistent transaction-DAG execution,
  checkpoints, and explicit crash injection.
- `src/app/`: the `hbfsim` command-line engine, physical configuration parser,
  and session protocol.
- `hbfsim_client/`: Python contracts and client for persistent simulation
  sessions.
- `configs/`: a complete HBM+HBF system profile and composable physical-tier
  overlays.

## Build

Requirements: CMake 3.20+ and a C++20 compiler.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

This builds the `hbfsim_core` library and the `hbfsim` executable. Inspect the
resolved geometry of the included system profile with:

```bash
./build/hbfsim \
  --system-config configs/systems/server-hbm128-hbf512.cfg \
  --describe-system
```

Physical overlays are applied in command-line order. For example, the
following resolves the base system with CXL memory as an enabled external
backing tier:

```bash
./build/hbfsim \
  --system-config configs/systems/server-hbm128-hbf512.cfg \
  --system-config configs/overlays/backing/cxl-memory.cfg \
  --enable-external true \
  --describe-system
```

Run `./build/hbfsim --help` for the transaction-session options. The Python
client can be installed with `python3 -m pip install -e .`.

## License

HBFSim is released under the MIT License. See `LICENSE`.
