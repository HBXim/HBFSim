# HBFSim Python client

`hbfsim_client` provides the semantic-free Python boundary to the persistent
`hbfsim` engine. It validates transaction DAGs, serializes the session
protocol, launches the simulator, and checks completion accounting.

```python
from pathlib import Path

from hbfsim_client import ResolvedSystemConfig, SimulationSession, Transaction

simulator = Path("build/hbfsim")
config = ResolvedSystemConfig.load((
    Path("configs/systems/server-hbm128-hbf512.cfg"),
)).resolve(simulator)

with SimulationSession(
    simulator_path=simulator,
    system_config=config,
    enable_hbm=True,
    enable_hbf=True,
) as session:
    result = session.run((
        Transaction(
            id="read-0",
            target="HBM",
            op="R",
            addr=0,
            bytes=64,
            issue_ns=0.0,
        ),
    ))
    print(result.elapsed_ns)
```

The package uses only the Python standard library.
