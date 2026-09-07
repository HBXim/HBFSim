"""Reusable Python client contracts for the HBFSim mapped-memory engine."""

from hbfsim_client.transaction_protocol import (
    HbfGeometry,
    NO_UPSTREAM_DIGEST,
    TRANSACTION_TARGETS,
    TransactionProtocolError,
    Transaction,
    TransactionBatch,
    hbf_dense_mapping_pages,
    hbf_link_bytes_by_stack,
)
from hbfsim_client.simulation_session import (
    BatchResult,
    SimulationSession,
    SimulationSessionError,
    ResolvedSystemConfig,
)
__version__ = "0.1.0"

__all__ = [
    "BatchResult",
    "HbfGeometry",
    "NO_UPSTREAM_DIGEST",
    "TRANSACTION_TARGETS",
    "TransactionProtocolError",
    "SimulationSession",
    "SimulationSessionError",
    "Transaction",
    "TransactionBatch",
    "ResolvedSystemConfig",
    "hbf_dense_mapping_pages",
    "hbf_link_bytes_by_stack",
]
