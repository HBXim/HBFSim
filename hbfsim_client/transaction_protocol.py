#!/usr/bin/env python3
"""Semantic-free transaction-graph protocol shared by HBFSim frontends."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
import hashlib
import math
from typing import Any, Iterable, Mapping


TRANSACTION_TRACE_SCHEMA = {
    "name": "hbfsim.transaction_graph",
    "version": 1,
}
# Digest value meaning "this batch has no upstream logical trace / routing
# sidecar"; the engine only echoes these digests.
NO_UPSTREAM_DIGEST = "0" * 64
TRANSACTION_TARGETS = (
    "HBM",
    "HBF_LOGICAL",
    "HBF_STATIC",
    "HBF_PHYSICAL",
    "D2D_HBF_TO_HBM",
    "D2D_HBM_TO_HBF",
    "DIRECT_HBF_TO_EXTERNAL",
    "DIRECT_EXTERNAL_TO_HBF",
    "EXTERNAL",
    "BARRIER",
)


class TransactionProtocolError(ValueError):
    """A transaction violates the HBFSim protocol contract."""


def require_integer(
    value: Any,
    description: str,
    *,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TransactionProtocolError(
            f"{description} must be an integer >= {minimum}"
        )
    return value


def _finite(value: Any, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TransactionProtocolError(f"{description} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise TransactionProtocolError(
            f"{description} must be finite and nonnegative"
        )
    return result


def require_safe_identifier(value: str, description: str) -> str:
    if not value or any(
        not (character.isalnum() or character in "_-.:/")
        for character in value
    ):
        raise TransactionProtocolError(
            f"{description} is not protocol-safe: {value!r}"
        )
    return value


def _format_float(value: float) -> str:
    normalized = _finite(value, "transaction floating-point field")
    return format(normalized, ".17g")


def _mix64(value: int) -> int:
    mask = 2**64 - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & mask


@dataclass(frozen=True)
class HbfGeometry:
    stacks: int
    channels_per_stack: int
    dies_per_channel: int
    planes_per_die: int
    blocks_per_plane: int
    pages_per_block: int
    page_size_bytes: int
    mapping_entries_per_page: int = 512

    def __post_init__(self) -> None:
        for name, value in (
            ("stacks", self.stacks),
            ("channels_per_stack", self.channels_per_stack),
            ("dies_per_channel", self.dies_per_channel),
            ("planes_per_die", self.planes_per_die),
            ("blocks_per_plane", self.blocks_per_plane),
            ("pages_per_block", self.pages_per_block),
            ("page_size_bytes", self.page_size_bytes),
            ("mapping_entries_per_page", self.mapping_entries_per_page),
        ):
            require_integer(value, f"HBF geometry {name}", minimum=1)

    @property
    def planes_per_stack(self) -> int:
        return (
            self.channels_per_stack
            * self.dies_per_channel
            * self.planes_per_die
        )

    @property
    def planes(self) -> int:
        return self.stacks * self.planes_per_stack

    @property
    def pages_per_plane(self) -> int:
        return self.blocks_per_plane * self.pages_per_block

    @property
    def capacity_bytes(self) -> int:
        return self.planes * self.pages_per_plane * self.page_size_bytes

    def plane_physical_addr(self, plane: int, page_in_plane: int) -> int:
        if not 0 <= plane < self.planes:
            raise TransactionProtocolError("HBF physical plane is out of range")
        if not 0 <= page_in_plane < self.pages_per_plane:
            raise TransactionProtocolError("HBF page-in-plane is out of range")
        ppn = plane * self.pages_per_plane + page_in_plane
        return ppn * self.page_size_bytes

    def stack_for_plane(self, plane: int) -> int:
        if not 0 <= plane < self.planes:
            raise TransactionProtocolError("HBF physical plane is out of range")
        return plane // self.planes_per_stack

    def stack_for_logical_page(self, lpn: int) -> int:
        require_integer(lpn, "HBF logical page")
        stripe, lane = divmod(lpn, self.stacks)
        group = stripe // self.mapping_entries_per_page
        rotation = _mix64(group) % self.stacks
        return (
            lane - (self.stacks - rotation)
            if lane >= self.stacks - rotation
            else lane + rotation
        )

    def canonical(self) -> dict[str, int]:
        return {
            "stacks": self.stacks,
            "channels_per_stack": self.channels_per_stack,
            "dies_per_channel": self.dies_per_channel,
            "planes_per_die": self.planes_per_die,
            "blocks_per_plane": self.blocks_per_plane,
            "pages_per_block": self.pages_per_block,
            "page_size_bytes": self.page_size_bytes,
            "mapping_entries_per_page": self.mapping_entries_per_page,
        }


def hbf_mapping_vpn_for_logical_page(
    lpn: int,
    geometry: HbfGeometry,
) -> int:
    """Return the stack-local persistent mapping page for one logical page."""

    require_integer(lpn, "HBF logical page")
    stripe, lane = divmod(lpn, geometry.stacks)
    group = stripe // geometry.mapping_entries_per_page
    rotation = _mix64(group) % geometry.stacks
    owner = (lane + rotation) % geometry.stacks
    return group * geometry.stacks + owner


def hbf_static_page_addr(
    source_page: int,
    geometry: HbfGeometry,
) -> int:
    """Map one logical page onto the FTL-bypass static physical fabric."""

    page = require_integer(source_page, "static HBF source page")
    capacity_pages = geometry.capacity_bytes // geometry.page_size_bytes
    if page >= capacity_pages:
        raise TransactionProtocolError(
            "static HBF source page exceeds physical page capacity"
        )
    stack = geometry.stack_for_logical_page(page)
    stack_page_index = page // geometry.stacks
    local_plane = stack_page_index % geometry.planes_per_stack
    page_in_plane = stack_page_index // geometry.planes_per_stack
    plane = stack * geometry.planes_per_stack + local_plane
    return geometry.plane_physical_addr(plane, page_in_plane)


def hbf_link_bytes_by_stack(
    address: int,
    byte_count: int,
    geometry: HbfGeometry,
) -> tuple[int, ...]:
    """Decompose an HBF-page-aligned link transfer by physical stack."""

    if (
        address % geometry.page_size_bytes
        or byte_count % geometry.page_size_bytes
    ):
        raise TransactionProtocolError("D2D transfer is not HBF-page aligned")
    first_lpn = address // geometry.page_size_bytes
    remaining = byte_count // geometry.page_size_bytes
    counts = [0] * geometry.stacks
    cursor = first_lpn
    group_pages = geometry.stacks * geometry.mapping_entries_per_page
    while remaining:
        group = cursor // group_pages
        group_end = (group + 1) * group_pages
        take = min(remaining, group_end - cursor)
        start_lane = cursor % geometry.stacks
        full, remainder = divmod(take, geometry.stacks)
        rotation = _mix64(group) % geometry.stacks
        for lane in range(geometry.stacks):
            counts[(lane + rotation) % geometry.stacks] += full
        for offset in range(remainder):
            lane = (start_lane + offset) % geometry.stacks
            counts[(lane + rotation) % geometry.stacks] += 1
        cursor += take
        remaining -= take
    if sum(counts) * geometry.page_size_bytes != byte_count:
        raise TransactionProtocolError("D2D stack decomposition lost bytes")
    return tuple(count * geometry.page_size_bytes for count in counts)


def hbf_dense_mapping_pages(
    first_lpn: int,
    page_count: int,
    geometry: HbfGeometry,
) -> int:
    """Return the exact mapping-page population of a dense logical range."""

    first = require_integer(first_lpn, "dense HBF first LPN")
    pages = require_integer(page_count, "dense HBF page count")
    if pages == 0:
        return 0
    last = first + pages - 1
    if last >= 2**63 // geometry.page_size_bytes:
        raise TransactionProtocolError("dense HBF range exceeds the user namespace")
    group_pages = geometry.stacks * geometry.mapping_entries_per_page
    first_group = first // group_pages
    last_group = last // group_pages
    if first_group == last_group:
        return min(geometry.stacks, pages)
    first_edge_pages = (first_group + 1) * group_pages - first
    last_edge_pages = last % group_pages + 1
    interior_groups = last_group - first_group - 1
    return (
        min(geometry.stacks, first_edge_pages)
        + interior_groups * geometry.stacks
        + min(geometry.stacks, last_edge_pages)
    )


@dataclass(frozen=True)
class Transaction:
    id: str
    target: str
    op: str | None
    addr: int
    bytes: int
    issue_ns: float
    duration_ns: float = 0.0
    dependencies: tuple[str, ...] = ()
    stack: int | None = None

    def validate(self) -> None:
        require_safe_identifier(self.id, "transaction id")
        if self.target not in TRANSACTION_TARGETS:
            raise TransactionProtocolError(f"unknown transaction target: {self.target}")
        _finite(self.issue_ns, f"{self.id}.issue_ns")
        _finite(self.duration_ns, f"{self.id}.duration_ns")
        if self.target == "BARRIER":
            if (
                self.op is not None
                or self.addr != 0
                or self.bytes != 0
                or self.stack is not None
            ):
                raise TransactionProtocolError(
                    f"transaction barrier {self.id} is malformed"
                )
        else:
            if self.op not in {"R", "W"} or self.bytes <= 0 or self.addr < 0:
                raise TransactionProtocolError(
                    f"memory transaction {self.id} is malformed"
                )
            if self.duration_ns != 0.0:
                raise TransactionProtocolError(
                    f"memory transaction {self.id} has a duration"
                )
            if self.addr > 2**64 - 1 or self.bytes > 2**64 - self.addr:
                raise TransactionProtocolError(
                    f"transaction {self.id} address range overflows"
                )
            if self.target.startswith(("D2D_", "DIRECT_")):
                require_integer(self.stack, f"{self.id}.stack")
            elif self.stack is not None:
                raise TransactionProtocolError(
                    f"non-link transaction {self.id} has a stack field"
                )
        if len(set(self.dependencies)) != len(self.dependencies):
            raise TransactionProtocolError(
                f"transaction {self.id} repeats a dependency"
            )
        for dependency in self.dependencies:
            require_safe_identifier(dependency, f"{self.id} dependency")
            if dependency == self.id:
                raise TransactionProtocolError(
                    f"transaction {self.id} depends on itself"
                )

    def protocol_line(self) -> str:
        self.validate()
        return self._protocol_line_unchecked()

    def _protocol_line_unchecked(self) -> str:
        """Serialize a transaction that its owning batch already validated."""

        dependencies = ",".join(self.dependencies) or "-"
        stack = "-" if self.stack is None else str(self.stack)
        op = "-" if self.op is None else self.op
        return (
            f"TX id={self.id} target={self.target} op={op} "
            f"addr={self.addr} bytes={self.bytes} "
            f"issue_ns={_format_float(self.issue_ns)} "
            f"duration_ns={_format_float(self.duration_ns)} "
            f"deps={dependencies} stack={stack}"
        )


def _lower_hex_sha256(value: Any, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TransactionProtocolError(
            f"{description} must be 64 lowercase hex characters"
        )
    return value


@dataclass(frozen=True)
class TransactionBatch:
    """One causally ordered transaction DAG submitted as a unit.

    ``frontier`` names the transactions whose completion defines the batch's
    ``elapsed_ns`` and the next batch's origin (``None`` = every
    transaction; an empty tuple = none, so only the batch's last arrival
    advances the origin). Work outside the frontier still executes and can
    still be depended upon later. ``retain`` is the complete set of earlier
    transaction ids that later batches may still name as dependencies; the
    engine forgets ids older than its dependency window unless they are
    retained. A tuple replaces the engine's retained set (an empty tuple
    clears it); ``None`` leaves it unchanged, so batches from another
    producer on the same session do not undo a remapper's declaration.
    ``completions`` asks the engine for per-transaction completion records
    (``False`` keeps only the aggregate receipt).
    """

    batch_id: int
    transactions: tuple[Transaction, ...]
    logical_trace_sha256: str = NO_UPSTREAM_DIGEST
    routing_sidecar_sha256: str = NO_UPSTREAM_DIGEST
    receipt: Mapping[str, Any] = field(default_factory=dict)
    frontier: tuple[str, ...] | None = None
    retain: tuple[str, ...] | None = None
    completions: bool = True

    def __post_init__(self) -> None:
        require_integer(self.batch_id, "transaction batch id")
        _lower_hex_sha256(
            self.logical_trace_sha256, "transaction batch logical trace digest"
        )
        _lower_hex_sha256(
            self.routing_sidecar_sha256,
            "transaction batch routing sidecar digest",
        )
        if not isinstance(self.receipt, Mapping):
            raise TransactionProtocolError("transaction batch receipt must be a mapping")
        if not isinstance(self.completions, bool):
            raise TransactionProtocolError(
                "transaction batch completions flag must be a bool"
            )
        if not self.transactions:
            raise TransactionProtocolError("transaction batch has no transactions")
        seen: set[str] = set()
        for transaction in self.transactions:
            transaction.validate()
            if transaction.id in seen:
                raise TransactionProtocolError(
                    f"transaction id is duplicated: {transaction.id}"
                )
            seen.add(transaction.id)
        if self.frontier is not None:
            if not isinstance(self.frontier, tuple):
                raise TransactionProtocolError(
                    "transaction batch frontier must be a tuple of ids or None"
                )
            if len(set(self.frontier)) != len(self.frontier):
                raise TransactionProtocolError(
                    "transaction batch frontier repeats an id"
                )
            for identifier in self.frontier:
                if identifier not in seen:
                    raise TransactionProtocolError(
                        f"transaction batch frontier names an id outside the "
                        f"batch: {identifier!r}"
                    )
        if self.retain is not None:
            if not isinstance(self.retain, tuple):
                raise TransactionProtocolError(
                    "transaction batch retain must be a tuple of ids or None"
                )
            if len(set(self.retain)) != len(self.retain):
                raise TransactionProtocolError(
                    "transaction batch retain repeats an id"
                )
            for identifier in self.retain:
                require_safe_identifier(identifier, "transaction batch retain id")

    @cached_property
    def transaction_trace_sha256(self) -> str:
        digest = hashlib.sha256()
        for transaction in self.transactions:
            digest.update(transaction._protocol_line_unchecked().encode("ascii"))
            digest.update(b"\n")
        return digest.hexdigest()

    def begin_line(self) -> str:
        """The protocol ``BEGIN`` header for this batch (without newline)."""

        fields = [
            f"BEGIN {self.batch_id} {self.logical_trace_sha256} "
            f"{self.transaction_trace_sha256}"
        ]
        if self.frontier is not None:
            fields.append("frontier=" + (",".join(self.frontier) or "-"))
        if self.retain is not None:
            fields.append("retain=" + (",".join(self.retain) or "-"))
        if not self.completions:
            fields.append("completions=0")
        return " ".join(fields)

    @property
    def frontier_transactions(self) -> int:
        return (
            len(self.transactions) if self.frontier is None else len(self.frontier)
        )

    def protocol_payload_chunks(
        self,
        *,
        target_bytes: int = 1024 * 1024,
    ) -> Iterable[str]:
        """Yield the exact transaction payload without materializing the batch."""

        require_integer(
            target_bytes,
            "transaction protocol chunk target",
            minimum=1,
        )
        buffered: list[str] = []
        buffered_bytes = 0
        for transaction in self.transactions:
            line = transaction._protocol_line_unchecked() + "\n"
            line_bytes = len(line)
            if buffered and buffered_bytes + line_bytes > target_bytes:
                yield "".join(buffered)
                buffered.clear()
                buffered_bytes = 0
            if line_bytes >= target_bytes:
                if buffered:
                    yield "".join(buffered)
                    buffered.clear()
                    buffered_bytes = 0
                yield line
            else:
                buffered.append(line)
                buffered_bytes += line_bytes
        if buffered:
            yield "".join(buffered)
