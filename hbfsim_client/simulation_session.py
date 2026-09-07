#!/usr/bin/env python3
"""Persistent client for HBFSim's semantic-free transaction protocol."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import select
import struct
import subprocess
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence, TextIO

from hbfsim_client.transaction_protocol import (
    HbfGeometry,
    TRANSACTION_TARGETS,
    Transaction,
    TransactionBatch,
    TransactionProtocolError,
    hbf_dense_mapping_pages,
    require_safe_identifier,
)


SESSION_SCHEMA = {"name": "hbfsim.simulation_session", "version": 1}
COMPLETION_SCHEMA = {
    "name": "hbfsim.simulation_batch_completion",
    "version": 1,
}
CHECKPOINT_SCHEMA = {
    "name": "hbfsim.simulation_checkpoint_completion",
    "version": 1,
}
CRASH_SCHEMA = {
    "name": "hbfsim.simulation_crash_completion",
    "version": 1,
}
HBF_PERSISTENT_IMAGE_SCHEMA = {
    "name": "hbfsim.hbf_persistent_image",
    "version": 2,
}
HBF_WEAR_SNAPSHOT_SCHEMA = {
    "name": "hbfsim.hbf_wear_snapshot",
    "version": 1,
}
PROTOCOL = "simulation-transaction-text-v2"
TIME_BASIS = "batch_relative_ns"
SOURCE_PROVENANCE_FIELDS = {
    "git_commit",
    "git_dirty",
    "tree_hash",
    "source_sha256",
    "provenance_source",
}
LATENCY_FIELDS = {
    "transactions",
    "logical_bytes",
    "physical_bytes",
    "queue_wait_work_ns",
    "service_work_ns",
    "latency_work_ns",
    "mean_queue_wait_ns",
    "mean_service_ns",
    "mean_latency_ns",
    "min_latency_ns",
    "max_latency_ns",
    "first_arrival_ns",
    "finish_ns",
    "active_span_ns",
    "effective_logical_GBps",
    "effective_physical_GBps",
}
TRANSACTION_COMPLETION_FIELDS = {
    "id",
    "arrival_ns",
    "start_ns",
    "finish_ns",
    "logical_bytes",
    "physical_bytes",
}
TRANSACTION_COMPLETION_DIGEST_ALGORITHM = (
    "sha256_id_ieee754bits_bytes_v1"
)
HBF_WAF_DEFINITION = (
    "physical_write_bytes/(logical_write_bytes+"
    "raw_physical_program_payload_bytes)"
)


class SimulationSessionError(RuntimeError):
    """The low-level simulator process violated its simulation-session contract."""


@dataclass(frozen=True)
class CompletionRecord:
    """One memory/link transaction's physical completion."""

    id: str
    arrival_ns: float
    start_ns: float
    finish_ns: float
    logical_bytes: int
    physical_bytes: int

    @property
    def latency_ns(self) -> float:
        return self.finish_ns - self.arrival_ns


@dataclass(frozen=True)
class BatchResult:
    """Typed view of one batch completion receipt.

    ``elapsed_ns`` follows the batch frontier (the transactions that gate
    the next batch); ``total_elapsed_ns`` covers every transaction of the
    batch, detached work included. ``completions`` is empty when the batch
    was submitted with ``completions=False``. ``receipt`` is the complete
    JSON receipt for anything not surfaced here.
    """

    batch_id: str
    sequence: int
    batch_origin_ns: float
    first_issue_ns: float
    blocking_finish_ns: float
    finish_ns: float
    elapsed_ns: float
    total_elapsed_ns: float
    frontier_transactions: int
    completions: tuple[CompletionRecord, ...]
    receipt: dict[str, Any]

    def completion(self, transaction_id: str) -> CompletionRecord:
        for record in self.completions:
            if record.id == transaction_id:
                return record
        raise KeyError(transaction_id)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path, description: str) -> dict[str, Any]:
    if path.is_symlink():
        raise SimulationSessionError(
            f"{description} must be a regular non-symlink file: {path}"
        )
    resolved = path.resolve()
    if not resolved.is_file():
        raise SimulationSessionError(
            f"{description} must be a regular non-symlink file: {resolved}"
        )
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _positive_integer(value: Any, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SimulationSessionError(f"{description} must be an integer > 0")
    return value


def _nonnegative_integer(value: Any, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SimulationSessionError(f"{description} must be an integer >= 0")
    return value


def _json_object(line: str, description: str) -> dict[str, Any]:
    try:
        value = json.loads(
            line,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number {token}")
            ),
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise SimulationSessionError(
            f"{description} is not strict JSON: {error}"
        ) from error
    if not isinstance(value, dict):
        raise SimulationSessionError(f"{description} is not a JSON object")
    return value


def _finite_nonnegative(value: Any, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SimulationSessionError(f"{description} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise SimulationSessionError(f"{description} must be finite and non-negative")
    return result


def _validate_latency_matrix(
    value: Any,
    *,
    description: str,
    expected_transactions: Mapping[str, Mapping[str, int]] | None = None,
) -> None:
    if not isinstance(value, Mapping) or set(value) != set(TRANSACTION_TARGETS):
        raise SimulationSessionError(f"{description} has invalid target coverage")
    nullable_fields = {
        "mean_queue_wait_ns",
        "mean_service_ns",
        "mean_latency_ns",
        "min_latency_ns",
        "max_latency_ns",
        "first_arrival_ns",
        "finish_ns",
        "active_span_ns",
        "effective_logical_GBps",
        "effective_physical_GBps",
    }
    for target in TRANSACTION_TARGETS:
        operations = value.get(target)
        if not isinstance(operations, Mapping) or set(operations) != {
            "read",
            "write",
        }:
            raise SimulationSessionError(
                f"{description}.{target} must contain exact read/write records"
            )
        for operation in ("read", "write"):
            row = operations.get(operation)
            path = f"{description}.{target}.{operation}"
            if not isinstance(row, Mapping) or set(row) != LATENCY_FIELDS:
                raise SimulationSessionError(f"{path} has an invalid metric schema")
            transactions = _nonnegative_integer(
                row.get("transactions"), f"{path}.transactions"
            )
            logical_bytes = _nonnegative_integer(
                row.get("logical_bytes"), f"{path}.logical_bytes"
            )
            physical_bytes = _nonnegative_integer(
                row.get("physical_bytes"), f"{path}.physical_bytes"
            )
            queue_work = _finite_nonnegative(
                row.get("queue_wait_work_ns"), f"{path}.queue_wait_work_ns"
            )
            service_work = _finite_nonnegative(
                row.get("service_work_ns"), f"{path}.service_work_ns"
            )
            latency_work = _finite_nonnegative(
                row.get("latency_work_ns"), f"{path}.latency_work_ns"
            )
            if not math.isclose(
                latency_work,
                queue_work + service_work,
                rel_tol=1e-10,
                abs_tol=1e-6,
            ):
                raise SimulationSessionError(f"{path} latency work does not conserve")
            if expected_transactions is not None and transactions != int(
                expected_transactions[target][operation]
            ):
                raise SimulationSessionError(
                    f"{path} transaction count diverged from transaction input"
                )
            if transactions == 0:
                if logical_bytes or physical_bytes or queue_work or service_work:
                    raise SimulationSessionError(f"{path} empty row has non-zero work")
                if any(row.get(field) is not None for field in nullable_fields):
                    raise SimulationSessionError(f"{path} empty row has defined moments")
                continue
            numeric = {
                field: _finite_nonnegative(row.get(field), f"{path}.{field}")
                for field in nullable_fields
            }
            if not math.isclose(
                numeric["mean_queue_wait_ns"] * transactions,
                queue_work,
                rel_tol=1e-10,
                abs_tol=1e-6,
            ) or not math.isclose(
                numeric["mean_service_ns"] * transactions,
                service_work,
                rel_tol=1e-10,
                abs_tol=1e-6,
            ) or not math.isclose(
                numeric["mean_latency_ns"] * transactions,
                latency_work,
                rel_tol=1e-10,
                abs_tol=1e-6,
            ):
                raise SimulationSessionError(f"{path} mean timing does not conserve")
            active_span = numeric["finish_ns"] - numeric["first_arrival_ns"]
            if active_span < 0 or not math.isclose(
                active_span,
                numeric["active_span_ns"],
                rel_tol=1e-10,
                abs_tol=1e-6,
            ):
                raise SimulationSessionError(f"{path} active span does not conserve")
            mean_below_min = (
                numeric["mean_latency_ns"] < numeric["min_latency_ns"]
                and not math.isclose(
                    numeric["mean_latency_ns"],
                    numeric["min_latency_ns"],
                    rel_tol=1e-10,
                    abs_tol=1e-6,
                )
            )
            mean_above_max = (
                numeric["mean_latency_ns"] > numeric["max_latency_ns"]
                and not math.isclose(
                    numeric["mean_latency_ns"],
                    numeric["max_latency_ns"],
                    rel_tol=1e-10,
                    abs_tol=1e-6,
                )
            )
            if mean_below_min or mean_above_max:
                raise SimulationSessionError(f"{path} latency moments are inconsistent")


def _transaction_completion_digest(
    completions: Sequence[Mapping[str, Any]],
) -> str:
    material = bytearray(
        b"hbfsim.simulation_transaction_completions.v1\n"
        + f"count={len(completions)}\n".encode("ascii")
    )
    for index, completion in enumerate(completions):
        identifier = str(completion["id"])
        material.extend(f"index={index}\n".encode("ascii"))
        material.extend(
            f"id={len(identifier)}:{identifier}\n".encode("ascii")
        )
        for field, label in (
            ("arrival_ns", "arrival_bits"),
            ("start_ns", "start_bits"),
            ("finish_ns", "finish_bits"),
        ):
            bits = struct.unpack(
                ">Q", struct.pack(">d", float(completion[field]))
            )[0]
            material.extend(f"{label}={bits:016x}\n".encode("ascii"))
        material.extend(
            f"logical_bytes={completion['logical_bytes']}\n".encode("ascii")
        )
        material.extend(
            f"physical_bytes={completion['physical_bytes']}\n".encode("ascii")
        )
    return hashlib.sha256(material).hexdigest()


def _validate_transaction_completions(
    value: Any,
    digest_value: Any,
    *,
    batch: TransactionBatch,
    latency_matrix: Mapping[str, Mapping[str, Mapping[str, Any]]],
    batch_origin_ns: float,
    batch_finish_ns: float,
) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        raise SimulationSessionError(
            "batch transaction_completions must be a list"
        )
    if not isinstance(digest_value, Mapping) or set(digest_value) != {
        "algorithm",
        "sha256",
    }:
        raise SimulationSessionError(
            "batch transaction completion digest schema is invalid"
        )
    if digest_value.get("algorithm") != TRANSACTION_COMPLETION_DIGEST_ALGORITHM:
        raise SimulationSessionError(
            "batch transaction completion digest algorithm is unsupported"
        )
    expected_by_id = {
        transaction.id: transaction
        for transaction in batch.transactions
        if transaction.target != "BARRIER"
    }
    if len(value) != len(expected_by_id):
        raise SimulationSessionError(
            "batch transaction completion count differs from memory input"
        )
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    aggregate: dict[tuple[str, str], dict[str, Any]] = {
        (target, operation): {
            "transactions": 0,
            "logical_bytes": 0,
            "physical_bytes": 0,
            "queue_wait_work_ns": 0.0,
            "service_work_ns": 0.0,
            "latency_work_ns": 0.0,
            "min_latency_ns": math.inf,
            "max_latency_ns": 0.0,
            "first_arrival_ns": math.inf,
            "finish_ns": 0.0,
        }
        for target in TRANSACTION_TARGETS
        for operation in ("read", "write")
    }
    for index, raw in enumerate(value):
        label = f"transaction_completions[{index}]"
        if not isinstance(raw, Mapping) or set(raw) != TRANSACTION_COMPLETION_FIELDS:
            raise SimulationSessionError(f"{label} has an invalid schema")
        identifier = raw.get("id")
        if not isinstance(identifier, str) or identifier not in expected_by_id:
            raise SimulationSessionError(f"{label} names an unknown input ID")
        if identifier in seen:
            raise SimulationSessionError(
                f"{label} repeats transaction ID {identifier}"
            )
        seen.add(identifier)
        transaction = expected_by_id[identifier]
        arrival = _finite_nonnegative(raw.get("arrival_ns"), f"{label}.arrival_ns")
        start = _finite_nonnegative(raw.get("start_ns"), f"{label}.start_ns")
        finish = _finite_nonnegative(raw.get("finish_ns"), f"{label}.finish_ns")
        logical_bytes = _nonnegative_integer(
            raw.get("logical_bytes"), f"{label}.logical_bytes"
        )
        physical_bytes = _nonnegative_integer(
            raw.get("physical_bytes"), f"{label}.physical_bytes"
        )
        if (
            logical_bytes != transaction.bytes
            or arrival < batch_origin_ns
            or arrival > start
            or start > finish
            or finish > batch_finish_ns + 1e-6
        ):
            raise SimulationSessionError(
                f"{label} bytes or arrival/start/finish ordering diverged"
            )
        row = {
            "id": identifier,
            "arrival_ns": arrival,
            "start_ns": start,
            "finish_ns": finish,
            "logical_bytes": logical_bytes,
            "physical_bytes": physical_bytes,
        }
        normalized.append(row)
        operation = "read" if transaction.op == "R" else "write"
        stats = aggregate[(transaction.target, operation)]
        queue = start - arrival
        service = finish - start
        latency = finish - arrival
        stats["transactions"] += 1
        stats["logical_bytes"] += logical_bytes
        stats["physical_bytes"] += physical_bytes
        stats["queue_wait_work_ns"] += queue
        stats["service_work_ns"] += service
        stats["latency_work_ns"] += latency
        stats["min_latency_ns"] = min(stats["min_latency_ns"], latency)
        stats["max_latency_ns"] = max(stats["max_latency_ns"], latency)
        stats["first_arrival_ns"] = min(stats["first_arrival_ns"], arrival)
        stats["finish_ns"] = max(stats["finish_ns"], finish)
    if seen != set(expected_by_id):
        raise SimulationSessionError(
            "batch transaction completions omit one or more memory input IDs"
        )
    claimed_digest = digest_value.get("sha256")
    if (
        not isinstance(claimed_digest, str)
        or len(claimed_digest) != 64
        or any(character not in "0123456789abcdef" for character in claimed_digest)
        or _transaction_completion_digest(normalized) != claimed_digest
    ):
        raise SimulationSessionError(
            "batch transaction completion digest does not reproduce"
        )
    for target in TRANSACTION_TARGETS:
        for operation in ("read", "write"):
            observed = aggregate[(target, operation)]
            expected = latency_matrix[target][operation]
            for field in (
                "transactions",
                "logical_bytes",
                "physical_bytes",
            ):
                if observed[field] != expected[field]:
                    raise SimulationSessionError(
                        f"transaction completions do not conserve {target}.{operation}.{field}"
                    )
            for field in (
                "queue_wait_work_ns",
                "service_work_ns",
                "latency_work_ns",
            ):
                if not math.isclose(
                    observed[field],
                    float(expected[field]),
                    rel_tol=1e-10,
                    abs_tol=1e-6,
                ):
                    raise SimulationSessionError(
                        f"transaction completions do not conserve {target}.{operation}.{field}"
                    )
            if observed["transactions"]:
                for field in (
                    "min_latency_ns",
                    "max_latency_ns",
                    "first_arrival_ns",
                    "finish_ns",
                ):
                    if not math.isclose(
                        observed[field],
                        float(expected[field]),
                        rel_tol=1e-10,
                        abs_tol=1e-6,
                    ):
                        raise SimulationSessionError(
                            f"transaction completions do not conserve {target}.{operation}.{field}"
                        )
    return tuple(normalized)


def _validate_device_accounting(
    value: Any,
    *,
    enable_hbm: bool,
    enable_hbf: bool,
    enable_external: bool,
    external_kind: str | None,
    description: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "hbm",
        "hbf",
        "external",
        "base_die_link",
        "hbf_external_direct_link",
    }:
        raise SimulationSessionError(f"{description} has invalid tier coverage")
    hbm = value.get("hbm")
    if (hbm is None) == enable_hbm or (
        hbm is not None and not isinstance(hbm, Mapping)
    ):
        raise SimulationSessionError(f"{description}.hbm availability diverged")
    hbf = value.get("hbf")
    if (hbf is None) == enable_hbf or (
        hbf is not None and not isinstance(hbf, Mapping)
    ):
        raise SimulationSessionError(f"{description}.hbf availability diverged")
    external = value.get("external")
    if (external is None) == enable_external or (
        external is not None and not isinstance(external, Mapping)
    ):
        raise SimulationSessionError(f"{description}.external availability diverged")
    if not isinstance(value.get("base_die_link"), Mapping):
        raise SimulationSessionError(f"{description}.base_die_link is missing")
    if not isinstance(value.get("hbf_external_direct_link"), Mapping):
        raise SimulationSessionError(
            f"{description}.hbf_external_direct_link is missing"
        )
    if isinstance(hbf, Mapping):
        logical = _nonnegative_integer(
            hbf.get("logical_write_bytes"),
            f"{description}.hbf.logical_write_bytes",
        )
        physical = _nonnegative_integer(
            hbf.get("physical_write_bytes"),
            f"{description}.hbf.physical_write_bytes",
        )
        raw_physical = _nonnegative_integer(
            hbf.get("raw_physical_program_payload_bytes"),
            f"{description}.hbf.raw_physical_program_payload_bytes",
        )
        decomposed = sum(
            _nonnegative_integer(hbf.get(field), f"{description}.hbf.{field}")
            for field in (
                "data_program_payload_bytes",
                "mapping_program_payload_bytes",
                "gc_relocation_payload_bytes",
            )
        )
        # Static wear-leveling migrations are a fourth physical-write
        # component; receipts from engines without the mechanism omit it.
        if hbf.get("static_wear_leveling_relocation_payload_bytes") is not None:
            decomposed += _nonnegative_integer(
                hbf.get("static_wear_leveling_relocation_payload_bytes"),
                f"{description}.hbf.static_wear_leveling_relocation_payload_bytes",
            )
        if physical != decomposed:
            raise SimulationSessionError(
                f"{description}.hbf physical write decomposition diverged"
            )
        if hbf.get("waf_definition") != HBF_WAF_DEFINITION:
            raise SimulationSessionError(
                f"{description}.hbf WAF definition diverged"
            )
        host_written = logical + raw_physical
        waf = hbf.get("waf")
        if host_written == 0:
            if waf is not None:
                raise SimulationSessionError(f"{description}.hbf zero-write WAF defined")
        elif not math.isclose(
            _finite_nonnegative(waf, f"{description}.hbf.waf"),
            physical / host_written,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise SimulationSessionError(f"{description}.hbf WAF does not conserve")
    if isinstance(external, Mapping):
        if external.get("kind") != external_kind:
            raise SimulationSessionError(
                f"{description}.external backing kind diverged"
            )
        counters = {
            field: _nonnegative_integer(
                external.get(field), f"{description}.external.{field}"
            )
            for field in (
                "read_requests",
                "write_requests",
                "read_bytes",
                "write_bytes",
                "page_run_requests",
                "page_run_segments",
                "page_run_pages",
                "m2s_payload_bytes",
                "m2s_protocol_bytes",
                "m2s_wire_bytes",
                "s2m_payload_bytes",
                "s2m_protocol_bytes",
                "s2m_wire_bytes",
            )
        }
        if (
            counters["page_run_segments"]
            > counters["read_requests"] + counters["write_requests"]
            or counters["page_run_requests"] > counters["page_run_segments"]
            or counters["page_run_segments"] > counters["page_run_pages"]
            or counters["page_run_requests"] > counters["page_run_pages"]
            or (
                counters["page_run_segments"] != 0
                and counters["page_run_requests"] == 0
            )
            or (
                counters["page_run_pages"] != 0
                and counters["page_run_requests"] == 0
            )
            or counters["m2s_payload_bytes"] != counters["write_bytes"]
            or counters["s2m_payload_bytes"] != counters["read_bytes"]
            or counters["m2s_wire_bytes"]
            != counters["m2s_payload_bytes"]
            + counters["m2s_protocol_bytes"]
            or counters["s2m_wire_bytes"]
            != counters["s2m_payload_bytes"]
            + counters["s2m_protocol_bytes"]
        ):
            raise SimulationSessionError(
                f"{description}.external transport bytes do not conserve"
            )
        for field in (
            "outstanding_wait_work_ns",
            "controller_queue_wait_work_ns",
            "controller_issue_busy_ns",
            "controller_processing_work_ns",
            "media_queue_wait_work_ns",
            "media_read_latency_work_ns",
            "media_write_latency_work_ns",
            "media_read_busy_ns",
            "media_write_busy_ns",
            "m2s_queue_wait_work_ns",
            "s2m_queue_wait_work_ns",
            "m2s_busy_ns",
            "s2m_busy_ns",
            "transport_propagation_work_ns",
        ):
            _finite_nonnegative(
                external.get(field), f"{description}.external.{field}"
            )
        if not isinstance(external.get("stage_work"), Mapping):
            raise SimulationSessionError(
                f"{description}.external stage-work receipt is missing"
            )
        device_cache = external.get("device_cache")
        if not isinstance(device_cache, Mapping):
            raise SimulationSessionError(
                f"{description}.external device-cache census is missing"
            )
        cache_counters = {
            field: _nonnegative_integer(
                device_cache.get(field),
                f"{description}.external.device_cache.{field}",
            )
            for field in EXTERNAL_DEVICE_CACHE_COUNTER_FIELDS
        }
        if (
            cache_counters["read_hits"] + cache_counters["read_misses"]
            > counters["read_requests"]
            or cache_counters["write_hits"] + cache_counters["write_misses"]
            > counters["write_requests"]
        ):
            raise SimulationSessionError(
                f"{description}.external device-cache census exceeds "
                "caller requests"
            )
        for field in EXTERNAL_DEVICE_CACHE_WORK_FIELDS:
            _finite_nonnegative(
                device_cache.get(field),
                f"{description}.external.device_cache.{field}",
            )


EXTERNAL_DEVICE_CACHE_COUNTER_FIELDS = (
    "read_hits",
    "read_misses",
    "write_hits",
    "write_misses",
    "read_for_ownership_segments",
    "read_for_ownership_bytes",
    "writeback_segments",
    "writeback_bytes",
    "prefetch_segments",
    "prefetch_bytes",
)
EXTERNAL_DEVICE_CACHE_WORK_FIELDS = (
    "latency_work_ns",
    "queue_wait_work_ns",
    "busy_ns",
    "flush_busy_ns",
    "prefetch_busy_ns",
    "writeback_gate_wait_work_ns",
)
QUIESCENCE_FIELDS = {
    "dirty_mapping_pages",
    "pending_dirty_mapping_events",
    "pending_lpn_updates",
    "pending_vpn_updates",
    "pending_commits",
    "write_buffer_entries",
    "inflight_buffered_generations",
    "pending_physical_programs",
    "pending_physical_erases",
}


def _normalized_quiescence(
    value: Any,
    description: str,
) -> dict[str, int | bool]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"verified", *QUIESCENCE_FIELDS}
        or not isinstance(value.get("verified"), bool)
    ):
        raise SimulationSessionError(
            f"{description} has an invalid quiescence schema"
        )
    normalized: dict[str, int | bool] = {
        "verified": bool(value["verified"])
    }
    for field in QUIESCENCE_FIELDS:
        normalized[field] = _nonnegative_integer(
            value.get(field), f"{description}.{field}"
        )
    all_clear = all(normalized[field] == 0 for field in QUIESCENCE_FIELDS)
    if normalized["verified"] is not all_clear:
        raise SimulationSessionError(
            f"{description} verified flag diverged from pending state"
        )
    return normalized


def _validate_quiescence(value: Any, description: str) -> None:
    normalized = _normalized_quiescence(value, description)
    if normalized["verified"] is not True:
        raise SimulationSessionError(f"{description} did not prove quiescence")


@dataclass(frozen=True)
class ResolvedSystemConfig:
    """The geometry fields the remapper must match to the simulator config."""

    paths: tuple[Path, ...]
    values: Mapping[str, str]
    artifacts: tuple[Mapping[str, Any], ...]
    logical_hbf_capacity_bytes: int | None = None

    def resolve(
        self, simulator_path: Path, *, enable_hbf: bool = True
    ) -> "ResolvedSystemConfig":
        command = [
            str(simulator_path.resolve()), "--describe-system", "--enable-hbf",
            "true" if enable_hbf else "false",
        ]
        for path in self.paths:
            command.extend(("--system-config", str(path)))
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=30
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise SimulationSessionError(f"cannot resolve HBFSim geometry: {error}") from error
        if completed.returncode:
            raise SimulationSessionError(
                f"cannot resolve HBFSim geometry: {completed.stderr.strip()}"
            )
        try:
            descriptor = json.loads(completed.stdout)
        except (ValueError, TypeError) as error:
            raise SimulationSessionError("invalid HBFSim resolved-system receipt") from error
        if not isinstance(descriptor, dict) or descriptor.get("schema") != {
            "name": "hbfsim.resolved_system", "version": 1,
        }:
            raise SimulationSessionError("unsupported HBFSim resolved-system receipt")
        capacity = _nonnegative_integer(
            descriptor.get("hbf_logical_capacity_bytes"), "resolved logical HBF capacity"
        )
        values = descriptor.get("values")
        required_values = {
            "hbm-capacity-bytes", "hbf-stacks", "hbf-channels",
            "hbf-dies-per-channel", "hbf-planes-per-die", "hbf-blocks-per-plane",
            "hbf-pages-per-block", "hbf-page-size", "hbf-mapping-entries-per-page",
            "hbf-mapping-mode", "hbf-ctrl-dram-bytes",
        }
        if (
            not isinstance(values, dict)
            or set(values) != required_values
            or not all(isinstance(value, str) for value in values.values())
        ):
            raise SimulationSessionError("invalid HBFSim resolved geometry values")
        resolved = replace(
            self, values={**self.values, **values},
            logical_hbf_capacity_bytes=capacity,
        )
        if capacity > resolved.hbf_geometry.capacity_bytes:
            raise SimulationSessionError("resolved logical HBF capacity exceeds raw capacity")
        return resolved

    @classmethod
    def load(cls, paths: Sequence[Path]) -> "ResolvedSystemConfig":
        if not paths:
            raise SimulationSessionError("at least one HBFSim system config is required")
        resolved_paths: list[Path] = []
        artifacts: list[Mapping[str, Any]] = []
        values: dict[str, str] = {}
        for raw_path in paths:
            path = raw_path.resolve()
            artifact = _artifact(path, "HBFSim system config")
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeError) as error:
                raise SimulationSessionError(
                    f"cannot read HBFSim system config {path}: {error}"
                ) from error
            for line_no, raw_line in enumerate(lines, start=1):
                line = raw_line.partition("#")[0].strip()
                if not line:
                    continue
                if "=" not in line:
                    raise SimulationSessionError(
                        f"{path}:{line_no} is not a key=value config record"
                    )
                key, value = (part.strip() for part in line.split("=", 1))
                if not key or not value:
                    raise SimulationSessionError(
                        f"{path}:{line_no} has an empty config key or value"
                    )
                values[key] = value
            resolved_paths.append(path)
            artifacts.append(artifact)
        return cls(
            paths=tuple(resolved_paths),
            values=values,
            artifacts=tuple(artifacts),
        )

    def integer(self, key: str, *, minimum: int = 1) -> int:
        raw = self.values.get(key)
        if raw is None:
            raise SimulationSessionError(
                f"resolved HBFSim config does not define {key}"
            )
        try:
            value = int(raw)
        except ValueError as error:
            raise SimulationSessionError(
                f"resolved HBFSim config {key} must be an integer"
            ) from error
        if str(value) != raw or value < minimum:
            raise SimulationSessionError(
                f"resolved HBFSim config {key} must be a canonical integer "
                f">= {minimum}"
            )
        return value

    def boolean(self, key: str, *, default: bool | None = None) -> bool:
        raw = self.values.get(key)
        if raw is None and default is not None:
            return default
        if raw == "true":
            return True
        if raw == "false":
            return False
        raise SimulationSessionError(
            f"resolved HBFSim config {key} must be true or false"
        )

    def floating(self, key: str, *, minimum: float | None = None) -> float:
        raw = self.values.get(key)
        if raw is None:
            raise SimulationSessionError(
                f"resolved HBFSim config does not define {key}"
            )
        try:
            value = float(raw)
        except ValueError as error:
            raise SimulationSessionError(
                f"resolved HBFSim config {key} must be numeric"
            ) from error
        if not math.isfinite(value) or (
            minimum is not None and value < minimum
        ):
            raise SimulationSessionError(
                f"resolved HBFSim config {key} must be finite"
                + ("" if minimum is None else f" and >= {minimum}")
            )
        return value

    @property
    def hbm_capacity_bytes(self) -> int:
        return self.integer("hbm-capacity-bytes")

    @property
    def hbm_burst_bytes(self) -> int:
        channel_width = self.integer("hbm-channel-width-bits")
        pseudo_channels = self.integer("hbm-pseudo-channels")
        burst_length = self.integer("hbm-burst-length")
        if channel_width % pseudo_channels:
            raise SimulationSessionError(
                "HBM channel width must divide evenly across pseudo-channels"
            )
        burst_bits = channel_width // pseudo_channels * burst_length
        if burst_bits % 8:
            raise SimulationSessionError("HBM burst geometry is not byte aligned")
        return burst_bits // 8

    @property
    def hbf_mapping_mode(self) -> str:
        mode = self.values.get("hbf-mapping-mode", "full-resident")
        if mode not in {"full-resident", "cached", "direct"}:
            raise SimulationSessionError(
                "resolved HBFSim config hbf-mapping-mode must be "
                "full-resident, cached, or direct"
            )
        return mode

    @property
    def hbf_thermal_identity(self) -> dict[str, Any]:
        """Resolved per-stack thermal model identity, mirroring the engine.

        The derived quantities replicate the simulator's own resolution
        (idle steady state, pacing budget, implied sustainable bandwidth)
        so receipts and preflights can bind the thermal contract without
        re-reading raw config text.
        """

        enabled = self.boolean("hbf-thermal-enable", default=False)
        start_state = self.values.get("hbf-thermal-start-state", "idle")
        if start_state not in {"idle", "throttle-ceiling"}:
            raise SimulationSessionError(
                "resolved HBFSim config hbf-thermal-start-state must be "
                "idle or throttle-ceiling"
            )
        if not enabled:
            if start_state != "idle":
                raise SimulationSessionError(
                    "hbf-thermal-start-state requires hbf-thermal-enable"
                )
            return {"enabled": False, "start_state": "idle"}
        ambient_c = self.floating("hbf-thermal-ambient-c")
        resistance = self.floating(
            "hbf-thermal-resistance-c-per-w", minimum=1e-12
        )
        capacitance = self.floating(
            "hbf-thermal-capacitance-j-per-c", minimum=1e-12
        )
        throttle_c = self.floating("hbf-thermal-throttle-c")
        release_c = self.floating("hbf-thermal-release-c")
        static_power = self.floating(
            "hbf-thermal-static-power-w", minimum=0.0
        )
        read_pj_per_bit = self.floating(
            "hbf-thermal-read-energy-pj-per-bit", minimum=1e-12
        )
        program_pj_per_bit = self.floating(
            "hbf-thermal-program-energy-pj-per-bit", minimum=1e-12
        )
        erase_uj_per_block = self.floating(
            "hbf-thermal-erase-energy-uj-per-block", minimum=0.0
        )
        explicit_power = self.floating(
            "hbf-thermal-throttle-power-w", minimum=0.0
        )
        neighbor_heat_c = (
            0.0
            if self.values.get("hbf-thermal-neighbor-heat-c") is None
            else self.floating(
                "hbf-thermal-neighbor-heat-c", minimum=0.0
            )
        )
        boundary_c = ambient_c + neighbor_heat_c
        idle_c = boundary_c + static_power * resistance
        if release_c >= throttle_c or idle_c >= release_c:
            raise SimulationSessionError(
                "resolved HBFSim thermal thresholds must satisfy "
                "idle < release < throttle"
            )
        pacing_power = (
            explicit_power
            if explicit_power > 0.0
            else (throttle_c - boundary_c) / resistance - static_power
        )
        if pacing_power <= 0.0:
            raise SimulationSessionError(
                "resolved HBFSim thermal pacing power is non-positive"
            )
        # bytes/s = W / (J/byte); pj/bit * 8e-3 = mJ/GB, so GB/s decimal.
        sustainable_read = pacing_power / (read_pj_per_bit * 8e-3)
        sustainable_program = pacing_power / (program_pj_per_bit * 8e-3)
        return {
            "enabled": True,
            "start_state": start_state,
            "ambient_c": ambient_c,
            "neighbor_heat_c": neighbor_heat_c,
            "boundary_temperature_c": boundary_c,
            "resistance_c_per_w": resistance,
            "capacitance_j_per_c": capacitance,
            "time_constant_s": resistance * capacitance,
            "throttle_c": throttle_c,
            "release_c": release_c,
            "static_power_w": static_power,
            "idle_temperature_c": idle_c,
            "boot_temperature_c": (
                throttle_c if start_state == "throttle-ceiling" else idle_c
            ),
            "read_energy_pj_per_bit": read_pj_per_bit,
            "program_energy_pj_per_bit": program_pj_per_bit,
            "erase_energy_uj_per_block": erase_uj_per_block,
            "pacing_power_w": pacing_power,
            "pacing_power_derivation": (
                "explicit"
                if explicit_power > 0.0
                else "throttle_minus_boundary_over_resistance_minus_static"
            ),
            "sustainable_read_GBps_per_stack": sustainable_read,
            "sustainable_program_GBps_per_stack": sustainable_program,
        }

    @property
    def hbf_ctrl_dram_bytes(self) -> int:
        if self.logical_hbf_capacity_bytes is not None:
            return self.integer("hbf-ctrl-dram-bytes", minimum=0)
        raw = self.values.get("hbf-ctrl-dram-bytes")
        denominator_raw = self.values.get(
            "hbf-ctrl-dram-capacity-denominator"
        )
        if raw is not None and denominator_raw is not None:
            raise SimulationSessionError(
                "resolved HBFSim config cannot define both explicit and "
                "capacity-derived controller DRAM"
            )
        if raw is not None:
            configured = self.integer("hbf-ctrl-dram-bytes", minimum=0)
            if configured > 0:
                return configured
        if denominator_raw is not None:
            denominator = self.integer(
                "hbf-ctrl-dram-capacity-denominator"
            )
            geometry = self.hbf_geometry
            budget_pages_per_stack = (
                geometry.planes_per_stack
                * geometry.pages_per_plane
                // denominator
            )
            if budget_pages_per_stack == 0:
                raise SimulationSessionError(
                    "controller-DRAM ratio yields less than one page per stack"
                )
            return (
                budget_pages_per_stack
                * geometry.page_size_bytes
                * geometry.stacks
            )
        if self.hbf_mapping_mode == "direct":
            # The exposed address space carries no L2P state; controller
            # DRAM holds only a configured write buffer, and direct mode
            # requires coalescing disabled.
            return 0
        if self.hbf_mapping_mode != "full-resident":
            raise SimulationSessionError(
                "resolved cached HBF mapping requires hbf-ctrl-dram-bytes"
            )
        geometry = self.hbf_geometry
        data_pages_per_stack = (
            geometry.planes_per_stack * geometry.pages_per_plane
        )
        mapping_pages_per_stack = (
            data_pages_per_stack
            + geometry.mapping_entries_per_page
            - 1
        ) // geometry.mapping_entries_per_page
        mapping_bytes = (
            mapping_pages_per_stack
            * geometry.page_size_bytes
            * geometry.stacks
        )
        write_buffer_bytes = 0
        if self.boolean("hbf-write-coalescing", default=False):
            write_buffer_bytes = (
                self.integer("hbf-write-buffer-pages")
                * geometry.page_size_bytes
                * geometry.stacks
            )
        return mapping_bytes + write_buffer_bytes

    @property
    def external_backing_identity(self) -> Mapping[str, Any]:
        kind = self.values.get("external-backing-kind")
        if kind not in {
            "on-package-lpddr",
            "host-dram",
            "cxl-memory",
            "nvme-ssd",
            "cxl-ssd",
        }:
            raise SimulationSessionError(
                "resolved HBFSim config has no supported external backing kind"
            )
        capacity_bytes = self.integer("external-backing-capacity-bytes")
        page_size_bytes = self.integer("external-backing-page-size")
        request_segment_bytes = self.integer(
            "external-backing-request-segment-bytes"
        )
        if capacity_bytes % page_size_bytes:
            raise SimulationSessionError(
                "external backing capacity must be a multiple of page size"
            )
        if (
            request_segment_bytes < page_size_bytes
            or request_segment_bytes % page_size_bytes
        ):
            raise SimulationSessionError(
                "external backing request segment must be a page-size "
                "multiple no smaller than one page"
            )
        media_channels = self.integer("external-backing-media-channels")
        media_read_queues = self.integer(
            "external-backing-media-read-queues"
        )
        media_write_queues = self.integer(
            "external-backing-media-write-queues"
        )
        if min(media_channels, media_read_queues, media_write_queues) <= 0:
            raise SimulationSessionError(
                "external backing media channels and directional queues "
                "must be positive"
            )
        identity: dict[str, Any] = {
            "kind": kind,
            "capacity_bytes": capacity_bytes,
            "page_size_bytes": page_size_bytes,
            "request_segment_bytes": request_segment_bytes,
            "media_channels": media_channels,
            "media_read_queues": media_read_queues,
            "media_write_queues": media_write_queues,
            "max_outstanding_requests": self.integer(
                "external-backing-max-outstanding-requests"
            ),
        }
        if self.boolean("external-backing-cache-enabled", default=False):
            # CXL-SSD device-cache identity mirrors the session binary's
            # ready echo exactly; the key is absent while the cache is
            # disabled so existing identities stay unchanged.
            cache_policy = self.values.get("external-backing-cache-policy")
            if cache_policy not in {"fifo", "lifo", "clock", "s3fifo"}:
                raise SimulationSessionError(
                    "external backing cache policy must be fifo, lifo, "
                    "clock, or s3fifo"
                )
            identity["device_cache"] = {
                "enabled": True,
                "capacity_bytes": self.integer(
                    "external-backing-cache-capacity-bytes"
                ),
                "ways": self.integer(
                    "external-backing-cache-ways", minimum=0
                ),
                "policy": cache_policy,
                "prefetch_degree": self.integer(
                    "external-backing-cache-prefetch-degree", minimum=0
                ),
                "prefetch_stride": self.integer(
                    "external-backing-cache-prefetch-stride"
                ),
            }
        return identity

    @property
    def hbf_geometry(self) -> HbfGeometry:
        return HbfGeometry(
            stacks=self.integer("hbf-stacks"),
            channels_per_stack=self.integer("hbf-channels"),
            dies_per_channel=self.integer("hbf-dies-per-channel"),
            planes_per_die=self.integer("hbf-planes-per-die"),
            blocks_per_plane=self.integer("hbf-blocks-per-plane"),
            pages_per_block=self.integer("hbf-pages-per-block"),
            page_size_bytes=self.integer("hbf-page-size"),
            mapping_entries_per_page=int(self.values.get("hbf-mapping-entries-per-page", "512")),
        )


class SimulationSession:
    """One persistent, digest-checked HBFSim memory-system execution plane."""

    def __init__(
        self,
        *,
        simulator_path: Path,
        system_config: ResolvedSystemConfig,
        enable_hbm: bool,
        enable_hbf: bool,
        enable_external: bool = False,
        hbm_capacity_bytes: int | None = None,
        static_hbf_blocks_per_plane: int = 0,
        published_hbf_blocks_per_plane: int = 0,
        initial_hbf_logical_first_lpn: int = 0,
        initial_hbf_logical_pages: int = 0,
        initial_hbf_persistent_image: Path | None = None,
        hbf_physical_heatmap: Path | None = None,
        hbf_physical_heatmap_bins: int = 0,
        read_timeout_s: float | None = None,
    ) -> None:
        self._simulator_artifact = _artifact(
            simulator_path, "HBFSim executable"
        )
        if read_timeout_s is not None and (
            isinstance(read_timeout_s, bool)
            or not isinstance(read_timeout_s, (int, float))
            or not math.isfinite(read_timeout_s)
            or read_timeout_s <= 0
        ):
            raise SimulationSessionError(
                "read timeout must be a positive number of seconds or None"
            )
        self._read_timeout_s = (
            None if read_timeout_s is None else float(read_timeout_s)
        )
        heatmap_bins = _nonnegative_integer(
            hbf_physical_heatmap_bins, "HBF physical heatmap bins"
        )
        if (hbf_physical_heatmap is None) != (heatmap_bins == 0):
            raise SimulationSessionError(
                "hbf_physical_heatmap and a positive hbf_physical_heatmap_bins "
                "must be given together"
            )
        if hbf_physical_heatmap is not None and not enable_hbf:
            raise SimulationSessionError(
                "the HBF physical heatmap stream requires an enabled HBF tier"
            )
        self._simulator_path = Path(self._simulator_artifact["path"])
        if not os.access(self._simulator_path, os.X_OK):
            raise SimulationSessionError(
                f"HBFSim executable is not executable: {self._simulator_path}"
            )
        if not enable_hbm and not enable_hbf and not enable_external:
            raise SimulationSessionError("simulation session must enable a memory tier")
        configured_hbm_capacity = system_config.hbm_capacity_bytes
        if enable_hbm:
            capacity = (
                configured_hbm_capacity
                if hbm_capacity_bytes is None
                else _positive_integer(hbm_capacity_bytes, "HBM capacity")
            )
        else:
            if hbm_capacity_bytes not in (None, 0):
                raise SimulationSessionError(
                    "a disabled HBM tier cannot have an effective capacity"
                )
            capacity = 0
        if (
            isinstance(static_hbf_blocks_per_plane, bool)
            or not isinstance(static_hbf_blocks_per_plane, int)
            or static_hbf_blocks_per_plane < 0
        ):
            raise SimulationSessionError(
                "static HBF blocks per plane must be an integer >= 0"
            )
        if not enable_hbf and static_hbf_blocks_per_plane:
            raise SimulationSessionError(
                "static HBF placement requires an enabled HBF tier"
            )
        if enable_hbf and (
            static_hbf_blocks_per_plane
            > system_config.hbf_geometry.blocks_per_plane
        ):
            raise SimulationSessionError(
                "static HBF placement exceeds the configured geometry"
            )
        published_blocks = _nonnegative_integer(
            published_hbf_blocks_per_plane,
            "published HBF blocks per plane",
        )
        if not enable_hbf and published_blocks:
            raise SimulationSessionError(
                "published HBF placement requires an enabled HBF tier"
            )
        if enable_hbf and (
            static_hbf_blocks_per_plane + published_blocks
            > system_config.hbf_geometry.blocks_per_plane
        ):
            raise SimulationSessionError(
                "static and published HBF placements exceed the configured geometry"
            )
        first_lpn = _nonnegative_integer(
            initial_hbf_logical_first_lpn,
            "initial HBF logical first LPN",
        )
        initial_pages = _nonnegative_integer(
            initial_hbf_logical_pages,
            "initial HBF logical pages",
        )
        if not enable_hbf and initial_pages:
            raise SimulationSessionError(
                "initial HBF logical image requires an enabled HBF tier"
            )
        if initial_pages == 0 and first_lpn != 0:
            raise SimulationSessionError(
                "initial HBF first LPN requires a non-empty image"
            )
        persistent_artifact = (
            None
            if initial_hbf_persistent_image is None
            else _artifact(
                initial_hbf_persistent_image,
                "initial HBF persistent image",
            )
        )
        if persistent_artifact is not None and not enable_hbf:
            raise SimulationSessionError(
                "initial HBF persistent image requires an enabled HBF tier"
            )
        if persistent_artifact is not None and (
            static_hbf_blocks_per_plane or first_lpn or initial_pages
        ):
            raise SimulationSessionError(
                "initial HBF persistent image is mutually exclusive with "
                "static or dense initial placement"
            )
        if enable_hbf and initial_pages:
            geometry = system_config.hbf_geometry
            end_bytes = (first_lpn + initial_pages) * geometry.page_size_bytes
            if end_bytes > 2**63:
                raise SimulationSessionError(
                    "initial HBF logical image exceeds the user address namespace"
                )
            total_pages = (
                geometry.planes
                * geometry.blocks_per_plane
                * geometry.pages_per_block
            )
            static_pages = (
                geometry.planes
                * static_hbf_blocks_per_plane
                * geometry.pages_per_block
            )
            published_pages = (
                geometry.planes
                * published_blocks
                * geometry.pages_per_block
            )
            minimum_mapping_pages = (
                0 if system_config.hbf_mapping_mode == "direct" else
                hbf_dense_mapping_pages(first_lpn, initial_pages, geometry)
            )
            if (
                static_pages
                + published_pages
                + initial_pages
                + minimum_mapping_pages
                > total_pages
            ):
                raise SimulationSessionError(
                    "initial HBF data, mapping, static, and published extents cannot fit "
                    "the configured raw geometry"
                )
        self._system_config = system_config
        self._enable_hbm = bool(enable_hbm)
        self._enable_hbf = bool(enable_hbf)
        self._enable_external = bool(enable_external)
        self._external_backing = (
            dict(system_config.external_backing_identity)
            if enable_external
            else None
        )
        self._hbm_capacity_bytes = capacity
        self._configured_hbm_capacity_bytes = configured_hbm_capacity
        self._hbf_mapping_mode = system_config.hbf_mapping_mode
        self._hbf_ctrl_dram_bytes = system_config.hbf_ctrl_dram_bytes
        self._hbf_ctrl_dram_capacity_denominator = (
            system_config.values.get(
                "hbf-ctrl-dram-capacity-denominator"
            )
        )
        self._static_hbf_blocks_per_plane = static_hbf_blocks_per_plane
        self._published_hbf_blocks_per_plane = published_blocks
        self._initial_hbf_logical_first_lpn = first_lpn
        self._initial_hbf_logical_pages = initial_pages
        self._initial_hbf_persistent_artifact = persistent_artifact
        command = [str(self._simulator_path)]
        for path in system_config.paths:
            command.extend(("--system-config", str(path)))
        command.extend(
            (
                "--enable-hbm",
                "true" if enable_hbm else "false",
                "--enable-hbf",
                "true" if enable_hbf else "false",
                "--enable-external",
                "true" if enable_external else "false",
                "--static-hbf-blocks-per-plane",
                str(static_hbf_blocks_per_plane),
                "--published-hbf-blocks-per-plane",
                str(published_blocks),
                "--initial-hbf-logical-first-lpn",
                str(first_lpn),
                "--initial-hbf-logical-pages",
                str(initial_pages),
            )
        )
        if enable_hbm:
            command.extend(("--hbm-capacity-bytes", str(capacity)))
        if persistent_artifact is not None:
            command.extend((
                "--initial-hbf-persistent-image",
                str(persistent_artifact["path"]),
            ))
        if hbf_physical_heatmap is not None:
            command.extend((
                "--hbf-physical-heatmap",
                str(Path(hbf_physical_heatmap).resolve()),
                "--hbf-physical-heatmap-bins",
                str(heatmap_bins),
            ))
        self._hbf_physical_heatmap = (
            None
            if hbf_physical_heatmap is None
            else {
                "path": str(Path(hbf_physical_heatmap).resolve()),
                "bins": heatmap_bins,
            }
        )
        self._stderr: TextIO = tempfile.TemporaryFile(
            mode="w+t", encoding="utf-8"
        )
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
        except OSError as error:
            self._stderr.close()
            raise SimulationSessionError(
                f"cannot start HBFSim simulation session: {error}"
            ) from error
        self._closed = False
        self._next_sequence = 0
        self._next_checkpoint_sequence = 0
        # Blocking frontier (origin of the next batch) and the completion of
        # every issued transaction, which may run past it.
        self._last_finish_ns = 0.0
        self._issued_work_frontier_ns = 0.0
        self._next_run_batch_id = 0
        self._stop_receipt: dict[str, Any] | None = None
        self._checkpoint_ids: set[str] = set()
        self._wear_snapshot_ids: set[str] = set()
        self._checkpoint_receipts: list[dict[str, Any]] = []
        self._expected_latency_transactions = {
            target: {"read": 0, "write": 0} for target in TRANSACTION_TARGETS
        }
        try:
            ready = self._read_response("simulation-session ready receipt")
            geometry = system_config.hbf_geometry
            raw_capacity_pages = (
                geometry.planes
                * geometry.blocks_per_plane
                * geometry.pages_per_block
                if enable_hbf
                else 0
            )
            static_pages = (
                geometry.planes
                * static_hbf_blocks_per_plane
                * geometry.pages_per_block
                if enable_hbf
                else 0
            )
            mapping_pages = (
                hbf_dense_mapping_pages(first_lpn, initial_pages, geometry)
                if enable_hbf and system_config.hbf_mapping_mode != "direct"
                else 0
            )
            expected_initial_image = {
                "mode": (
                    "preloaded_mutable_dense" if initial_pages else "none"
                ),
                "first_lpn": self._initial_hbf_logical_first_lpn,
                "pages": self._initial_hbf_logical_pages,
                "physical_data_pages": self._initial_hbf_logical_pages,
                "physical_mapping_pages": mapping_pages,
                "static_pages": static_pages,
                "free_pages_after_setup": (
                    raw_capacity_pages
                    - static_pages
                    - self._initial_hbf_logical_pages
                    - mapping_pages
                ),
                "raw_capacity_pages": raw_capacity_pages,
            }
            persistent_ready = ready.get("initial_hbf_persistent_image")
            if persistent_artifact is None:
                persistent_ready_valid = persistent_ready is None
            else:
                persistent_ready_valid = (
                    isinstance(persistent_ready, Mapping)
                    and persistent_ready.get("schema")
                    == HBF_PERSISTENT_IMAGE_SCHEMA
                    and persistent_ready.get("path")
                    == persistent_artifact["path"]
                    and persistent_ready.get("bytes")
                    == persistent_artifact["bytes"]
                    and persistent_ready.get("sha256")
                    == persistent_artifact["sha256"]
                    and all(
                        isinstance(persistent_ready.get(field), int)
                        and not isinstance(persistent_ready.get(field), bool)
                        and persistent_ready[field] >= 0
                        for field in (
                            "logical_data_pages",
                            "mapping_pages",
                            "compact_logical_data_pages",
                            "compact_mapping_pages",
                            "static_pages",
                            "raw_pages",
                            "free_pages_after_restore",
                            "raw_capacity_pages",
                            "block_erase_count_sum",
                        )
                    )
                    and persistent_ready["raw_capacity_pages"]
                    == raw_capacity_pages
                    and persistent_ready["raw_pages"]
                    == (
                        geometry.planes
                        * published_blocks
                        * geometry.pages_per_block
                    )
                    and persistent_ready.get("encoding")
                    in {"materialized_v2", "compact_v2"}
                    and (
                        persistent_ready["encoding"] == "compact_v2"
                    )
                    == (
                        persistent_ready["compact_logical_data_pages"] > 0
                    )
                )
            engine_source = ready.get("source")
            dependency_window = ready.get("dependency_window_batches")
            direct_lane = ready.get("hbf_external_direct_link")
            if (
                ready.get("schema") != SESSION_SCHEMA
                or ready.get("result") != "ready"
                or ready.get("protocol") != PROTOCOL
                or ready.get("time_basis") != TIME_BASIS
                or not isinstance(engine_source, Mapping)
                or set(engine_source) != SOURCE_PROVENANCE_FIELDS
                or not isinstance(engine_source.get("git_commit"), str)
                or not isinstance(engine_source.get("git_dirty"), bool)
                or not isinstance(engine_source.get("tree_hash"), str)
                or not isinstance(engine_source.get("source_sha256"), str)
                or len(engine_source["source_sha256"]) != 64
                or any(character not in "0123456789abcdef"
                       for character in engine_source["source_sha256"])
                or engine_source.get("provenance_source") != "build-time"
                or isinstance(dependency_window, bool)
                or not isinstance(dependency_window, int)
                or dependency_window < 1
                or not (direct_lane is None or isinstance(direct_lane, Mapping))
                or ready.get("nonterminal_hbf_checkpoint") is not True
                or ready.get("hbf_wear_snapshot") is not True
                or ready.get("enable_hbm") is not self._enable_hbm
                or ready.get("enable_hbf") is not self._enable_hbf
                or ready.get("enable_external") is not self._enable_external
                or ready.get("hbf_mapping_mode") != self._hbf_mapping_mode
                or ready.get("hbf_ctrl_dram_bytes")
                != self._hbf_ctrl_dram_bytes
                or ready.get("hbf_ctrl_dram_capacity_denominator")
                != (
                    0
                    if self._hbf_ctrl_dram_capacity_denominator is None
                    else int(self._hbf_ctrl_dram_capacity_denominator)
                )
                or ready.get("static_hbf_blocks_per_plane")
                != self._static_hbf_blocks_per_plane
                or ready.get("published_hbf_blocks_per_plane")
                != self._published_hbf_blocks_per_plane
                or ready.get("initial_hbf_logical_image")
                != expected_initial_image
                or not persistent_ready_valid
                or ready.get("external_backing") != self._external_backing
            ):
                raise SimulationSessionError(
                    "HBFSim simulation-session ready receipt does not match the request"
                )
            self._initial_hbf_setup = expected_initial_image
            self._initial_hbf_persistent_setup = deepcopy(persistent_ready)
            self._engine_source = dict(engine_source)
            self._dependency_window_batches = int(dependency_window)
            self._hbf_external_direct_link = (
                None if direct_lane is None else deepcopy(dict(direct_lane))
            )
        except BaseException:
            self._force_close()
            raise

    def _stderr_text(self) -> str:
        try:
            self._stderr.flush()
            self._stderr.seek(0)
            return self._stderr.read().strip()
        except (OSError, ValueError):
            return ""

    @property
    def completed_frontier_ns(self) -> float:
        """Blocking completion frontier: the origin of the next batch."""

        return self._last_finish_ns

    @property
    def issued_work_frontier_ns(self) -> float:
        """Completion of every issued transaction, detached work included."""

        return self._issued_work_frontier_ns

    @property
    def dependency_window_batches(self) -> int:
        """Completed batches whose ids stay resolvable without ``retain``."""

        return self._dependency_window_batches

    @property
    def engine_source(self) -> dict[str, Any]:
        """Source-tree provenance reported by the engine at start."""

        return dict(self._engine_source)

    def _wait_for_output(self, description: str) -> None:
        if self._read_timeout_s is None:
            return
        assert self._process.stdout is not None
        deadline = time.monotonic() + self._read_timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._force_close()
                raise SimulationSessionError(
                    f"HBFSim produced no {description} within "
                    f"{self._read_timeout_s:g} s"
                )
            try:
                readable, _, _ = select.select(
                    [self._process.stdout], [], [], remaining
                )
            except (OSError, ValueError) as error:
                raise SimulationSessionError(
                    f"cannot wait for {description}: {error}"
                ) from error
            if readable:
                return

    def _read_response(self, description: str) -> dict[str, Any]:
        assert self._process.stdout is not None
        self._wait_for_output(description)
        line = self._process.stdout.readline()
        if not line:
            code = self._process.poll()
            detail = self._stderr_text()
            suffix = f": {detail}" if detail else ""
            raise SimulationSessionError(
                f"HBFSim ended before {description} (exit={code}){suffix}"
            )
        if not line.endswith("\n"):
            raise SimulationSessionError(f"{description} is not newline terminated")
        return _json_object(line, description)

    def submit(self, batch: TransactionBatch) -> dict[str, Any]:
        if self._closed:
            raise SimulationSessionError("cannot submit to a closed simulation session")
        if self._initial_hbf_persistent_artifact is not None and any(
            transaction.target == "HBF_PHYSICAL" and transaction.op == "W"
            for transaction in batch.transactions
        ):
            raise SimulationSessionError(
                "restored published extent is read-only"
            )
        external_transactions = tuple(
            transaction
            for transaction in batch.transactions
            if transaction.target == "EXTERNAL"
        )
        assert self._process.stdin is not None
        try:
            self._process.stdin.write(batch.begin_line() + "\n")
            for chunk in batch.protocol_payload_chunks():
                self._process.stdin.write(chunk)
            self._process.stdin.write(f"END {batch.batch_id}\n")
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            detail = self._stderr_text()
            suffix = f": {detail}" if detail else ""
            raise SimulationSessionError(
                f"cannot submit transaction batch {batch.batch_id}{suffix}"
            ) from error
        completion = self._read_response(
            f"transaction batch {batch.batch_id} completion"
        )
        if (
            completion.get("schema") == COMPLETION_SCHEMA
            and completion.get("result") == "error"
            and completion.get("batch_id") == str(batch.batch_id)
        ):
            # The engine rejected the batch before touching device state;
            # the session, its sequence, and its frontier are unchanged.
            message = completion.get("message")
            raise SimulationSessionError(
                f"HBFSim rejected batch {batch.batch_id}: "
                f"{message if isinstance(message, str) else 'unknown input error'}"
            )
        expected_census = {
            target: {"transactions": 0, "bytes": 0}
            for target in TRANSACTION_TARGETS
        }
        for transaction in batch.transactions:
            expected_census[transaction.target]["transactions"] += 1
            expected_census[transaction.target]["bytes"] += transaction.bytes
        memory_transactions = sum(
            transaction.target != "BARRIER"
            for transaction in batch.transactions
        )
        barriers = len(batch.transactions) - memory_transactions
        dependency_edges = sum(
            len(transaction.dependencies) for transaction in batch.transactions
        )
        expected_scalars = {
            "sequence": self._next_sequence,
            "transactions": len(batch.transactions),
            "memory_transactions": memory_transactions,
            "barriers": barriers,
            "dependency_edges": dependency_edges,
            "frontier_transactions": batch.frontier_transactions,
            "transaction_bytes": sum(
                transaction.bytes for transaction in batch.transactions
            ),
        }
        if (
            completion.get("schema") != COMPLETION_SCHEMA
            or completion.get("result") != "pass"
            or completion.get("batch_id") != str(batch.batch_id)
            or completion.get("logical_trace_sha256")
            != batch.logical_trace_sha256
            or completion.get("transaction_trace_sha256")
            != batch.transaction_trace_sha256
            or completion.get("time_basis") != TIME_BASIS
            or completion.get("by_target") != expected_census
            or any(completion.get(key) != value for key, value in expected_scalars.items())
        ):
            raise SimulationSessionError(
                f"HBFSim completion receipt diverged for batch {batch.batch_id}"
            )
        expected_latency_transactions = {
            target: {"read": 0, "write": 0} for target in TRANSACTION_TARGETS
        }
        for transaction in batch.transactions:
            if transaction.target == "BARRIER":
                continue
            operation = "read" if transaction.op == "R" else "write"
            expected_latency_transactions[transaction.target][operation] += 1
        _validate_latency_matrix(
            completion.get("transaction_latency_by_target"),
            description=f"batch {batch.batch_id} transaction latency",
            expected_transactions=expected_latency_transactions,
        )
        _validate_device_accounting(
            completion.get("device_delta"),
            enable_hbm=self._enable_hbm,
            enable_hbf=self._enable_hbf,
            enable_external=self._enable_external,
            external_kind=(
                None
                if self._external_backing is None
                else str(self._external_backing["kind"])
            ),
            description=f"batch {batch.batch_id} device delta",
        )
        if self._enable_external:
            external_delta = completion["device_delta"]["external"]
            assert isinstance(external_delta, Mapping)
            assert self._external_backing is not None
            external_page_size = int(self._external_backing["page_size_bytes"])
            external_segment_size = int(
                self._external_backing["request_segment_bytes"]
            )
            external_pages = sum(
                (
                    transaction.addr % external_page_size
                    + transaction.bytes
                    + external_page_size
                    - 1
                )
                // external_page_size
                for transaction in external_transactions
            )
            external_segments = sum(
                (
                    transaction.addr % external_segment_size
                    + transaction.bytes
                    + external_segment_size
                    - 1
                )
                // external_segment_size
                for transaction in external_transactions
            )
            if (
                external_delta.get("page_run_requests")
                != len(external_transactions)
                or external_delta.get("page_run_segments") != external_segments
                or external_delta.get("page_run_pages") != external_pages
            ):
                raise SimulationSessionError(
                    f"HBFSim external page-run receipt does not conserve "
                    f"external traffic for batch {batch.batch_id}"
                )
        hbm_engine = completion.get("hbm_engine")
        hbm_engine_fields = ("requests", "bursts")
        if not isinstance(hbm_engine, Mapping) or set(hbm_engine) != set(
            hbm_engine_fields
        ):
            raise SimulationSessionError(
                f"HBFSim completion has no exact HBM-engine receipt for "
                f"batch {batch.batch_id}"
            )
        normalized_hbm: dict[str, int] = {}
        for key in hbm_engine_fields:
            value = hbm_engine.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SimulationSessionError(
                    f"HBFSim completion HBM-engine {key} is invalid"
                )
            normalized_hbm[key] = value
        hbm_transactions = tuple(
            transaction
            for transaction in batch.transactions
            if transaction.target == "HBM"
        )
        burst_bytes = self._system_config.hbm_burst_bytes
        hbm_bursts = sum(
            (transaction.addr % burst_bytes + transaction.bytes +
             burst_bytes - 1) // burst_bytes
            for transaction in hbm_transactions
        )
        if (
            normalized_hbm["requests"] != len(hbm_transactions)
            or normalized_hbm["bursts"] != hbm_bursts
        ):
            raise SimulationSessionError(
                f"HBFSim HBM-engine receipt does not conserve HBM "
                f"traffic for batch {batch.batch_id}"
            )
        read_engine = completion.get("hbf_read_engine")
        read_engine_fields = (
            "page_run_requests",
            "page_run_pages",
            "page_run_physical_requests",
            "page_run_static_requests",
            "page_run_logical_requests",
            "streaming_read_buffer_bypass_pages",
            "scalar_read_requests",
            "scalar_read_pages",
        )
        if not isinstance(read_engine, Mapping) or set(read_engine) != set(
            read_engine_fields
        ):
            raise SimulationSessionError(
                f"HBFSim completion has no exact read-engine receipt for "
                f"batch {batch.batch_id}"
            )
        normalized_engine: dict[str, int] = {}
        for key in read_engine_fields:
            value = read_engine.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SimulationSessionError(
                    f"HBFSim completion read-engine {key} is invalid"
                )
            normalized_engine[key] = value
        hbf_reads = tuple(
            transaction
            for transaction in batch.transactions
            if transaction.target in {
                "HBF_LOGICAL",
                "HBF_STATIC",
                "HBF_PHYSICAL",
            }
            and transaction.op == "R"
        )
        page_size = self._system_config.hbf_geometry.page_size_bytes
        hbf_read_pages = sum(
            (transaction.addr % page_size + transaction.bytes + page_size - 1)
            // page_size
            for transaction in hbf_reads
        )
        if (
            normalized_engine["page_run_requests"]
            + normalized_engine["scalar_read_requests"]
            != len(hbf_reads)
            or normalized_engine["page_run_pages"]
            + normalized_engine["scalar_read_pages"]
            != hbf_read_pages
            or normalized_engine["page_run_physical_requests"]
            + normalized_engine["page_run_static_requests"]
            + normalized_engine["page_run_logical_requests"]
            != normalized_engine["page_run_requests"]
            or normalized_engine["streaming_read_buffer_bypass_pages"]
            > hbf_read_pages
        ):
            raise SimulationSessionError(
                f"HBFSim read-engine receipt does not conserve HBF "
                f"reads for batch {batch.batch_id}"
            )
        numeric: dict[str, float] = {}
        for key in (
            "batch_origin_ns",
            "first_issue_ns",
            "finish_ns",
            "blocking_finish_ns",
            "elapsed_ns",
            "total_elapsed_ns",
        ):
            value = completion.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SimulationSessionError(
                    f"HBFSim completion {key} is not numeric"
                )
            normalized = float(value)
            if not math.isfinite(normalized) or normalized < 0:
                raise SimulationSessionError(
                    f"HBFSim completion {key} is invalid"
                )
            numeric[key] = normalized
        if (
            not math.isclose(
                numeric["batch_origin_ns"],
                self._last_finish_ns,
                rel_tol=1e-12,
                abs_tol=1e-9,
            )
            or numeric["first_issue_ns"] < numeric["batch_origin_ns"]
            or numeric["finish_ns"] < numeric["first_issue_ns"]
            or numeric["blocking_finish_ns"] < numeric["first_issue_ns"]
            or numeric["blocking_finish_ns"] > numeric["finish_ns"] + 1e-6
            or not math.isclose(
                numeric["elapsed_ns"],
                numeric["blocking_finish_ns"] - numeric["batch_origin_ns"],
                rel_tol=1e-12,
                abs_tol=1e-6,
            )
            or not math.isclose(
                numeric["total_elapsed_ns"],
                numeric["finish_ns"] - numeric["batch_origin_ns"],
                rel_tol=1e-12,
                abs_tol=1e-6,
            )
        ):
            raise SimulationSessionError(
                f"HBFSim completion clock diverged for batch {batch.batch_id}"
            )
        latency_matrix = completion.get("transaction_latency_by_target")
        assert isinstance(latency_matrix, Mapping)
        if batch.completions:
            _validate_transaction_completions(
                completion.get("transaction_completions"),
                completion.get("transaction_completions_digest"),
                batch=batch,
                latency_matrix=latency_matrix,  # type: ignore[arg-type]
                batch_origin_ns=numeric["batch_origin_ns"],
                batch_finish_ns=numeric["finish_ns"],
            )
        elif (
            completion.get("transaction_completions") is not None
            or completion.get("transaction_completions_digest") is not None
        ):
            raise SimulationSessionError(
                f"HBFSim exported completions for batch {batch.batch_id} "
                "although none were requested"
            )
        self._last_finish_ns = numeric["blocking_finish_ns"]
        self._issued_work_frontier_ns = max(
            self._issued_work_frontier_ns, numeric["finish_ns"]
        )
        self._next_sequence += 1
        for target, operations in expected_latency_transactions.items():
            for operation, count in operations.items():
                self._expected_latency_transactions[target][operation] += count
        return deepcopy(completion)

    def run(
        self,
        transactions: Iterable[Transaction],
        *,
        frontier: Iterable[str] | None = None,
        retain: Iterable[str] | None = None,
        completions: bool = True,
    ) -> BatchResult:
        """Submit one auto-numbered batch and return a typed result.

        ``frontier`` names the transactions whose completion gates the next
        batch (``None`` = all, an empty iterable = none); ``retain`` is the
        complete set of earlier ids later batches may still depend on
        (``None`` leaves the engine's retained set unchanged, an empty
        iterable clears it).
        """

        batch = TransactionBatch(
            batch_id=self._next_run_batch_id,
            transactions=tuple(transactions),
            frontier=None if frontier is None else tuple(frontier),
            retain=None if retain is None else tuple(retain),
            completions=completions,
        )
        receipt = self.submit(batch)
        self._next_run_batch_id += 1
        records = tuple(
            CompletionRecord(
                id=str(row["id"]),
                arrival_ns=float(row["arrival_ns"]),
                start_ns=float(row["start_ns"]),
                finish_ns=float(row["finish_ns"]),
                logical_bytes=int(row["logical_bytes"]),
                physical_bytes=int(row["physical_bytes"]),
            )
            for row in (receipt.get("transaction_completions") or ())
        )
        return BatchResult(
            batch_id=str(receipt["batch_id"]),
            sequence=int(receipt["sequence"]),
            batch_origin_ns=float(receipt["batch_origin_ns"]),
            first_issue_ns=float(receipt["first_issue_ns"]),
            blocking_finish_ns=float(receipt["blocking_finish_ns"]),
            finish_ns=float(receipt["finish_ns"]),
            elapsed_ns=float(receipt["elapsed_ns"]),
            total_elapsed_ns=float(receipt["total_elapsed_ns"]),
            frontier_transactions=int(receipt["frontier_transactions"]),
            completions=records,
            receipt=receipt,
        )

    def checkpoint(self, checkpoint_id: str) -> dict[str, Any]:
        """Persist pending HBF state and causally continue the same session."""

        return self._checkpoint(checkpoint_id, image_output=None)

    def hbf_wear_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        """Read exact per-writable-block P/E counts without changing state."""

        if self._closed:
            raise SimulationSessionError(
                "cannot snapshot wear from a closed simulation session"
            )
        if not self._enable_hbf:
            raise SimulationSessionError(
                "simulation wear snapshot requires an enabled HBF tier"
            )
        try:
            normalized_id = require_safe_identifier(
                snapshot_id, "simulation wear snapshot id"
            )
        except TransactionProtocolError as error:
            raise SimulationSessionError(str(error)) from error
        if normalized_id in self._wear_snapshot_ids:
            raise SimulationSessionError(
                f"simulation wear snapshot id is duplicated: {normalized_id}"
            )
        assert self._process.stdin is not None
        try:
            self._process.stdin.write(f"WEAR_SNAPSHOT {normalized_id}\n")
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            detail = self._stderr_text()
            suffix = f": {detail}" if detail else ""
            raise SimulationSessionError(
                f"cannot request wear snapshot {normalized_id}{suffix}"
            ) from error
        receipt = self._read_response(f"wear snapshot {normalized_id}")
        counts = receipt.get("block_erase_counts")
        geometry = self._system_config.hbf_geometry
        expected_blocks = (
            geometry.planes
            * (geometry.blocks_per_plane - self._static_hbf_blocks_per_plane)
        )
        if (
            receipt.get("schema") != HBF_WEAR_SNAPSHOT_SCHEMA
            or receipt.get("result") != "pass"
            or receipt.get("snapshot_id") != normalized_id
            or receipt.get("writable_blocks") != expected_blocks
            or not isinstance(counts, list)
            or len(counts) != expected_blocks
            or any(
                isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
                or count > 2**32 - 1
                for count in counts
            )
            or receipt.get("block_erase_count_sum") != sum(counts)
        ):
            raise SimulationSessionError(
                f"HBFSim wear snapshot diverged for {normalized_id}"
            )
        frontier = _finite_nonnegative(
            receipt.get("completed_frontier_ns"),
            f"wear snapshot {normalized_id} completed frontier",
        )
        if not math.isclose(
            frontier, self._last_finish_ns, rel_tol=1e-12, abs_tol=1e-6
        ):
            raise SimulationSessionError(
                f"HBFSim wear snapshot frontier diverged for {normalized_id}"
            )
        self._wear_snapshot_ids.add(normalized_id)
        return deepcopy(receipt)

    def checkpoint_image(
        self,
        checkpoint_id: str,
        image_output: Path,
    ) -> dict[str, Any]:
        """Checkpoint and atomically export exact quiescent HBF media state."""

        requested = Path(image_output)
        if requested.is_symlink() or requested.exists():
            raise SimulationSessionError(
                f"persistent image output already exists: {requested}"
            )
        resolved = requested.resolve()
        if not resolved.parent.is_dir():
            raise SimulationSessionError(
                "persistent image output parent must be a directory"
            )
        return self._checkpoint(checkpoint_id, image_output=resolved)

    def _checkpoint(
        self,
        checkpoint_id: str,
        *,
        image_output: Path | None,
    ) -> dict[str, Any]:

        if self._closed:
            raise SimulationSessionError(
                "cannot checkpoint a closed simulation session"
            )
        if not self._enable_hbf:
            raise SimulationSessionError(
                "simulation checkpoint requires an enabled HBF tier"
            )
        try:
            normalized_id = require_safe_identifier(
                checkpoint_id, "simulation checkpoint id"
            )
        except TransactionProtocolError as error:
            raise SimulationSessionError(str(error)) from error
        if normalized_id in self._checkpoint_ids:
            raise SimulationSessionError(
                f"simulation checkpoint id is duplicated: {normalized_id}"
            )
        assert self._process.stdin is not None
        try:
            command = (
                f"CHECKPOINT {normalized_id}\n"
                if image_output is None
                else (
                    f"CHECKPOINT_IMAGE {normalized_id} "
                    f"{str(image_output).encode('utf-8').hex()}\n"
                )
            )
            self._process.stdin.write(command)
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            detail = self._stderr_text()
            suffix = f": {detail}" if detail else ""
            raise SimulationSessionError(
                f"cannot submit checkpoint {normalized_id}{suffix}"
            ) from error
        completion = self._read_response(
            f"checkpoint {normalized_id} completion"
        )
        if (
            completion.get("schema") != CHECKPOINT_SCHEMA
            or completion.get("result") != "pass"
            or completion.get("checkpoint_id") != normalized_id
            or completion.get("sequence") != self._next_checkpoint_sequence
            or completion.get("has_hbf") is not True
            or completion.get("causal_for_subsequent_batches") is not True
            or completion.get("completion_note")
            not in {"drained-pending-hbf-state", "no-pending-hbf-state"}
        ):
            raise SimulationSessionError(
                f"HBFSim checkpoint receipt diverged for {normalized_id}"
            )
        arrival = _finite_nonnegative(
            completion.get("arrival_frontier_ns"),
            f"checkpoint {normalized_id} arrival frontier",
        )
        finish = _finite_nonnegative(
            completion.get("finish_ns"),
            f"checkpoint {normalized_id} finish",
        )
        elapsed = _finite_nonnegative(
            completion.get("elapsed_ns"),
            f"checkpoint {normalized_id} elapsed time",
        )
        # A checkpoint persists every issued transaction, so it starts once
        # all of them (detached work included) have completed.
        if (
            not math.isclose(
                arrival,
                self._issued_work_frontier_ns,
                rel_tol=1e-12,
                abs_tol=1e-6,
            )
            or finish < arrival
            or not math.isclose(
                elapsed, finish - arrival, rel_tol=1e-12, abs_tol=1e-6
            )
        ):
            raise SimulationSessionError(
                f"HBFSim checkpoint timing diverged for {normalized_id}"
            )
        physical_bytes = _nonnegative_integer(
            completion.get("physical_bytes"),
            f"checkpoint {normalized_id} physical bytes",
        )
        device_delta = completion.get("device_delta")
        _validate_device_accounting(
            device_delta,
            enable_hbm=self._enable_hbm,
            enable_hbf=True,
            enable_external=self._enable_external,
            external_kind=(
                None
                if self._external_backing is None
                else str(self._external_backing["kind"])
            ),
            description=f"checkpoint {normalized_id} device delta",
        )
        assert isinstance(device_delta, Mapping)
        hbf = device_delta.get("hbf")
        assert isinstance(hbf, Mapping)
        if physical_bytes != (
            _nonnegative_integer(
                hbf.get("physical_read_bytes"),
                f"checkpoint {normalized_id} HBF physical reads",
            )
            + _nonnegative_integer(
                hbf.get("physical_write_bytes"),
                f"checkpoint {normalized_id} HBF physical writes",
            )
        ):
            raise SimulationSessionError(
                f"HBFSim checkpoint physical bytes diverged for {normalized_id}"
            )
        _validate_quiescence(
            completion.get("quiescence"),
            f"checkpoint {normalized_id} quiescence",
        )
        image_receipt = completion.get("persistent_image")
        if image_output is None:
            if image_receipt is not None:
                raise SimulationSessionError(
                    f"checkpoint {normalized_id} exported an unrequested image"
                )
        else:
            artifact = _artifact(
                image_output,
                f"checkpoint {normalized_id} persistent image",
            )
            if (
                not isinstance(image_receipt, Mapping)
                or image_receipt.get("schema")
                != HBF_PERSISTENT_IMAGE_SCHEMA
                or image_receipt.get("path") != artifact["path"]
                or image_receipt.get("bytes") != artifact["bytes"]
                or image_receipt.get("sha256") != artifact["sha256"]
            ):
                raise SimulationSessionError(
                    f"checkpoint {normalized_id} image artifact diverged"
                )
        self._last_finish_ns = finish
        self._issued_work_frontier_ns = finish
        self._checkpoint_ids.add(normalized_id)
        self._next_checkpoint_sequence += 1
        receipt = deepcopy(completion)
        self._checkpoint_receipts.append(receipt)
        return deepcopy(receipt)

    def crash(self, crash_id: str) -> dict[str, Any]:
        """End at a command boundary without checkpointing or terminal drain."""

        if self._closed:
            raise SimulationSessionError(
                "cannot inject a crash into a closed simulation session"
            )
        try:
            normalized_id = require_safe_identifier(
                crash_id, "simulation crash id"
            )
        except TransactionProtocolError as error:
            raise SimulationSessionError(str(error)) from error

        assert self._process.stdin is not None
        try:
            self._process.stdin.write(f"CRASH {normalized_id}\n")
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            detail = self._stderr_text()
            suffix = f": {detail}" if detail else ""
            self._force_close()
            raise SimulationSessionError(
                f"cannot inject crash {normalized_id}{suffix}"
            ) from error

        try:
            completion = self._read_response(
                f"crash {normalized_id} completion"
            )
            if (
                completion.get("schema") != CRASH_SCHEMA
                or completion.get("result") != "crashed"
                or completion.get("crash_id") != normalized_id
                or completion.get("completed_batches") != self._next_sequence
                or completion.get("completed_checkpoints")
                != self._next_checkpoint_sequence
                or completion.get("injection_boundary")
                != "after_previous_completed_protocol_command"
                or completion.get("terminal_drain_performed") is not False
            ):
                raise SimulationSessionError(
                    f"HBFSim crash receipt diverged for {normalized_id}"
                )
            frontier = _finite_nonnegative(
                completion.get("completed_frontier_ns"),
                f"crash {normalized_id} completed frontier",
            )
            if not math.isclose(
                frontier,
                self._last_finish_ns,
                rel_tol=1e-12,
                abs_tol=1e-6,
            ):
                raise SimulationSessionError(
                    f"HBFSim crash frontier diverged for {normalized_id}"
                )
            _validate_device_accounting(
                completion.get("device_workload_totals"),
                enable_hbm=self._enable_hbm,
                enable_hbf=self._enable_hbf,
                enable_external=self._enable_external,
                external_kind=(
                    None
                    if self._external_backing is None
                    else str(self._external_backing["kind"])
                ),
                description=f"crash {normalized_id} device totals",
            )
            _normalized_quiescence(
                completion.get("quiescence_at_injection"),
                f"crash {normalized_id} quiescence",
            )
            try:
                return_code = self._process.wait(timeout=5)
            except subprocess.TimeoutExpired as error:
                raise SimulationSessionError(
                    f"HBFSim did not terminate after crash {normalized_id}"
                ) from error
            if return_code != 0:
                detail = self._stderr_text()
                suffix = f": {detail}" if detail else ""
                raise SimulationSessionError(
                    f"HBFSim crash process failed (exit={return_code}){suffix}"
                )
            self._stop_receipt = deepcopy(completion)
            return deepcopy(completion)
        finally:
            self._force_close()

    def source_receipt(self) -> dict[str, Any]:
        return {
            "protocol": PROTOCOL,
            "time_basis": TIME_BASIS,
            "dependency_window_batches": self._dependency_window_batches,
            "simulator_executable": deepcopy(self._simulator_artifact),
            # Verbatim engine provenance: commit, dirty flag, tree hash, and
            # whether git answered at run time or the build snapshot was used.
            "engine_source": deepcopy(self._engine_source),
            "system_configs": deepcopy(list(self._system_config.artifacts)),
            "execution_options": {
                "enable_hbm": self._enable_hbm,
                "enable_hbf": self._enable_hbf,
                "enable_external": self._enable_external,
                "external_backing": deepcopy(self._external_backing),
                "hbf_external_direct_link": deepcopy(
                    self._hbf_external_direct_link
                ),
                "hbf_physical_heatmap": deepcopy(self._hbf_physical_heatmap),
                "configured_hbm_capacity_bytes": (
                    self._configured_hbm_capacity_bytes
                ),
                "effective_hbm_capacity_bytes": self._hbm_capacity_bytes,
                "hbf_mapping_mode": self._hbf_mapping_mode,
                "hbf_ctrl_dram_bytes": self._hbf_ctrl_dram_bytes,
                "hbf_ctrl_dram_capacity_denominator": (
                    0
                    if self._hbf_ctrl_dram_capacity_denominator is None
                    else int(self._hbf_ctrl_dram_capacity_denominator)
                ),
                "static_hbf_blocks_per_plane": (
                    self._static_hbf_blocks_per_plane
                ),
                "published_hbf_blocks_per_plane": (
                    self._published_hbf_blocks_per_plane
                ),
                "initial_hbf_logical_image": {
                    **deepcopy(self._initial_hbf_setup),
                    "bytes": (
                        self._initial_hbf_logical_pages
                        * self._system_config.hbf_geometry.page_size_bytes
                    ),
                    "installation_accounting": "setup_excluded_from_serving",
                    "capacity_accounting": "data_and_mapping_pages_reserved",
                },
                "initial_hbf_persistent_image": deepcopy(
                    self._initial_hbf_persistent_setup
                ),
            },
            "lifecycle_checkpoints": deepcopy(self._checkpoint_receipts),
            "final_measurement": deepcopy(self._stop_receipt),
        }

    def _force_close(self) -> None:
        if getattr(self, "_closed", True):
            return
        self._closed = True
        process = self._process
        try:
            if process.stdin is not None:
                process.stdin.close()
        except OSError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if process.stdout is not None:
            process.stdout.close()
        self._stderr.close()

    def close(self) -> None:
        if self._closed:
            return
        error: BaseException | None = None
        try:
            if self._process.poll() is None:
                assert self._process.stdin is not None
                self._process.stdin.write("QUIT\n")
                self._process.stdin.flush()
                stopped = self._read_response("simulation-session stop receipt")
                if (
                    stopped.get("schema") != SESSION_SCHEMA
                    or stopped.get("result") != "stopped"
                    or stopped.get("completed_batches") != self._next_sequence
                    or stopped.get("completed_checkpoints")
                    != self._next_checkpoint_sequence
                ):
                    raise SimulationSessionError(
                        "HBFSim simulation-session stop receipt is malformed"
                    )
                stopped_frontier = _finite_nonnegative(
                    stopped.get("completed_frontier_ns"),
                    "simulation-session stop completed frontier",
                )
                if not math.isclose(
                    stopped_frontier,
                    self._last_finish_ns,
                    rel_tol=1e-12,
                    abs_tol=1e-6,
                ):
                    raise SimulationSessionError(
                        "HBFSim simulation-session stop frontier diverged"
                    )
                drained_frontier = _finite_nonnegative(
                    stopped.get("drained_frontier_ns"),
                    "simulation-session drained frontier",
                )
                if drained_frontier < stopped_frontier:
                    raise SimulationSessionError(
                        "HBFSim simulation-session drain preceded serving completion"
                    )
                _validate_latency_matrix(
                    stopped.get("transaction_latency_by_target"),
                    description="simulation-session cumulative transaction latency",
                    expected_transactions=self._expected_latency_transactions,
                )
                _validate_device_accounting(
                    stopped.get("device_workload_totals"),
                    enable_hbm=self._enable_hbm,
                    enable_hbf=self._enable_hbf,
                    enable_external=self._enable_external,
                    external_kind=(
                        None
                        if self._external_backing is None
                        else str(self._external_backing["kind"])
                    ),
                    description="simulation-session cumulative device accounting",
                )
                drain = stopped.get("end_of_session_drain")
                if not isinstance(drain, Mapping):
                    raise SimulationSessionError(
                        "HBFSim simulation-session drain receipt is missing"
                    )
                drain_serving = _finite_nonnegative(
                    drain.get("serving_frontier_ns"),
                    "simulation-session drain serving frontier",
                )
                drain_finish = _finite_nonnegative(
                    drain.get("finish_ns"),
                    "simulation-session drain finish",
                )
                drain_tail = _finite_nonnegative(
                    drain.get("tail_ns"),
                    "simulation-session drain tail",
                )
                if (
                    drain.get("has_hbf") is not self._enable_hbf
                    or drain.get("serving_timing_excludes_drain") is not True
                    or not math.isclose(
                        drain_serving,
                        self._issued_work_frontier_ns,
                        rel_tol=1e-12,
                        abs_tol=1e-6,
                    )
                    or drain_serving + 1e-6 < stopped_frontier
                    or not math.isclose(
                        drain_finish,
                        drained_frontier,
                        rel_tol=1e-12,
                        abs_tol=1e-6,
                    )
                    or not math.isclose(
                        drain_tail,
                        drain_finish - drain_serving,
                        rel_tol=1e-12,
                        abs_tol=1e-6,
                    )
                ):
                    raise SimulationSessionError(
                        "HBFSim simulation-session drain timing does not conserve"
                    )
                _nonnegative_integer(
                    drain.get("drain_physical_bytes"),
                    "simulation-session drain physical bytes",
                )
                _validate_device_accounting(
                    drain.get("device_delta"),
                    enable_hbm=self._enable_hbm,
                    enable_hbf=self._enable_hbf,
                    enable_external=self._enable_external,
                    external_kind=(
                        None
                        if self._external_backing is None
                        else str(self._external_backing["kind"])
                    ),
                    description="simulation-session drain device delta",
                )
                _validate_quiescence(
                    drain.get("quiescence"),
                    "simulation-session end-of-session quiescence",
                )
                if not isinstance(stopped.get("measurement_semantics"), Mapping):
                    raise SimulationSessionError(
                        "HBFSim simulation-session measurement semantics are missing"
                    )
                self._stop_receipt = deepcopy(stopped)
            elif self._process.returncode != 0:
                detail = self._stderr_text()
                suffix = f": {detail}" if detail else ""
                raise SimulationSessionError(
                    "HBFSim simulation session exited before graceful shutdown"
                    f" (exit={self._process.returncode}){suffix}"
                )
        except BaseException as caught:
            error = caught
        finally:
            self._force_close()
        if error is not None:
            raise error

    def __enter__(self) -> "SimulationSession":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        # Never chain a shutdown failure over the exception already in
        # flight: on an error path the engine is stopped without the
        # receipt-validating graceful close.
        if exc_type is None:
            self.close()
        else:
            self._force_close()

    def __del__(self) -> None:
        try:
            self._force_close()
        except Exception:  # noqa: BLE001 - finalizers must not raise
            pass
