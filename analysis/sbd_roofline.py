"""Analytical roofline for the SBD FlexAttention study.

This is not benchmark code. It applies the FLOP/byte accounting and MI300X
hardware constants from AMD Primus to mask statistics measured by the external
workload.

Primus source, pinned for reproducibility:
https://github.com/AMD-AGI/Primus/blob/0232423472112c612735ca06b35e4b6da5bfcdb2/primus/core/projection/simulation_backends/sdpa_simulator.py
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Hardware:
    peak_bf16_tflops: float = 1307.0
    peak_hbm_gbps: float = 5300.0


@dataclass(frozen=True)
class Workload:
    batch_size: int = 1
    query_heads: int = 16
    kv_heads: int = 4
    query_tokens: int = 4096
    kv_tokens: int = 4096
    head_dim_qk: int = 128
    head_dim_v: int = 128
    bytes_per_element: int = 2


@dataclass(frozen=True)
class Projection:
    phase: str
    pairs_per_head: int
    flops: float
    minimal_hbm_bytes: float
    arithmetic_intensity_flops_per_byte: float
    compute_roof_us: float
    memory_roof_us: float
    roof_us: float
    measured_us: float
    achieved_tflops: float
    peak_efficiency_percent: float
    latency_over_roof: float


def minimal_hbm_bytes(workload: Workload) -> tuple[int, int]:
    """Return forward/backward compulsory bytes.

    Forward follows Primus ``SDPASimulator._flops_bytes`` exactly. Backward
    adds the dQ write omitted by that helper; the correction does not change
    the bound for this workload because backward remains compute-bound.
    """

    w = workload
    b = w.bytes_per_element

    q = w.batch_size * w.query_heads * w.query_tokens * w.head_dim_qk * b
    k = w.batch_size * w.kv_heads * w.kv_tokens * w.head_dim_qk * b
    v = w.batch_size * w.kv_heads * w.kv_tokens * w.head_dim_v * b
    o = w.batch_size * w.query_heads * w.query_tokens * w.head_dim_v * b
    lse = w.batch_size * w.query_heads * w.query_tokens * 4

    forward = q + k + v + o + lse
    backward = q + k + v + o + o + lse + q + k + v
    return forward, backward


def phase_flops(phase: str, pairs_per_head: int, workload: Workload) -> int:
    """Apply Primus's FlashAttention forward/backward operation counts."""

    w = workload
    head_pairs = w.batch_size * w.query_heads * pairs_per_head
    if phase == "forward":
        # QK^T + PV + softmax.
        flops_per_pair = 2 * w.head_dim_qk + 2 * w.head_dim_v + 5
    elif phase == "backward":
        # QK^T recompute + dP + dV + dQ + dK + softmax backward.
        flops_per_pair = (
            2 * w.head_dim_qk
            + 2 * w.head_dim_v
            + 2 * w.head_dim_v
            + 2 * w.head_dim_qk
            + 2 * w.head_dim_qk
            + 5
        )
    else:
        raise ValueError(f"unknown phase: {phase}")
    return head_pairs * flops_per_pair


def project(
    phase: str,
    pairs_per_head: int,
    measured_ms: float,
    workload: Workload,
    hardware: Hardware,
) -> Projection:
    fwd_bytes, bwd_bytes = minimal_hbm_bytes(workload)
    hbm_bytes = fwd_bytes if phase == "forward" else bwd_bytes
    flops = phase_flops(phase, pairs_per_head, workload)

    compute_ms = flops / (hardware.peak_bf16_tflops * 1e12) * 1e3
    memory_ms = hbm_bytes / (hardware.peak_hbm_gbps * 1e9) * 1e3
    roof_ms = max(compute_ms, memory_ms)
    achieved_tflops = flops / (measured_ms * 1e-3) / 1e12

    return Projection(
        phase=phase,
        pairs_per_head=pairs_per_head,
        flops=flops,
        minimal_hbm_bytes=hbm_bytes,
        arithmetic_intensity_flops_per_byte=flops / hbm_bytes,
        compute_roof_us=compute_ms * 1e3,
        memory_roof_us=memory_ms * 1e3,
        roof_us=roof_ms * 1e3,
        measured_us=measured_ms * 1e3,
        achieved_tflops=achieved_tflops,
        peak_efficiency_percent=achieved_tflops / hardware.peak_bf16_tflops * 100,
        latency_over_roof=measured_ms / roof_ms,
    )


def main() -> None:
    hardware = Hardware()
    workload = Workload()

    pair_counts = {
        "semantic_allowed": 1_065_984,
        "block64_selected": 320 * 64 * 64,
        "block256_selected": 32 * 256 * 256,
    }
    measured_ms = {
        "block64_selected": {
            "forward": 0.2604,
            "backward": 0.5544,
            "training": 0.8147,
        },
        "block256_selected": {
            "forward": 0.2323,
            "backward": 1.1754,
            "training": 1.4077,
        },
    }

    results = {}
    for name, timings in measured_ms.items():
        phases = [
            project(phase, pair_counts[name], timings[phase], workload, hardware)
            for phase in ("forward", "backward")
        ]
        total_flops = sum(item.flops for item in phases)
        total_measured_us = timings["training"] * 1e3
        total_roof_us = sum(item.roof_us for item in phases)
        results[name] = {
            "phases": [asdict(item) for item in phases],
            "training": {
                "flops": total_flops,
                "measured_us": total_measured_us,
                "roof_us": total_roof_us,
                "achieved_tflops": total_flops / (total_measured_us * 1e-6) / 1e12,
                "peak_efficiency_percent": total_roof_us / total_measured_us * 100,
                "latency_over_roof": total_measured_us / total_roof_us,
            },
        }

    payload = {
        "model": "Primus-derived global BF16/HBM roofline",
        "hardware": asdict(hardware),
        "workload": asdict(workload),
        "ridge_point_flops_per_byte": hardware.peak_bf16_tflops * 1e3 / hardware.peak_hbm_gbps,
        "pair_counts": pair_counts,
        "block64_tile_inflation_over_semantic": (
            pair_counts["block64_selected"] / pair_counts["semantic_allowed"]
        ),
        "results": results,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
