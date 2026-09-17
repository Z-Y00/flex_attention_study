#!/usr/bin/env python3
"""Driver for the SBD FlexAttention workload in ``bench.py``.

``bench.py`` is a verbatim extraction of the fork's mask module; it defines the
mask and the ``fused_flex_attention`` call site but no harness.  This driver
supplies the workload shape, the HIP-event timing loop, the correctness check
and the tuning sweep used by this study.

Workload (matches results/gfx942-sbd-primus-v26.2.md):

    B=1, Hq=16, Hkv=4, head_dim=128, S=2048 (unified 2S=4096),
    4 documents of 512 base tokens, diffusion_block_size=16, BF16.

Usage:

    python3 bench_driver.py                       # sweep default block sizes
    python3 bench_driver.py --blocks 64
    python3 bench_driver.py --check               # correctness vs block 256
    python3 bench_driver.py --build-time          # mask construction timing
    python3 bench_driver.py --dense --blocks      # no-mask control
    python3 bench_driver.py --sweep fwd --blocks 16 \
        --grid '{"fwd_BLOCK_M":[],"fwd_BLOCK_N":[]}'   # tune launch params
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import traceback
from typing import Optional

import torch
from torch.nn.attention.flex_attention import flex_attention

import bench
from bench import build_global_layer_block_mask, fused_flex_attention

DEFAULT_BLOCKS = (256, 128, 64, 32, 16)


# --------------------------------------------------------------------------- #
# Workload construction
# --------------------------------------------------------------------------- #


def make_doc_ids(B: int, seq_len: int, num_docs: int, device: torch.device) -> torch.Tensor:
    """Evenly packed documents, as the packed dataloader produces them."""
    assert seq_len % num_docs == 0, "num_docs must divide the base sequence length"
    doc_starts = torch.zeros((B, seq_len), device=device, dtype=torch.bool)
    doc_starts[:, :: seq_len // num_docs] = True
    return bench.make_doc_ids_from_offsets(doc_starts)


def make_qkv(cfg, device: torch.device, seed: int = 0):
    """Q/K/V for the unified 2S sequence, bhsd with GQA."""
    gen = torch.Generator(device=device).manual_seed(seed)
    total_len = 2 * cfg.seq_len

    def rand(heads):
        return torch.randn(
            (cfg.batch, heads, total_len, cfg.head_dim),
            device=device,
            dtype=torch.bfloat16,
            generator=gen,
        ).requires_grad_(True)

    return rand(cfg.hq), rand(cfg.hkv), rand(cfg.hkv)


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #


def time_ms(fn, warmup: int, iters: int, repeat: int = 1) -> float:
    """Per-call wall time from HIP/CUDA events, best of ``repeat`` batches.

    The node is shared, so a single batch picks up interference from unrelated
    work. The minimum over repeated batches is the least contaminated estimate
    of the kernel's own cost.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    best = float("inf")
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) / iters)
    return best


def measure_block(cfg, mask_block_size: int, kernel_options: Optional[dict] = None) -> dict:
    """Forward and forward+backward latency for one mask block size."""
    device = torch.device("cuda")
    doc_ids = make_doc_ids(cfg.batch, cfg.seq_len, cfg.num_docs, device)
    mask = build_global_layer_block_mask(
        doc_ids,
        diffusion_block_size=cfg.diffusion_block_size,
        mask_block_size=mask_block_size,
    )
    q, k, v = make_qkv(cfg, device)
    dout = torch.randn_like(q)

    if kernel_options is None:
        call = fused_flex_attention
    else:
        # Same call as bench.fused_flex_attention, with explicit kernel options
        # so a candidate lowering default can be measured before it is baked
        # into the patch.
        @torch.compile(dynamic=False)
        def call(q, k, v, mask=None, _opts=tuple(sorted(kernel_options.items()))):
            return flex_attention(
                q, k, v, block_mask=mask, enable_gqa=True, kernel_options=dict(_opts)
            )

    def forward():
        return call(q, k, v, mask)

    def forward_backward():
        out = call(q, k, v, mask)
        out.backward(dout, retain_graph=False)

    fwd = time_ms(forward, cfg.warmup, cfg.iters, cfg.repeat)
    fwd_bwd = time_ms(forward_backward, cfg.warmup, cfg.iters, cfg.repeat)

    return {
        "block": mask_block_size,
        "forward_ms": fwd,
        "forward_backward_ms": fwd_bwd,
        "backward_ms": fwd_bwd - fwd,
        "sparsity": block_stats(mask),
    }


def measure_dense(cfg) -> dict:
    """Same shapes with no BlockMask at all.

    The lowering defaults this study changes are also reached by dense
    FlexAttention, so this is the non-regression control.
    """
    device = torch.device("cuda")
    q, k, v = make_qkv(cfg, device)
    dout = torch.randn_like(q)

    def forward():
        return fused_flex_attention(q, k, v, None)

    def forward_backward():
        fused_flex_attention(q, k, v, None).backward(dout)

    fwd = time_ms(forward, cfg.warmup, cfg.iters, cfg.repeat)
    fwd_bwd = time_ms(forward_backward, cfg.warmup, cfg.iters, cfg.repeat)
    return {
        "forward_ms": fwd,
        "forward_backward_ms": fwd_bwd,
        "backward_ms": fwd_bwd - fwd,
    }


def block_stats(mask) -> dict:
    """Selected partial/full block counts, for the roofline and for reporting."""
    partial = int(mask.kv_num_blocks.sum().item())
    full = int(mask.full_kv_num_blocks.sum().item()) if mask.full_kv_num_blocks is not None else 0
    q_blocks, kv_blocks = mask.kv_num_blocks.shape[-1], mask.kv_indices.shape[-1]
    total = q_blocks * kv_blocks
    return {
        "partial_blocks": partial,
        "full_blocks": full,
        "total_blocks": total,
        "selected_fraction": (partial + full) / total if total else 0.0,
    }


# --------------------------------------------------------------------------- #
# Correctness
# --------------------------------------------------------------------------- #


def check_against_reference(cfg, mask_block_size: int, reference_block: int = 256) -> dict:
    """Max abs deviation of out/dQ/dK/dV versus the coarse-block result.

    Both sides describe the same token-level mask, so any difference is BF16
    tile-order noise.
    """
    device = torch.device("cuda")
    doc_ids = make_doc_ids(cfg.batch, cfg.seq_len, cfg.num_docs, device)

    def run(block):
        mask = build_global_layer_block_mask(
            doc_ids,
            diffusion_block_size=cfg.diffusion_block_size,
            mask_block_size=block,
        )
        q, k, v = make_qkv(cfg, device, seed=1234)
        dout = torch.randn(
            q.shape, device=device, dtype=torch.bfloat16,
            generator=torch.Generator(device=device).manual_seed(99),
        )
        out = fused_flex_attention(q, k, v, mask)
        out.backward(dout)
        return out.detach(), q.grad, k.grad, v.grad

    ref = run(reference_block)
    got = run(mask_block_size)
    names = ("output", "dQ", "dK", "dV")
    return {
        name: (a.float() - b.float()).abs().max().item()
        for name, a, b in zip(names, ref, got)
    }


# --------------------------------------------------------------------------- #
# Mask construction cost
# --------------------------------------------------------------------------- #


def measure_build(cfg, mask_block_size: int) -> dict:
    """Eager versus compiled analytical BlockMask builder (a caller-side win)."""
    device = torch.device("cuda")
    doc_ids = make_doc_ids(cfg.batch, cfg.seq_len, cfg.num_docs, device)
    kwargs = dict(
        diffusion_block_size=cfg.diffusion_block_size,
        mask_block_size=mask_block_size,
    )

    eager = time_ms(lambda: build_global_layer_block_mask(doc_ids, **kwargs), 3, 20, cfg.repeat)

    compiled = torch.compile(build_global_layer_block_mask, dynamic=False, fullgraph=True)
    warm = time_ms(lambda: compiled(doc_ids, **kwargs), 3, 20, cfg.repeat)

    return {
        "block": mask_block_size,
        "eager_ms": eager,
        "compiled_ms": warm,
        "speedup": eager / warm if warm else float("inf"),
    }


# --------------------------------------------------------------------------- #
# Kernel-option sweep
# --------------------------------------------------------------------------- #

# Candidate lowering defaults to explore for a sparse block. Anything that wins
# here is a candidate value for the patch's derived defaults. The ``fwd_``/
# ``bwd_`` prefixes are stripped by the respective lowerings, so the forward and
# backward tiles can be tuned independently.
FWD_GRID = {
    "fwd_BLOCK_M": [16, 32, 64, 128],
    "fwd_BLOCK_N": [16, 32, 64, 128],
    "fwd_num_warps": [1, 2, 4, 8],
    "fwd_num_stages": [1, 2],
}

BWD_GRID = {
    "bwd_BLOCK_M1": [16, 32, 64, 128],
    "bwd_BLOCK_N1": [16, 32, 64, 128],
    "bwd_BLOCK_M2": [16, 32, 64, 128],
    "bwd_BLOCK_N2": [16, 32, 64, 128],
    "bwd_num_warps": [1, 2, 4, 8],
    "bwd_num_stages": [1, 2],
}


def _viable(opts: dict, block: int) -> bool:
    """Drop candidates the template or the block-sparse indexing would reject."""
    tiles = {k.split("_", 1)[1]: v for k, v in opts.items() if "BLOCK_" in k}
    # Every tile must divide the sparse block exactly.
    if any(block % value != 0 for value in tiles.values()):
        return False
    # Static assertions in flex_backwards.py.jinja.
    if "BLOCK_N1" in tiles and tiles["BLOCK_N1"] % tiles["BLOCK_M1"] != 0:
        return False
    if "BLOCK_M2" in tiles and tiles["BLOCK_M2"] % tiles["BLOCK_N2"] != 0:
        return False
    return True


def sweep_kernel_options(
    cfg, mask_block_size: int, grid: dict, fixed: Optional[dict] = None, key: str = "forward_ms"
) -> list:
    """Measure each candidate option set; failures are recorded, not fatal."""
    keys = sorted(grid)
    results = []
    for values in itertools.product(*(grid[k] for k in keys)):
        opts = dict(zip(keys, values))
        if not _viable(opts, mask_block_size):
            continue
        opts.update(fixed or {})
        try:
            torch._dynamo.reset()
            res = measure_block(cfg, mask_block_size, kernel_options=opts)
            res["kernel_options"] = opts
            results.append(res)
            print(
                f"  {opts} -> fwd {res['forward_ms']:.4f} ms, "
                f"fwd+bwd {res['forward_backward_ms']:.4f} ms, "
                f"bwd {res['backward_ms']:.4f} ms",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - a rejected config is a data point
            print(f"  {opts} -> FAILED: {type(exc).__name__}: {exc}", flush=True)
    results.sort(key=lambda r: r[key])
    if results:
        best = results[0]
        print(f"  BEST by {key}: {best['kernel_options']} -> {best[key]:.4f} ms", flush=True)
    return results


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--hq", type=int, default=16)
    p.add_argument("--hkv", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=2048, help="base AR length S; mask spans 2S")
    p.add_argument("--num-docs", type=int, default=4)
    p.add_argument("--diffusion-block-size", type=int, default=16)
    p.add_argument("--blocks", type=int, nargs="*", default=list(DEFAULT_BLOCKS))
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--repeat", type=int, default=3, help="timing batches per measurement; best wins")
    p.add_argument("--check", action="store_true", help="correctness versus block 256")
    p.add_argument("--dense", action="store_true", help="no-mask control, for non-regression")
    p.add_argument("--build-time", action="store_true", help="mask construction timing")
    p.add_argument(
        "--sweep",
        choices=("fwd", "bwd"),
        default=None,
        help="tune forward or backward lowering options for each --blocks value",
    )
    p.add_argument(
        "--fixed-options",
        type=json.loads,
        default=None,
        help='JSON kernel options pinned during a sweep, e.g. \'{"fwd_num_warps": 4}\'',
    )
    p.add_argument(
        "--grid",
        type=json.loads,
        default=None,
        help='JSON overriding entries of the swept grid, e.g. \'{"fwd_BLOCK_M": [32]}\'. '
        "An empty list drops that axis, leaving the lowering default in place.",
    )
    p.add_argument("--json", type=str, default=None, help="write results to this path")
    cfg = p.parse_args(argv)

    if not torch.cuda.is_available():
        print("No ROCm/CUDA device available.", file=sys.stderr)
        return 1

    print(f"device:  {torch.cuda.get_device_name(0)}")
    print(f"torch:   {torch.__version__}")
    try:
        import triton

        print(f"triton:  {triton.__version__}")
    except ImportError:
        pass
    print(
        f"workload: B={cfg.batch} Hq={cfg.hq} Hkv={cfg.hkv} D={cfg.head_dim} "
        f"S={cfg.seq_len} (2S={2 * cfg.seq_len}) docs={cfg.num_docs} "
        f"diffusion_block={cfg.diffusion_block_size} dtype=bf16"
    )
    print()

    report = {"blocks": [], "build": [], "check": {}, "sweep": [], "dense": None}

    if cfg.dense:
        torch._dynamo.reset()
        d = measure_dense(cfg)
        print(
            f"dense (no mask): forward {d['forward_ms']:.4f} ms, "
            f"forward+backward {d['forward_backward_ms']:.4f} ms, "
            f"backward {d['backward_ms']:.4f} ms"
        )
        report["dense"] = d
        print()

    if cfg.sweep:
        grid = dict(BWD_GRID if cfg.sweep == "bwd" else FWD_GRID)
        grid.update(cfg.grid or {})
        grid = {k: v for k, v in grid.items() if v}
        # A forward sweep is ranked on forward time; a backward sweep on the
        # backward time it actually changes.
        key = "backward_ms" if cfg.sweep == "bwd" else "forward_ms"
        for block in cfg.blocks:
            print(f"{cfg.sweep} kernel-option sweep, mask block {block}:")
            report["sweep"].append(
                {
                    "block": block,
                    "direction": cfg.sweep,
                    "results": sweep_kernel_options(
                        cfg, block, grid, fixed=cfg.fixed_options, key=key
                    ),
                }
            )
            print()
    else:
        for block in cfg.blocks:
            torch._dynamo.reset()
            try:
                res = measure_block(cfg, block)
            except Exception as exc:  # noqa: BLE001 - unsupported block is the finding
                print(f"block {block:>3}: FAILED: {type(exc).__name__}: {exc}")
                traceback.print_exc(limit=3)
                report["blocks"].append({"block": block, "error": repr(exc)})
                continue
            s = res["sparsity"]
            print(
                f"block {block:>3}: forward {res['forward_ms']:.4f} ms, "
                f"forward+backward {res['forward_backward_ms']:.4f} ms, "
                f"backward {res['backward_ms']:.4f} ms, "
                f"selected {s['selected_fraction'] * 100:.2f}%"
            )
            report["blocks"].append(res)

    if cfg.build_time:
        print()
        for block in cfg.blocks:
            torch._dynamo.reset()
            b = measure_build(cfg, block)
            print(
                f"build {block:>3}: eager {b['eager_ms']:.4f} ms, "
                f"compiled {b['compiled_ms']:.4f} ms, {b['speedup']:.1f}x"
            )
            report["build"].append(b)

    if cfg.check:
        print()
        for block in cfg.blocks:
            if block == 256:
                continue
            torch._dynamo.reset()
            d = check_against_reference(cfg, block)
            print(f"check {block:>3} vs 256: " + ", ".join(f"{k} {v:.8f}" for k, v in d.items()))
            report["check"][str(block)] = d

    if cfg.json:
        with open(cfg.json, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nwrote {cfg.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
