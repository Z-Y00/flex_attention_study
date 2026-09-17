# ROCm FlexAttention small sparse-block study

This repository contains a minimal PyTorch/ROCm patch and reproducible Docker
recipe for fine-grained `BlockMask` training on AMD Instinct MI300X (`gfx942`).

The workload's mask module (`bench.py`) is mounted from a separate checkout and
is not stored here. `bench_driver.py` is the harness this study uses to drive
it: it supplies the workload shape, the HIP-event timing loop, the kernel-option
sweeps, the dense control and the correctness check.

## Finding

PyTorch's installed ROCm FlexAttention lowering filters candidate Triton
configurations using the architecture's dense-attention tile sizes before
adapting them to the supplied sparse block size. For 64x64 and 16x16
`BlockMask` metadata every candidate is discarded, and lowering raises:

```text
NoValidChoicesError: No choices to select
ValueError: Q and KV block size must be divisible by BLOCK_M and BLOCK_N
```

gfx942 offers exactly one candidate per direction at head_dim 128 / BF16, so
there is no autotuning fallback once that candidate is rejected.

The patch:

1. Shrinks forward and backward Triton tile defaults to the sparse block size.
2. Validates the derived tile sizes rather than the original dense defaults,
   including the backward template's own static assertions.
3. Re-tunes the launch parameters for the shrunk tile, on ROCm only.
4. Preserves explicit user kernel options; the changes are all `setdefault`s.

Step 3 turned out to matter far more than step 1. The shipped `num_warps=8` is
wrong for head_dim 128 in both directions at every block size measured, and for
the dense no-mask case too. Measured optima:

```text
forward:   one warp per 2048 tile elements     (128x64 -> 4, 64x64 -> 2, 32x32 -> 1)
backward:  one warp per 16 rows of the smallest tile (64 -> 4, 32 -> 2, 16 -> 1)
backward:  a second pipeline stage below 32 rows
```

## Results

Current default image is Primus v26.2 (PyTorch 2.10). Forward + backward:

```text
                 unpatched      patched
block 256:        2.5534 ms     1.1942 ms
block 128:        2.2186 ms     0.9989 ms
block  64:        (fails)       0.7054 ms
block  32:        (fails)       0.5957 ms
block  16:        (fails)       0.5241 ms
dense (no mask):  7.2387 ms     2.5132 ms
```

Block 16 is 4.87x faster than the workload's default block 256 on the unpatched
image. The dense control improves 2.88x rather than regressing, because the
patch reaches it through the same mistuned defaults.

See [results/gfx942-sbd-primus-v26.2.md](results/gfx942-sbd-primus-v26.2.md)
for the sweeps behind those numbers, block statistics, mask-construction
measurements and correctness results.

Earlier images:
[PyTorch 2.12](results/gfx942-sbd-pytorch2.12.md) (where block 64 was the
optimum and only the backward failed) and
[PyTorch 2.9](results/gfx942-sbd-pytorch2.9.md).

The
[Primus-derived analytical roofline](results/gfx942-sbd-analytical-roofline.md)
uses exact selected block counts to compare the optimized forward and backward
latencies with MI300X BF16/HBM limits. The calculation is reproducible with
`analysis/sbd_roofline.py`; it is an analytical model, not benchmark code.

## Build

```bash
docker build \
  -t flex-attention-study:v26.2-small-blocks \
  .
```

The default base image is:

```text
rocm/primus:v26.2
```

Target another image by overriding both the base and the matching patch:

```bash
docker build \
  --build-arg BASE_IMAGE=rocm/primus:v26.7-pytorch2.12-te2.17 \
  --build-arg PATCH=pytorch2.12-flexattention-small-sparse-blocks.patch \
  -t flex-attention-study:pytorch2.12-small-blocks \
  .
```

## Run the benchmark

Mount the directory holding `bench.py` and `bench_driver.py` instead of copying
them into this repository:

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video \
  --ipc=host --shm-size=32g \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -e HIP_VISIBLE_DEVICES=0 \
  -e TORCHINDUCTOR_MAX_AUTOTUNE=0 \
  -v /path/to/external/benchmark/directory:/workspace \
  -w /workspace \
  flex-attention-study:v26.2-small-blocks \
  python3 bench_driver.py --blocks 256 128 64 32 16 --repeat 5
```

Useful driver modes:

```bash
python3 bench_driver.py --dense --blocks          # no-mask non-regression control
python3 bench_driver.py --check                   # versus the block-256 result
python3 bench_driver.py --build-time              # mask construction cost
python3 bench_driver.py --sweep fwd --blocks 16 \
  --grid '{"fwd_BLOCK_M":[],"fwd_BLOCK_N":[]}'    # re-derive the warp rule
```

For this workload the caller should request `mask_block_size=16` and compile the
analytical mask builder. On a shared GPU, use `--repeat` and treat differences
smaller than the node's spread as noise.

## Status

This is an upstream-study prototype, not a production image:

- Validated on gfx942 with BF16, GQA, the SBD mask and a dense control.
- The warp and stage rules were fitted to gfx942 / BF16 / head_dim 128. They
  are gated to ROCm but not to a head dimension, and no other head dimension,
  dtype or architecture has been measured.
- Only one Triton config exists per direction on this stack, so the rules are
  untested against a multi-config architecture.
- The patch needs broader PyTorch test coverage across dtypes, architectures,
  head dimensions, mask block shapes and user-specified kernel options before
  upstream submission.

See [UPSTREAM.md](UPSTREAM.md) for source provenance and synchronization steps.
