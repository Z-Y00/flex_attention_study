# ROCm FlexAttention small sparse-block study

This repository contains a minimal PyTorch/ROCm patch and reproducible Docker
recipe for fine-grained `BlockMask` training on AMD Instinct MI300X (`gfx942`).

It intentionally does **not** contain the workload benchmark. The benchmark is
mounted from a separate checkout when testing.

## Finding

PyTorch's installed ROCm FlexAttention backward lowering filters candidate
Triton configurations using the architecture's dense-attention tile sizes
before adapting them to the supplied sparse block size. For 64x64 and 16x16
`BlockMask` metadata, every candidate is discarded and backward raises:

```text
NoValidChoicesError: No choices to select
```

The patch:

1. Shrinks default forward and backward Triton tiles to the sparse block size.
2. Validates the derived tile sizes rather than the original dense defaults.
3. Uses four warps for sparse blocks of at most 64 tokens on gfx942.
4. Preserves explicit user kernel options and leaves 128+-token and dense
   defaults unchanged.

The current default image is Primus with PyTorch 2.12. Sparse block size 64 is
still best overall:

```text
block 256: forward 0.2323 ms, forward+backward 1.4077 ms
block  64: forward 0.2604 ms, forward+backward 0.8147 ms
block  32: forward 0.3197 ms, forward+backward 0.8908 ms
block  16: forward 0.4031 ms, forward+backward 1.0095 ms
```

Block 64 reduces measured training attention time by 1.73x relative to the
working 256-block baseline. PyTorch 2.12 substantially improves the baseline
backward, while fine blocks trade slower forward traversal for much faster
backward sparsity.

See [results/gfx942-sbd-pytorch2.12.md](results/gfx942-sbd-pytorch2.12.md) for
the current setup and
[results/gfx942-sbd-pytorch2.9.md](results/gfx942-sbd-pytorch2.9.md) for the
previous image comparison. Both reports include block statistics,
mask-construction measurements, and correctness results.

## Build

```bash
docker build \
  -t flex-attention-study:pytorch2.12-small-blocks \
  .
```

The default base image is:

```text
rocm/primus:v26.7-pytorch2.12-te2.17
```

Override it when testing another image:

```bash
docker build \
  --build-arg BASE_IMAGE=<image> \
  -t flex-attention-study:pytorch2.12-small-blocks \
  .
```

## Run an external benchmark

Mount the directory containing your benchmark instead of copying it into this
repository:

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add 44 --group-add 110 \
  --ipc=host --shm-size=32g \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -e HIP_VISIBLE_DEVICES=0 \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e TORCHINDUCTOR_MAX_AUTOTUNE=0 \
  -v /path/to/external/benchmark/directory:/workspace \
  -w /workspace \
  flex-attention-study:pytorch2.12-small-blocks \
  python3 benchmark.py
```

The workload must request `mask_block_size=64` to use the recommended
granularity. The patch also makes block sizes 32 and 16 executable, but both
were slower for forward+backward in the measured PyTorch 2.12 configuration.

## Status

This is an upstream-study prototype, not a production image:

- Validated on gfx942 with BF16, GQA, and the SBD mask.
- Dense forward performance was checked separately and did not regress.
- The patch needs broader PyTorch test coverage across dtypes, architectures,
  head dimensions, mask block shapes, and user-specified kernel options before
  upstream submission.

See [UPSTREAM.md](UPSTREAM.md) for source provenance and synchronization steps.
