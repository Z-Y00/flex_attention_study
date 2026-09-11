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

For the studied SBD workload, sparse block size 64 is best overall:

```text
block 256: forward 0.2072 ms, forward+backward 2.6714 ms
block  64: forward 0.1525 ms, forward+backward 0.7142 ms
block  16: forward 0.1437 ms, forward+backward 0.7649 ms
```

Block 64 reduces measured training attention time by 3.74x relative to the
working 256-block baseline. Block 16 has slightly faster forward but worse
backward, so it is not the recommended default.

See [results/gfx942-sbd.md](results/gfx942-sbd.md) for the complete setup,
block statistics, mask-construction measurements, and correctness results.

## Build

```bash
docker build \
  -t flex-attention-study:pytorch-small-blocks \
  .
```

The default base image is:

```text
rocm/sgl-dev:v0.5.17-rocm720-mi30x-20260819
```

Override it when testing another image:

```bash
docker build \
  --build-arg BASE_IMAGE=<image> \
  -t flex-attention-study:pytorch-small-blocks \
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
  flex-attention-study:pytorch-small-blocks \
  python3 benchmark.py
```

The workload must request `mask_block_size=64` to use the recommended
granularity. The patch also makes block size 16 executable, but it was slower
for forward+backward in the measured configuration.

## Status

This is an upstream-study prototype, not a production image:

- Validated on gfx942 with BF16, GQA, and the SBD mask.
- Dense forward performance was checked separately and did not regress.
- The patch needs broader PyTorch test coverage across dtypes, architectures,
  head dimensions, mask block shapes, and user-specified kernel options before
  upstream submission.

See [UPSTREAM.md](UPSTREAM.md) for source provenance and synchronization steps.
