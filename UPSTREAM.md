# Upstream provenance and synchronization

## Patched source

The patch modifies:

```text
torch/_inductor/kernel/flex/flex_attention.py
```

Base runtime:

```text
Image:
  rocm/primus:v26.7-pytorch2.12-te2.17

Image digest:
  sha256:68b7eb7d4db99ecfa18bd7972e5b6d8b78e56219f0a144a8ef5e9a993e57f195

PyTorch:
  2.12.0+rocm10.0.0
  git 8c5ddc4002609fc54816dac039975bc39d4a6665

Triton:
  3.8.0

ROCm/HIP:
  7.15.26333
```

Permanent PyTorch source revision:

```text
https://github.com/pytorch/pytorch/commit/8c5ddc4002609fc54816dac039975bc39d4a6665
```

Affected functions:

```text
torch._inductor.kernel.flex.flex_attention.flex_attention
torch._inductor.kernel.flex.flex_attention.flex_attention_backward
```

Patch:

```text
patches/pytorch2.12-flexattention-small-sparse-blocks.patch
```

The earlier PyTorch 2.9 study is retained as:

```text
patches/pytorch2.9-flexattention-small-sparse-blocks.patch
results/gfx942-sbd-pytorch2.9.md
```

## Related upstream work

This study was informed by:

- PyTorch FlexAttention Triton lowering:
  <https://github.com/pytorch/pytorch/tree/main/torch/_inductor/kernel/flex>
- Target-dependent ROCm FlexAttention configurations:
  <https://github.com/pytorch/pytorch/pull/181283>
- FlyDSL FlexAttention forward:
  <https://github.com/pytorch/pytorch/pull/194309>
- FlyDSL FlexAttention backward:
  <https://github.com/pytorch/pytorch/pull/193854>
- Primus-Turbo FlexAttention compatibility layer:
  <https://github.com/AMD-AGI/Primus-Turbo/pull/472>

No benchmark source or Primus/FlyDSL kernel source is copied into this
repository.

## Local delta

The patch changes only Triton-choice construction:

1. Forward default `BLOCK_M/BLOCK_N` values are capped by the corresponding
   sparse Q/KV block sizes.
2. Backward eligibility is evaluated after deriving actual
   `BLOCK_M1/BLOCK_N1/BLOCK_M2/BLOCK_N2` values.
3. Derived backward defaults are capped by sparse Q/KV block sizes.
4. Fine sparse blocks use at most four warps; explicit user choices still win.

It does not modify:

- Triton forward or backward template math
- BlockMask construction or representation
- Mask/score graph lowering
- Dense attention interfaces
- CUDA/CuTe or FlyDSL backends

## Synchronizing to a newer PyTorch revision

1. Inspect the newer forward/backward choice loops:

```bash
python - <<'PY'
import inspect
import torch._inductor.kernel.flex.flex_attention as flex

print(inspect.getsource(flex.flex_attention))
print(inspect.getsource(flex.flex_attention_backward))
PY
```

2. Check whether upstream already:

- Caps every candidate's tile sizes to the sparse block dimensions.
- Validates the final tile values rather than the original dense config.
- Provides gfx942 backward choices for 64-token sparse blocks.
- Preserves explicit `fwd_`/`bwd_` kernel options.

3. Dry-run the patch:

```bash
patch --dry-run --batch --forward -p1 \
  -d <python-site-packages> \
  < patches/pytorch2.12-flexattention-small-sparse-blocks.patch
```

4. If the source moved, rebase the small logical change rather than adjusting
   hunk offsets blindly.

5. Re-run:

- Dense no-mask forward and backward
- Sparse blocks 256, 128, 64, 32, and 16
- MHA, MQA, and GQA
- FP16 and BF16
- Causal, document, SBD, fully masked, and irregular partial masks
- Explicit user kernel options
- Max-autotune enabled and disabled

6. Update the image digest, PyTorch/Triton versions, patch context, and results
in this repository.

## Licensing

The repository contains only a small patch against PyTorch source. PyTorch is
BSD-3-Clause licensed. Preserve upstream headers and follow PyTorch's
contribution and CLA requirements when preparing the final pull request.
