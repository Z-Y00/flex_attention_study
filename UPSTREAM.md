# Upstream provenance and synchronization

## Patched source

The patch modifies:

```text
torch/_inductor/kernel/flex/flex_attention.py
```

Current base runtime:

```text
Image:
  rocm/primus:v26.2

Image digest:
  sha256:aa6fe22be0aff81de1e67d5754daf2e5d838ed781dbe592f10ceddf614b782d9

PyTorch:
  2.10.0a0+git449b176
  git 449b1768410104d3ed79d3bcfe4ba1d65c7f22c0

Triton:
  3.6.0

ROCm/HIP:
  7.2.26015
```

Permanent PyTorch source revision:

```text
https://github.com/pytorch/pytorch/commit/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0
```

Affected functions:

```text
torch._inductor.kernel.flex.flex_attention.flex_attention
torch._inductor.kernel.flex.flex_attention.flex_attention_backward
```

Patch:

```text
patches/pytorch2.10-flexattention-small-sparse-blocks.patch
```

Earlier studies are retained as:

```text
patches/pytorch2.12-flexattention-small-sparse-blocks.patch
results/gfx942-sbd-pytorch2.12.md
  rocm/primus:v26.7-pytorch2.12-te2.17
  sha256:68b7eb7d4db99ecfa18bd7972e5b6d8b78e56219f0a144a8ef5e9a993e57f195
  torch 2.12.0+rocm10.0.0 git 8c5ddc4002609fc54816dac039975bc39d4a6665
  triton 3.8.0, ROCm/HIP 7.15.26333

patches/pytorch2.9-flexattention-small-sparse-blocks.patch
results/gfx942-sbd-pytorch2.9.md
```

Each patch targets the PyTorch revision in its own image; select one with the
Dockerfile's `PATCH` build argument.

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
   `BLOCK_M1/BLOCK_N1/BLOCK_M2/BLOCK_N2` values, and also checks the backward
   template's `BLOCK_N1 % BLOCK_M1` and `BLOCK_M2 % BLOCK_N2` static assertions.
3. Derived backward defaults are capped by sparse Q/KV block sizes.
4. On ROCm, `num_warps` and `num_stages` defaults are re-derived from the
   shrunk tile rather than inherited from the dense config:
   - forward: one warp per 2048 tile elements;
   - backward: one warp per 16 rows of the smallest tile;
   - backward: at least two pipeline stages when that tile is 32 rows or less.
   Each is clamped to the config's own value and to at least one warp.

Steps 1 through 3 are unconditional, because they decide whether a fine
`BlockMask` can lower at all. Step 4 is gated on `torch.version.hip` because it
was fitted to gfx942 measurements (see
`results/gfx942-sbd-primus-v26.2.md`). All four are `setdefault`s, so explicit
`fwd_`/`bwd_` kernel options still take priority.

It does not modify:

- Triton forward or backward template math
- BlockMask construction or representation
- Mask/score graph lowering
- Dense attention interfaces or numerics
- CUDA/CuTe or FlyDSL backends

It does change dense FlexAttention's *launch parameters* on ROCm. A dense call
builds a full `BlockMask` with a 128-token block size, so the gfx942 forward
tile of 128x64 reaches step 4 and drops from eight warps to four. This is
deliberate and measured: the dense no-mask control improves from 7.2387 ms to
2.5132 ms in forward + backward. Any newer base image must re-run that control
before this patch is trusted on it.

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
- Provides gfx942 choices for 64-token and finer sparse blocks.
- Sizes `num_warps`/`num_stages` from the tile actually being launched.
- Preserves explicit `fwd_`/`bwd_` kernel options.

3. List the architecture's candidates, since the rules assume the tile shapes
   they produce and gfx942 currently offers exactly one per direction:

```bash
python - <<'PY'
import torch
from torch._inductor.choices import InductorChoices

c = InductorChoices()
print(c.get_flex_attention_fwd_configs(128, torch.bfloat16, "cuda"))
print(c.get_flex_attention_bwd_configs(128, torch.bfloat16, "cuda"))
PY
```

4. Dry-run the patch:

```bash
patch --dry-run --batch --forward -p1 \
  -d <python-site-packages> \
  < patches/pytorch2.10-flexattention-small-sparse-blocks.patch
```

5. If the source moved, rebase the small logical change rather than adjusting
   hunk offsets blindly.

6. Re-derive the warp and stage rules before trusting them on a new stack;
   they are fitted constants, not portable facts:

```bash
python3 bench_driver.py --sweep fwd --blocks 256 128 64 32 16 --repeat 5 \
  --grid '{"fwd_BLOCK_M":[],"fwd_BLOCK_N":[]}'
python3 bench_driver.py --sweep bwd --blocks 256 128 64 32 16 --repeat 5 \
  --grid '{"bwd_BLOCK_M1":[],"bwd_BLOCK_N1":[],"bwd_BLOCK_M2":[],"bwd_BLOCK_N2":[]}'
```

7. Re-run:

- Dense no-mask forward and backward (`--dense`), which this patch also affects
- Sparse blocks 256, 128, 64, 32, and 16
- MHA, MQA, and GQA
- FP16 and BF16
- Causal, document, SBD, fully masked, and irregular partial masks
- Explicit user kernel options
- Max-autotune enabled and disabled

8. Update the image digest, PyTorch/Triton versions, patch context, and results
in this repository.

## Licensing

The repository contains only a small patch against PyTorch source. PyTorch is
BSD-3-Clause licensed. Preserve upstream headers and follow PyTorch's
contribution and CLA requirements when preparing the final pull request.
