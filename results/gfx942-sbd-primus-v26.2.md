# gfx942 SBD FlexAttention results: Primus v26.2 (PyTorch 2.10)

## Environment

```text
GPU:       AMD Instinct MI300X (gfx942)
Image:     rocm/primus:v26.2
Digest:    sha256:aa6fe22be0aff81de1e67d5754daf2e5d838ed781dbe592f10ceddf614b782d9
Torch:     2.10.0a0+git449b176
Torch git: 449b1768410104d3ed79d3bcfe4ba1d65c7f22c0
Triton:    3.6.0
ROCm/HIP:  7.2.26015
Autotune:  disabled
Timing:    HIP events, 3 warmups, 50 iterations, best of 5 batches
```

Workload:

```text
B=1
Hq=16
Hkv=4
head_dim=128
base sequence S=2048
unified AR+diffusion sequence=4096
documents=4, each 512 base tokens
diffusion block size=16
dtype=BF16
```

The mask module is the fork extraction in `bench.py`; the harness that supplies
the shape, timing loop, sweeps and checks is `bench_driver.py`.

Measurements were taken on a shared node. Every GPU carried roughly 6.5 GB of
resident memory from unrelated long-running containers, all idle at 0% compute.
A single measurement batch varied by up to ~35% on this node, so every number
below is the best of five batches; only effects far larger than that spread are
treated as real.

## Available Triton configurations

gfx942 offers exactly one candidate in each direction at head_dim 128 / BF16:

```text
forward:  block_m=128, block_n=64,                     num_stages=1, num_warps=8
backward: block_m1=64, block_n1=128, block_m2=128, block_n2=64,
                                                       num_stages=1, num_warps=8
```

With a single candidate there is no autotuning fallback: if that one config is
rejected, lowering fails outright.

## Unpatched image

```text
block 256:
  forward:            0.2340 ms
  forward + backward: 2.5534 ms
  backward:           2.3193 ms

block 128:
  forward:            0.1766 ms
  forward + backward: 2.2186 ms
  backward:           2.0420 ms

block 64, 32, 16:
  LoweringException: ValueError: Q and KV block size must be divisible by
  BLOCK_M and BLOCK_N.

dense (no mask):
  forward:            0.7563 ms
  forward + backward: 7.2387 ms
  backward:           6.4824 ms
```

On this base the *forward* lowering already fails for fine blocks, earlier than
on PyTorch 2.12, because the sole forward candidate has `block_m=128` and there
is no second config to fall back to. The backward filter would reject its sole
candidate as well.

## Patched image

```text
block 256:
  forward:            0.1582 ms
  forward + backward: 1.1942 ms
  backward:           1.0360 ms

block 128:
  forward:            0.1160 ms
  forward + backward: 0.9989 ms
  backward:           0.8829 ms

block 64:
  forward:            0.0971 ms
  forward + backward: 0.7054 ms
  backward:           0.6083 ms

block 32:
  forward:            0.0964 ms
  forward + backward: 0.5957 ms
  backward:           0.4992 ms

block 16:
  forward:            0.1035 ms
  forward + backward: 0.5241 ms
  backward:           0.4207 ms

dense (no mask):
  forward:            0.5424 ms
  forward + backward: 2.5132 ms
  backward:           1.9709 ms
```

Speedups in forward + backward:

```text
block 256, patched vs unpatched:            2.14x
block 16 vs the workload default block 256: 4.87x
block 16 vs best working unpatched (128):   4.23x
dense, patched vs unpatched:                2.88x
```

Unlike the PyTorch 2.12 study, block 16 is the optimum here, not block 64. Each
step down in mask block size keeps reducing selected area (12.50% at 256 to
6.45% at 16), and once the launch parameters match the smaller tile the
traversal overhead no longer cancels that saving.

## Why the launch parameters dominate

Sweeping `num_warps` and `num_stages` with the tiles left at the derived values
separates the two effects the patch has. The tile capping is what makes fine
blocks lower at all; the launch parameters are what make everything fast.

Forward, best `num_warps` per derived tile (`num_stages=1` won everywhere):

```text
tile 128x64 (block 256, 128):  4 warps    0.1582 / 0.1171 ms   (8 warps: 0.2299 / 0.1740)
tile  64x64 (block 64):        2 warps    0.0976 ms            (8 warps: 0.2365)
tile  32x32 (block 32):        1 warp     0.0986 ms            (8 warps: 0.2909)
tile  16x16 (block 16):        1 warp     0.1043 ms            (8 warps: 0.2875)
```

The optimum tracks tile *area*, at roughly one warp per 2048 elements.

Backward, best `num_warps` / `num_stages` per derived tile set:

```text
tiles 64/128/128/64 (block 256):  4 warps, 1 stage   1.0584 ms  (8 warps: 2.3532)
tiles 64/128/128/64 (block 128):  4 warps, 1 stage   0.8867 ms  (8 warps: 2.0291)
tiles 64/64/64/64   (block 64):   4 warps, 1 stage   0.6039 ms  (8 warps: 1.5090)
tiles 32/32/32/32   (block 32):   2 warps, 2 stages  0.4846 ms  (8 warps: 1.0757)
tiles 16/16/16/16   (block 16):   1 warp,  2 stages  0.4226 ms  (8 warps: 0.8927)
```

Backward tracks the *smallest* tile dimension instead, at one warp per 16 rows,
and below 32 rows a second pipeline stage pays for itself (block 16: 0.5790 ms
at one stage against 0.4226 ms at two).

The shipped `num_warps=8` is therefore wrong for this head dimension in both
directions, at every block size measured and for the dense no-mask case too.
That is why the dense control improves 2.88x rather than regressing: the patch
reaches it through the same defaults.

Tile *shape* was also swept at fixed warps. Every candidate landed inside the
node's measurement spread, so the patch leaves tile shape at the capped config
value and changes only what the sweeps showed to be a large effect.

## Correctness

Max absolute deviation against the block-256 result, which describes the same
token-level mask:

```text
block 128: output 0.00390625  dQ 0.00781250  dK 0.03125000  dV 0.01562500
block  64: output 0.00781250  dQ 0.00781250  dK 0.03125000  dV 0.01562500
block  32: output 0.00781250  dQ 0.00781250  dK 0.03125000  dV 0.01562500
block  16: output 0.00781250  dQ 0.00976562  dK 0.03125000  dV 0.01562500
```

These are BF16 tile-order differences of the same magnitude the PyTorch 2.12
study saw. An upstream submission still needs an FP32-reference comparison over
more shapes and masks.

## BlockMask construction

Compiling the analytical builder is a caller-side change, not part of the patch:

```text
block 256: eager 2.5158 ms, compiled 0.1053 ms, 23.9x
block 128: eager 2.5089 ms, compiled 0.0989 ms, 25.4x
block  64: eager 2.2525 ms, compiled 0.1079 ms, 20.9x
block  32: eager 2.5061 ms, compiled 0.1067 ms, 23.5x
block  16: eager 2.5272 ms, compiled 0.1380 ms, 18.3x
```

At block 16 the eager builder costs 2.53 ms, about five times the tuned
attention call itself, so compiling it matters more here than it did at block
256:

```python
compiled_builder = torch.compile(
    build_global_layer_block_mask,
    dynamic=False,
    fullgraph=True,
)
```

## Recommendation

```text
mask_block_size=16
compiled analytical BlockMask builder
patched sparse-compatible and re-tuned ROCm Triton choices
```

Attention forward + backward for this workload goes from 2.5534 ms to 0.5241 ms,
a 4.87x reduction. Adding the compiled mask builder replaces a 2.53 ms eager
build with 0.14 ms.

## Caveats

- The warp and stage rules were fitted to gfx942, BF16, head_dim 128. They are
  gated to ROCm but not to a head dimension, and no other head dimension, dtype
  or architecture was measured.
- Only one Triton config exists per direction on this stack, so these results
  say nothing about how the rules interact with a multi-config architecture.
- Measurements come from a shared node; see the spread note above.
