# gfx942 SBD FlexAttention results: Primus PyTorch 2.12

## Environment

```text
GPU:       AMD Instinct MI300X (gfx942)
Image:     rocm/primus:v26.7-pytorch2.12-te2.17
Digest:    sha256:68b7eb7d4db99ecfa18bd7972e5b6d8b78e56219f0a144a8ef5e9a993e57f195
Torch:     2.12.0+rocm10.0.0
Torch git: 8c5ddc4002609fc54816dac039975bc39d4a6665
Triton:    3.8.0
ROCm/HIP:  7.15.26333
Autotune:  disabled
Timing:    HIP events, 3 warmups, 50 measured iterations
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

The workload benchmark is mounted externally and is intentionally not stored in
this repository.

## Unpatched image

PyTorch 2.12 improves the working 256-block baseline considerably relative to
the previous PyTorch 2.9 image:

```text
block 256:
  forward:              0.2411 ms
  forward + backward:   1.4132 ms
  approximate backward: 1.1721 ms
```

Fine-grained masks still fail before kernel generation:

```text
block 64:
  NoValidChoicesError in flex_attention_backward

block 16:
  NoValidChoicesError in flex_attention_backward
```

The backward lowering rejects every architecture configuration using its
original dense tile dimensions before adapting those dimensions to the sparse
BlockMask.

## Patched image

Final default behavior after deriving sparse-compatible tiles and selecting
four warps for blocks at most 64:

```text
block 256:
  forward:              0.2323 ms
  forward + backward:   1.4077 ms
  approximate backward: 1.1754 ms

block 64:
  forward:              0.2604 ms
  forward + backward:   0.8147 ms
  approximate backward: 0.5544 ms

block 32:
  forward:              0.3197 ms
  forward + backward:   0.8908 ms
  approximate backward: 0.5711 ms

block 16:
  forward:              0.4031 ms
  forward + backward:   1.0095 ms
  approximate backward: 0.6064 ms
```

Block 64 relative to the patched block-256 control:

```text
forward: 0.89x (12% slower)
forward + backward: 1.73x faster
approximate backward: 2.12x faster
```

The smaller sparse block reduces wasted backward score work enough to dominate
its additional metadata/traversal overhead. Blocks 32 and 16 continue reducing
the selected token area, but their extra program/list overhead outweighs the
saved math.

## Correctness

Equivalent SBD masks at block 64 versus the existing block-256 result:

```text
output max abs: 0.00781250
dQ max abs:     0.00781250
dK max abs:     0.01562500
dV max abs:     0.00781250
```

Block 16 versus block 256:

```text
output max abs: 0.00781250
dQ max abs:     0.00781250
dK max abs:     0.03125000
dV max abs:     0.00781250
```

These are expected BF16 tile-order differences. An upstream submission still
needs FP32-reference coverage over more shapes and masks.

## BlockMask construction

For the recommended block size 64:

```text
warm eager builder:    2.6360 ms
warm compiled builder: 0.1373 ms
speedup:                  19.2x
```

Compile the workload's analytical mask builder separately:

```python
compiled_builder = torch.compile(
    build_global_layer_block_mask,
    dynamic=False,
    fullgraph=True,
)
```

This optimization belongs in the caller and is not part of the Docker/PyTorch
patch.

## Comparison with PyTorch 2.9

PyTorch 2.12 roughly halves the original 256-block training baseline:

```text
PyTorch 2.9 block 256:  2.6714 ms
PyTorch 2.12 block 256: 1.4077 ms
```

The older image gained more from the patch in relative terms (3.74x), while the
new Primus image still gains 1.73x and has the faster absolute result.

## Recommendation

Use:

```text
mask_block_size=64
compiled analytical BlockMask builder
patched sparse-compatible ROCm backward choices
```

For this workload, the focused PyTorch change remains preferable to a separate
attention backend.
