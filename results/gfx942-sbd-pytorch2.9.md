# gfx942 SBD FlexAttention results

## Environment

```text
GPU:       AMD Instinct MI300X (gfx942)
Image:     rocm/sgl-dev:v0.5.17-rocm720-mi30x-20260819
Torch:     2.9.1+rocm7.2.0.git7e1940d4
Triton:    3.7.0
ROCm/HIP:  7.2.26015-fc0010cf6a
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

The workload source was mounted externally and is intentionally not stored in
this repository.

## Block sparsity

Exact token-level mask:

```text
allowed token pairs: 1,065,984
total token pairs:  16,777,216
allowed density:         6.3538%
```

Block size 256:

```text
partial blocks: 24
full blocks:     8
empty blocks:  224
selected area / allowed area: 1.967x
```

Block size 64:

```text
partial blocks:   96
full blocks:     224
empty blocks:  3,776
selected area / allowed area: 1.230x
```

Block size 16:

```text
partial blocks:    128
full blocks:     4,096
empty blocks:   61,312
selected area / allowed area: 1.014x
```

## Unpatched behavior

Sparse block 256 is the smallest working training configuration with the
installed defaults:

```text
forward:             0.2120 ms
forward + backward:  2.6464 ms
approximate backward: 2.4344 ms
```

Blocks 64 and 16 compile for forward when compatible forward options are
provided, but backward fails before kernel generation:

```text
NoValidChoicesError: No choices to select
target: flex_attention_backward
```

Controls:

```text
All 256 block pairs treated as partial:
  forward:             1.2116 ms
  forward + backward: 13.8502 ms

Dense no-mask attention:
  forward:             0.8071 ms
  forward + backward:  8.2791 ms
```

The existing full/partial/empty BlockMask separation therefore provides a
5.71x forward and 5.23x training speedup over evaluating all blocks as partial.

## Patched default behavior

Final tuned patch:

```text
block 256:
  forward:              0.2072 ms
  forward + backward:   2.6714 ms
  approximate backward: 2.4642 ms

block 64:
  forward:              0.1525 ms
  forward + backward:   0.7142 ms
  approximate backward: 0.5616 ms

block 16:
  forward:              0.1437 ms
  forward + backward:   0.7649 ms
  approximate backward: 0.6211 ms

dense no-mask, isolated:
  forward:              0.7921 ms
```

Relative to block 256, block 64 is:

```text
1.36x faster in forward
3.74x faster in forward + backward
4.39x faster in approximate backward
```

Block 16 lowers forward by another 0.0088 ms but increases total training time
by 0.0507 ms. Block 64 is therefore the recommended SBD training granularity.

## Warp tuning

Retaining the gfx942 dense default of eight warps is poor for fine sparse
blocks:

```text
block 64, 8 warps:
  forward:             0.2522 ms
  forward + backward:  2.0645 ms

block 64, 4 warps:
  forward:             0.1515 ms
  forward + backward:  0.7105 ms

block 16, 8 warps:
  forward:             0.2828 ms
  forward + backward:  1.2801 ms

block 16, 4 warps:
  forward:             0.1442 ms
  forward + backward:  0.7647 ms
```

The patch caps the default at four warps only when the sparse block size is at
most 64. Explicit user choices remain unchanged.

## Correctness

The 256-block result was used as the existing-kernel reference for equivalent
SBD masks.

Block 64:

```text
output max abs: 0.00390625
dQ max abs:     0.00781250
dK max abs:     0.03125000
dV max abs:     0.00781250
```

Block 16:

```text
output max abs: 0.00781250
dQ max abs:     0.00781250
dK max abs:     0.03125000
dV max abs:     0.00781250
```

These differences are consistent with BF16 tile-order variation. Broader
upstream tests against an FP32 materialized reference are still required.

## BlockMask construction

The analytical workload builder is dominated by many small eager launches:

```text
warm eager wall time:    2.5289 ms
warm compiled wall time: 0.1534 ms
speedup:                    16.5x
```

The compiled form is:

```python
compiled_builder = torch.compile(
    build_global_layer_block_mask,
    dynamic=False,
    fullgraph=True,
)
```

This is an application-level optimization and is not part of the Docker patch.

## Conclusion

The default Triton FlexAttention implementation is already effective once:

1. The analytical BlockMask builder is compiled.
2. Backward choices are adapted to 64-token sparse blocks.
3. Fine sparse blocks use four warps on gfx942.

For this workload, that focused PyTorch patch is likely sufficient and avoids
the engineering cost of a separate ROCm attention implementation.
