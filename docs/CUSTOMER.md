# Accelerating the SBD FlexAttention workload on MI300X

How to apply this study's patch to the semi-autoregressive block-diffusion (SBD)
training workload and get the measured speedup.

All numbers below were measured on AMD Instinct MI300X (`gfx942`) with
`rocm/primus:v26.2`, at B=1, Hq=16, Hkv=4, head_dim=128, S=2048 (unified
sequence 2S=4096), 4 documents, diffusion block size 16, BF16, Inductor
autotuning disabled. Timings are HIP events, best of five batches of 50
iterations. See [../results/gfx942-sbd-primus-v26.2.md](../results/gfx942-sbd-primus-v26.2.md)
for the full sweeps behind them.

## What you get

Two tiers. The patch alone helps without touching your code; the full speedup
needs a one-line change at the mask-construction call site.

| Configuration | Attention fwd+bwd | Speedup | Code change |
| --- | --- | --- | --- |
| Today: stock `rocm/primus:v26.2`, `mask_block_size=256` | 2.5534 ms | — | — |
| Patched image only | 1.1942 ms | **2.14x** | none |
| Patched + `mask_block_size=16` | 0.5241 ms | **4.87x** | one argument |

Intermediate block sizes also work once patched: 128 gives 0.9989 ms, 64 gives
0.7054 ms, 32 gives 0.5957 ms. On the stock image, every block size below 128
fails to compile at all.

## Step 1: get the patched runtime

### Option A: build the image

Preferred for training jobs.

```bash
git clone git@github.com:Z-Y00/flex_attention_study.git
cd flex_attention_study
git checkout primus-v26.2-small-sparse-blocks
docker build -t primus-v26.2-flex-small-blocks .
```

The Dockerfile defaults to `rocm/primus:v26.2` and the matching PyTorch 2.10
patch. For a different base image, override both:

```bash
docker build \
  --build-arg BASE_IMAGE=rocm/primus:v26.7-pytorch2.12-te2.17 \
  --build-arg PATCH=pytorch2.12-flexattention-small-sparse-blocks.patch \
  -t primus-v26.7-flex-small-blocks .
```

### Option B: patch an existing container or venv in place

No rebuild required.

```bash
patch --batch --forward -p1 \
  -d /opt/venv/lib/python3.12/site-packages \
  < patches/pytorch2.10-flexattention-small-sparse-blocks.patch
```

Adjust the `-d` path if your PyTorch lives elsewhere; it should be the
directory containing `torch/`.

### Confirm the patch is live

```bash
python3 -c "import torch._inductor.kernel.flex.flex_attention as f; \
print('patched' if hasattr(f, '_sparse_fwd_num_warps') else 'NOT patched')"
```

No cache clearing is needed. A TorchInductor/Triton cache warmed by the
unpatched build was tested against the patched build and the new lowering still
took effect.

## Step 2: ask for a finer mask block

This is where most of the gain is. In the fork this is the `_build_block_mask`
call in `CMoEModel.forward`, which currently takes the default of 256:

```python
block_mask = build_global_layer_block_mask(
    doc_ids,
    diffusion_block_size=args.diffusion_block_size,
    mask_block_size=16,          # was 256
)
```

`mask_block_size` must divide the AR sequence length `S`. This is asserted in
`make_sbd_doc_block_mask`, so that no mask block straddles the AR/diffusion
midpoint. 16 divides 2048.

Making this change *without* Step 1 raises:

```text
ValueError: Q and KV block size must be divisible by BLOCK_M and BLOCK_N.
```

so apply the patch first.

## Step 3 (optional): compile the mask builder

Independent of the patch, and worth more at block 16 than it was at block 256,
because the attention call it sits next to is now much cheaper:

```python
compiled_builder = torch.compile(
    build_global_layer_block_mask,
    dynamic=False,
    fullgraph=True,
)
```

This takes the builder from 2.5272 ms to 0.1380 ms, an 18x reduction. At block
16 the eager builder otherwise costs roughly five times the tuned attention
call itself. It only matters if you rebuild the mask every step rather than
caching it across steps.

## Step 4: verify on your own shapes

`bench_driver.py` needs the workload's mask module, `bench.py`, beside it.

```bash
python3 bench_driver.py --blocks 256 16 --repeat 5   # speedup
python3 bench_driver.py --check                      # numerics against block 256
python3 bench_driver.py --dense --blocks             # dense no-mask control
```

Pass `--seq-len`, `--hq`, `--hkv`, `--head-dim`, `--num-docs` and
`--diffusion-block-size` to match your real configuration.

`--check` reports the maximum absolute deviation against the block-256 result,
which describes the same token-level mask. Expect around 1e-2 for dK and less
elsewhere: BF16 tile-order differences, not a change in what is computed.

On a shared GPU, use `--repeat` and disregard differences smaller than the
node's run-to-run spread.

## Rollback

```bash
patch -R -p1 \
  -d /opt/venv/lib/python3.12/site-packages \
  < patches/pytorch2.10-flexattention-small-sparse-blocks.patch
```

Or simply run the stock image. The patch touches one file,
`torch/_inductor/kernel/flex/flex_attention.py`, and only its Triton-choice
construction: no template math, no BlockMask representation, no interfaces.

## Before you deploy

- **This also changes dense FlexAttention on ROCm.** A dense call builds a full
  BlockMask with a 128-token block size internally, so it reaches the same
  re-tuned warp count. Here that is a 2.88x improvement (7.2387 ms to 2.5132 ms
  forward+backward), not a regression, but any other FlexAttention call site in
  your model is affected too and is worth spot-checking.
- **The warp and pipeline-stage rules were fitted to gfx942, BF16, head_dim
  128.** They are gated to ROCm but not to a head dimension. If you run other
  head dimensions, re-derive them with the sweeps in
  [../UPSTREAM.md](../UPSTREAM.md) before trusting the defaults.
- **Explicit kernel options still win.** Everything the patch sets is a
  `setdefault`, so any `fwd_`/`bwd_` kernel options you already pass are
  untouched.
- **Only one Triton config exists per direction on this stack** at head_dim 128
  and BF16, so there is no autotuning fallback. This is why the stock image
  fails outright rather than picking a slower kernel.
- **This is an upstream-study prototype, not a supported product.** It needs
  broader dtype, architecture and head-dimension coverage before it can go
  upstream to PyTorch.
