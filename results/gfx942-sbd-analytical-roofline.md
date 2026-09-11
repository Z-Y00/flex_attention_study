# gfx942 SBD FlexAttention analytical roofline

## Scope

This report estimates how far the optimized block-64 result is from a global
MI300X roofline. It is analytical only: no hardware counters or profiler traces
are used.

The model reuses AMD Primus commit
[`0232423472112c612735ca06b35e4b6da5bfcdb2`](https://github.com/AMD-AGI/Primus/commit/0232423472112c612735ca06b35e4b6da5bfcdb2):

- [MI300X BF16 and HBM peaks](https://github.com/AMD-AGI/Primus/blob/0232423472112c612735ca06b35e4b6da5bfcdb2/primus/core/projection/simulation_backends/sdpa_simulator.py#L100-L132)
- [FlashAttention forward/backward FLOP and compulsory-byte formulas](https://github.com/AMD-AGI/Primus/blob/0232423472112c612735ca06b35e4b6da5bfcdb2/primus/core/projection/simulation_backends/sdpa_simulator.py#L688-L737)

Primus's stock SDPA interface accepts dense/causal geometry but not arbitrary
BlockMask lists. Applying its dense-causal simulator directly would therefore
model the wrong workload. This study keeps the Primus operation accounting and
hardware profile, but substitutes the exact full/partial block counts from the
external SBD mask builder.

The reproducible calculation is in
[`analysis/sbd_roofline.py`](../analysis/sbd_roofline.py). It contains no
benchmark workload code.

## Inputs

```text
GPU:                    AMD Instinct MI300X (gfx942)
Peak BF16 matrix rate:  1307 TFLOP/s
Peak HBM bandwidth:     5300 GB/s
Roofline ridge point:   246.60 FLOP/byte

B:                      1
Hq / Hkv:               16 / 4
Q / KV tokens:          4096 / 4096
QK / V head dimension:  128 / 128
dtype:                  BF16
```

The HBM term is a compulsory-traffic lower bound: each logical Q/K/V/O tensor
is transferred once, with GQA K/V sized using `Hkv=4`. The backward byte count
also includes the dQ write omitted by the current Primus helper. This correction
does not change the final backward roof because the phase remains
compute-bound.

## Exact sparse work

The real mask builder reports:

```text
Mathematically allowed pairs per head:      1,065,984
Block-64 partial blocks:                           96
Block-64 full blocks:                             224
Block-64 selected tile pairs:               1,310,720
Selected / mathematically allowed:             1.2296x

Block-256 partial blocks:                          24
Block-256 full blocks:                              8
Block-256 selected tile pairs:              2,097,152
Selected / mathematically allowed:             1.9673x
```

The block-64 roofline uses selected tile pairs, not only allowed cells. Partial
tiles still execute QK and PV matrix work before their token mask is applied.
Relative to block 256, block 64 removes 37.5% of selected matrix work.

## Optimized block-64 result

Forward:

```text
Executed work:          10.8423 GFLOP
Compulsory HBM traffic: 42.2052 MB
Arithmetic intensity:  256.89 FLOP/byte
Compute floor:          8.296 us
HBM floor:              7.963 us
Roofline floor:         8.296 us (compute-bound, barely above ridge)

Measured latency:       260.4 us
Achieved throughput:    41.64 TFLOP/s
Peak fraction:          3.19%
Latency / roofline:     31.39x
```

Backward:

```text
Executed work:          26.9484 GFLOP
Compulsory HBM traffic: 84.1482 MB
Arithmetic intensity:  320.25 FLOP/byte
Compute floor:          20.619 us
HBM floor:              15.877 us
Roofline floor:         20.619 us (compute-bound)

Measured latency:       554.4 us
Achieved throughput:    48.61 TFLOP/s
Peak fraction:          3.72%
Latency / roofline:     26.89x
```

Forward plus backward:

```text
Executed work:          37.7907 GFLOP
Analytical floor:       28.914 us
Measured latency:       814.7 us
Achieved throughput:    46.39 TFLOP/s
Peak fraction:          3.55%
Latency / roofline:     28.18x
```

## What the optimization changed

The block-256 control reaches 5.71% of peak in forward but only 2.81% in
backward. Block 64 reaches 3.19% and 3.72%, respectively.

That explains the measured behavior:

- Forward does 37.5% less selected matrix work, but its efficiency falls enough
  that it is still about 12% slower.
- Backward does 37.5% less selected work and improves effective utilization,
  producing the measured 2.12x speedup.
- Combined effective throughput rises from 42.95 to 46.39 TFLOP/s while the
  much larger reduction in executed work supplies most of the 1.73x training
  speedup.

## Interpretation

The optimized kernel is far from the global hardware roof, but `28.18x` is not
a credible available speedup target. The global roof assumes that a small,
irregular sparse launch can sustain full-chip BF16 peak and that all compulsory
traffic receives perfect cache reuse. It does not price:

- finite wave/workgroup quantization across 304 CUs;
- short 64x64 matrix tiles and instruction-pipeline fill;
- sparse-list traversal and partial-block mask evaluation;
- repeated K/V loads that miss cache;
- launch, reduction, synchronization, or GQA gradient-accumulation overhead;
- VGPR/LDS occupancy constraints.

The actionable conclusion is narrower: block 64 wins by avoiding matrix work,
not by approaching peak utilization. The next kernel opportunity is to recover
small-tile utilization—especially in forward—while retaining 64-token sparse
selection granularity. Examples include processing multiple sparse entries per
program, amortizing metadata/mask work, or independently tuning forward and
backward traversal rather than making the sparse block coarser again.

