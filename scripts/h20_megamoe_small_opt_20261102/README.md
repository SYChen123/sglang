# H20-3e small-shape SM90 W4A8 MegaMoE experiments

Build the TVM-FFI wheel from the native SM90 register-path commit in
the sibling DeepGEMM repository. Point `PACKAGE` at the unpacked wheel.
The benchmark also requires the Humming package and four idle H20-3e GPUs.

```bash
EXPERIMENT=scripts/h20_megamoe_small_opt_20261102
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH="$PACKAGE" \
  python -m torch.distributed.run --standalone --nproc-per-node=4 \
  "$EXPERIMENT/bench_small.py" --native-register-pack \
  --output "$RESULTS/native" --rows 8 32 --capacity 9600 --sms 78 \
  --tp-baseline --scale-jitter 2 --rounds 7 --iters 30
```

The native packer selects the register kernel and its matching physical
layout. SGLang serving continues to use the ordinary SM90 layout.
The archived experimental packages and their ablation/profiling drivers
still accept the three explicit packing flags. Keep those flags separate
from `--native-register-pack`.

The benchmark compares the max-rank latency of the routed MoE chain,
including pre-dispatch and collectives, under CUDA Graph replay. Router
and shared expert are outside the timed scope. Routing and scale inputs
are synthetic. The observer profile records rank-local GPU times and
does not synchronize GPU clocks.

## What limits the fused path

The fused schedule can remove launches and overlap independent movement.
Per expert, however, L2 consumes the output of L1 and cannot complete before
that activation exists. In the current schedule the A loader waits for the
whole L1 output tile mask, and combine follows the cross-rank scatter fence.
Weight loads, FP4-to-FP8 expansion and WGMMA are still paid for, and producer
warps and stage storage compete for registers, shared memory and occupancy.
Humming can choose independent gate/up and down tactics. Under CUDA Graph
replay its separate kernels also pay less host dispatch overhead.

Measured on the same four-card synthetic workload, the shared-LUT control
was 488/942 us and the fastest Humming was 356/753 us at 8/32 token per rank.
At 8, A producer polling was 1.8–3.2 us on the late CTA per rank, and FP4
unpack occupied 40–70 us on the late warpgroup. Even eliminating the measured
L2 unpack entirely would leave part of the gap. At 32, A polling was
135–456 us in the B observer profile. Wave 12 reduced the waiting sharply,
but reduced chain latency by only 17–28 us in repeated tests because producer
work and wait overlap. The per-tile independent readiness, early-weight
barrier, LUT pre-expansion and active-wave experiments did not improve both
shapes. All completed supply variants matched the shared-LUT control bitwise
on the four-rank test at 8 and 32 rows per rank.

## Next experiments

1. Measure the same post-router chain with fixed EP4 ownership and matched
   packing/precision. Split out useful GPU GEMM work and exposed communication
   by comparing the complete fused path, isolated L1/L2, and Humming EP4 with
   both its original and a compact EP transport. Keep TP4/EP1 as a deployment
   reference since it changes expert ownership.
2. At 8, compare a dedicated small-token expert kernel that shares the packed
   FP4 format and uses an indexed compute schedule. Focus on FP4-to-FP8 issue,
   effective weight bytes per active expert, tile padding and occupancy for
   **both** expert projections. Start with separated stages to establish the
   compute floor before adding overlap.
3. At 32, test producer-driven readiness using an order shared by all CTA roles
   or a separate stage queue. Avoid polling every K block. Keep a whole-mask
   fast path and preserve generation and publication ordering for graph replay.
   Wave 12 is the current scheduling candidate for this shape, with regression
   coverage needed for uneven routes and other sizes.
4. After the compute floor is competitive, prototype per-destination stripe
   readiness for scatter/combine. Compare the overlap against notification,
   reduction and occupancy costs. Validate stable top-k reduction ordering and
   empty-rank replay before a service benchmark.

Original H20 records are under the remote
`/home/workspace/code/sglang/dsv4-logs/1103-megamoe-dispatch-sfb/` experiment
root. See `supply-observer/`, `supply-wave12/`, `supply-ablation/` and
`supply-repeat/`. Keep device traces and compiled wheels out of Git.
