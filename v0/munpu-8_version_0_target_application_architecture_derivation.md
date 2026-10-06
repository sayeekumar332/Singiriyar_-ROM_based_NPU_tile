# µNPU-8 — Version_0
## Target Application and Architecture Derivation

**Status:** Version 0 — Architecture Baseline  
**Project:** µNPU-8  
**Focus:** µROM-controlled INT8 GEMV accelerator for LLM autoregressive decode

---

## 1. Executive Summary

µNPU-8 Version 0 is a small programmable neural-processing accelerator targeting **LLM autoregressive decode**.

The design isolates one representative kernel:

`Y = XW`

where `X` is a single-token activation vector(represented as 1 x K vector), `W` is a learned weight matrix (represented as K x N), and `Y` is the output vector (represented as 1 X N vector). This is effectively an INT8 **GEMV** operation.

                                  So 1 x N = (1 x K) x (K x N)

                                   K = number of elements or features in X ; N = number of elements or features in Y
Representative workload:

- `K = N = 4096`
- INT8 input activations
- INT8 input weights
- INT32 output accumulation
- `B = 1, 2, 4, 8` concurrent decode requests

The central architectural hypothesis is:

> During concurrent LLM decode, a weight tile can be loaded once into a small local buffer and reused across multiple active requests, reducing external weight-memory traffic and potentially improving effective compute utilization.

The Version 0 architecture therefore combines:

- a **4×4 parallel INT8 MAC array**
- a **4 KiB candidate local weight buffer**
- local activation/output storage
- **µROM + µPC + field decoder**
- weight-tile-major execution
- cross-request weight reuse

The µROM is specifically used as a programmable **data-reuse and execution scheduler**.

---

# 2. Target Application

## 2.1 LLM Autoregressive Decode

LLM inference can broadly be viewed as:

1. Prefill
2. Autoregressive decode

Version 0 focuses only on decode.

For a simplified linear layer:

`Y = XW`

with:

`X ∈ R^(1×K)`

`W ∈ R^(K×N)`

`Y ∈ R^(1×N)`

each output is:

`y[j] = sum(k=0..K-1) X[k] * W[k,j]`

For one token this has the computational structure of GEMV.

---

## 2.2 Why This Workload Is Interesting

For `K = N = 4096`:

- Weight count = `4096 × 4096 = 16,777,216`
- INT8 weight storage = **16 MiB**
- MAC count per request = **16.78 million MACs**

Multiple concurrent requests use the same model weights.

A request-by-request implementation can therefore repeatedly stream the same weight matrix.

If `B` compatible requests share the same weights, ideal weight traffic changes from approximately:

`B × K × N`

to:

`K × N`

The ideal traffic reduction is:

`R = 1 - 1/B`

| Concurrent Requests | Baseline Weight Traffic | Ideal Reuse Traffic | Ideal Reduction |
|---:|---:|---:|---:|
| 1 | 16 MiB | 16 MiB | 0% |
| 2 | 32 MiB | 16 MiB | 50% |
| 4 | 64 MiB | 16 MiB | 75% |
| 8 | 128 MiB | 16 MiB | 87.5% |

These are **ideal weight-traffic reductions**, not guaranteed end-to-end speedups.

---

# 3. Architecture Derivation

## 3.1 Workload → Bottleneck

The representative layer combines:

- large INT8 weight storage
- substantial MAC computation
- relatively small per-token activation data
- repeated use of identical weights across concurrent requests

The Version 0 bottleneck hypothesis is therefore:

> **Repeated movement of model weights can limit useful compute throughput during autoregressive decode.**

This is a hypothesis to be experimentally validated.

---

## 3.2 Bottleneck → Optimization

Instead of:

```text
Request 0
    Load W
    Compute

Request 1
    Load W
    Compute

Request 2
    Load W
    Compute
```

Version 0 proposes:

```text
Load W tile
     |
     +---- Request 0
     +---- Request 1
     +---- Request 2
     +---- Request 3
     |
Load next W tile
```

The corresponding loop transformation is:

```text
Baseline:
for request
    for weight_tile
        compute

Proposed:
for weight_tile
    for request
        compute
```

The second ordering is the fundamental cross-request reuse mechanism.

---

# 4. Compute Architecture

## 4.1 4×4 INT8 MAC Array

Version 0 uses:

`4 × 4 = 16`

parallel INT8 MAC units.

Peak arithmetic throughput:

**16 MAC/cycle**

The block is intentionally described as a **parallel MAC array**, not a systolic array. A systolic interconnect will only be introduced if analysis demonstrates a measurable benefit for this GEMV workload.

---

## 4.2 4×4 Compute Tile

A compute tile contains:

`W_t ∈ R^(4×4)`

and:

`X_t = [x0, x1, x2, x3]`

The array computes four partial output values:

`y[j]_partial = sum(i=0..3) X[i] × W[i,j]`

Conceptually:

```text
             W0   W1   W2   W3
          +----+----+----+----+
 x0 ----> |MAC |MAC |MAC |MAC |
          +----+----+----+----+
 x1 ----> |MAC |MAC |MAC |MAC |
          +----+----+----+----+
 x2 ----> |MAC |MAC |MAC |MAC |
          +----+----+----+----+
 x3 ----> |MAC |MAC |MAC |MAC |
          +----+----+----+----+
             |    |    |    |
             v    v    v    v
             Y0   Y1   Y2   Y3
```

---

# 5. Accumulation

Each output must accumulate across all `K = 4096` input elements.

There are:

`4096 / 4 = 1024`

4-element K groups.

Thus one output group is accumulated across **1024 4×4 compute operations**.

The conceptual PE operation is:

`ACC[i,j] <- ACC[i,j] + X[i] × W[i,j]`

INT32 accumulation state remains local while the K dimension is traversed.

---

# 6. Ideal Compute Latency

Total MACs:

`4096 × 4096 = 16,777,216`

At 16 MAC/cycle:

`16,777,216 / 16 = 1,048,576 cycles`

Therefore:

> **Ideal arithmetic lower bound = 1,048,576 cycles per 4096×4096 request.**

This is **not yet the final worst-case functional latency**.

It excludes:

- weight-buffer refills
- activation loads
- output stores
- memory stalls
- control overhead
- pipeline fill/drain
- SRAM banking bubbles

The final worst-case functional latency will be derived from the cycle-accurate RTN and frozen memory/controller implementation.

---

# 7. Local Weight Buffer

## 7.1 Purpose

The local weight buffer separates external memory traffic from high-bandwidth MAC feeding.

Conceptually:

```text
External Memory
      |
      | Load once
      v
Local Weight Buffer
      |
      +---- Request 0
      +---- Request 1
      +---- Request 2
      +---- Request 3
```

---

## 7.2 Candidate Sizes

| Resident Weight Tile | Storage |
|---|---:|
| 32×32 | 1 KiB |
| **64×64** | **4 KiB** |
| 128×64 | 8 KiB |

Version 0 uses **4 KiB** as the initial architectural design point.

This is an experimental choice, not a claim of global optimality.

---

## 7.3 Why 64×64?

A 64×64 INT8 tile contains:

`64 × 64 = 4096 bytes = 4 KiB`

It contains:

- `64 / 4 = 16` K subtiles
- `64 / 4 = 16` output subtiles
- `16 × 16 = 256` logical 4×4 compute tiles

This gives a useful reuse window while keeping the Gen-0 design small.

---

# 8. Local Memory Bandwidth

A 4×4 compute operation consumes:

### Activation

4 INT8 values:

**4 bytes/cycle**

### Weight

16 INT8 values:

**16 bytes/cycle**

Therefore the local compute interface needs approximately:

**20 operand bytes/cycle**

to sustain the ideal 16-MAC/cycle datapath.

This is a **local-buffer bandwidth requirement**, not necessarily an external-memory bandwidth requirement.

The whole point of the weight buffer is to avoid requiring external memory to continuously supply the same weights for every request.

---

# 9. Weight-Stationary Execution

Version 0 uses a weight-tile-major schedule:

```text
LOAD_WEIGHT_TILE

for request = 0 .. B-1
    LOAD_ACTIVATION
    COMPUTE
end

NEXT_WEIGHT_TILE
```

The defining property is:

> **A resident weight tile remains available while multiple concurrent requests consume it.**

The exact nesting of output tiles and K tiles will be finalized during RTN derivation.

---

# 10. µROM Control Architecture

The µROM is not present merely as a hardwired-control replacement.

Its purpose is to encode the programmable **data-movement and compute schedule**.

Conceptually:

```text
             +--------------------+
             |        µROM        |
             |   Microprogram     |
             +---------+----------+
                       |
                      µPC
                       |
                Field Decoder
                       |
       +---------------+---------------+
       |               |               |
       v               v               v
 Weight Control   Activation Control  MAC Control
```

Representative operations:

```text
LOAD_WEIGHT_TILE
WAIT_WEIGHT_READY
CLEAR_ACC
LOAD_ACTIVATION
COMPUTE
ACCUMULATE
NEXT_K_TILE
NEXT_REQUEST
STORE_RESULT
DONE
```

The final microinstruction width and field allocation will be derived from the RTN.

---

# 11. Architectural Block Diagram

```text
                         LLM Decode Request(s)
                                  |
                                  v
                       +----------------------+
                       | Activation Interface |
                       +----------+-----------+
                                  |
                                  v
                       +----------------------+
                       | Activation Buffer    |
                       +----------+-----------+
                                  |
                +-----------------+----------------+
                |                                  |
                |            µROM Control          |
                |       +--------------------+      |
                +------>| µPC + Field Decode |      |
                        +---------+----------+      |
                                  |                 |
                                  v                 |
                       +----------------------+     |
                       | Weight Load Control  |     |
                       +----------+-----------+     |
                                  |                 |
External Weight Memory            |                 |
        |                         v                 |
        |                +------------------+       |
        +--------------->| 4 KiB Weight     |       |
                         | Local Buffer     |       |
                         +--------+---------+       |
                                  |                 |
                                  v                 |
                         +------------------+       |
                         | 4×4 INT8 MAC     |<------+
                         | Array            |
                         +--------+---------+
                                  |
                                  v
                         +------------------+
                         | INT32 Accumulators|
                         +--------+---------+
                                  |
                                  v
                         +------------------+
                         | Output Buffer    |
                         +--------+---------+
                                  |
                                  v
                           Output Interface
```

---

# 12. Baseline vs Proposed

## Baseline B0

```text
Request 0
   |
Load W
   |
Compute
   |
Request 1
   |
Load W
   |
Compute
   |
...
```

Ideal weight traffic:

`B × K × N`

## Proposed R1

```text
Load W tile
     |
     +---- Compute Request 0
     +---- Compute Request 1
     +---- Compute Request 2
     +---- Compute Request 3
     |
Evict W tile
     |
Load next W tile
```

Ideal weight traffic:

`K × N`

for a group of `B` compatible requests.

---

# 13. Advantages

### Reduced external weight traffic

Ideal reduction:

`R = 1 - 1/B`

At B=4:

**75%**

At B=8:

**87.5%**

These figures describe weight traffic only.

### Small compute engine

Only 16 MAC units are required, keeping Gen-0 tractable for RTL, verification, synthesis, timing, and PPA exploration.

### Programmable dataflow

The µROM permits experiments with:

- request-major scheduling
- weight-major scheduling
- tile ordering
- prefetching
- double buffering

without redesigning the fundamental datapath.

### Strong baseline comparison

The project naturally provides a B0→R1 experiment:

- external weight bytes
- compute utilization
- total latency
- local-buffer overhead
- PPA

---

# 14. Limitations and Design Trade-offs

## No benefit at B=1

Cross-request reuse provides no benefit for a single request.

## Reuse depends on scheduling

The ideal reduction assumes compatible requests can be grouped and scheduled together.

Real serving systems have dynamic arrivals, different sequence lengths, scheduling constraints, and latency/SLO requirements.

## Buffer cost

Larger local storage increases area, routing, bandwidth, and control complexity.

## Compute can become the bottleneck

Reducing weight traffic does not guarantee proportional latency improvement. Once memory pressure is reduced, the 4×4 compute array may become the limiting resource.

## Local SRAM bandwidth can become the next bottleneck

The MAC array requires approximately 20 operand bytes/cycle. Insufficient banking or ports can introduce bubbles and reduce utilization.

---

# 15. Architecture Decision Matrix

| Decision | Version 0 |
|---|---|
| Target | LLM autoregressive decode |
| Kernel | INT8 GEMV |
| Matrix | 4096×4096 |
| Requests | B=1/2/4/8 |
| Compute | 4×4 MAC |
| Peak | 16 MAC/cycle |
| Accumulation | INT32 |
| Dataflow | Weight-tile-major |
| Reuse | Cross-request |
| Weight buffer | 4 KiB candidate |
| Activation buffer | Local |
| Output state | Local INT32 |
| Control | µROM + µPC + decoder |
| Baseline | Request-major weight streaming |
| Proposed | Weight-major reuse |
| Final µinstruction width | To be derived |
| SRAM banking | To be derived |
| Double buffering | To be evaluated |
| Final cycle latency | To be derived from RTN |

---

# 16. Frozen vs Unfrozen

## Frozen at the Architectural Level

1. LLM autoregressive decode
2. INT8 GEMV
3. 4096×4096 representative layer
4. 4×4 parallel INT8 MAC array
5. INT32 accumulation
6. Cross-request weight reuse
7. Weight-tile-major execution
8. 4 KiB local weight buffer as initial design point
9. µROM + µPC + field decoder
10. B=1/2/4/8 evaluation points

## Not Yet Frozen

1. SRAM banking
2. SRAM port organization
3. Activation-buffer implementation
4. Output-buffer implementation
5. PE interconnect
6. Pipeline registers
7. Double buffering
8. External memory protocol
9. µinstruction width
10. µinstruction field allocation
11. Loop-controller implementation
12. Final cycle-by-cycle latency

These are deliberately derived from the RTN and performance model.

---

# 17. Development Methodology

```text
Real workload
     |
     v
Workload characterization
     |
     v
Bottleneck hypothesis
     |
     v
Analytical model
     |
     v
Architecture
     |
     v
RTN
     |
     v
Microinstruction definition
     |
     v
Microarchitecture
     |
     v
SystemVerilog RTL
     |
     v
Verification
     |
     v
Synthesis / STA
     |
     v
PPA measurement
     |
     v
Architecture refinement
```

Open-source accelerators and published research may be used as architectural references when a measured bottleneck warrants investigation. They will not be copied blindly.

---

# 18. Version 0 Success Criteria

Version 0 should demonstrate:

### Functional correctness

`Y_RTL = Y_Golden`

for supported configurations.

### Weight-traffic reduction

Measurable reduction relative to the request-major baseline.

### Compute utilization

Fraction of cycles in which the 4×4 array performs useful MACs.

### Latency

Measured `START → DONE` latency in cycles.

### PPA

Measure:

- area
- critical path
- timing slack / Fmax
- buffer overhead

### Trade-off analysis

Compare:

- B=1/2/4/8
- buffer sizes
- dataflow choices
- baseline vs proposed

---

# 19. Worst-Case Functional Latency Status

The final worst-case functional latency is **not yet frozen** because the exact memory interface, banking, microprogram, and pipeline schedule are not yet defined.

Current arithmetic lower bound for one 4096×4096 request:

**1,048,576 cycles**

at an ideal sustained rate of:

**16 MAC/cycle**

The final RTL specification must explicitly report:

`L_WC = L_load + L_compute + L_activation + L_store + L_control + L_pipeline`

or the appropriate overlapped equivalent once memory/compute overlap is implemented.

---

# 20. Next Phase

## Cycle-Accurate RTN Derivation

The next phase will derive:

1. 4×4 tile execution
2. 64×64 resident weight-tile traversal
3. K-loop behavior
4. Output-tile traversal
5. Request-loop behavior
6. Weight-buffer addressing
7. Activation-buffer addressing
8. INT32 accumulator lifetime
9. µPC transitions
10. WAIT/STALL behavior
11. LOAD/COMPUTE/STORE timing
12. Exact worst-case functional latency in cycles

Only after this should the µinstruction width and RTL structure be frozen.

---

## Version 0 Architectural Thesis

> **µNPU-8 is a small programmable INT8 GEMV accelerator for LLM autoregressive decode. Its defining architectural optimization is to retain weight tiles in a local buffer and reuse them across concurrent decode requests. A µROM-controlled execution schedule separates the data-reuse policy from the fixed 4×4 INT8 MAC datapath, enabling systematic experimentation with weight tiling, request interleaving, and memory/computation scheduling.**
