# µNPU-8
## Programmable Multi-Clock INT8 Matrix-Multiplication Neural Processing Unit

**Document Status:** Architecture Specification — Baseline v1.0  
**Implementation Target:** 15-day RTL/verification project  
**Initial Compute Configuration:** 4×4 INT8 MAC array  
**Architectural Scaling Target:** Multi-tile AI accelerator

---

## 1. Purpose

µNPU-8 is a compact, synthesizable neural-processing accelerator designed to demonstrate the architectural and RTL-design principles used in modern AI accelerator ASICs.

The baseline accelerator executes tiled INT8 matrix multiplication:

**C = A × B**

where:

- A contains signed INT8 operands.
- B contains signed INT8 operands.
- C contains signed INT32 accumulated results.

The architecture combines:

- programmable microcoded control,
- a 4×4 systolic MAC array,
- local scratchpad memories,
- multiple independent clock domains,
- CDC-safe command and data transfer,
- independent reset domains,
- cycle-accurate execution,
- functional verification,
- synthesis/timing analysis,
- and measurable PPA characteristics.

The architecture is intentionally divided into a **minimal implementable NPU tile** and a **scalable system architecture** so that the baseline can be completed within approximately 15 days while preserving a credible path toward substantially larger AI-compute systems.

---

## 2. Architectural Goals

The primary goals are:

1. Provide a programmable accelerator for tiled INT8 GEMM.
2. Demonstrate hardware/software command sequencing.
3. Separate control and compute functions into independent clock domains.
4. Demonstrate robust CDC and RDC handling.
5. Exploit spatial and temporal data reuse through a systolic array.
6. Minimize unnecessary external-memory traffic.
7. Provide deterministic cycle-level execution behavior.
8. Enable quantitative latency and throughput analysis.
9. Enable synthesis-based area and timing analysis.
10. Demonstrate PPA-oriented architectural trade-offs.
11. Establish a reusable NPU compute-tile abstraction.
12. Preserve a clear architectural path toward multi-tile AI acceleration.

---

## 3. Architectural Scope

The architecture is divided into two levels.

### 3.1 Baseline implementation

```text
1 tile
4×4 MAC array
INT8 × INT8 → INT32
Local scratchpad memories
Microcoded controller
3 clock domains
CDC/RDC
GEMM execution
Verification
Synthesis
Timing analysis
PPA analysis
```

### 3.2 Future scalable architecture

The architectural boundary additionally permits:

```text
Multiple NPU tiles
Tile-to-tile communication
NoC
Distributed memory
Larger systolic arrays
Additional tensor operations
Collective operations
Advanced scheduling
Multiple numerical precisions
High-bandwidth memory
```

These future capabilities are architectural targets and are **not part of the 15-day baseline implementation**.

---

## 4. Top-Level Baseline Architecture

```text
                         ┌──────────────────────────────┐
                         │        HOST / SYSTEM         │
                         │          clk_host            │
                         │                              │
                         │   Command / Status IF        │
                         └──────────────┬───────────────┘
                                        │
                                Host → Control CDC
                                        │
                                  Async Command FIFO
                                        │
                                        ▼
                         ┌──────────────────────────────┐
                         │       CONTROL DOMAIN         │
                         │          clk_ctrl            │
                         │                              │
                         │  Command Processor           │
                         │  Microcode Sequencer         │
                         │  Configuration Registers     │
                         │  Execution Controller        │
                         └──────────────┬───────────────┘
                                        │
                                Control / Event CDC
                                        │
                                        ▼
                         ┌──────────────────────────────┐
                         │       COMPUTE DOMAIN         │
                         │        clk_compute           │
                         │                              │
                         │       Data FIFO              │
                         │          │                   │
                         │          ▼                   │
                         │    ┌───────────────┐         │
                         │    │ 4 × 4 Systolic│         │
                         │    │   MAC Array   │         │
                         │    └───────┬───────┘         │
                         │            │                 │
                         │      INT32 Accumulators      │
                         │            │                 │
                         │            ▼                 │
                         │      C Scratchpad            │
                         └──────────────────────────────┘
```

---

## 5. NPU Tile Abstraction

The most important architectural abstraction is the **NPU Tile**.

The baseline implementation contains exactly one tile:

```text
NPU_TILE[0]
```

The tile contains:

```text
┌─────────────────────────────────┐
│             NPU TILE            │
│                                 │
│  Microcode Controller           │
│          │                      │
│          ▼                      │
│  CDC / Data Movement            │
│          │                      │
│          ▼                      │
│  4×4 Systolic MAC Array         │
│          │                      │
│          ▼                      │
│  Accumulator / Scratchpad       │
│                                 │
└─────────────────────────────────┘
```

The tile boundary shall remain stable as the architecture scales.

This means that future scaling can replicate:

```text
NPU_TILE[0]
NPU_TILE[1]
NPU_TILE[2]
...
NPU_TILE[N]
```

without fundamentally redesigning the internal compute engine.

---

## 6. Computational Model

The fundamental operation is:

**C[i][j] = Σ A[i][k] × B[k][j]**

with:

```text
A          : signed INT8
B          : signed INT8
Product    : signed INT16
Accumulator: signed INT32
C          : signed INT32
```

Initial workload:

```text
M = 8
K = 8
N = 8
```

Initial compute array:

```text
4 × 4 = 16 PEs
```

The 8×8×8 GEMM is therefore decomposed into 4×4 tiles.

---

## 7. Processing Element

Each PE performs:

**ACC ← ACC + A × B**

Conceptually:

```text
             A operand
                 │
                 ▼
            ┌─────────┐
B operand → │ INT8 ×  │
            │  INT8   │
            └────┬────┘
                 │
               INT16
                 │
                 ▼
            ┌─────────┐
            │  INT32  │
            │  ACC    │
            └─────────┘
```

Each PE provides:

- signed INT8 multiplication,
- INT32 accumulation,
- operand forwarding,
- accumulator clear,
- valid/control handling.

---

## 8. Systolic Compute Array

Sixteen PEs form a 4×4 systolic array.

```text
             A0       A1       A2       A3
              │        │        │        │
              ▼        ▼        ▼        ▼

B0 ───────► PE00 ───► PE01 ───► PE02 ───► PE03
              │        │        │        │
              ▼        ▼        ▼        ▼
B1 ───────► PE10 ───► PE11 ───► PE12 ───► PE13
              │        │        │        │
              ▼        ▼        ▼        ▼
B2 ───────► PE20 ───► PE21 ───► PE22 ───► PE23
              │        │        │        │
              ▼        ▼        ▼        ▼
B3 ───────► PE30 ───► PE31 ───► PE32 ───► PE33
```

The array provides:

- spatial data reuse,
- local partial-sum storage,
- predictable execution,
- scalable PE replication.

The array dimensions shall be treated as architectural parameters even though the baseline implementation uses 4×4.

---

## 9. Scalable Compute-Array Model

The architectural abstraction is:

```text
PE_ARRAY_ROWS
PE_ARRAY_COLS
```

The baseline is:

```text
PE_ARRAY_ROWS = 4
PE_ARRAY_COLS = 4
```

Potential future configurations include:

```text
4×4
8×8
16×16
32×32
```

Scaling the array increases:

- compute throughput,
- area,
- local data bandwidth requirements,
- routing complexity,
- power,
- timing pressure.

Therefore compute scaling must be accompanied by corresponding memory and interconnect scaling.

---

## 10. Scratchpad Memory Architecture

The baseline contains three logical scratchpads:

```text
A Scratchpad
B Scratchpad
C Scratchpad
```

Initial capacities:

```text
A : 256 × 8-bit
B : 256 × 8-bit
C : 256 × 32-bit
```

The basic movement is:

```text
External/System Memory
          │
          │ LOAD
          ▼
   Local Scratchpad
          │
          │ COMPUTE
          ▼
      MAC Array
          │
          │ STORE
          ▼
   Local C Scratchpad
          │
          │ STORE
          ▼
External/System Memory
```

The scratchpad abstraction is intentionally explicit so that future architectures can replace or augment it with larger hierarchical memory systems.

---

## 11. Future Memory Hierarchy

The architecture reserves a hierarchical memory model:

```text
              Global / External Memory
                       │
                       ▼
                Shared Memory
                       │
                       ▼
                  Tile SRAM
                       │
                       ▼
                 PE-local data
                       │
                       ▼
                    MAC PE
```

The baseline implements only the tile-local memory level.

Future systems may introduce:

- larger shared SRAM,
- distributed SRAM,
- HBM,
- DRAM,
- cache-like structures,
- software-managed tensor buffers.

The compute tile must remain independent of the implementation details of the future global memory system.

---

## 12. Control Architecture

The control domain contains:

```text
Command FIFO
     │
     ▼
Command Processor
     │
     ▼
Microcode Sequencer
     │
     ├── Microcode ROM
     ├── µPC
     └── Execution Controller
```

Initial microoperations:

```text
µOP_NOP
µOP_LOAD
µOP_MOVE
µOP_CLEAR
µOP_COMPUTE
µOP_ACCUMULATE
µOP_STORE
µOP_SYNC
µOP_WAIT
µOP_DONE
```

The instruction set is deliberately expressed in terms of **tensor movement and execution primitives**, rather than hard-coding the accelerator around one specific neural-network operator.

---

## 13. Programmability Model

The architectural separation is:

```text
WHAT is computed
       │
       ▼
Compute datapath

WHEN it is computed
       │
       ▼
Microcode

WHERE data resides
       │
       ▼
Memory/data-movement controls
```

This allows the same datapath to execute different sequences without modifying the compute RTL.

GEMM is the baseline workload, while future workloads may use the same primitives for:

- matrix-vector operations,
- convolution,
- projection,
- attention-related matrix operations,
- other tensor kernels.

---

## 14. Clock-Domain Architecture

Three independent clock domains are mandatory in the baseline:

```text
clk_host
clk_ctrl
clk_compute
```

Responsibilities:

| Domain | Responsibility |
|---------------|---------------------------------|
| `clk_host`    | System command/status interface |
| `clk_ctrl`    | Scheduling and microcode        |
| `clk_compute` | Tensor computation              |

Initial verification frequencies:

```text
clk_host     = 100 MHz
clk_ctrl     = 200 MHz
clk_compute  = 400 MHz
```

The verification environment shall also exercise non-harmonically related clock periods.

---

## 15. CDC Architecture

CDC is a first-class architectural subsystem.

### Host → Control

```text
HOST
 │
 ▼
Async Command FIFO
 │
 ▼
CONTROL
```

### Control → Compute

```text
CONTROL
 │
 ▼
CDC-safe command/event transfer
 │
 ▼
COMPUTE
```

### Compute → Control

```text
COMPUTE
 │
 ▼
CDC-safe completion/status transfer
 │
 ▼
CONTROL
```

### Tensor/Data Transfer

```text
CONTROL
 │
 ▼
Async Data FIFO
 │
 ▼
COMPUTE
```

The architecture therefore demonstrates multiple CDC mechanisms rather than relying exclusively on simple two-flop synchronization.

---

## 16. Reset Architecture

Independent resets are defined:

```text
rst_host_n
rst_ctrl_n
rst_compute_n
```

The architecture must define behavior for independent reset events.

In particular:

```text
compute reset
      ↓
control must not interpret stale COMPUTE_DONE
```

and:

```text
control reset
      ↓
compute must not execute an unintended stale command
```

Reset release behavior is therefore treated as part of the CDC/RDC architecture.

---

## 17. Tile Interface

Each NPU tile shall conceptually expose five classes of interfaces:

```text
COMMAND
DATA-IN
DATA-OUT
CONTROL
STATUS
```

A future tile-to-tile interface can therefore be introduced without redesigning the internal MAC array.

Conceptually:

```text
             ┌──────────────┐
COMMAND ────►│              │
DATA-IN ────►│   NPU TILE   │───► DATA-OUT
CONTROL ────►│              │───► STATUS
             └──────────────┘
```

The baseline may implement these using simple valid/ready protocols.

Future implementations may map the same logical interface to a NoC.

---

## 18. Future NoC Boundary

A Network-on-Chip is **not implemented in the 15-day baseline**.

However, the architecture explicitly reserves the following boundary:

```text
                NPU TILE
                   │
             Tile Interface
                   │
                   ▼
              Future NoC
```

This allows future scaling to:

```text
        ┌─────────┐
        │  NoC    │
        └────┬────┘
             │
     ┌───────┼────────┐
     ▼       ▼        ▼
   TILE0   TILE1    TILE2
```

The NoC may eventually provide:

- tile-to-tile communication,
- memory requests,
- tensor transfers,
- synchronization,
- broadcast,
- reduction,
- routing.

---

## 19. Tile Identification

Every tile has a logical `TILE_ID`.

Baseline:

```text
TILE_ID = 0
```

Future system:

```text
TILE_ID ∈ [0, N-1]
```

A future global address may conceptually contain:

```text
GLOBAL_ADDRESS
      │
      ├── TILE_ID
      ├── BANK_ID
      └── LOCAL_OFFSET
```

This provides a scalable addressing abstraction without requiring a NoC implementation in the baseline.

---

## 20. Synchronization Model

The scalable architecture reserves explicit synchronization primitives:

```text
µOP_WAIT
µOP_SIGNAL
µOP_BARRIER
µOP_SYNC
```

This supports future coordinated execution:

```text
TILE 0 ─────┐
TILE 1 ─────┤
TILE 2 ─────┼──► BARRIER
TILE 3 ─────┘
```

The baseline may use these primitives only for local synchronization.

---

## 21. Future Collective Operations

For multi-tile scaling, the architectural programming model reserves:

```text
BROADCAST
GATHER
SCATTER
REDUCE
REDUCE_SUM
ALL_REDUCE
BARRIER
```

These operations are not part of the initial RTL implementation.

They are architectural extensions intended to support distributed tensor computation.

---

## 22. Scaling Hierarchy

The architecture is defined across four generations.

### Generation 0 — µNPU-8 baseline

```text
1 tile
4×4 MAC
INT8
INT32 accumulation
3 clock domains
CDC/RDC
local SRAM
microcode
GEMM
```

### Generation 1 — Larger tile

```text
1 tile
8×8 / 16×16 MAC
larger local memory
additional precision modes
additional tensor operations
```

### Generation 2 — Multi-tile accelerator

```text
Multiple NPU tiles
Tile-to-tile communication
Distributed memory
Synchronization
Collective operations
NoC
```

### Generation 3 — Large-scale AI compute system

```text
Large tile count
High-bandwidth memory
Advanced NoC
Distributed scheduling
Multiple numerical precisions
Large-model execution
System-level memory hierarchy
```

Only **Generation 0** is within the 15-day implementation commitment.

---

## 23. Numerical Precision Scalability

The baseline uses:

```text
INT8 × INT8 → INT32
```

The architecture should avoid making the PE interface permanently dependent on INT8.

Future configurations may support:

```text
INT4
INT8
INT16
BF16
FP16
```

where appropriate.

The precision extension should occur primarily within the compute datapath while preserving the higher-level tile programming and memory interfaces.

---

## 24. Workload Scalability

GEMM is the first workload because it provides a compact and measurable demonstration of tensor acceleration.

The long-term workload abstraction is:

```text
Tensor Workload
      │
      ├── GEMM
      ├── MatVec
      ├── Convolution
      ├── Projection
      ├── Attention operations
      └── Future tensor kernels
```

The architecture therefore separates:

```text
Tensor operation
```

from:

```text
Physical compute implementation
```

---

## 25. Performance Model

The following metrics are mandatory.

### Functional latency

```text
START accepted → DONE
```

reported in **cycles**.

### Throughput

```text
completed MAC operations / compute cycle
```

### PE utilization

```text
active PE cycles
──────────────────── × 100
available PE cycles
```

### Memory traffic

Measure:

```text
bytes transferred
per GEMM
```

and:

```text
bytes / MAC
```

### CDC overhead

Measure:

```text
command-transfer latency
data-transfer latency
completion-transfer latency
```

The performance model must distinguish pure compute latency from system-level transfer overhead.

---

## 26. Worst-Case Functional Latency

The final implementation shall report:

```text
START → DONE = N cycles
```

for each supported workload/configuration.

The latency shall be decomposed into:

```text
Command acceptance
+
Host→Control CDC
+
Operand loading
+
Accumulator initialization
+
Systolic computation
+
Tile transitions
+
Result storage
+
Compute→Control completion CDC
=
Worst-case functional latency
```

The final value shall be measured from RTL simulation.

---

## 27. PPA Analysis

The architecture shall support quantitative analysis of:

```text
Area
Power proxy
Performance
Timing
```

At minimum:

```text
Gate/cell count
Register count
Combinational logic
Critical path
Slack
Estimated Fmax
```

The design shall compare at least two microarchitectural configurations.

The purpose is to demonstrate the relationship:

```text
Architecture
     ↓
Microarchitecture
     ↓
RTL
     ↓
PPA
```

---

## 28. Verification Architecture

Verification shall cover:

```text
Functional correctness
CDC correctness
RDC correctness
Protocol correctness
Microcode correctness
Performance correctness
```

The verification environment will contain:

```text
Python golden model
Directed tests
Randomized tests
Scoreboard
SystemVerilog assertions
Functional coverage
CDC stress tests
Reset tests
```

Clock frequencies and relative phases shall be varied independently.

---

## 29. Scalability Verification Philosophy

The verification environment should be parameterized around:

```text
PE rows
PE columns
matrix dimensions
clock frequencies
tile count
```

where practical.

The baseline verifies:

```text
1 tile
4×4 array
8×8×8 GEMM
```

Future regression configurations may verify:

```text
1 tile / 8×8
1 tile / 16×16
multiple tiles
```

without fundamentally replacing the verification methodology.

---

## 30. Architectural Error Conditions

The architecture defines behavior for:

- invalid microinstruction,
- command FIFO overflow,
- command FIFO underflow,
- data FIFO overflow,
- data FIFO underflow,
- illegal matrix dimensions,
- compute requested while busy,
- reset during computation,
- reset during CDC transfer,
- invalid tile identifier,
- synchronization timeout.

The baseline implementation may use a compact error/status mechanism.

---

## 31. Artificial-Intelligence-System Scaling Philosophy

The purpose of scalability is not to claim that µNPU-8 is itself an artificial-superintelligence machine.

Instead, the architecture is designed around the principle:

```text
Compute Tile
     ↓
Replicated Compute Tiles
     ↓
Connected Accelerator
     ↓
Distributed AI Compute Fabric
```

A future large AI system would require substantially more than MAC throughput.

Its architecture would also require:

```text
Compute
+
Memory capacity
+
Memory bandwidth
+
Interconnect
+
Scheduling
+
Synchronization
+
Programmability
+
Numerical precision
+
Reliability
+
Compiler/software support
```

µNPU-8 therefore serves as a **scalable compute-tile prototype**, not as a claim of complete AGI/ASI hardware.

---

## 32. Architectural Design Principle

The fundamental design principle is:

**Scale the architecture by replication and hierarchy rather than by rewriting the compute engine.**

Therefore:

```text
Small System

       ┌─────────┐
       │ NPU TILE│
       └─────────┘
```

can evolve into:

```text
Medium System

 ┌─────────┐   ┌─────────┐
 │ TILE 0  │───│ TILE 1  │
 └─────────┘   └─────────┘
```

and ultimately:

```text
Large System

       ┌───────────────────┐
       │       NoC         │
       └───────────────────┘
        │   │   │   │   │
       T0  T1  T2  T3  ... TN
```

while preserving the conceptual NPU tile abstraction.

---

## 33. Baseline Implementation Boundary

The following features are **mandatory**:

```text
✓ 4×4 systolic MAC
✓ INT8 × INT8 → INT32
✓ Scratchpad memories
✓ Microcoded control
✓ Three clock domains
✓ Async command/data transfer
✓ CDC synchronization
✓ Independent reset domains
✓ CDC/RDC verification
✓ 8×8 GEMM
✓ Python golden model
✓ SVA
✓ Cycle-level performance model
✓ Yosys synthesis
✓ OpenSTA timing analysis
✓ PPA comparison
```

The following are **architecturally reserved but not mandatory**:

```text
○ 8×8 / 16×16 array
○ Multiple tiles
○ NoC
○ HBM
○ Distributed memory
○ Collective operations
○ INT4/BF16/FP16
○ Transformer-specific operators
○ Global scheduler
```

---

## 34. Architectural Deliverables

The project shall produce:

1. Architectural specification.
2. Top-level block diagram.
3. NPU tile specification.
4. Compute-array specification.
5. Memory hierarchy specification.
6. Microcode programming model.
7. Clock-domain architecture.
8. CDC architecture.
9. RDC/reset architecture.
10. Tile interface specification.
11. Scalability specification.
12. Canonical RTN.
13. Microoperation specification.
14. RTL implementation.
15. Verification plan.
16. Performance model.
17. Synthesis/timing analysis.
18. PPA analysis.
19. Final interview presentation.

---

## 35. Design Flow

The project shall follow:

```text
                    AI WORKLOAD
                         │
                         ▼
                  ARCHITECTURE
                         │
                         ▼
                MICROARCHITECTURE
                         │
                         ▼
                       RTN
                         │
                         ▼
                   MICROOPS
                         │
                         ▼
                       RTL
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
          FUNCTIONAL             CDC/RDC
          VERIFICATION           VERIFICATION
              │                     │
              └──────────┬──────────┘
                         ▼
                     SYNTHESIS
                         │
                         ▼
                       STA
                         │
                         ▼
                       PPA
                         │
                         ▼
                  PERFORMANCE
                         │
                         ▼
              SCALABILITY ANALYSIS
```

---

## 36. Final Architectural Statement

µNPU-8 is a **programmable, multi-clock, CDC-aware INT8 neural-processing tile** designed as the smallest practical implementation of a scalable AI accelerator architecture.

The 15-day implementation focuses on one fully functional 4×4 compute tile executing 8×8×8 INT8 GEMM.

The architecture deliberately establishes stable abstractions for:

- compute,
- memory,
- control,
- microcode,
- CDC,
- synchronization,
- tile identification,
- communication,
- and performance measurement.

These abstractions permit future expansion from a single compute tile into a multi-tile AI accelerator connected through a scalable interconnect and hierarchical memory system.

The baseline therefore demonstrates complete RTL-design competence while preserving a technically credible path toward much larger AI-compute architectures.
