# MPI-Based Parallel Computing

> Implementations of parallel communication and computation patterns in C using MPI, benchmarked on the **PARAM Rudra** HPC cluster. Covers deadlock-free point-to-point pipelines, 3D stencil computation with communication–computation overlap, and scalability analysis across multi-node deployments.

---

## Assignment 1 — Point-to-Point MPI Pipeline

### Overview
Simulates a parallel data-processing pipeline using **strictly point-to-point MPI** communication. Each process operates on a linear topology, acting simultaneously as a sender (to ranks D1 and D2 distances away) and a receiver, performing element-wise computations over T iterations.

### Algorithm
- **Parity-based Odd-Even Scheduling** — Processes are grouped into blocks of size D. Even blocks send first; odd blocks receive first. This prevents deadlock without non-blocking calls.
- **Binary-Tree Reduction** — Global maximum values are aggregated at Rank 0 in O(log P) steps, since `MPI_Reduce` is forbidden for data values.

### Compilation & Execution
```bash
mpicc -o src src.c -lm
mpirun -np <P> ./src <M> <D1> <D2> <T> <seed>

# Example
mpirun -np 16 ./src 262144 2 4 10 1000
```

### Configurations
| Parameter | Values |
|-----------|--------|
| P (processes) | 8, 16, 32 |
| M (doubles) | 262144, 1048576 |
| D1 / D2 / T / seed | 2 / 4 / 10 / 1000 |

Run each configuration 5 times. Use 16 processes per node.

### Results & Observations
Execution time **increases monotonically** with process count for both data sizes. Communication overhead grows faster than the parallel speedup gain, making inter-process communication the dominant factor. Low variance across runs confirms the trend is stable.

---

## Assignment 2 — 3D Stencil with Halo Exchange

### Overview
Performs a **3D stencil computation** on a domain distributed across a `px × py × pz` process grid. Each process owns a local sub-domain of size `nx × ny × nz` and iterates for T steps, performing halo exchanges and counting isovalue crossings across edges.

### Algorithm & Optimizations
- **Communication–Computation Overlap** — All six halo exchanges are posted as non-blocking `MPI_Isend`/`MPI_Irecv` before computation begins. Interior stencil runs concurrently with transfers; boundary stencil runs only after `MPI_Waitall`. For d=7 on a 120³ sub-domain, ~95% of points are interior.
- **Three-Direction Isocount** — Each cell checks only the −x, −y, −z neighbours, ensuring every undirected edge is counted exactly once without bookkeeping overhead.
- **MPI Derived Datatypes (Zero-Copy Halo)** — `MPI_Type_vector` descriptors are registered once at startup for each face slab, enabling direct transfer without manual pack/unpack buffers.
- **Additional** — Precomputed reciprocal `inv_d = 1/d` replaces division; double-buffer pointer swap eliminates full-domain memory copies each iteration.

### Compilation & Execution
```bash
mpicc -O3 -o stencil src.c -lm
mpirun -np <P> ./stencil <d> <ppn> <px> <py> <pz> <nx> <ny> <nz> <T> <seed> <F> <isovalue>

# Example (P=32, nx=ny=nz=120)
mpirun -np 32 ./stencil 7 32 4 4 2 120 120 120 5 1000 2 500
```

### Configurations
| P | px × py × pz | ppn |
|---|-------------|-----|
| 32 | 4 × 4 × 2 | 32 |
| 48 | 6 × 4 × 2 | 48 |
| 64 | 4 × 4 × 4 | 32 |
| 96 | 6 × 4 × 4 | 48 |

Sub-domain sizes: `nx=ny=nz=120` and `nx=ny=nz=240`. Fixed: d=7, T=5, seed=1000, F=2, isovalue=500. Run each 5 times.

### Results & Observations
- The 240³ configuration is ~4–5× slower than 120³ despite being 8× larger in volume — confirming that communication–computation overlap successfully hides a portion of halo latency.
- A notable jump in execution time occurs at P=64, where the job transitions from single-node (ppn=48) to multi-node (ppn=32), revealing that **inter-node communication is the primary bottleneck**.
- Higher variance at P=96 for 120³ is attributed to inter-node interconnect contention.

---

## Output Format

**Assignment 1** — single line per run:
```
<maxD1> <maxD2> <time_seconds>
```

**Assignment 2** — T+1 lines per run:
```
<count_field1> <count_field2>   # timestep 1
...
<count_field1> <count_field2>   # timestep T
<time_seconds>
```
