# PGC — Parallel & GPU Computing Lab Repository

Lab experiments (LAB 1 to LAB 6) and the Evaluation 1 mini-project for the **Parallel & GPU Computing** course. Every folder is self-contained: its own `README.md`, source code, chart script and result screenshots.

## Repository Structure

| Folder | Experiment | Technologies |
|---|---|---|
| [`LAB1`](LAB1/) | Performance analysis of matrix multiplication across parallel computing models | Sequential, OpenMP, MPI, CUDA |
| [`LAB2`](LAB2/) | Multithreaded programming in C: Pthreads vs. OpenMP | Pthreads, OpenMP, mutex, race conditions |
| [`LAB3`](LAB3/) | OpenMP synchronization and concurrency benchmark | critical, atomic, lock, barrier, semaphore |
| [`LAB4`](LAB4/) | Cache coherence (MSI, MESI, MOESI) and NUMA memory access performance | C, Python simulator, false sharing, NUMA |
| [`LAB5`](LAB5/) | Neural network layers in CUDA | CUDA kernels, ReLU, sigmoid |
| [`LAB6`](LAB6/) | CUDA memory optimization: coalesced vs. uncoalesced access | CUDA, shared memory, matrix transpose |
| [`EVALUATION1`](EVALUATION1/) | Topic 5: Distributed vector processing using MPI | MPI, speedup and efficiency analysis |

## How to Use

Open a folder and follow the `README.md` inside it. Each one has its own build and run instructions.

```text
PGC/
├── LAB1/ ... LAB6/        # one folder per lab
│   ├── README.md
│   ├── src/ or code/      # source files
│   ├── scripts/           # chart generation scripts
│   └── images/ or graphs/ # screenshots and charts
└── EVALUATION1/           # Evaluation 1 mini-project
```
