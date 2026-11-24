# Experiment 7 -- Profiling and GPU Utilization

## Aim

To profile a GPU workload, identify idle gaps and data-transfer
overhead, and improve GPU utilization by overlapping host-to-device
transfers with computation.

## Environment

-   OS: Windows 11
-   GPU: NVIDIA GeForce RTX 5060 Ti (16 GB)
-   Python: 3.12.10
-   PyTorch: 2.14.1+cu132
-   CUDA runtime reported by PyTorch: 13.2
-   CUDA Toolkit (`nvcc`): 13.4

## Project files

``` text
Experiment7/
├── gpu_workload.py
├── analyze_trace.py
├── average_gpu.py
├── trace_baseline.json
├── trace_optimized.json
├── smi_baseline.csv
├── smi_optimized.csv
├── screenshots/
│   ├── baseline_perfetto.png
│   ├── optimized_perfetto.png
│   ├── baseline_terminal.png
│   └── optimized_terminal.png
└── README.md
```

Keep only files that actually exist in your repository. Add other
screenshots using clear names.

## Implementation

### Baseline

-   Uses pageable host memory.
-   Uses blocking host-to-device/device-to-host transfers.
-   Executes work on the default CUDA stream.

### Optimized

-   Uses pinned host memory.
-   Uses two CUDA streams and double buffering.
-   Uses CUDA events to coordinate buffer reuse.
-   Uses non-blocking transfers to overlap data movement and
    computation.

## How to run

Open Command Prompt in this folder and activate the virtual environment:

``` bat
.venv\Scripts\activate
```

Run the baseline workload and save its profiler trace:

``` bat
python gpu_workload.py --mode baseline --profile
```

Run the optimized workload and save its profiler trace:

``` bat
python gpu_workload.py --mode optimized --profile
```

Analyze both traces:

``` bat
python analyze_trace.py trace_baseline.json trace_optimized.json
```

Open the JSON traces in [Perfetto](https://ui.perfetto.dev/) to inspect
CUDA kernel activity, memory copies, overlap, and idle gaps.

## Results

### End-to-end execution time

The following values are the mean of two non-profiled runs:

  Metric                    Baseline          Optimized
  ---------------- ----------------- ------------------
  Total time                 3.893 s            1.950 s
  Time per batch            19.47 ms            9.75 ms
  Throughput         51.35 batches/s   102.55 batches/s
  Checksum              4.198830e+09       4.198830e+09

**Observed speedup:** approximately **2.00×**. The optimized version
reduced total execution time by approximately **49.9%** while preserving
the checksum.

### Profiler trace comparison

  -----------------------------------------------------------------------
  Metric                               Baseline                 Optimized
  ------------------- ------------------------- -------------------------
  Profiled window                     199.65 ms                 108.04 ms

  Kernel busy time                     92.96 ms                  93.57 ms

  GPU utilization in                      46.6%                     86.6%
  profiled window                               

  Copy busy time                      100.33 ms                  49.50 ms

  Copy/compute                          0.00 ms                  38.39 ms
  overlap                                       

  Idle gaps greater          9 (93.24 ms total)         9 (1.30 ms total)
  than 50 µs                                    
  -----------------------------------------------------------------------

The profiler shows that pinned memory reduced host-to-device transfer
time and the optimized implementation overlapped copies with GPU
computation. The kernel work remained broadly similar; the improvement
came mainly from reducing transfer and idle time.

> Note: The GPU utilization percentages above are from the profiler's
> measured trace window. Separate `nvidia-smi` CSV averages should only
> be reported if the samples were collected strictly during each
> workload run; samples containing idle time are not a valid direct
> comparison.

## Screenshots to include

Add your actual screenshots to the `screenshots/` folder: 1. Baseline
workload terminal output. 2. Optimized workload terminal output. 3.
Baseline Perfetto timeline showing transfers and kernels. 4. Optimized
Perfetto timeline showing transfer/compute overlap. 5. Analyzer output
comparing both traces. 6. `nvidia-smi` monitoring output or CSV summary,
if correctly captured.

Use the exact filenames in the project tree above, or update the tree to
match your files.

## Main observations

-   The baseline profile contains pageable host-to-device copies and no
    measured copy/compute overlap.
-   The optimized profile uses pinned memory and shows approximately
    38.39 ms of copy/compute overlap.
-   Profiler-window GPU utilization increased from 46.6% to 86.6%.
-   The optimized execution achieved approximately 2× throughput.
-   Matching checksums indicate that both modes produced the same
    accumulated result for this workload.

## Conclusion

The experiment demonstrated that pinned host memory, non-blocking data
transfers, CUDA streams, double buffering, and CUDA events can reduce
transfer-related idle time and improve GPU utilization. In this run, the
optimized implementation achieved approximately 2× speedup with a
matching checksum.

## Reproducibility note

Timing and utilization values depend on the GPU, driver, system load,
and workload parameters. The values in this README are the measurements
recorded for this experiment's run.
