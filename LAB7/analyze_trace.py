
"""
Experiment 7: Analyze PyTorch profiler traces
and report GPU utilization.
"""

import collections
import json
import sys

GAP_US = 50

COMPUTE_HINTS = (
    "gemm",
    "sgemm",
    "cutlass",
    "cublas",
    "conv",
    "wmma"
)


# ---------------------------------------------------------
# Merge overlapping intervals
# ---------------------------------------------------------

def merge(iv):

    iv = sorted(iv)
    out = []

    for s, e in iv:

        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)

        else:
            out.append([s, e])

    return out


# ---------------------------------------------------------
# Calculate total duration
# ---------------------------------------------------------

def total(iv):
    return sum(e - s for s, e in iv)


# ---------------------------------------------------------
# Calculate copy and kernel overlap
# ---------------------------------------------------------

def overlap(a, b):

    i = 0
    j = 0
    t = 0

    while i < len(a) and j < len(b):

        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])

        if e > s:
            t += e - s

        if a[i][1] < b[j][1]:
            i += 1

        else:
            j += 1

    return t


# ---------------------------------------------------------
# Kernel classification
# ---------------------------------------------------------

def label(name):

    low = name.lower()

    if any(h in low for h in COMPUTE_HINTS):
        return "compute-bound"

    return "memory-bound"


# ---------------------------------------------------------
# Analyze one trace file
# ---------------------------------------------------------

def analyze(path):

    with open(path) as f:
        data = json.load(f)

    events = (
        data["traceEvents"]
        if isinstance(data, dict)
        else data
    )

    kern = []
    copy = []

    knames = collections.defaultdict(lambda: [0, 0.0])
    cnames = collections.defaultdict(lambda: [0, 0.0])

    for e in events:

        if e.get("ph") != "X":
            continue

        cat = e.get("cat", "")
        s = e["ts"]
        d = e.get("dur", 0)

        if cat == "kernel":

            kern.append([s, s + d])

            knames[e["name"][:60]][0] += 1
            knames[e["name"][:60]][1] += d

        elif cat in ("gpu_memcpy", "gpu_memset"):

            copy.append([s, s + d])

            cnames[e["name"][:60]][0] += 1
            cnames[e["name"][:60]][1] += d

    if not kern:

        print(f"{path}: no GPU kernels found in trace")
        return None

    k = merge(kern)
    c = merge(copy)

    start = min(
        k[0][0],
        c[0][0] if c else k[0][0]
    )

    end = max(
        k[-1][1],
        c[-1][1] if c else k[-1][1]
    )

    span = end - start

    busy = total(k)

    gaps = [
        k[i + 1][0] - k[i][1]
        for i in range(len(k) - 1)
    ]

    big = [g for g in gaps if g > GAP_US]

    print("=" * 72)
    print(path)
    print("=" * 72)

    print(f"Profiled window             : {span / 1000:9.2f} ms")
    print(f"Kernel busy time            : {busy / 1000:9.2f} ms")

    utilization = 100 * busy / span if span > 0 else 0

    print(f"GPU utilization             : {utilization:9.1f} %")
    print(f"Copy busy time              : {total(c) / 1000:9.2f} ms")
    print(f"Copy overlapped with kernel : {overlap(k, c) / 1000:9.2f} ms")

    print(
        f"Idle gaps > {GAP_US} us       : {len(big):9d} "
        f"(total {sum(big) / 1000:.2f} ms)"
    )

    print("\nKernels")
    print(f"{'calls':>6} {'total ms':>10} {'label':<14} name")

    for n, (cnt, dur) in sorted(
        knames.items(),
        key=lambda x: -x[1][1]
    ):

        print(
            f"{cnt:6d} {dur / 1000:10.2f} "
            f"{label(n):<14} {n}"
        )

    print("\nMemory copies")
    print(f"{'calls':>6} {'total ms':>10} name")

    for n, (cnt, dur) in sorted(
        cnames.items(),
        key=lambda x: -x[1][1]
    ):

        print(
            f"{cnt:6d} {dur / 1000:10.2f} {n}"
        )

    print()

    return (
        span,
        busy,
        total(c),
        overlap(k, c),
        len(big)
    )


# ---------------------------------------------------------
# Main function
# ---------------------------------------------------------

def main():

    if len(sys.argv) < 2:

        raise SystemExit(
            "Usage: python analyze_trace.py trace_*.json"
        )

    rows = []

    for p in sys.argv[1:]:

        r = analyze(p)

        if r:
            rows.append((p, r))

    if len(rows) > 1:

        print("SUMMARY")

        print(
            f"{'file':<24}"
            f"{'window ms':>10}"
            f"{'util %':>8}"
            f"{'overlap ms':>12}"
            f"{'gaps':>6}"
        )

        for p, (span, busy, _, ov, g) in rows:

            utilization = 100 * busy / span if span > 0 else 0

            print(
                f"{p:<24}"
                f"{span / 1000:10.1f}"
                f"{utilization:8.1f}"
                f"{ov / 1000:12.1f}"
                f"{g:6d}"
            )


if __name__ == "__main__":
    main()
