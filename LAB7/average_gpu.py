
import csv
import os

files = [
    "smi_baseline_run.csv",
    "smi_optimized_run.csv"
]

for filename in files:
    if not os.path.exists(filename):
        print(filename, "NOT FOUND")
        continue

    values = []

    with open(filename, "r") as f:
        reader = csv.reader(f)
        next(reader, None)

        for row in reader:
            if row:
                try:
                    values.append(float(row[0].strip().replace("%", "")))
                except ValueError:
                    continue

    if values:
        avg = sum(values) / len(values)
        print("\nFile:", filename)
        print("Readings:", len(values))
        print("Average GPU utilization:", round(avg, 2), "%")
    else:
        print(filename, "has no valid readings")
