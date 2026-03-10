"""Plot training curves from nanostate CSV logs.

Usage:
  python plot.py                              # plot most recent log
  python plot.py logs/lm_20240101_120000.csv  # plot specific log
  python plot.py logs/lm_*.csv                # overlay multiple runs
"""

import csv
import glob
import os

import matplotlib.pyplot as plt


def read_log(path):
    """Read a CSV log file into a dict of lists."""
    with open(path) as f:
        reader = csv.DictReader(f)
        cols = {col: [] for col in reader.fieldnames}
        for row in reader:
            for col in reader.fieldnames:
                try:
                    cols[col].append(float(row[col]))
                except (ValueError, TypeError):
                    cols[col].append(row[col])
    return cols


def detect_task(cols):
    """Figure out which task from the CSV columns."""
    if "val_bpb" in cols:
        return "lm"
    elif "accuracy" in cols:
        return "dna"
    elif "val_mse" in cols:
        return "ts"
    return "unknown"


def plot_runs(paths, out_path="logs/curves.png"):
    """Plot one or more training runs, overlaid."""
    runs = []
    for p in paths:
        cols = read_log(p)
        task = detect_task(cols)
        label = os.path.basename(p).replace(".csv", "")
        runs.append((cols, task, label))

    # all runs should be same task for overlay to make sense
    tasks = set(r[1] for r in runs)
    if len(tasks) > 1:
        print(f"Warning: mixing tasks {tasks}, plots may not be comparable")

    task = runs[0][1]

    if task == "lm":
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for cols, _, label in runs:
            axes[0].plot(cols["step"], cols["train_loss"], label=f"{label} train", alpha=0.7)
            axes[0].plot(cols["step"], cols["val_loss"], label=f"{label} val", linestyle="--", alpha=0.7)
            axes[1].plot(cols["step"], cols["val_bpb"], label=label)
        axes[0].set_xlabel("step")
        axes[0].set_ylabel("cross-entropy loss")
        axes[0].set_title("Loss")
        axes[0].legend(fontsize=8)
        axes[1].set_xlabel("step")
        axes[1].set_ylabel("bits per byte")
        axes[1].set_title("Validation BPB")
        axes[1].legend(fontsize=8)

    elif task == "dna":
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for cols, _, label in runs:
            axes[0].plot(cols["step"], cols["train_loss"], label=f"{label} train", alpha=0.7)
            axes[0].plot(cols["step"], cols["val_loss"], label=f"{label} val", linestyle="--", alpha=0.7)
            axes[1].plot(cols["step"], cols["accuracy"], label=label)
        axes[0].set_xlabel("step")
        axes[0].set_ylabel("cross-entropy loss")
        axes[0].set_title("Loss")
        axes[0].legend(fontsize=8)
        axes[1].set_xlabel("step")
        axes[1].set_ylabel("accuracy")
        axes[1].set_title("Validation Accuracy")
        axes[1].legend(fontsize=8)

    elif task == "ts":
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        for cols, _, label in runs:
            axes[0].plot(cols["step"], cols["train_loss"], label=f"{label} train", alpha=0.7)
            axes[0].plot(cols["step"], cols["val_loss"], label=f"{label} val", linestyle="--", alpha=0.7)
            axes[1].plot(cols["step"], cols["val_mse"], label=label)
            axes[2].plot(cols["step"], cols["val_mae"], label=label)
        axes[0].set_xlabel("step")
        axes[0].set_ylabel("MSE loss")
        axes[0].set_title("Loss")
        axes[0].legend(fontsize=8)
        axes[1].set_xlabel("step")
        axes[1].set_ylabel("MSE")
        axes[1].set_title("Validation MSE")
        axes[1].legend(fontsize=8)
        axes[2].set_xlabel("step")
        axes[2].set_ylabel("MAE")
        axes[2].set_title("Validation MAE")
        axes[2].legend(fontsize=8)
    else:
        print("Unknown task type in logs")
        return

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    plt.show()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="*", help="CSV log files to plot")
    parser.add_argument("-o", "--output", default=None, help="Output path for the plot PNG")
    args = parser.parse_args()

    if args.logs:
        paths = []
        for arg in args.logs:
            expanded = glob.glob(arg)
            paths.extend(expanded if expanded else [arg])
    else:
        all_logs = sorted(glob.glob("logs/*.csv"), key=os.path.getmtime)
        if not all_logs:
            print("No logs found in logs/")
            return
        paths = [all_logs[-1]]
        print(f"Plotting most recent: {paths[0]}")

    out_path = args.output or os.path.join("logs", "curves.png")
    plot_runs(paths, out_path)


if __name__ == "__main__":
    main()
