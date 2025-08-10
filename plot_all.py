#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Automate a sweep over (nc, snr) settings:
- Launch an external script for each pair
- Parse the reported best accuracy
- Save all results to a CSV
- Plot Top-1 accuracy vs. nc for each SNR

Outputs:
  - results_<timestamp>.csv
  - file_name.pdf
  - plot_<timestamp>.log
"""

import subprocess
import csv
import time
import json
import os
import logging
import sys
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd

# Add a custom font (Times New Roman) from a local .ttf file
fm.fontManager.addfont('./times.ttf')

# Timestamp string used in output file names
ts = time.strftime("%Y-%m-%d-%H-%M-%S")

# Path to the external script to run for each experiment
SCRIPT = "file_name.py"

# Hyperparameters
EPOCHS = 100
BATCH = 128

# List of nc (dimensionality) values to sweep
NC_LIST = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

# List of snr values to sweep (in dB)
SNR_LIST = [0, 3, 5]

# Paths for log file, result CSV file, and output figure
LOG_PATH = f"plot_{ts}.log"
RESULT_CSV = f"results_{ts}.csv"
FIG_PATH = f"file_name.pdf"

# Logging setup (both file and console)
logger = logging.getLogger("plot")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s,%(msecs)03d %(message)s", "%Y-%m-%d %H:%M:%S")
for h in (logging.FileHandler(LOG_PATH, encoding="utf-8"),
          logging.StreamHandler(sys.stdout)):
    h.setFormatter(fmt);
    logger.addHandler(h)
logger.propagate = False
logger.info(f"Logging to {LOG_PATH}")


# Single-run launcher
def run_one(nc, snr):
    cmd = [
        "python", SCRIPT,
        "--nc", str(nc),
        "--snr", str(snr),
        "--epochs", str(EPOCHS),
        "--batch", str(BATCH)
    ]
    logger.info(f"Launch: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)

    logger.info(proc.stdout.strip())
    if proc.stderr:
        logger.warning(proc.stderr.strip())

    for line in reversed(proc.stdout.splitlines()):
        if "best_acc" in line:
            best_acc = float(line.split("best_acc:")[-1].strip())
            return best_acc
    raise RuntimeError("best_acc not found")


def plot_results():
    """
    Read results from RESULT_CSV and generate a line plot
    showing Top-1 accuracy vs. nc for each snr value.
    """
    # Configure global matplotlib styles
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "mathtext.fontset": "cm",
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    # Read the results CSV into a pandas DataFrame
    df = pd.read_csv(RESULT_CSV)
    fig, ax = plt.subplots(figsize=(5, 4.7))

    # Marker and color settings for each snr value
    markers = {0: "s", 3: "^", 5: "o"}
    colors = {0: "#F5C267", 3: "#A6D6A6", 5: "#5AB2FF"}

    # Plot a separate curve for each SNR
    for snr in SNR_LIST:
        sub = df[df.snr_db == snr].sort_values("nc")
        ax.plot(sub.nc, sub.best_acc,
                marker=markers[snr], color=colors[snr],
                label=f"SNR = {snr} dB",
                solid_capstyle='round')

    # Axis labels
    ax.set_xlabel(r"Dimensionality of Semantic Representation $z$")
    ax.set_ylabel(r"Top-1 Accuracy")

    # Add caption text below the plot
    ax.text(
        0.5, -0.185,
        "(d) CIFAR-100 / ResNet-34 (conv5_x-avgpool, fc)",
        transform=ax.transAxes,
        ha='center', va='top',
        fontsize=15,
        fontweight='normal',
        fontfamily='Times New Roman'
    )

    # Y-axis range
    ax.set_ylim(0, 0.8)
    # Alternative: ax.set_ylim(0.05, 0.95)

    # X-axis: log2 scale with explicit ticks
    ax.set_xscale("log", base=2)
    ax.set_xticks(NC_LIST)
    ax.set_xlim(min(NC_LIST) * 0.8, max(NC_LIST) * 1.2)

    # Tick style
    ax.tick_params(
        which='both',
        direction='in',
        top=False,
        bottom=False,
        left=False,
        right=False,
        color='gray',
        pad=6
    )

    # Add dotted grid lines
    ax.grid(True,
            which="both",
            linestyle=":",
            color="gray"
            )

    # Legend configuration
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=3,
        frameon=False,
        columnspacing=1,
        borderaxespad=0.08
    )

    # Make borderlines thicker
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    # Adjust layout to avoid clipping
    fig.tight_layout()

    # Save the plot
    # fig.savefig(FIG_PATH, dpi=600)  # Option for high-res PNG
    fig.savefig(FIG_PATH, format="pdf", bbox_inches="tight")


def main():
    """
    Run all experiments for each (nc, snr) combination,
    record results in RESULT_CSV, then plot them.
    """
    with open(RESULT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["nc", "snr_db", "best_acc"])  # CSV header

        for snr in SNR_LIST:
            for nc in NC_LIST:
                logger.info(f"Start using nc={nc}, snr={snr}")
                best_acc = run_one(nc, snr)
                writer.writerow([nc, snr, best_acc])
                f.flush()
                logger.info(f"Finished with nc={nc}, snr={snr}, best_acc={best_acc:.3f}")

    logger.info(f"All experiments done. Results saved to {RESULT_CSV}")
    plot_results()


if __name__ == "__main__":
    main()
