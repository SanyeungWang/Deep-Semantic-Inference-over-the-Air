#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
This script plots the relationship between beta and the normalized computation time
and saves the figure in both PDF and PNG formats.

The plot uses Times New Roman font with custom sizing for titles, labels, ticks, and legend.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Add a custom font (Times New Roman) from a local .ttf file
fm.fontManager.addfont('./times.ttf')

# Update global plotting parameters
plt.rcParams.update({
    "font.family": "Times New Roman",  # Set the global font family
    "mathtext.fontset": "cm",  # Use Computer Modern for math text
    "font.size": 13,  # Base font size
    "axes.titlesize": 15,  # Axis title font size
    "axes.labelsize": 19,  # Axis label font size
    "legend.fontsize": 17,  # Legend font size
    "xtick.labelsize": 18,  # X-axis tick label size
    "ytick.labelsize": 18,  # Y-axis tick label size
})

# Define beta range (from 1e-5 to 1) and take its log10
beta = np.linspace(1e-5, 1, 300)
log_beta = np.log10(beta)


# Linear function: y = b + beta * a
def y_func(beta, b, a):
    return b + beta * a


# Data series for SP-2, SP-3, SP-4
y1 = y_func(beta, b=229310464 / 1159448576, a=2072037376 / 1159448576)
y2 = y_func(beta, b=515571712 / 1159448576, a=847300608 / 1159448576)
y3 = y_func(beta, b=953876480 / 1159448576, a=251709440 / 1159448576)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(5, 5))

# Plot each curve with custom colors and labels
ax.plot(log_beta, y1, label=r'SP-2', color='#F5C267', linewidth=2)
ax.plot(log_beta, y2, label=r'SP-3', color='#A6D6A6', linewidth=2)
ax.plot(log_beta, y3, label=r'SP-4', color='#5AB2FF', linewidth=2)

# Set X and Y ticks
ax.set_xticks(np.arange(-5, 0.1, 1))  # X-axis from -5 to 0, step size = 1
ax.set_yticks(np.arange(0.25, 2.01, 0.25))  # Y-axis from 0.25 to 2.0, step size = 0.25

# Set axis labels
ax.set_xlabel(r'$\log_{10}(\beta)$')
ax.set_ylabel(r'Normalized Computation Time')

# Adjust border line width for all spines
for spine in ax.spines.values():
    spine.set_linewidth(1.2)

# Customize tick parameters
ax.tick_params(
    which='both', direction='in',
    top=False, bottom=False, left=False, right=False,
    color='gray', pad=6
)

# Add dotted grid lines
ax.grid(True, which="both", linestyle=":", color="gray", alpha=0.7)

# Configure legend
ax.legend(
    loc="upper center",  # Place legend above the plot
    bbox_to_anchor=(0.5, 1.12),  # Adjust legend position
    ncol=3,  # 3 columns
    frameon=False,  # Remove legend frame
    columnspacing=1,  # Space between legend columns
    borderaxespad=0.3,  # Padding between axes and legend
    handlelength=1.5  # Length of legend line markers
)

# Adjust layout to prevent overlaps
fig.tight_layout()

# Save figure as PDF and high-resolution PNG
fig.savefig("tcomp_vs_log_beta.pdf", format="pdf", bbox_inches="tight")
fig.savefig("tcomp_vs_log_beta.png", dpi=600, bbox_inches="tight")
