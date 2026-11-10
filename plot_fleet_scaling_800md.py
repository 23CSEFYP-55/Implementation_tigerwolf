#!/usr/bin/env python3
"""
Fleet Scalability Analysis Plotter for 800 Mobile Devices (MDs).
Visualizes performance scaling of CAR + AP_DYNAMIC across fleet sizes:
30, 60, 100, 150, 200, 300, 400 UAVs under heavy 800-MD workload.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "images", "benchmarks")
os.makedirs(OUTPUT_DIR, exist_ok=True)

fleet_sizes_800 = np.array([30, 60, 100, 150, 200, 300, 400])
success_rates_800 = np.array([18.13, 36.35, 46.60, 68.55, 77.54, 93.63, 97.07])
throughput_mb_800 = np.array([857.35, 1711.37, 2196.31, 3239.63, 3664.02, 4418.85, 4581.45])
runtime_ms_800 = np.array([0.1168, 0.1568, 0.2647, 0.2100, 0.2651, 0.3440, 0.3973])
deadline_slack_800 = np.array([0.2048, 0.3384, 0.4516, 0.5499, 0.6061, 0.6815, 0.7088])

# 100 MD baseline data for comparison
fleet_sizes_100 = np.array([30, 45, 60, 80, 100])
success_rates_100 = np.array([42.41, 57.01, 73.35, 87.66, 100.00])

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axs = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
plt.subplots_adjust(wspace=0.25, hspace=0.30, left=0.08, right=0.95, top=0.91, bottom=0.08)

# Panel 1: Success Rate Scaling (100 MD vs 800 MD)
ax1 = axs[0, 0]
ax1.plot(fleet_sizes_800, success_rates_800, marker='o', linewidth=2.8, markersize=8, color="#d62728", label="800 MDs (Heavy Load)")
ax1.plot(fleet_sizes_100, success_rates_100, marker='s', linewidth=2.2, markersize=7, color="#1f77b4", linestyle="--", label="100 MDs (Baseline)")
ax1.axhline(100.0, color="#2ca02c", linestyle=":", alpha=0.7, label="100% Saturation Bound")
for x, y in zip(fleet_sizes_800, success_rates_800):
    ax1.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 9), ha='center', fontweight='bold', fontsize=8, color="#d62728")
ax1.set_title("(a) Success Rate Scaling: 800 MDs vs 100 MDs", fontsize=13, fontweight='bold', pad=10)
ax1.set_xlabel("Number of UAVs in Fleet", fontsize=11)
ax1.set_ylabel("Scheduling Success Rate (%)", fontsize=11)
ax1.set_ylim(10, 108)
ax1.legend(loc="lower right", fontsize=9)
ax1.grid(True, linestyle="--", alpha=0.6)

# Panel 2: Total Throughput (MB) under 800 MDs
ax2 = axs[0, 1]
bars = ax2.bar(fleet_sizes_800.astype(str), throughput_mb_800, color="#2ca02c", alpha=0.85, width=0.55, edgecolor="black")
for bar, val in zip(bars, throughput_mb_800):
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 70, f"{val:.0f} MB", ha='center', va='bottom', fontweight='bold', fontsize=8)
ax2.set_title("(b) Cumulative Processed Throughput under 800 MDs", fontsize=13, fontweight='bold', pad=10)
ax2.set_xlabel("Number of UAVs in Fleet", fontsize=11)
ax2.set_ylabel("Total Data Processed (MB)", fontsize=11)
ax2.set_ylim(0, 5200)
ax2.grid(True, linestyle="--", alpha=0.6)

# Panel 3: Deadline Slack Safety Margin under 800 MDs
ax3 = axs[1, 0]
ax3.plot(fleet_sizes_800, deadline_slack_800, marker='^', linewidth=2.5, markersize=8, color="#9467bd", label="Avg Deadline Slack (S)")
for x, y in zip(fleet_sizes_800, deadline_slack_800):
    ax3.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 9), ha='center', fontweight='bold', fontsize=8, color="#9467bd")
ax3.set_title("(c) Deadline Safety Margin Expansion (800 MDs)", fontsize=13, fontweight='bold', pad=10)
ax3.set_xlabel("Number of UAVs in Fleet", fontsize=11)
ax3.set_ylabel("Average Deadline Slack Margin", fontsize=11)
ax3.set_ylim(0.10, 0.80)
ax3.legend(loc="upper left", fontsize=9)
ax3.grid(True, linestyle="--", alpha=0.6)

# Panel 4: Scheduling Runtime Overhead (ms) under 800 MDs
ax4 = axs[1, 1]
ax4.plot(fleet_sizes_800, runtime_ms_800, marker='D', linewidth=2.5, markersize=8, color="#ff7f0e", label="AP_DYNAMIC Batch Overhead")
for x, y in zip(fleet_sizes_800, runtime_ms_800):
    ax4.annotate(f"{y:.3f} ms", (x, y), textcoords="offset points", xytext=(0, 9), ha='center', fontweight='bold', fontsize=8, color="#ff7f0e")
ax4.axhline(1.0, color="gray", linestyle=":", label="1.0 ms 5G URLLC Limit")
ax4.set_title("(d) Sub-Millisecond Runtime Invariance (30 to 400 UAVs)", fontsize=13, fontweight='bold', pad=10)
ax4.set_xlabel("Number of UAVs in Fleet", fontsize=11)
ax4.set_ylabel("Scheduling Overhead per Batch (ms)", fontsize=11)
ax4.set_ylim(0.05, 1.15)
ax4.legend(loc="upper left", fontsize=9)
ax4.grid(True, linestyle="--", alpha=0.6)

fig.suptitle("FANET-MEC Fleet Scalability Benchmark under Heavy 800 Mobile Device (MD) Workload", fontsize=16, fontweight='bold', y=0.97)

out_file = os.path.join(OUTPUT_DIR, "fleet_scalability_800md.png")
plt.savefig(out_file, dpi=300)
plt.close()
print(f"[SUCCESS] Saved 800-MD fleet scalability figure to {out_file}")
