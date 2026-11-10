#!/usr/bin/env python3
"""
Fleet Scalability Analysis Plotter for FANET-MEC.
Visualizes performance scaling of CAR + AP_DYNAMIC across fleet sizes:
30, 45, 60, 80, 100 UAVs.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "images", "benchmarks")
os.makedirs(OUTPUT_DIR, exist_ok=True)

fleet_sizes = np.array([30, 45, 60, 80, 100])
success_rates = np.array([42.41, 57.01, 73.35, 87.66, 100.00])
throughput_mb = np.array([2150.06, 2686.50, 3462.79, 4141.33, 4720.69])
fairness_jain = np.array([0.7967, 0.8230, 0.8292, 0.9083, 0.9589])
deadline_slack = np.array([0.1009, 0.1606, 0.2116, 0.2896, 0.3594])
runtime_ms = np.array([0.1575, 0.2253, 0.2313, 0.2164, 0.2697])

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axs = plt.subplots(2, 2, figsize=(14, 11), dpi=300)
plt.subplots_adjust(wspace=0.25, hspace=0.30, left=0.08, right=0.95, top=0.91, bottom=0.08)

colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

# Panel 1: Success Rate & Throughput
ax1 = axs[0, 0]
ax1.plot(fleet_sizes, success_rates, marker='o', linewidth=2.8, markersize=8, color="#1f77b4", label="Success Rate (%)")
ax1.axhline(100.0, color="#2ca02c", linestyle="--", alpha=0.7, label="100% Saturation Bound")
for x, y in zip(fleet_sizes, success_rates):
    ax1.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold', fontsize=9, color="#1f77b4")
ax1.set_title("(a) Scheduling Success Rate vs Fleet Size", fontsize=13, fontweight='bold', pad=10)
ax1.set_xlabel("Number of UAVs in Swarm", fontsize=11)
ax1.set_ylabel("Scheduling Success Rate (%)", fontsize=11)
ax1.set_ylim(30, 108)
ax1.legend(loc="lower right", fontsize=9)
ax1.grid(True, linestyle="--", alpha=0.6)

# Panel 2: Total Throughput (MB)
ax2 = axs[0, 1]
bars = ax2.bar(fleet_sizes.astype(str), throughput_mb, color="#2ca02c", alpha=0.85, width=0.55, edgecolor="black")
for bar, val in zip(bars, throughput_mb):
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 70, f"{val:.0f} MB", ha='center', va='bottom', fontweight='bold', fontsize=9)
ax2.set_title("(b) Cumulative System Throughput (MB)", fontsize=13, fontweight='bold', pad=10)
ax2.set_xlabel("Number of UAVs in Swarm", fontsize=11)
ax2.set_ylabel("Total Data Processed (MB)", fontsize=11)
ax2.set_ylim(0, 5400)
ax2.grid(True, linestyle="--", alpha=0.6)

# Panel 3: Swarm Fairness & Deadline Slack Margin
ax3 = axs[1, 0]
line1 = ax3.plot(fleet_sizes, fairness_jain, marker='s', linewidth=2.4, markersize=8, color="#ff7f0e", label="Jain's Fairness Index (J)")
ax3.set_ylabel("Jain's Fairness Index", fontsize=11, color="#ff7f0e")
ax3.set_ylim(0.70, 1.02)
ax3.tick_params(axis='y', labelcolor="#ff7f0e")

ax3_twin = ax3.twinx()
line2 = ax3_twin.plot(fleet_sizes, deadline_slack, marker='^', linewidth=2.4, markersize=8, color="#9467bd", linestyle="-.", label="Avg Deadline Slack (S)")
ax3_twin.set_ylabel("Average Deadline Slack Margin", fontsize=11, color="#9467bd")
ax3_twin.set_ylim(0.05, 0.42)
ax3_twin.tick_params(axis='y', labelcolor="#9467bd")

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax3.legend(lines, labels, loc="lower right", fontsize=9)
ax3.set_title("(c) Workload Fairness & Deadline Safety Margins", fontsize=13, fontweight='bold', pad=10)
ax3.set_xlabel("Number of UAVs in Swarm", fontsize=11)
ax3.grid(True, linestyle="--", alpha=0.6)

# Panel 4: Scheduling Runtime Overhead (ms)
ax4 = axs[1, 1]
ax4.plot(fleet_sizes, runtime_ms, marker='D', linewidth=2.5, markersize=8, color="#d62728", label="Batch Overhead (ms)")
for x, y in zip(fleet_sizes, runtime_ms):
    ax4.annotate(f"{y:.3f} ms", (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold', fontsize=9, color="#d62728")
ax4.axhline(1.0, color="gray", linestyle=":", label="1.0 ms Real-Time TTI Limit")
ax4.set_title("(d) AP_DYNAMIC Scheduling Overhead vs Fleet Size", fontsize=13, fontweight='bold', pad=10)
ax4.set_xlabel("Number of UAVs in Swarm", fontsize=11)
ax4.set_ylabel("Runtime Overhead per Batch (ms)", fontsize=11)
ax4.set_ylim(0.10, 1.15)
ax4.legend(loc="upper left", fontsize=9)
ax4.grid(True, linestyle="--", alpha=0.6)

fig.suptitle("FANET-MEC Fleet Scalability Benchmark: CAR Trajectory + AP_DYNAMIC Scheduler", fontsize=16, fontweight='bold', y=0.97)

out_file = os.path.join(OUTPUT_DIR, "fleet_scalability_analysis.png")
plt.savefig(out_file, dpi=300)
plt.close()
print(f"[SUCCESS] Saved fleet scalability analysis figure to {out_file}")
