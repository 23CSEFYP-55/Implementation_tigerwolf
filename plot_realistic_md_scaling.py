#!/usr/bin/env python3
"""
Comparative Scalability & Spatial Analysis: Realistic MD Distribution vs Uniform Random.
Plots fleet scaling performance curves and visualizes the realistic urban topology.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "images", "benchmarks")
os.makedirs(OUTPUT_DIR, exist_ok=True)

fleet_sizes = np.array([30, 45, 60, 80, 100])

# Benchmark empirical data
uniform_success = np.array([42.41, 57.01, 73.35, 87.66, 100.00])
realistic_success = np.array([41.56, 71.64, 88.81, 99.06, 100.00])

uniform_throughput = np.array([2150.06, 2686.50, 3462.79, 4141.33, 4720.69])
realistic_throughput = np.array([1962.29, 3382.02, 4191.88, 4676.82, 4720.69])

realistic_fairness = np.array([0.7768, 0.6968, 0.8207, 0.8438, 0.9544])
uniform_fairness = np.array([0.7967, 0.8230, 0.8292, 0.9083, 0.9589])

realistic_runtime = np.array([0.1718, 0.2030, 0.2149, 0.2386, 0.2273])
realistic_slack = np.array([0.1038, 0.1304, 0.1768, 0.2603, 0.3613])

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axs = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
plt.subplots_adjust(wspace=0.25, hspace=0.32, left=0.08, right=0.95, top=0.91, bottom=0.08)

# Panel 1: Success Rate Comparison
ax1 = axs[0, 0]
ax1.plot(fleet_sizes, uniform_success, marker='s', linestyle='--', color='#7f7f7f', linewidth=2.0, markersize=7, label="Uniform Random MDs")
ax1.plot(fleet_sizes, realistic_success, marker='o', linestyle='-', color='#1f77b4', linewidth=2.8, markersize=8, label="Realistic Urban MDs (Hubs+Corridors)")
ax1.axhline(100.0, color="#2ca02c", linestyle=":", alpha=0.6, label="100% Saturation Bound")

# Annotate improvements
for i, (x, u_y, r_y) in enumerate(zip(fleet_sizes, uniform_success, realistic_success)):
    diff = r_y - u_y
    if diff > 1.0:
        ax1.annotate(f"+{diff:.1f}%\n({r_y:.1f}%)", (x, r_y), textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold', fontsize=9, color="#1f77b4")
    elif diff < -0.1:
        ax1.annotate(f"{r_y:.1f}%", (x, r_y), textcoords="offset points", xytext=(0, -18), ha='center', fontweight='bold', fontsize=9, color="#1f77b4")
    else:
        ax1.annotate(f"{r_y:.1f}%", (x, r_y), textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold', fontsize=9, color="#1f77b4")

ax1.set_title("(a) Scheduling Success Rate: Realistic vs Uniform MDs", fontsize=12, fontweight='bold', pad=10)
ax1.set_xlabel("Number of UAVs in Swarm", fontsize=11)
ax1.set_ylabel("Scheduling Success Rate (%)", fontsize=11)
ax1.set_ylim(30, 112)
ax1.legend(loc="lower right", fontsize=9)
ax1.grid(True, linestyle="--", alpha=0.6)

# Panel 2: Total Throughput Comparison
ax2 = axs[0, 1]
x_indices = np.arange(len(fleet_sizes))
bar_width = 0.35

rects1 = ax2.bar(x_indices - bar_width/2, uniform_throughput, bar_width, label='Uniform Random MDs', color='#aec7e8', edgecolor='gray')
rects2 = ax2.bar(x_indices + bar_width/2, realistic_throughput, bar_width, label='Realistic Urban MDs', color='#2ca02c', edgecolor='black')

for i, (u_val, r_val) in enumerate(zip(uniform_throughput, realistic_throughput)):
    diff = r_val - u_val
    txt = f"+{diff:.0f} MB" if diff > 0 else f"{diff:.0f} MB"
    color = "darkgreen" if diff > 0 else "gray"
    ax2.text(x_indices[i] + bar_width/2, r_val + 70, f"{r_val:.0f}\n({txt})", ha='center', va='bottom', fontweight='bold', fontsize=8, color=color)

ax2.set_xticks(x_indices)
ax2.set_xticklabels(fleet_sizes.astype(str))
ax2.set_title("(b) Cumulative Throughput: Realistic vs Uniform MDs", fontsize=12, fontweight='bold', pad=10)
ax2.set_xlabel("Number of UAVs in Swarm", fontsize=11)
ax2.set_ylabel("Total Data Processed (MB)", fontsize=11)
ax2.set_ylim(0, 5600)
ax2.legend(loc="upper left", fontsize=9)
ax2.grid(True, linestyle="--", alpha=0.6)

# Panel 3: Swarm Fairness & Deadline Slack Margin (Realistic)
ax3 = axs[1, 0]
line1 = ax3.plot(fleet_sizes, realistic_fairness, marker='s', linewidth=2.4, markersize=8, color="#ff7f0e", label="Jain's Fairness (Realistic)")
ax3.plot(fleet_sizes, uniform_fairness, marker='x', linestyle='--', linewidth=1.8, markersize=7, color="#ffbb78", label="Jain's Fairness (Uniform)")
ax3.set_ylabel("Jain's Fairness Index", fontsize=11, color="#d95f02")
ax3.set_ylim(0.60, 1.02)
ax3.tick_params(axis='y', labelcolor="#d95f02")

ax3_twin = ax3.twinx()
line2 = ax3_twin.plot(fleet_sizes, realistic_slack, marker='^', linewidth=2.4, markersize=8, color="#9467bd", linestyle="-.", label="Avg Deadline Slack (Realistic)")
ax3_twin.set_ylabel("Average Deadline Slack Margin", fontsize=11, color="#9467bd")
ax3_twin.set_ylim(0.05, 0.45)
ax3_twin.tick_params(axis='y', labelcolor="#9467bd")

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax3.legend(lines, labels, loc="lower right", fontsize=9)
ax3.set_title("(c) Workload Fairness & Deadline Safety Margins", fontsize=12, fontweight='bold', pad=10)
ax3.set_xlabel("Number of UAVs in Swarm", fontsize=11)
ax3.grid(True, linestyle="--", alpha=0.6)

# Panel 4: Scheduling Runtime Overhead (ms)
ax4 = axs[1, 1]
ax4.plot(fleet_sizes, realistic_runtime, marker='D', linewidth=2.5, markersize=8, color="#d62728", label="AP_DYNAMIC Batch Runtime (ms)")
for x, y in zip(fleet_sizes, realistic_runtime):
    ax4.annotate(f"{y:.3f} ms", (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold', fontsize=9, color="#d62728")
ax4.axhline(1.0, color="gray", linestyle=":", label="1.0 ms Real-Time TTI Limit")
ax4.set_title("(d) AP_DYNAMIC Runtime Overhead vs Fleet Size", fontsize=12, fontweight='bold', pad=10)
ax4.set_xlabel("Number of UAVs in Swarm", fontsize=11)
ax4.set_ylabel("Runtime Overhead per Batch (ms)", fontsize=11)
ax4.set_ylim(0.10, 1.15)
ax4.legend(loc="upper left", fontsize=9)
ax4.grid(True, linestyle="--", alpha=0.6)

fig.suptitle("Performance Evaluation under Realistic Spatial MD Distribution (Hubs + Arterial Corridors)", fontsize=15, fontweight='bold', y=0.97)

out_file = os.path.join(OUTPUT_DIR, "fleet_scalability_realistic_md.png")
plt.savefig(out_file, dpi=300)
plt.close()
print(f"[SUCCESS] Saved realistic MD scalability figure to {out_file}")
