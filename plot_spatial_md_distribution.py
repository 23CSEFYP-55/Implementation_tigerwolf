#!/usr/bin/env python3
"""
Plots the Realistic Spatial Distribution of 100 Mobile Devices (MDs).
Visualizes the 4 Urban Demand Hubs, Arterial Transit Corridors, and Low-Discrepancy Peripheral Grid.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "images", "benchmarks")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Replicate the exact Java logic with same seed
class JavaRandom:
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)

    def next_gaussian(self):
        return self.rng.standard_normal()

    def next_double(self):
        return self.rng.uniform(0.0, 1.0)

    def next_int(self, n):
        return self.rng.integers(0, n)

rng = JavaRandom(42)

hubs = np.array([
    [500.0, 500.0],
    [1500.0, 500.0],
    [500.0, 1500.0],
    [1500.0, 1500.0]
])

corridor_starts = np.array([
    [500.0, 500.0],
    [500.0, 1500.0],
    [500.0, 500.0],
    [1500.0, 500.0],
    [500.0, 500.0],
    [500.0, 1500.0]
])
corridor_ends = np.array([
    [1500.0, 500.0],
    [1500.0, 1500.0],
    [500.0, 1500.0],
    [1500.0, 1500.0],
    [1500.0, 1500.0],
    [1500.0, 500.0]
])

num_mds = 100
hub_count = int(num_mds * 0.40)
corridor_count = int(num_mds * 0.36)
grid_count = num_mds - hub_count - corridor_count

coords = []
types = []

# 1. Urban Hubs
for i in range(hub_count):
    hub_idx = i % 4
    x = hubs[hub_idx, 0] + rng.next_gaussian() * 130.0
    y = hubs[hub_idx, 1] + rng.next_gaussian() * 130.0
    coords.append([np.clip(x, 50.0, 1950.0), np.clip(y, 50.0, 1950.0)])
    types.append("Urban Hubs (40%)")

# 2. Corridors
for i in range(corridor_count):
    c_idx = i % 6
    t = rng.next_double()
    lateral = rng.next_gaussian() * 35.0
    dx = corridor_ends[c_idx, 0] - corridor_starts[c_idx, 0]
    dy = corridor_ends[c_idx, 1] - corridor_starts[c_idx, 1]
    length = np.hypot(dx, dy)
    norm_x = -dy / length
    norm_y = dx / length
    x = corridor_starts[c_idx, 0] + t * dx + lateral * norm_x
    y = corridor_starts[c_idx, 1] + t * dy + lateral * norm_y
    coords.append([np.clip(x, 50.0, 1950.0), np.clip(y, 50.0, 1950.0)])
    types.append("Transit Corridors (36%)")

# 3. Peripheral Grid
cell_w = 2000.0 / 5.0
cell_h = 2000.0 / 5.0
for i in range(grid_count):
    gx = i % 5
    gy = (i // 5) % 5
    x = (gx + 0.15 + 0.70 * rng.next_double()) * cell_w
    y = (gy + 0.15 + 0.70 * rng.next_double()) * cell_h
    coords.append([np.clip(x, 50.0, 1950.0), np.clip(y, 50.0, 1950.0)])
    types.append("Peripheral Grid (24%)")

coords = np.array(coords)
types = np.array(types)

plt.figure(figsize=(9, 9), dpi=300)
ax = plt.gca()

# Draw corridors
for s, e in zip(corridor_starts, corridor_ends):
    ax.plot([s[0], e[0]], [s[1], e[1]], color='royalblue', linestyle='--', alpha=0.35, linewidth=3)

# Draw hubs
for h in hubs:
    circle = plt.Circle((h[0], h[1]), 250, color='crimson', fill=True, alpha=0.10, linestyle=':')
    ax.add_patch(circle)

# Scatter MDs
scatter_colors = {
    "Urban Hubs (40%)": "#e41a1c",
    "Transit Corridors (36%)": "#377eb8",
    "Peripheral Grid (24%)": "#4daf4a"
}

for cat, color in scatter_colors.items():
    mask = (types == cat)
    ax.scatter(coords[mask, 0], coords[mask, 1], c=color, label=cat, s=65, edgecolors='black', linewidths=0.6, alpha=0.9, zorder=5)

ax.set_xlim(0, 2000)
ax.set_ylim(0, 2000)
ax.set_aspect('equal')
ax.set_xlabel("X Coordinate (meters)", fontsize=12, fontweight='bold')
ax.set_ylabel("Y Coordinate (meters)", fontsize=12, fontweight='bold')
ax.set_title("Realistic Spatial Distribution of 100 Mobile Devices (MDs)\n$2000\\,\\text{m} \\times 2000\\,\\text{m}$ Urban Theater Topology", fontsize=13, fontweight='bold', pad=12)
ax.legend(loc="upper right", frameon=True, fontsize=10)
ax.grid(True, linestyle="--", alpha=0.5)

out_file = os.path.join(OUTPUT_DIR, "realistic_md_spatial_topology.png")
plt.savefig(out_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"[SUCCESS] Saved spatial topology map to {out_file}")
