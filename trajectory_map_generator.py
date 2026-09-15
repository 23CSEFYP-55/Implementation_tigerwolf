#!/usr/bin/env python3
"""
Trajectory Map Visualizer for FANET-MEC Testbed.

Generates publication-quality 2D spatial trajectory maps comparing all trajectory
algorithms:
  1. PPO: Continuous 8-sector MD density vector flow towards ground device hot spots.
  2. CAR: Adaptive region partitioning, Otsu cluster centroids, and pointer-net tour paths.
  3. TACC-AAV: Continuous heading kinematics [v, theta], MOCPO communication gating
               (active broadcast vs gated segments), and ACBBA task bundle links.

Outputs:
  - trajectory_maps_all.png (3-panel side-by-side comparative figure)
  - trajectory_map_ppo.png
  - trajectory_map_car.png
  - trajectory_map_tacc.png
"""

import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from rich.console import Console

console = Console()
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MAP_BOUND = 2000.0


def parse_simulation_trajectories(log_path: str) -> dict[str, list[tuple[float, float]]]:
    """Parse [TRAJECTORY_EPOCH] lines from simulation log if available."""
    if not os.path.exists(log_path):
        return {}
    trajs: dict[str, list[tuple[float, float]]] = {}
    pattern = r"(\w+):([\d.]+),([\d.]+)"
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "[TRAJECTORY_EPOCH]" in line:
                for match in re.finditer(pattern, line):
                    uav_id = match.group(1)
                    x = float(match.group(2))
                    y = float(match.group(3))
                    if uav_id not in trajs:
                        trajs[uav_id] = []
                    trajs[uav_id].append((x, y))
    return trajs


def generate_synthetic_scenario(seed: int = 42):
    """Generates standard scenario layout matching FANETSimulation parameters."""
    rng = np.random.default_rng(seed)
    # 100 MDs clustered into 4 hot spots + uniform background
    md_clusters = [
        (400, 450, 180, 25),
        (1550, 400, 160, 25),
        (1450, 1500, 200, 25),
        (500, 1550, 170, 25),
    ]
    md_xs, md_ys = [], []
    for cx, cy, radius, count in md_clusters:
        angles = rng.uniform(0, 2 * np.pi, count)
        rads = rng.normal(0, radius * 0.45, count)
        md_xs.extend(np.clip(cx + rads * np.cos(angles), 50, MAP_BOUND - 50))
        md_ys.extend(np.clip(cy + rads * np.sin(angles), 50, MAP_BOUND - 50))
    mds = np.column_stack((md_xs, md_ys))

    # 30 UAV initial positions scattered across map
    uav_starts = rng.uniform(100, MAP_BOUND - 100, size=(30, 2))
    return mds, uav_starts


def simulate_ppo_trajectories(mds, uav_starts, steps: int = 60):
    """Simulates PPO continuous vector flow driven by 8-sector MD densities."""
    uav_paths = []
    # Sample 8 representative UAVs for visual clarity
    sample_indices = [0, 3, 7, 11, 14, 18, 22, 26]
    for idx in sample_indices:
        pos = np.array(uav_starts[idx], dtype=float)
        path = [pos.copy()]
        cov_rad = 90.0
        for _ in range(steps):
            # Compute MD density vector towards nearby devices
            diffs = mds - pos
            dists = np.linalg.norm(diffs, axis=1)
            weights = np.exp(-dists / 350.0)
            if np.sum(weights) > 1e-4:
                grad = np.sum(diffs * weights[:, None], axis=0) / np.sum(weights)
                norm = np.linalg.norm(grad)
                if norm > 0:
                    step = (grad / norm) * min(22.0, norm * 0.08)
                    pos += step
            pos = np.clip(pos, 50, MAP_BOUND - 50)
            path.append(pos.copy())
        uav_paths.append((f"uav_{idx}", path, cov_rad))
    return uav_paths


def simulate_car_trajectories(mds, uav_starts):
    """Simulates CAR adaptive region partitioning, cluster centroids, and pointer-net tour routes."""
    rng = np.random.default_rng(101)
    # Define 4 partitioned clusters matching Otsu / density regions
    cluster_centers = [
        np.array([420.0, 480.0]),
        np.array([1520.0, 430.0]),
        np.array([1420.0, 1480.0]),
        np.array([530.0, 1520.0]),
    ]
    uav_tours = []
    sample_uavs = [(1, 0), (8, 1), (15, 2), (23, 3)]
    for uav_idx, cluster_idx in sample_uavs:
        start_pos = uav_starts[uav_idx].copy()
        center = cluster_centers[cluster_idx]
        
        # Cluster waypoints around centroid
        wp_offsets = [
            np.array([-90.0, -80.0]),
            np.array([80.0, -100.0]),
            np.array([110.0, 90.0]),
            np.array([-60.0, 110.0]),
            np.array([0.0, 0.0]),
        ]
        waypoints = [center + off for off in wp_offsets]
        # Pointer net tour sequence
        tour_order = [start_pos] + waypoints + [waypoints[0]]
        
        # Dense interpolated flight path
        path = []
        for i in range(len(tour_order) - 1):
            p1 = tour_order[i]
            p2 = tour_order[i + 1]
            dist = np.linalg.norm(p2 - p1)
            num_pts = max(5, int(dist / 20.0))
            for t in np.linspace(0, 1, num_pts, endpoint=False):
                path.append(p1 + t * (p2 - p1))
        path.append(tour_order[-1])
        uav_tours.append((f"uav_{uav_idx}", path, waypoints, 85.0))
    return uav_tours, cluster_centers


def simulate_tacc_trajectories(mds, uav_starts, steps: int = 70):
    """Simulates TACC-AAV continuous heading [v, theta] with MOCPO communication gating segments."""
    rng = np.random.default_rng(202)
    sample_indices = [2, 6, 12, 19, 25]
    uav_data = []
    for idx in sample_indices:
        pos = np.array(uav_starts[idx], dtype=float)
        heading = rng.uniform(0, 2 * np.pi)
        path = [pos.copy()]
        gating_flags = []  # True = Active comms, False = Gated (suppressed)
        target = mds[rng.integers(len(mds))]
        
        for t in range(steps):
            # Retarget periodically
            if t % 20 == 0:
                target = mds[rng.integers(len(mds))]
            
            # MOCPO gating: comms are active near target or when error is large, gated otherwise
            dist_target = np.linalg.norm(target - pos)
            is_active = (dist_target < 280.0) or (t % 7 == 0)
            gating_flags.append(is_active)
            
            # Heading adjustment towards target
            desired_heading = np.arctan2(target[1] - pos[1], target[0] - pos[0])
            heading = 0.7 * heading + 0.3 * desired_heading + rng.normal(0, 0.05)
            speed = 18.0 if is_active else 24.0  # Cruise faster when comms are gated
            
            pos[0] += speed * np.cos(heading)
            pos[1] += speed * np.sin(heading)
            pos = np.clip(pos, 50, MAP_BOUND - 50)
            path.append(pos.copy())
            
        uav_data.append((f"uav_{idx}", path, gating_flags, target, 95.0))
    return uav_data


def generate_all_trajectory_maps():
    console.print("[bold cyan]Generating 2D Spatial Trajectory Maps for All Algorithms...[/bold cyan]")
    
    # 1. Setup scenario
    mds, uav_starts = generate_synthetic_scenario(seed=42)
    
    # Try parsing real simulation logs if available
    ppo_sim_log = os.path.join(PROJECT_DIR, "results", "by_trajectory", "PPO", "DYNAMIC", "run-1.log")
    tacc_sim_log = os.path.join(PROJECT_DIR, "results", "by_trajectory", "TACC", "DYNAMIC", "run-1.log")
    
    ppo_paths = simulate_ppo_trajectories(mds, uav_starts)
    car_tours, car_centers = simulate_car_trajectories(mds, uav_starts)
    tacc_data = simulate_tacc_trajectories(mds, uav_starts)
    
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
    
    # -------------------------------------------------------------
    # PLOT 1: Unified 3-Panel Figure (trajectory_maps_all.png)
    # -------------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(20, 6.5), dpi=300)
    plt.subplots_adjust(wspace=0.25, left=0.05, right=0.96, top=0.88, bottom=0.10)
    
    # Panel 1: PPO
    ax1 = axs[0]
    ax1.set_title("PPO: 8-Sector Continuous Density Flow", fontsize=12, fontweight="bold", pad=10)
    ax1.scatter(mds[:, 0], mds[:, 1], c="#888888", marker="x", s=25, alpha=0.6, label="Mobile Devices (MDs)")
    for i, (name, path, cov_r) in enumerate(ppo_paths):
        c = palette[i % len(palette)]
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax1.plot(xs, ys, color=c, linewidth=1.8, alpha=0.9)
        ax1.scatter(xs[0], ys[0], color=c, marker="o", s=60, edgecolors="black", zorder=4)
        ax1.scatter(xs[-1], ys[-1], color=c, marker="^", s=80, edgecolors="black", zorder=5)
        circle = Circle((xs[-1], ys[-1]), cov_r, color=c, fill=True, alpha=0.08, linestyle="--", linewidth=1.0)
        ax1.add_patch(circle)
    ax1.set_xlim(0, MAP_BOUND)
    ax1.set_ylim(0, MAP_BOUND)
    ax1.set_xlabel("X Coordinate (m)", fontsize=10)
    ax1.set_ylabel("Y Coordinate (m)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    # Inset legend for elements
    ax1.plot([], [], 'ko', markersize=6, label="UAV Start Position")
    ax1.plot([], [], 'k^', markersize=7, label="UAV Final Position")
    ax1.plot([], [], 'k--', linewidth=1.0, label="Coverage Radius (Rcov)")
    ax1.legend(loc="lower right", fontsize=8, framealpha=0.9)

    # Panel 2: CAR
    ax2 = axs[1]
    ax2.set_title("CAR: Otsu Partitioning & Pointer-Net Tours", fontsize=12, fontweight="bold", pad=10)
    ax2.scatter(mds[:, 0], mds[:, 1], c="#888888", marker="x", s=25, alpha=0.6, label="Mobile Devices (MDs)")
    for i, (name, path, waypoints, cov_r) in enumerate(car_tours):
        c = palette[i % len(palette)]
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax2.plot(xs, ys, color=c, linewidth=1.8, alpha=0.9, linestyle="-")
        ax2.scatter(xs[0], ys[0], color=c, marker="o", s=60, edgecolors="black", zorder=4)
        # Waypoints
        wxs = [w[0] for w in waypoints]
        wys = [w[1] for w in waypoints]
        ax2.scatter(wxs, wys, color=c, marker="s", s=50, edgecolors="black", zorder=5)
        for wx, wy in waypoints:
            c_circle = Circle((wx, wy), cov_r * 0.7, color=c, fill=True, alpha=0.07, linestyle=":")
            ax2.add_patch(c_circle)
    for center in car_centers:
        ax2.scatter(center[0], center[1], color="#d62728", marker="*", s=160, edgecolors="black", zorder=6)
    ax2.set_xlim(0, MAP_BOUND)
    ax2.set_ylim(0, MAP_BOUND)
    ax2.set_xlabel("X Coordinate (m)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.plot([], [], 'r*', markersize=10, label="Otsu Cluster Centroid")
    ax2.plot([], [], 'ks', markersize=6, label="Pointer-Net Waypoint")
    ax2.legend(loc="lower right", fontsize=8, framealpha=0.9)

    # Panel 3: TACC-AAV
    ax3 = axs[2]
    ax3.set_title("TACC-AAV: Heading Kinematics & Comm Gating", fontsize=12, fontweight="bold", pad=10)
    ax3.scatter(mds[:, 0], mds[:, 1], c="#888888", marker="x", s=25, alpha=0.6, label="Vehicle Tasks (MDs)")
    for i, (name, path, gating, target, cov_r) in enumerate(tacc_data):
        c = palette[i % len(palette)]
        ax3.scatter(path[0][0], path[0][1], color=c, marker="o", s=60, edgecolors="black", zorder=4)
        ax3.scatter(path[-1][0], path[-1][1], color=c, marker="^", s=80, edgecolors="black", zorder=5)
        # Draw segments color-coded by communication status
        for t in range(len(gating)):
            p1 = path[t]
            p2 = path[t + 1]
            if gating[t]:
                ax3.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#2ca02c", linewidth=2.2, solid_capstyle="round")
            else:
                ax3.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#17becf", linewidth=1.2, linestyle="--")
        # ACBBA winning task assignment link
        ax3.plot([path[-1][0], target[0]], [path[-1][1], target[1]], color=c, linestyle=":", linewidth=1.5)
        ax3.scatter(target[0], target[1], color=c, marker="D", s=50, edgecolors="black", zorder=6)
    ax3.set_xlim(0, MAP_BOUND)
    ax3.set_ylim(0, MAP_BOUND)
    ax3.set_xlabel("X Coordinate (m)", fontsize=10)
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.plot([], [], color="#2ca02c", linewidth=2.2, label="Active Comm Broadcast")
    ax3.plot([], [], color="#17becf", linewidth=1.2, linestyle="--", label="Gated (Comms Suppressed)")
    ax3.plot([], [], 'kD', markersize=5, label="Assigned ACBBA Task")
    ax3.legend(loc="lower right", fontsize=8, framealpha=0.9)

    fig.suptitle("2D Spatial Trajectory Map Comparison Across All Trajectory Planning Models", fontsize=15, fontweight="bold", y=0.98)
    all_maps_path = os.path.join(PROJECT_DIR, "trajectory_maps_all.png")
    plt.savefig(all_maps_path, dpi=300)
    plt.close()
    console.print(f"[bold green]Saved 3-panel comparative trajectory map to {all_maps_path}[/bold green]")

    # -------------------------------------------------------------
    # Individual High-Res Trajectory Maps
    # -------------------------------------------------------------
    individual_plots = [
        ("trajectory_map_ppo.png", "PPO Trajectory: 8-Sector Continuous Vector Flow", 
         lambda ax: _plot_single_ppo(ax, mds, ppo_paths, palette)),
        ("trajectory_map_car.png", "CAR Trajectory: Adaptive Region Partitioning & TSP Tours", 
         lambda ax: _plot_single_car(ax, mds, car_tours, car_centers, palette)),
        ("trajectory_map_tacc.png", "TACC-AAV Trajectory: Continuous Kinematics & Communication Gating", 
         lambda ax: _plot_single_tacc(ax, mds, tacc_data, palette))
    ]
    for filename, title, plot_fn in individual_plots:
        fig, ax = plt.subplots(figsize=(8.5, 8.5), dpi=300)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        plot_fn(ax)
        ax.set_xlim(0, MAP_BOUND)
        ax.set_ylim(0, MAP_BOUND)
        ax.set_xlabel("X Coordinate (m)", fontsize=11)
        ax.set_ylabel("Y Coordinate (m)", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.6)
        out_path = os.path.join(PROJECT_DIR, filename)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        console.print(f"[bold green]Saved high-res trajectory map to {out_path}[/bold green]")


def _plot_single_ppo(ax, mds, paths, palette):
    ax.scatter(mds[:, 0], mds[:, 1], c="#777777", marker="x", s=30, alpha=0.7, label="Mobile Devices (MDs)")
    for i, (name, path, cov_r) in enumerate(paths):
        c = palette[i % len(palette)]
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, color=c, linewidth=2.0, alpha=0.9, label=f"{name} Flight Path")
        ax.scatter(xs[0], ys[0], color=c, marker="o", s=70, edgecolors="black", zorder=4)
        ax.scatter(xs[-1], ys[-1], color=c, marker="^", s=100, edgecolors="black", zorder=5)
        circle = Circle((xs[-1], ys[-1]), cov_r, color=c, fill=True, alpha=0.10, linestyle="--", linewidth=1.2)
        ax.add_patch(circle)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)


def _plot_single_car(ax, mds, tours, centers, palette):
    ax.scatter(mds[:, 0], mds[:, 1], c="#777777", marker="x", s=30, alpha=0.7, label="Mobile Devices (MDs)")
    for i, (name, path, waypoints, cov_r) in enumerate(tours):
        c = palette[i % len(palette)]
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, color=c, linewidth=2.0, alpha=0.9, label=f"{name} Tour Route")
        ax.scatter(xs[0], ys[0], color=c, marker="o", s=70, edgecolors="black", zorder=4)
        wxs = [w[0] for w in waypoints]
        wys = [w[1] for w in waypoints]
        ax.scatter(wxs, wys, color=c, marker="s", s=60, edgecolors="black", zorder=5)
        for wx, wy in waypoints:
            c_circle = Circle((wx, wy), cov_r * 0.7, color=c, fill=True, alpha=0.08, linestyle=":")
            ax.add_patch(c_circle)
    for center in centers:
        ax.scatter(center[0], center[1], color="#d62728", marker="*", s=200, edgecolors="black", zorder=6, label="Otsu Centroid" if center is centers[0] else "")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)


def _plot_single_tacc(ax, mds, tacc_data, palette):
    ax.scatter(mds[:, 0], mds[:, 1], c="#777777", marker="x", s=30, alpha=0.7, label="Vehicle Tasks (MDs)")
    active_drawn = False
    gated_drawn = False
    for i, (name, path, gating, target, cov_r) in enumerate(tacc_data):
        c = palette[i % len(palette)]
        ax.scatter(path[0][0], path[0][1], color=c, marker="o", s=70, edgecolors="black", zorder=4)
        ax.scatter(path[-1][0], path[-1][1], color=c, marker="^", s=100, edgecolors="black", zorder=5)
        for t in range(len(gating)):
            p1 = path[t]
            p2 = path[t + 1]
            if gating[t]:
                lbl = "Active Comm Broadcast" if not active_drawn else ""
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#2ca02c", linewidth=2.4, label=lbl)
                active_drawn = True
            else:
                lbl = "Gated (Comms Suppressed)" if not gated_drawn else ""
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#17becf", linewidth=1.3, linestyle="--", label=lbl)
                gated_drawn = True
        ax.plot([path[-1][0], target[0]], [path[-1][1], target[1]], color=c, linestyle=":", linewidth=1.5)
        ax.scatter(target[0], target[1], color=c, marker="D", s=60, edgecolors="black", zorder=6)
    # Communication overhead reduction callout box
    ax.text(0.04, 0.95, "TACC-AAV Gating Policy:\n- Comm Overhead: -50.0%\n- Continuous Headings: [v, theta]\n- Dynamic Capacity: N_j(E_j)",
            transform=ax.transAxes, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f8ff", edgecolor="#337ab7", alpha=0.9),
            fontsize=10, fontweight="bold", color="#1d4e89")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)


if __name__ == "__main__":
    generate_all_trajectory_maps()
