#!/usr/bin/env python3
"""
Generates all empirical comparison figures for H-MET (Hierarchical Macro-Micro Elastic Trajectory)
vs Normal CAR, PPO, and TACC-AAV by directly parsing simulator logs in results/hmet_eval/.
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(PROJECT_DIR, "results", "hmet_eval")
TRAJ_BY_DIR = os.path.join(PROJECT_DIR, "results", "by_trajectory")
BENCH_OUT = os.path.join(PROJECT_DIR, "images", "benchmarks")
TRAJ_OUT = os.path.join(PROJECT_DIR, "images", "trajectories")
os.makedirs(BENCH_OUT, exist_ok=True)
os.makedirs(TRAJ_OUT, exist_ok=True)


def parse_summary(log_path: str) -> dict:
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    patterns = {
        "completed": r"System Total Tasks Completed:\s*(\d+)",
        "throughput_mb": r"System Total Throughput:\s*([\d.]+)\s*MB",
        "latency_ms": r"Average System Latency:\s*([\d.]+)\s*ms",
        "scheduled": r"Total Tasks Successfully Scheduled\s*:\s*(\d+)",
        "dropped": r"Total Tasks Dropped\s*:\s*(\d+)",
        "success_rate": r"Scheduling Success Rate\s*:\s*([\d.]+)%",
        "overhead_ms": r"Average Scheduling Overhead\s*:\s*([\d.]+) ms",
        "fairness": r"Average Jain's Fairness\s*:\s*([\d.]+)",
        "churn": r"Average Allocation Churn\s*:\s*([\d.]+)",
        "slack": r"Average Deadline Slack\s*:\s*([\d.]+)",
        "uav_util": r"Average UAV Utilization\s*│\s*([\d.]+)",
        "cap_util": r"Average Capacity Util\s*│\s*([\d.]+)",
    }
    res = {}
    for k, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            res[k] = float(m.group(1))
    return res


def parse_epochs(log_path: str) -> list[dict]:
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    pattern = (
        r"\[METRIC_EPOCH\] Epoch:(\d+), Generated:(\d+), Scheduled:(\d+), Dropped:\d+, "
        r"RuntimeMs:([\d.]+), UavUtil:([\d.]+), CapUtil:([\d.]+), RepairQ:(\d+), "
        r"Invalid:(\d+), Reassigned:(\d+)(?:, Fairness:([\d.]+), Churn:([\d.]+), Slack:([\d.]+))?"
    )
    epochs = []
    for line in text.splitlines():
        m = re.search(pattern, line)
        if m:
            gen = max(1, int(m.group(2)))
            sched = int(m.group(3))
            epochs.append({
                "epoch": int(m.group(1)),
                "throughput": sched / gen,
                "runtime": float(m.group(4)),
                "uav_util": float(m.group(5)),
                "cap_util": float(m.group(6)),
                "repair_q": float(m.group(7)),
                "fairness": float(m.group(10)) if m.group(10) else 0.0,
                "churn": float(m.group(11)) if m.group(11) else 0.0,
                "slack": float(m.group(12)) if m.group(12) else 0.0,
            })
    return epochs


def parse_trajectories(log_path: str) -> dict[str, list[tuple[float, float]]]:
    trajs: dict[str, list[tuple[float, float]]] = {}
    pattern = r"(uav_\d+):([\d.]+),([\d.]+)"
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "[TRAJECTORY_EPOCH]" in line:
                for m in re.finditer(pattern, line):
                    uid = m.group(1)
                    trajs.setdefault(uid, []).append((float(m.group(2)), float(m.group(3))))
    return trajs


def plot_fleet_scaling():
    fleet_sizes = [30, 45, 60, 80, 100]
    car_stats = [parse_summary(os.path.join(EVAL_DIR, f"car_ap_{n}.log")) for n in fleet_sizes]
    hmet_stats = [parse_summary(os.path.join(EVAL_DIR, f"hmet_ap_{n}.log")) for n in fleet_sizes]

    print("\n=== Fleet Scaling Summary (CAR vs H-MET + AP_DYNAMIC) ===")
    for n, c, h in zip(fleet_sizes, car_stats, hmet_stats):
        print(
            f"N={n:3d} | CAR: SR={c.get('success_rate',0):6.2f}%, Thr={c.get('throughput_mb',0):7.2f}MB, "
            f"OH={c.get('overhead_ms',0):.4f}ms, J={c.get('fairness',0):.4f}, S={c.get('slack',0):.4f} | "
            f"HMET: SR={h.get('success_rate',0):6.2f}%, Thr={h.get('throughput_mb',0):7.2f}MB, "
            f"OH={h.get('overhead_ms',0):.4f}ms, J={h.get('fairness',0):.4f}, S={h.get('slack',0):.4f}"
        )

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axs = plt.subplots(2, 2, figsize=(15, 11), dpi=250)
    plt.subplots_adjust(wspace=0.25, hspace=0.30, left=0.08, right=0.94, top=0.90, bottom=0.08)

    car_sr = [s.get("success_rate", 0) for s in car_stats]
    hmet_sr = [s.get("success_rate", 0) for s in hmet_stats]
    car_thr = [s.get("throughput_mb", 0) for s in car_stats]
    hmet_thr = [s.get("throughput_mb", 0) for s in hmet_stats]
    car_fair = [s.get("fairness", 0) for s in car_stats]
    hmet_fair = [s.get("fairness", 0) for s in hmet_stats]
    car_slack = [s.get("slack", 0) for s in car_stats]
    hmet_slack = [s.get("slack", 0) for s in hmet_stats]
    car_oh = [s.get("overhead_ms", 0) for s in car_stats]
    hmet_oh = [s.get("overhead_ms", 0) for s in hmet_stats]

    # (a) Scheduling Success Rate
    ax1 = axs[0, 0]
    ax1.plot(fleet_sizes, car_sr, marker='s', linestyle='--', color='#1f77b4', lw=2.2, ms=7, label="Normal CAR + AP_DYNAMIC")
    ax1.plot(fleet_sizes, hmet_sr, marker='o', linestyle='-', color='#d62728', lw=2.8, ms=8, label="H-MET + AP_DYNAMIC (Proposed)")
    for x, yc, yh in zip(fleet_sizes, car_sr, hmet_sr):
        diff = yh - yc
        sign = "+" if diff >= 0 else ""
        if x == 30:
            ax1.annotate(f"{yh:.1f}%\n({sign}{diff:.1f}%)", (x, yh), textcoords="offset points", xytext=(24, -16),
                         ha='center', fontsize=8.5, fontweight='bold', color='#555555')
            ax1.annotate(f"{yc:.1f}%", (x, yc), textcoords="offset points", xytext=(-2, 12),
                         ha='center', fontsize=8, color='#1f77b4')
        elif x == 45:
            ax1.annotate(f"{yh:.1f}%\n({sign}{diff:.1f}%)", (x, yh), textcoords="offset points", xytext=(-10, 10),
                         ha='center', fontsize=8.5, fontweight='bold', color='#d62728')
            ax1.annotate(f"{yc:.1f}%", (x, yc), textcoords="offset points", xytext=(16, -15),
                         ha='center', fontsize=8, color='#1f77b4')
        else:
            ax1.annotate(f"{yh:.1f}%\n({sign}{diff:.1f}%)", (x, yh), textcoords="offset points", xytext=(0, 9),
                         ha='center', fontsize=8.5, fontweight='bold', color='#d62728' if diff >= 0 else '#555555')
            ax1.annotate(f"{yc:.1f}%", (x, yc), textcoords="offset points", xytext=(0, -15),
                         ha='center', fontsize=8, color='#1f77b4')
    ax1.set_title("(a) Scheduling Success Rate vs Swarm Size", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Number of UAVs (N)")
    ax1.set_ylabel("Success Rate (%)")
    ax1.set_ylim(35, 112)
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # (b) Cumulative Throughput (MB)
    ax2 = axs[0, 1]
    idx = np.arange(len(fleet_sizes))
    w = 0.35
    ax2.bar(idx - w/2, car_thr, w, color='#6baed6', edgecolor='black', label="Normal CAR + AP_DYNAMIC")
    ax2.bar(idx + w/2, hmet_thr, w, color='#de2d26', edgecolor='black', label="H-MET + AP_DYNAMIC (Proposed)")
    for i, (yc, yh) in enumerate(zip(car_thr, hmet_thr)):
        ax2.text(idx[i] - w/2, yc + 60, f"{yc:.0f}", ha='center', va='bottom', fontsize=8, color='#08519c')
        ax2.text(idx[i] + w/2, yh + 60, f"{yh:.0f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#a50f15')
    ax2.set_xticks(idx)
    ax2.set_xticklabels([str(n) for n in fleet_sizes])
    ax2.set_title("(b) Cumulative System Throughput (MB)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Number of UAVs (N)")
    ax2.set_ylabel("Data Processed (MB)")
    ax2.set_ylim(0, 5400)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.6)

    # (c) Jain's Fairness & Deadline Slack
    ax3 = axs[1, 0]
    l1 = ax3.plot(fleet_sizes, car_fair, marker='s', ls='--', color='#ff7f0e', lw=2.0, label="Fairness (CAR)")
    l2 = ax3.plot(fleet_sizes, hmet_fair, marker='o', ls='-', color='#d95f02', lw=2.5, label="Fairness (H-MET)")
    ax3.set_ylabel("Jain's Fairness Index", color='#d95f02')
    ax3.set_ylim(0.35, 1.05)
    ax3_t = ax3.twinx()
    l3 = ax3_t.plot(fleet_sizes, car_slack, marker='^', ls='--', color='#9467bd', lw=1.8, label="Deadline Slack (CAR)")
    l4 = ax3_t.plot(fleet_sizes, hmet_slack, marker='^', ls='-', color='#6a3d9a', lw=2.5, label="Deadline Slack (H-MET)")
    ax3_t.set_ylabel("Deadline Slack Margin", color='#6a3d9a')
    ax3_t.set_ylim(0.0, 0.42)
    lines = l1 + l2 + l3 + l4
    ax3.legend(lines, [l.get_label() for l in lines], loc="lower left", fontsize=8)
    ax3.set_title("(c) Fairness Equity vs Deadline Slack", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Number of UAVs (N)")
    ax3.grid(True, linestyle="--", alpha=0.6)

    # (d) Real-Time Scheduling Runtime Overhead
    ax4 = axs[1, 1]
    ax4.plot(fleet_sizes, car_oh, marker='s', ls='--', color='#2ca02c', lw=2.0, label="Normal CAR + AP_DYNAMIC")
    ax4.plot(fleet_sizes, hmet_oh, marker='D', ls='-', color='#d62728', lw=2.5, label="H-MET + AP_DYNAMIC")
    ax4.axhline(1.0, color='gray', ls=':', lw=1.5, label="1.0 ms 5G Real-Time TTI Limit")
    for x, yc, yh in zip(fleet_sizes, car_oh, hmet_oh):
        ax4.annotate(f"{yh:.3f}", (x, yh), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=8.5, fontweight='bold', color='#d62728')
    ax4.set_title("(d) Scheduling Batch Overhead vs 5G TTI Budget", fontsize=12, fontweight='bold')
    ax4.set_xlabel("Number of UAVs (N)")
    ax4.set_ylabel("Runtime Overhead (ms)")
    ax4.set_ylim(0.10, 1.15)
    ax4.legend(loc="upper left", fontsize=9)
    ax4.grid(True, linestyle="--", alpha=0.6)

    fig.suptitle("Fleet Scalability Evaluation: H-MET vs Normal CAR (100 Realistic Urban MDs)", fontsize=15, fontweight='bold', y=0.96)
    out = os.path.join(BENCH_OUT, "hmet_fleet_scaling_comparison.png")
    plt.savefig(out, dpi=250)
    plt.close()
    print(f"[SUCCESS] Saved {out}")


def plot_timeseries():
    car_ep = parse_epochs(os.path.join(EVAL_DIR, "car_ap_30.log"))
    hmet_ep = parse_epochs(os.path.join(EVAL_DIR, "hmet_ap_30.log"))
    x = [e["epoch"] for e in hmet_ep]

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), dpi=250)
    plt.subplots_adjust(wspace=0.22, hspace=0.30, left=0.07, right=0.96, top=0.90, bottom=0.08)

    panels = [
        ("throughput", "Per-Epoch Task Throughput Ratio", "(a) Instantaneous Scheduling Throughput (30 UAVs)", axs[0, 0]),
        ("cap_util", "Fleet Buffer Capacity Utilization", "(b) Compute Capacity Utilization Over Time", axs[0, 1]),
        ("fairness", "Jain's Fairness Index (J)", "(c) Swarm Workload Fairness Dynamics", axs[1, 0]),
        ("slack", "Average Deadline Slack Margin (S)", "(d) Deadline Safety Margin Over Time", axs[1, 1]),
    ]

    for key, ylabel, title, ax in panels:
        cy = [e[key] for e in car_ep]
        hy = [e[key] for e in hmet_ep]
        ax.plot(x, cy, color="#1f77b4", ls="--", lw=1.8, alpha=0.85, label=f"Normal CAR (avg={np.mean(cy):.3f})")
        ax.plot(x, hy, color="#d62728", ls="-", lw=2.3, alpha=0.95, label=f"H-MET (avg={np.mean(hy):.3f})")
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("Scheduling Epoch (1000 ms/epoch)")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, ls="--", alpha=0.5)

    fig.suptitle("Epoch-by-Epoch Dynamics at Resource-Constrained Fleet (N=30 UAVs, M=100 MDs)", fontsize=15, fontweight='bold', y=0.96)
    out = os.path.join(BENCH_OUT, "hmet_vs_car_timeseries.png")
    plt.savefig(out, dpi=250)
    plt.close()
    print(f"[SUCCESS] Saved {out}")


def plot_scheduler_ablation():
    scheds = ["DYNAMIC", "MOGS", "LEXICOGRAPHIC", "BIDDING", "MARL", "AP_DYNAMIC"]
    car_srs, hmet_srs = [], []
    car_ohs, hmet_ohs = [], []

    print("\n=== Cross-Scheduler Ablation at N=30 UAVs (CAR vs H-MET) ===")
    for s in scheds:
        if s == "AP_DYNAMIC":
            c = parse_summary(os.path.join(EVAL_DIR, "car_ap_30.log"))
            h = parse_summary(os.path.join(EVAL_DIR, "hmet_ap_30.log"))
        else:
            c = parse_summary(os.path.join(TRAJ_BY_DIR, "CAR", s, "run-1.log"))
            h = parse_summary(os.path.join(EVAL_DIR, f"hmet_{s}_30.log"))
        car_srs.append(c.get("success_rate", 0.0))
        hmet_srs.append(h.get("success_rate", 0.0))
        car_ohs.append(c.get("overhead_ms", 0.0))
        hmet_ohs.append(h.get("overhead_ms", 0.0))
        print(f"{s:<14} | CAR: SR={car_srs[-1]:6.2f}%, OH={car_ohs[-1]:7.4f}ms | H-MET: SR={hmet_srs[-1]:6.2f}%, OH={hmet_ohs[-1]:7.4f}ms")

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axs = plt.subplots(1, 2, figsize=(15, 6), dpi=250)
    plt.subplots_adjust(wspace=0.22, left=0.07, right=0.96, top=0.84, bottom=0.16)

    idx = np.arange(len(scheds))
    w = 0.36

    ax1 = axs[0]
    ax1.bar(idx - w/2, car_srs, w, color='#6baed6', edgecolor='black', label="Normal CAR Trajectory")
    ax1.bar(idx + w/2, hmet_srs, w, color='#de2d26', edgecolor='black', label="H-MET Trajectory (Proposed)")
    for i, (yc, yh) in enumerate(zip(car_srs, hmet_srs)):
        ax1.text(idx[i] - w/2, yc + 1.0, f"{yc:.1f}%", ha='center', va='bottom', fontsize=8, rotation=35)
        ax1.text(idx[i] + w/2, yh + 1.0, f"{yh:.1f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#a50f15', rotation=35)
    ax1.set_xticks(idx)
    ax1.set_xticklabels(scheds, rotation=20, ha='right', fontweight='bold')
    ax1.set_ylim(0, 82)
    ax1.set_ylabel("Scheduling Success Rate (%)")
    ax1.set_title("(a) Scheduling Success Rate Across All 6 Schedulers", fontsize=12, fontweight='bold')
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, ls="--", alpha=0.5, axis='y')

    ax2 = axs[1]
    ax2.bar(idx - w/2, car_ohs, w, color='#6baed6', edgecolor='black', label="Normal CAR Trajectory")
    ax2.bar(idx + w/2, hmet_ohs, w, color='#de2d26', edgecolor='black', label="H-MET Trajectory (Proposed)")
    ax2.axhline(1.0, color='black', ls=':', lw=1.8, label="1.0 ms Real-Time 5G TTI Bound")
    for i, (yc, yh) in enumerate(zip(car_ohs, hmet_ohs)):
        ax2.text(idx[i] + w/2, yh * 1.12, f"{yh:.2f}ms", ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax2.set_yscale("log")
    ax2.set_ylim(0.08, 32)
    ax2.set_xticks(idx)
    ax2.set_xticklabels(scheds, rotation=20, ha='right', fontweight='bold')
    ax2.set_ylabel("Batch Runtime Overhead (ms, log scale)")
    ax2.set_title("(b) Real-Time Scheduling Latency vs 1.0 ms Budget", fontsize=12, fontweight='bold')
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, ls="--", alpha=0.5, axis='y')

    fig.suptitle("Trajectory x Scheduler Cross-Ablation: H-MET vs Normal CAR (N=30 UAVs, M=100 MDs)", fontsize=14, fontweight='bold', y=0.95)
    out = os.path.join(BENCH_OUT, "hmet_scheduler_ablation.png")
    plt.savefig(out, dpi=250)
    plt.close()
    print(f"[SUCCESS] Saved {out}")


def plot_real_spatial_trajectories():
    car_traj = parse_trajectories(os.path.join(EVAL_DIR, "car_ap_30.log"))
    hmet_traj = parse_trajectories(os.path.join(EVAL_DIR, "hmet_ap_30.log"))

    # Compute total distance traveled across fleet to quantify elimination of terminal hover freeze
    def avg_path_len(tmap):
        lengths = []
        for pts in tmap.values():
            arr = np.array(pts)
            d = np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1))
            lengths.append(d)
        return float(np.mean(lengths)), float(np.max(lengths))

    car_avg_d, car_max_d = avg_path_len(car_traj)
    hmet_avg_d, hmet_max_d = avg_path_len(hmet_traj)
    print(f"\n=== Real Trajectory Mobility Stats ===")
    print(f"Normal CAR : Mean Patrol Distance = {car_avg_d:.1f} m, Max = {car_max_d:.1f} m")
    print(f"H-MET      : Mean Patrol Distance = {hmet_avg_d:.1f} m, Max = {hmet_max_d:.1f} m")

    hubs = np.array([[500, 500], [1500, 500], [500, 1500], [1500, 1500]])
    corridors = [
        ([500, 1500], [500, 500]),
        ([500, 1500], [1500, 1500]),
        ([500, 500], [500, 1500]),
        ([1500, 1500], [500, 1500]),
        ([500, 1500], [500, 1500]),
    ]

    fig, axs = plt.subplots(1, 2, figsize=(16, 7.5), dpi=250)
    plt.subplots_adjust(wspace=0.16, left=0.06, right=0.97, top=0.86, bottom=0.10)
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf"]

    # Pick top 8 active UAVs in H-MET for clean visual clarity
    sorted_uavs = sorted(
        hmet_traj.keys(),
        key=lambda u: np.sum(np.linalg.norm(np.diff(np.array(hmet_traj[u]), axis=0), axis=1)),
        reverse=True,
    )[:10]

    for ax, tmap, title, avg_d in [
        (axs[0], car_traj, f"(a) Normal CAR: Open-Loop Static Tours\n(Mean Distance = {car_avg_d:.0f} m — Terminal Hover Freeze)", car_avg_d),
        (axs[1], hmet_traj, f"(b) H-MET [PROPOSED]: Cyclic Loop + Micro Elastic Drift\n(Mean Distance = {hmet_avg_d:.0f} m — Active Demand Pursuit)", hmet_avg_d),
    ]:
        for xs, ys in corridors:
            ax.plot(xs, ys, color="gray", ls="--", lw=3, alpha=0.25)
        for hx, hy in hubs:
            ax.add_patch(Circle((hx, hy), 250, color="crimson", alpha=0.07))
            ax.scatter(hx, hy, color="crimson", marker="*", s=180, zorder=3, edgecolors="black")

        for i, uid in enumerate(sorted_uavs):
            pts = np.array(tmap.get(uid, []))
            if len(pts) == 0:
                continue
            c = palette[i % len(palette)]
            ax.plot(pts[:, 0], pts[:, 1], color=c, lw=2.0, alpha=0.9)
            ax.scatter(pts[0, 0], pts[0, 1], color=c, marker="o", s=55, edgecolors="black", zorder=5)
            ax.scatter(pts[-1, 0], pts[-1, 1], color=c, marker="^", s=80, edgecolors="black", zorder=6)

        ax.set_xlim(0, 2000)
        ax.set_ylim(0, 2000)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=10)
        ax.set_xlabel("X Coordinate (m)")
        ax.set_ylabel("Y Coordinate (m)")
        ax.plot([], [], 'ko', ms=6, label="Epoch 1 Start")
        ax.plot([], [], 'k^', ms=7, label="Epoch 100 Final")
        ax.plot([], [], 'r*', ms=9, label="Urban Hub Center")
        ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
        ax.grid(True, ls="--", alpha=0.5)

    fig.suptitle("Empirical 2D Flight Paths Parsed Directly from iFogSim2 [TRAJECTORY_EPOCH] Logs (N=30 UAVs)", fontsize=14, fontweight='bold', y=0.96)
    out = os.path.join(TRAJ_OUT, "hmet_spatial_trajectory_map.png")
    plt.savefig(out, dpi=250)
    plt.close()
    print(f"[SUCCESS] Saved {out}")


if __name__ == "__main__":
    plot_fleet_scaling()
    plot_timeseries()
    plot_scheduler_ablation()
    plot_real_spatial_trajectories()
