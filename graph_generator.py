#!/usr/bin/env python3
"""
FANET-MEC Cross-Trajectory and Cross-Scheduler Graph Generator.

Performs a full Cartesian cross-product of:
  Trajectory Algorithms: PPO, CAR, TACC
  Task Allocations (Schedulers): DYNAMIC, MOGS, LEXICOGRAPHIC, BIDDING, MARL
  (Total: 15 Combinations)

Organizes and reads simulation outputs from:
  results/by_trajectory/<TRAJECTORY>/<SCHEDULER>/run-<n>.log

Generates:
  - 3-panel comparative time-series plots (PPO | CAR | TACC side-by-side):
      1. throughput_comparison.png
      2. runtime_comparison.png
      3. utilization_comparison.png
      4. capacity_utilization_comparison.png
      5. queue_length_comparison.png
      6. repair_queue_comparison.png
      7. fairness_comparison.png
      8. churn_comparison.png
      9. slack_margin_comparison.png
  - Macro cross-product comparative bar chart:
      10. trajectory_cross_summary.png
"""

import os
import sys
import re
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table

console = Console()
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "by_trajectory")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "images", "benchmarks")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAJECTORIES = ["PPO", "CAR", "TACC"]
SCHEDULERS = {
    "DYNAMIC": "#2ca02c",        # Green
    "MOGS": "#e377c2",           # Magenta/Pink
    "LEXICOGRAPHIC": "#17becf",  # Cyan
    "BIDDING": "#ff7f0e",        # Orange
    "MARL": "#1f77b4"            # Blue
}

FORCE_RUN = "--run" in sys.argv
NUM_RUNS = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 1


def parse_epoch_metrics(text: str) -> list[dict]:
    """Parses [METRIC_EPOCH] telemetry lines from simulation log."""
    epochs = []
    pattern = (
        r"\[METRIC_EPOCH\] Epoch:\d+, Generated:(\d+), Scheduled:(\d+), Dropped:\d+, "
        r"RuntimeMs:([\d.]+), UavUtil:([\d.]+), CapUtil:([\d.]+), RepairQ:(\d+), "
        r"Invalid:(\d+), Reassigned:(\d+)(?:, Fairness:([\d.]+), Churn:([\d.]+), Slack:([\d.]+))?"
    )
    for line in text.splitlines():
        match = re.search(pattern, line)
        if match:
            generated = max(1, int(match.group(1)))
            scheduled = int(match.group(2))
            throughput = scheduled / generated
            fairness = float(match.group(9)) if match.group(9) is not None else 1.0
            churn = float(match.group(10)) if match.group(10) is not None else 0.0
            slack = float(match.group(11)) if match.group(11) is not None else 0.0
            epochs.append({
                'throughput': throughput,
                'runtime': float(match.group(3)),
                'uav_util': float(match.group(4)),
                'cap_util': float(match.group(5)),
                'queue': float(match.group(1)) - scheduled,
                'repair_queue': float(match.group(6)),
                'invalid': float(match.group(7)),
                'reassigned': float(match.group(8)),
                'fairness': fairness,
                'churn': churn,
                'slack': slack
            })
    return epochs


def parse_summary_metrics(text: str) -> dict:
    """Extracts final summary KPIs from simulation log."""
    summary = {}
    patterns = {
        "success_rate": r"Scheduling Success Rate\s*:\s*([\d.]+)%",
        "throughput_mb": r"System Total Throughput:\s*([\d.]+)\s*MB",
        "overhead_ms": r"Average Scheduling Overhead\s*:\s*([\d.]+) ms",
        "fairness": r"Average Jain's Fairness\s*:\s*([\d.]+)",
        "churn": r"Average Allocation Churn\s*:\s*([\d.]+)",
        "slack": r"Average Deadline Slack\s*:\s*([\d.]+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            summary[key] = float(m.group(1))
    return summary


def ensure_simulation_data(traj: str, sched: str) -> tuple[list[dict], dict]:
    """Retrieves or executes simulation for trajectory x scheduler combo."""
    combo_dir = os.path.join(RESULTS_DIR, traj, sched)
    os.makedirs(combo_dir, exist_ok=True)
    log_path = os.path.join(combo_dir, "run-1.log")
    
    need_sim = FORCE_RUN or not os.path.exists(log_path) or os.path.getsize(log_path) == 0
    if not need_sim:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        epochs = parse_epoch_metrics(content)
        summary = parse_summary_metrics(content)
        if len(epochs) >= 80:
            return epochs, summary
        else:
            need_sim = True

    console.print(f"  [yellow]Executing simulation:[/yellow] {sched} x {traj} ...")
    cmd = ["./run_single_sim.sh", sched, traj]
    res = subprocess.run(cmd, cwd=PROJECT_DIR, capture_output=True, text=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(res.stdout)
        if res.stderr:
            f.write("\n--- STDERR ---\n" + res.stderr)
            
    epochs = parse_epoch_metrics(res.stdout)
    summary = parse_summary_metrics(res.stdout)
    return epochs, summary


def generate_graphs():
    console.print("[bold cyan]================================================================[/bold cyan]")
    console.print("[bold cyan] FANET-MEC: Cross-Multiplying All Trajectories x Schedulers [/bold cyan]")
    console.print("[bold cyan]================================================================[/bold cyan]\n")

    # Data structure: data[traj][sched] = {'epochs': [...], 'summary': {...}}
    dataset: dict[str, dict[str, dict]] = {t: {} for t in TRAJECTORIES}
    
    for traj in TRAJECTORIES:
        console.print(f"[bold magenta]Trajectory Algorithm: {traj}[/bold magenta]")
        for sched in SCHEDULERS.keys():
            epochs, summary = ensure_simulation_data(traj, sched)
            dataset[traj][sched] = {
                'epochs': epochs,
                'summary': summary
            }
            count = len(epochs)
            succ = summary.get('success_rate', 0.0)
            console.print(f"  [green]✓[/green] {sched:<14} -> {count} epochs, Success Rate: {succ:.2f}%")
        console.print()

    # Define metric specifications for time-series figures
    metrics_to_plot = [
        ("throughput", "Throughput (Scheduled / Generated)", "Average Throughput Over Time", "throughput_comparison.png", True),
        ("runtime", "Scheduling Overhead (ms)", "Average Scheduling Runtime Over Time", "runtime_comparison.png", True),
        ("uav_util", "UAV Fleet Utilization", "Average Fleet Utilization Over Time", "utilization_comparison.png", True),
        ("cap_util", "Capacity Utilization", "Average Capacity Utilization Over Time", "capacity_utilization_comparison.png", True),
        ("queue", "Unassigned Tasks (Queue Length)", "Average Queue Length Over Time", "queue_length_comparison.png", True),
        ("repair_queue", "Repair Queue Length", "Dynamic Matching Repair Queue Over Time", "repair_queue_comparison.png", False),
        ("fairness", "Jain's Fairness Index", "Swarm Workload Balance Over Time (Jain's Index)", "fairness_comparison.png", True),
        ("churn", "Allocation Churn Rate", "Task Allocation Churn Rate Over Time", "churn_comparison.png", True),
        ("slack", "Deadline Slack Margin", "Average Deadline Slack Margin Over Time", "slack_margin_comparison.png", True),
    ]

    console.print("[bold cyan]Rendering 3-Panel Side-by-Side Comparative Figures...[/bold cyan]")

    for key, ylabel, title, filename, include_all in metrics_to_plot:
        fig, axs = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True, dpi=200)
        plt.subplots_adjust(wspace=0.15, left=0.06, right=0.96, top=0.86, bottom=0.12)
        
        # Calculate global y-bounds for honest comparison across panels
        all_vals = []
        for traj in TRAJECTORIES:
            for sched in SCHEDULERS.keys():
                if not include_all and sched != "DYNAMIC":
                    continue
                epochs = dataset[traj].get(sched, {}).get('epochs', [])
                if epochs:
                    all_vals.extend([e[key] for e in epochs])
        
        if all_vals:
            y_min = max(0.0, np.min(all_vals) * 0.9)
            y_max = np.max(all_vals) * 1.1 if np.max(all_vals) > 0 else 1.0
        else:
            y_min, y_max = 0.0, 1.0

        for col_idx, traj in enumerate(TRAJECTORIES):
            ax = axs[col_idx]
            traj_label = f"TACC-AAV" if traj == "TACC" else f"{traj} Trajectory"
            ax.set_title(f"Trajectory: {traj_label}", fontsize=12, fontweight="bold", pad=8)
            ax.set_xlabel("Scheduling Cycle (Epoch)", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.set_ylim(y_min, y_max)

            for sched, color in SCHEDULERS.items():
                if not include_all and sched != "DYNAMIC":
                    continue
                epochs = dataset[traj].get(sched, {}).get('epochs', [])
                if not epochs:
                    continue
                y_series = [e[key] for e in epochs]
                x_series = list(range(1, len(y_series) + 1))
                ax.plot(x_series, y_series, label=sched, color=color, linewidth=1.6)

            if col_idx == 2:
                ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

        fig.suptitle(f"{title} (Cross-Trajectory Comparison)", fontsize=14, fontweight="bold", y=0.96)
        out_file = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(out_file, dpi=200)
        plt.close()
        console.print(f"  [green]✓[/green] Saved 3-panel cross-trajectory plot -> [bold]{filename}[/bold]")

    # ------------------------------------------------------------------
    # Macro Cross-Product Summary Chart (trajectory_cross_summary.png)
    # ------------------------------------------------------------------
    console.print("\n[bold cyan]Rendering Macro Cross-Product Summary Bar Chart...[/bold cyan]")
    fig, axs = plt.subplots(2, 2, figsize=(16, 11), dpi=200)
    plt.subplots_adjust(hspace=0.32, wspace=0.18, left=0.07, right=0.96, top=0.92, bottom=0.08)
    
    kpis = [
        ("success_rate", "Scheduling Success Rate (%)", "Scheduling Success Rate by Trajectory & Allocation", axs[0, 0]),
        ("overhead_ms", "Scheduling Overhead (ms)", "Scheduling Algorithm Runtime Overhead (ms)", axs[0, 1]),
        ("fairness", "Jain's Fairness Index", "Swarm Workload Balancing (Jain's Fairness)", axs[1, 0]),
        ("slack", "Deadline Slack Margin", "Average Deadline Slack Margin", axs[1, 1]),
    ]
    
    x = np.arange(len(TRAJECTORIES))
    width = 0.15  # Width of each scheduler bar
    
    for metric_key, ylabel, panel_title, ax in kpis:
        ax.set_title(panel_title, fontsize=12, fontweight="bold", pad=8)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f"TACC-AAV" if t == "TACC" else f"{t}" for t in TRAJECTORIES], fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5, axis='y')
        
        for idx, (sched, color) in enumerate(SCHEDULERS.items()):
            offset = (idx - 2) * width
            vals = []
            for traj in TRAJECTORIES:
                summary = dataset[traj].get(sched, {}).get('summary', {})
                val = summary.get(metric_key, 0.0)
                # Fallback to epoch average if summary string wasn't emitted
                if val == 0.0 and metric_key in ('fairness', 'slack'):
                    epochs = dataset[traj].get(sched, {}).get('epochs', [])
                    if epochs:
                        val = float(np.mean([e[metric_key] for e in epochs]))
                vals.append(val)
            
            bars = ax.bar(x + offset, vals, width, label=sched, color=color, edgecolor="black", alpha=0.88)
            # Label bar tops
            for bar in bars:
                h = bar.get_height()
                fmt = f"{h:.1f}%" if "Rate" in ylabel else (f"{h:.3f}" if h < 1.0 else f"{h:.2f}")
                ax.annotate(fmt,
                            xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=7.5, rotation=0)

        ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    fig.suptitle("Full Cross-Multiplication Benchmark Summary (3 Trajectories × 5 Schedulers)", fontsize=15, fontweight="bold", y=0.97)
    summary_path = os.path.join(OUTPUT_DIR, "trajectory_cross_summary.png")
    plt.savefig(summary_path, dpi=200)
    plt.close()
    console.print(f"  [bold green]✓ Saved macro cross-summary chart -> trajectory_cross_summary.png[/bold green]\n")

    # Print summary table in terminal
    tbl = Table(title="Trajectory x Scheduler Cross-Product Benchmark Summary", show_header=True)
    tbl.add_column("Trajectory", style="bold magenta")
    tbl.add_column("Scheduler", style="cyan")
    tbl.add_column("Success Rate (%)", justify="right")
    tbl.add_column("Overhead (ms)", justify="right")
    tbl.add_column("Fairness", justify="right")
    tbl.add_column("Slack", justify="right")

    for traj in TRAJECTORIES:
        for sched in SCHEDULERS.keys():
            summary = dataset[traj].get(sched, {}).get('summary', {})
            sr = summary.get('success_rate', 0.0)
            ov = summary.get('overhead_ms', 0.0)
            fa = summary.get('fairness', 0.0)
            sl = summary.get('slack', 0.0)
            tbl.add_row(traj, sched, f"{sr:.2f}%", f"{ov:.4f}", f"{fa:.4f}", f"{sl:.4f}")

    console.print(tbl)


if __name__ == "__main__":
    generate_graphs()
