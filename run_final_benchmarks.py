#!/usr/bin/env python3
"""
Runs missing simulation benchmarks for AP_DYNAMIC and AD_CAR combinations,
storing outputs in results/by_trajectory/<TRAJ>/<SCHED>/run-1.log.
"""

import os
import subprocess
import time
from rich.console import Console

console = Console()
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "by_trajectory")

RUN_LIST = [
    ("CAR", "AP_DYNAMIC"),
    ("TACC", "AP_DYNAMIC"),
    ("AD_CAR", "AP_DYNAMIC"),
    ("AD_CAR", "DYNAMIC"),
    ("AD_CAR", "LEXICOGRAPHIC"),
    ("AD_CAR", "MOGS"),
    ("AD_CAR", "BIDDING"),
]

def run_all():
    console.print("[bold cyan]=======================================================[/bold cyan]")
    console.print("[bold cyan] Running Final Optimization Benchmarks [/bold cyan]")
    console.print("[bold cyan]=======================================================[/bold cyan]\n")

    for traj, sched in RUN_LIST:
        out_dir = os.path.join(RESULTS_DIR, traj, sched)
        os.makedirs(out_dir, exist_ok=True)
        log_path = os.path.join(out_dir, "run-1.log")

        if os.path.exists(log_path) and os.path.getsize(log_path) > 1000:
            console.print(f"  [green]Skipping (already exists):[/green] {traj} x {sched}")
            continue

        console.print(f"  [yellow]Running simulation:[/yellow] {traj} x {sched} ...")
        t0 = time.time()
        cmd = ["./run_single_sim.sh", sched, traj]
        res = subprocess.run(cmd, cwd=PROJECT_DIR, capture_output=True, text=True)
        elapsed = time.time() - t0

        with open(log_path, "w", encoding="utf-8") as f:
            f.write(res.stdout)
            if res.stderr:
                f.write("\n--- STDERR ---\n" + res.stderr)

        console.print(f"  [bold green]✓ Done ({elapsed:.1f}s):[/bold green] {traj} x {sched} -> {log_path}")

    console.print("\n[bold green]All benchmark simulations completed successfully![/bold green]")

if __name__ == "__main__":
    run_all()
