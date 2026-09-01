#!/usr/bin/env python3
"""
Interactive mix-and-match FANET benchmarker.

Run any combination of task schedulers x trajectory models and compare them.

Usage:
    .venv/bin/python run_compare.py           # interactive menu
    .venv/bin/python run_compare.py --sched MOGS,MARL --traj both

The servers required by a combo are started once (and reused across runs),
then shut down when the script exits. Full raw logs are kept under:
    results/compare_<timestamp>/<scheduler>_<trajectory>/run-<n>.log
Aggregated results also go to results/compare_<timestamp>/results.csv
"""

from __future__ import annotations

import datetime as dt
import os
import re
import socket
import subprocess
import sys
import time

from rich.console import Console
from rich.panel import Panel
from rich.progress import (BarColumn, Progress, TextColumn,
                           TaskProgressColumn, TimeElapsedColumn)
from rich.table import Table

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHON = os.path.join(PROJECT_DIR, ".venv", "bin", "python")
SERVERS = {
    "PPO": ("ppo_agent/tf_ppo_server.py", 5500),
    "CAR": ("ppo_agent/car_ppo_server.py", 5510),
    "MARL": ("ppo_agent/marl_allocation_server.py", 5501),
}

SCHEDULERS = ["MOGS", "LEXICOGRAPHIC", "BIDDING", "MARL", "DYNAMIC"]
TRAJECTORIES = ["PPO", "CAR"]

JAVA_CP = f"out/production/iFogSim:jars/*:jars/commons-math3-3.5/*"

# (label, regex, higher_is_better)
METRICS = [
    ("Tasks Completed", r"System Total Tasks Completed:\s*(\d+)", True),
    ("Throughput (MB)", r"System Total Throughput:\s*([\d.]+)\s*MB", True),
    ("Avg Latency (ms)", r"Average System Latency:\s*([\d.]+)\s*ms", False),
    ("Tasks Dropped", r"Total Tasks Dropped\s*:\s*(\d+)", False),
    ("Success Rate (%)", r"Scheduling Success Rate\s*:\s*([\d.]+)%", True),
    ("Sched Overhead (ms)", r"Average Scheduling Overhead\s*:\s*([\d.]+) ms per batch", False),
    ("P2P Offloads", r"System Total Offloaded Tasks\s*:\s*(\d+)", True),
    ("Average Throughput-Thr", r"│ Average Throughput\s*│\s*([\d.]+)\s*│", True),
    ("UAV Utilization", r"│ Average UAV Utilization\s*│\s*([\d.]+)\s*│", True),
    ("Capacity Utilization", r"│ Average Capacity Util\s*│\s*([\d.]+)\s*│", True),
    ("Unassigned Tasks", r"│ Average Unassigned Tasks\s*│\s*([\d.]+)\s*│", False),
]
# DYNAMIC-scheduler-only metrics (shown when available)
DYNAMIC_METRICS = [
    ("Repair Queue Len", r"│ Average Repair Queue\s*│\s*([\d.]+)\s*│", False),
    ("Invalid Assigns", r"│ Average Invalid Assigns\s*│\s*([\d.]+)\s*│", False),
    ("Reassigned Tasks", r"│ Average Reassigned Tasks\s*│\s*([\d.]+)\s*│", False),
]

console = Console()


def parse_metrics(log_path: str) -> dict:
    """Pull every available metric value out of one simulation stdout log."""
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    values: dict = {}
    for label, pattern, _ in METRICS + DYNAMIC_METRICS:
        m = re.search(pattern, text)
        if m:
            values[label] = float(m.group(1))
    return values


def port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        try:
            s.connect(("127.0.0.1", port))
            return True
        except OSError:
            return False


def wait_port(port: int, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if port_open(port):
            return True
        time.sleep(0.5)
    return False


def compile_java() -> None:
    with console.status("[bold yellow]Compiling Java sources...", spinner="dots"):
        cmd = ["javac", "--release", "21", "-d", "out/production/iFogSim",
               "-cp", "jars/*:jars/commons-math3-3.5/*",
               "-sourcepath", "src",
               "src/org/fog/test/perfeval/FANETSimulation.java"]
        res = subprocess.run(cmd, cwd=PROJECT_DIR, capture_output=True, text=True)
        if res.returncode != 0:
            console.print("[bold red]Compilation failed:[/bold red]")
            console.print(res.stderr[-2000:])
            sys.exit(1)
    console.print("[bold green]Compilation successful.[/bold green]")


def pick_choice(prompt: str, options: list[str], default: list[str] | None = None,
                any_keyword: str = "all") -> list[str]:
    """Print a numbered menu and read a comma/space-separated selection."""
    console.print(Panel(
        "\n".join(f"[cyan]{i + 1}[/cyan]  {name}" for i, name in enumerate(options)),
        title=prompt, expand=False))
    default_s = ",".join(default) if default is not None else ""
    hint = f" (default: {default_s})" if default is not None else ""
    while True:
        raw = input(f"> numbers, or '{any_keyword}'{hint}: ").strip().lower()
        if raw in ("", "a", any_keyword) and default is not None:
            return list(default)
        if raw in ("", "a", any_keyword):
            return list(options)
        parts = re.split(r"[,\s]+", raw)
        chosen, bad = [], []
        for p in parts:
            try:
                idx = int(p) - 1
                chosen.append(options[idx])
            except (ValueError, IndexError):
                bad.append(p)
        if chosen and not bad:
            return chosen
        console.print(f"[red]Invalid: {', '.join(bad) or raw}. Try again.[/red]")


def free_ports(ports: list[int]) -> None:
    """Kill any stale process currently bound to one of our benchmark ports."""
    for port in ports:
        try:
            out = subprocess.check_output(["lsof", "-ti", f"tcp:{port}"],
                                          text=True, stderr=subprocess.DEVNULL)
            for pid in out.split():
                os.kill(int(pid), 15)
            time.sleep(0.5)
        except subprocess.CalledProcessError:
            pass


def start_servers(needed: set[str], log_dir: str) -> list[subprocess.Popen]:
    """Kill stale owners of our ports, then start every required server fresh."""
    free_ports([SERVERS[name][1] for name in needed])
    pids: list[subprocess.Popen] = []
    for name in sorted(needed, key=lambda n: SERVERS[n][1]):
        script, port = SERVERS[name]
        log = open(os.path.join(log_dir, f"{name}.log"), "wb")
        proc = subprocess.Popen(
            [VENV_PYTHON, "-u", script],
            cwd=PROJECT_DIR, stdout=log, stderr=subprocess.STDOUT)
        pids.append(proc)
        if not wait_port(port):
            console.print(f"[bold red]Server {name} did not come up on port {port}.[/bold red]")
            continue
        console.print(f"[green]Started {name} server on port {port} "
                      f"[dim](log: {os.path.basename(log.name)})[/dim].[/green]")
    return pids


def run_simulation(scheduler: str, trajectory: str, log_path: str) -> dict:
    cmd = ["java", "-cp", JAVA_CP, "org.fog.test.perfeval.FANETSimulation",
           scheduler, trajectory]
    with open(log_path, "w") as f:
        res = subprocess.run(cmd, cwd=PROJECT_DIR, stdout=f, stderr=subprocess.STDOUT)
    if res.returncode != 0:
        return {}
    return parse_metrics(log_path)


def main() -> None:
    console.print(Panel(
        "[bold green]iFogSim FANET mix-and-match benchmarker[/bold green]\n"
        "Compare any task scheduler x trajectory model combination.",
        expand=False))

    # 1. Configuration (skippable via CLI flags)
    argv = sys.argv[1:]
    opts: dict[str, str] = {}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--") and "=" in a:
            k, v = a[2:].split("=", 1)
            opts[k] = v
        elif a.startswith("--") and i + 1 < len(argv):
            opts[a[2:]] = argv[i + 1]
            i += 1
        i += 1
    if opts:
        scheds = [s.upper() for s in opts.get("sched", "").split(",") if s]
        scheds = [s for s in scheds if s in SCHEDULERS] or SCHEDULERS
        trajs = {"ppo": ["PPO"], "car": ["CAR"], "both": ["PPO", "CAR"]} \
            .get((opts.get("traj", "") or "both").lower(), ["PPO", "CAR"])
        try:
            num_runs = int(opts.get("runs", 3)) if opts else 3
        except ValueError:
            num_runs = 3
    else:
        scheds = pick_choice("Select schedulers", SCHEDULERS, default=SCHEDULERS)
        console.print()
        traj_raw = input("Trajectory models  [ppo / car / both] (default: both): ").strip().lower()
        trajs = {"ppo": ["PPO"], "car": ["CAR"], "both": ["PPO", "CAR"]}.get(traj_raw, ["PPO", "CAR"])
        raw = input("Runs per combination [int, default: 3]: ").strip()
        try:
            num_runs = max(1, int(raw))
        except ValueError:
            num_runs = 3

    combos = [f"{s}/{t}" for s in scheds for t in trajs]
    if not combos:
        console.print("[red]No combinations selected.[/red]")
        return

    # 2. Compile + spin up servers once
    compile_java()
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(PROJECT_DIR, "results", f"compare_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    server_log_dir = os.path.join(out_dir, "servers")
    os.makedirs(server_log_dir, exist_ok=True)

    needed = {"MARL", "PPO" if "PPO" in trajs else "", "CAR" if "CAR" in trajs else ""}
    needed.discard("")
    procs = start_servers(needed, server_log_dir)

    console.print(Panel(
        "[bold]Plan:[/bold]\n" + "\n".join(f"  • {c}  x {num_runs} run(s)" for c in combos),
        expand=False))

    # 3. Run every combination
    results: dict[str, list[dict]] = {c: [] for c in combos}
    total_runs = len(combos) * num_runs
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(), TimeElapsedColumn(), console=console,
    ) as progress:
        task = progress.add_task(f"[cyan]{total_runs} simulation runs", total=total_runs)
        for sched in scheds:
            for traj in trajs:
                combo = f"{sched}/{traj}"
                combo_dir = os.path.join(out_dir, f"{sched}_{traj}")
                os.makedirs(combo_dir, exist_ok=True)
                for run_no in range(1, num_runs + 1):
                    log_path = os.path.join(combo_dir, f"run-{run_no}.log")
                    progress.update(task, description=f"[cyan]{combo} run {run_no}/{num_runs}")
                    metrics = run_simulation(sched, traj, log_path)
                    if metrics:
                        results[combo].append(metrics)
                    else:
                        console.print(f"[yellow]Warning: {combo} run {run_no} produced no metrics "
                                      f"(see {log_path}).[/yellow]")
                    progress.advance(task)

    # 4. Aggregate + persist
    csv_path = os.path.join(out_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(["combo", "run", *[label for label, _, _ in METRICS + DYNAMIC_METRICS]])
        for combo, runs in results.items():
            for i, m in enumerate(runs, 1):
                writer.writerow([combo, i,
                                 *[m.get(label, "") for label, _, _ in METRICS + DYNAMIC_METRICS]])

    # 5. Comparison table with best/worst highlighting
    summary: dict[str, dict] = {}
    for combo, runs in results.items():
        summary[combo] = {label: [m.get(label) for m in runs if m.get(label) is not None]
                          for label, _, _ in METRICS + DYNAMIC_METRICS}

    def avg(xs):
        xs = list(xs)
        return sum(xs) / len(xs) if xs else None

    table = Table(title=f"BENCHMARK COMPARISON — {num_runs} run(s) per combo",
                  header_style="bold magenta")
    table.add_column("Metric", style="dim", width=26)
    for combo in combos:
        table.add_column(combo, justify="right")

    for metric in METRICS:
        label, _, higher = metric
        combos_vals = {c: avg(v for v in summary[c][label]) for c in combos}
        cvals = {c: v for c, v in combos_vals.items() if v is not None}
        best = max(cvals.values()) if cvals and higher else min(cvals.values()) if cvals else None
        worst = min(cvals.values()) if cvals and higher else max(cvals.values()) if cvals else None
        row = [label]
        for combo in combos:
            v = combos_vals[combo]
            if v is None:
                row.append("—")
                continue
            text = f"{v:.3f}"
            if best is not None and abs(v - best) < 1e-9 and len(set(cvals.values())) > 1:
                row.append(f"[bold green]{text}[/bold green]")
            elif worst is not None and abs(v - worst) < 1e-9 and len(set(cvals.values())) > 1:
                row.append(f"[red]{text}[/red]")
            else:
                row.append(text)
        table.add_row(*row)

    console.print()
    console.print(table)

    # DYNAMIC-only metrics (if present in the logs)
    any_dyn = any(summary[c][d[0]] for c in combos for d in DYNAMIC_METRICS
                  if summary[c].get(d[0]))
    if any_dyn:
        dyn_table = Table(title="Dynamic Repair Scheduler Metrics (only when scheduling is DYNAMIC)",
                          header_style="bold cyan")
        dyn_table.add_column("Metric", style="dim", width=26)
        for combo in combos:
            dyn_table.add_column(combo, justify="right")
        for label, _, _ in DYNAMIC_METRICS:
            row = [label]
            for combo in combos:
                xs = summary[combo][label]
                row.append(f"{avg(xs):.3f}" if xs else "—")
            dyn_table.add_row(*row)
        console.print()
        console.print(dyn_table)

    console.print(f"\nRaw logs: [cyan]{out_dir}[/cyan]")
    console.print(f"CSV:     [cyan]{csv_path}[/cyan]")

    # 6. Cleanup: terminate the servers we started, then free the ports
    for proc in procs:
        proc.terminate()
    for proc in procs:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    free_ports([SERVERS[name][1] for name in needed])
    console.print("[dim]Benchmark servers stopped.[/dim]")


if __name__ == "__main__":
    if not os.path.exists(VENV_PYTHON):
        console.print("[red]No .venv found. Run 'python3 -m venv .venv && "
                      ".venv/bin/pip install torch numpy rich' first.[/red]")
        sys.exit(1)
    main()