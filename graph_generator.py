import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.progress import track

console = Console()
NUM_RUNS = 3 # Adjust for smoother graphs, takes longer

SCHEDULERS = {
    "MOGS": "magenta",
    "LEXICOGRAPHIC": "cyan",
    "BIDDING": "orange",
    "MARL": "blue",
    "DYNAMIC": "green"
}

def parse_epoch_metrics(output):
    """Parses [METRIC_EPOCH] lines from simulation output."""
    epochs = []
    # e.g. [METRIC_EPOCH] Epoch:1, Generated:100, Scheduled:20, Dropped:80, RuntimeMs:1.5000, UavUtil:0.2, CapUtil:0.05, RepairQ:0, Invalid:0, Reassigned:0
    pattern = r"\[METRIC_EPOCH\] Epoch:\d+, Generated:(\d+), Scheduled:(\d+), Dropped:\d+, RuntimeMs:([\d.]+), UavUtil:([\d.]+), CapUtil:([\d.]+), RepairQ:(\d+), Invalid:(\d+), Reassigned:(\d+)"
    for line in output.split('\n'):
        match = re.search(pattern, line)
        if match:
            generated = max(1, int(match.group(1)))
            scheduled = int(match.group(2))
            throughput = scheduled / generated
            epochs.append({
                'throughput': throughput,
                'runtime': float(match.group(3)),
                'uav_util': float(match.group(4)),
                'cap_util': float(match.group(5)),
                'queue': float(match.group(1)) - scheduled,
                'repair_queue': float(match.group(6)),
                'invalid': float(match.group(7)),
                'reassigned': float(match.group(8))
            })
    return epochs

def generate_graphs():
    console.print(f"[bold cyan]Running {NUM_RUNS} simulations per algorithm to generate graphs...[/bold cyan]\n")
    
    all_metrics = {s: [] for s in SCHEDULERS.keys()}
    
    for sched in SCHEDULERS.keys():
        for i in track(range(NUM_RUNS), description=f"[{SCHEDULERS[sched]}]{sched}...[/{SCHEDULERS[sched]}]"):
            result = subprocess.run(["./run_single_sim.sh", sched], capture_output=True, text=True)
            run_metrics = parse_epoch_metrics(result.stdout)
            if run_metrics:
                all_metrics[sched].append(run_metrics)
                
    # Average across runs
    averaged_metrics = {}
    for sched in SCHEDULERS.keys():
        if not all_metrics[sched]: continue
        
        # Ensure all runs have same length (pad with last value if needed, or truncate to min length)
        min_len = min(len(r) for r in all_metrics[sched])
        if min_len == 0: continue
        
        truncated_runs = [r[:min_len] for r in all_metrics[sched]]
        
        averaged_metrics[sched] = {
            'throughput': np.mean([[e['throughput'] for e in run] for run in truncated_runs], axis=0),
            'runtime': np.mean([[e['runtime'] for e in run] for run in truncated_runs], axis=0),
            'uav_util': np.mean([[e['uav_util'] for e in run] for run in truncated_runs], axis=0),
            'cap_util': np.mean([[e['cap_util'] for e in run] for run in truncated_runs], axis=0),
            'queue': np.mean([[e['queue'] for e in run] for run in truncated_runs], axis=0),
            'repair_queue': np.mean([[e['repair_queue'] for e in run] for run in truncated_runs], axis=0),
            'invalid': np.mean([[e['invalid'] for e in run] for run in truncated_runs], axis=0),
            'reassigned': np.mean([[e['reassigned'] for e in run] for run in truncated_runs], axis=0)
        }

    metrics_to_plot = [
        ("throughput", "Throughput (Scheduled / Generated)", "Average Throughput Over Time", "throughput_comparison.png", True),
        ("runtime", "Runtime (ms)", "Average Runtime Over Time", "runtime_comparison.png", True),
        ("uav_util", "UAV Utilization", "Average UAV Utilization Over Time", "utilization_comparison.png", True),
        ("cap_util", "Capacity Utilization", "Average Capacity Utilization Over Time", "capacity_utilization_comparison.png", True),
        ("queue", "Unassigned Tasks (Queue Length)", "Average Queue Length Over Time", "queue_length_comparison.png", True),
        ("repair_queue", "Repair Queue Length", "Average Repair Queue Over Time", "repair_queue_comparison.png", False),
    ]

    for key, ylabel, title, filename, include_all in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        
        for sched, color in SCHEDULERS.items():
            if sched not in averaged_metrics: continue
            if not include_all and sched != "DYNAMIC": continue
                
            epochs = range(len(averaged_metrics[sched][key]))
            plt.plot(epochs, averaged_metrics[sched][key], label=sched, color=color)
            
        plt.title(title)
        plt.xlabel("Scheduling Cycle (Epoch)")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename)
        console.print(f"[bold green]Saved {title} graph to {filename}[/bold green]")

if __name__ == "__main__":
    generate_graphs()
