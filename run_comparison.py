import subprocess
import re
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.panel import Panel

console = Console()
NUM_RUNS = 100

SCHEDULERS = ["MOGS", "LEXICOGRAPHIC", "BIDDING", "MARL", "DYNAMIC"]

def run_experiment(scheduler_type, progress, task_id):
    total_tasks_completed = []
    total_throughput_mb = []
    average_latency_ms = []
    
    total_tasks_dropped = []
    scheduling_success_rate = []
    average_scheduling_overhead_ms = []
    total_system_offloaded = []
    
    # New metrics
    avg_uav_utilization = []
    avg_capacity_utilization = []
    avg_unassigned_tasks = []
    avg_repair_queue = []
    avg_invalid_assigns = []
    avg_reassigned_tasks = []
    avg_throughput_ratio = []
    
    for i in range(1, NUM_RUNS + 1):
        progress.update(task_id, description=f"[cyan]Running {scheduler_type} {i}/{NUM_RUNS}...")
        
        # Run the simulation
        result = subprocess.run(["./run_single_sim.sh", scheduler_type], capture_output=True, text=True)
        output = result.stdout
    
        # Parse results using regex
        tasks_match = re.search(r"System Total Tasks Completed:\s*(\d+)", output)
        throughput_match = re.search(r"System Total Throughput:\s*([\d.]+)\s*MB", output)
        latency_match = re.search(r"Average System Latency:\s*([\d.]+)\s*ms", output)
        
        dropped_match = re.search(r"Total Tasks Dropped\s*:\s*(\d+)", output)
        success_rate_match = re.search(r"Scheduling Success Rate\s*:\s*([\d.]+)%", output)
        overhead_match = re.search(r"Average Scheduling Overhead\s*:\s*([\d.]+)", output)
        offload_match = re.search(r"System Total Offloaded Tasks\s*:\s*(\d+)", output)
        
        # Parse new benchmark metrics
        uav_util_match = re.search(r"│ Average UAV Utilization\s*│\s*([\d.]+)\s*│", output)
        cap_util_match = re.search(r"│ Average Capacity Util\s*│\s*([\d.]+)\s*│", output)
        unassigned_match = re.search(r"│ Average Unassigned Tasks\s*│\s*([\d.]+)\s*│", output)
        throughput_ratio_match = re.search(r"│ Average Throughput\s*│\s*([\d.]+)\s*│", output)
        
        repair_match = re.search(r"│ Average Repair Queue\s*│\s*([\d.]+)\s*│", output)
        invalid_match = re.search(r"│ Average Invalid Assigns\s*│\s*([\d.]+)\s*│", output)
        reassigned_match = re.search(r"│ Average Reassigned Tasks\s*│\s*([\d.]+)\s*│", output)
        
        if tasks_match and throughput_match and latency_match and dropped_match and success_rate_match and overhead_match and offload_match:
            total_tasks_completed.append(int(tasks_match.group(1)))
            total_throughput_mb.append(float(throughput_match.group(1)))
            average_latency_ms.append(float(latency_match.group(1)))
            total_tasks_dropped.append(int(dropped_match.group(1)))
            scheduling_success_rate.append(float(success_rate_match.group(1)))
            average_scheduling_overhead_ms.append(float(overhead_match.group(1)))
            total_system_offloaded.append(int(offload_match.group(1)))
            
        if uav_util_match and cap_util_match and unassigned_match and throughput_ratio_match:
            avg_uav_utilization.append(float(uav_util_match.group(1)))
            avg_capacity_utilization.append(float(cap_util_match.group(1)))
            avg_unassigned_tasks.append(float(unassigned_match.group(1)))
            avg_throughput_ratio.append(float(throughput_ratio_match.group(1)))
            
        if scheduler_type == "DYNAMIC":
            if repair_match: avg_repair_queue.append(float(repair_match.group(1)))
            if invalid_match: avg_invalid_assigns.append(float(invalid_match.group(1)))
            if reassigned_match: avg_reassigned_tasks.append(float(reassigned_match.group(1)))
        
        progress.advance(task_id)
    
    # Calculate averages
    if total_tasks_completed:
        results = {
            'tasks': sum(total_tasks_completed) / len(total_tasks_completed),
            'throughput_mb': sum(total_throughput_mb) / len(total_throughput_mb),
            'latency': sum(average_latency_ms) / len(average_latency_ms),
            'dropped': sum(total_tasks_dropped) / len(total_tasks_dropped),
            'succ_rate': sum(scheduling_success_rate) / len(scheduling_success_rate),
            'overhead': sum(average_scheduling_overhead_ms) / len(average_scheduling_overhead_ms),
            'offloads': sum(total_system_offloaded) / len(total_system_offloaded),
            'uav_util': sum(avg_uav_utilization) / max(1, len(avg_uav_utilization)),
            'cap_util': sum(avg_capacity_utilization) / max(1, len(avg_capacity_utilization)),
            'unassigned': sum(avg_unassigned_tasks) / max(1, len(avg_unassigned_tasks)),
            'throughput_ratio': sum(avg_throughput_ratio) / max(1, len(avg_throughput_ratio))
        }
        
        if scheduler_type == "DYNAMIC":
            results['repair_queue'] = sum(avg_repair_queue) / max(1, len(avg_repair_queue))
            results['invalid_assigns'] = sum(avg_invalid_assigns) / max(1, len(avg_invalid_assigns))
            results['reassigned'] = sum(avg_reassigned_tasks) / max(1, len(avg_reassigned_tasks))
        else:
            results['repair_queue'] = 0.0
            results['invalid_assigns'] = 0.0
            results['reassigned'] = 0.0
            
        return results
    return None

def main():
    console.print(Panel("[bold green]iFogSim FANET Scheduler Benchmarking[/bold green]\nStarting comparative analysis of task scheduling algorithms.", expand=False))
    
    with console.status("[bold yellow]Compiling Java source...") as status:
        subprocess.run(["javac", "--release", "21", "-d", "out/production/iFogSim", "-cp", "jars/*:jars/commons-math3-3.5/*", "-sourcepath", "src", "src/org/fog/test/perfeval/FANETSimulation.java"], check=True)
        status.update("[bold green]Compilation successful!")
    
    all_results = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        
        tasks = {}
        for sched in SCHEDULERS:
            tasks[sched] = progress.add_task(f"[cyan]Starting {sched}...", total=NUM_RUNS)
            
        for sched in SCHEDULERS:
            res = run_experiment(sched, progress, tasks[sched])
            if res:
                all_results[sched] = res
            progress.update(tasks[sched], description=f"[green]{sched} Complete!")
        
    console.print()
    
    if all_results:
        table = Table(title=f"FINAL BENCHMARK COMPARISON ({NUM_RUNS} RUNS EACH)", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="dim", width=25)
        for sched in SCHEDULERS:
            table.add_column(sched, justify="right")
        
        metrics = [
            ('Success Rate (%)', 'succ_rate', '{:.2f}'),
            ('Throughput Ratio', 'throughput_ratio', '{:.4f}'),
            ('Sched Overhead (ms)', 'overhead', '{:.4f}'),
            ('Average Latency (ms)', 'latency', '{:.2f}'),
            ('Tasks Completed', 'tasks', '{:.2f}'),
            ('Data Processed (MB)', 'throughput_mb', '{:.4f}'),
            ('Tasks Dropped', 'dropped', '{:.2f}'),
            ('P2P Offloads', 'offloads', '{:.2f}'),
            ('UAV Utilization', 'uav_util', '{:.4f}'),
            ('Capacity Utilization', 'cap_util', '{:.4f}'),
            ('Unassigned Tasks (Queue)', 'unassigned', '{:.2f}'),
            ('Repair Queue Length', 'repair_queue', '{:.2f}'),
            ('Invalid Assignments', 'invalid_assigns', '{:.2f}'),
            ('Reassigned Tasks', 'reassigned', '{:.2f}')
        ]
        
        for label, key, format_str in metrics:
            row = [label]
            for sched in SCHEDULERS:
                val = all_results.get(sched, {}).get(key, 0.0)
                if ("repair" in key or "invalid" in key or "reassigned" in key) and sched != "DYNAMIC":
                    row.append("N/A")
                else:
                    row.append(format_str.format(val))
            table.add_row(*row)
            
        console.print(table)
    else:
        console.print("[bold red]Failed to gather complete benchmark data.[/bold red]")

if __name__ == "__main__":
    main()
