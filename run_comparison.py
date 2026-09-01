import subprocess
import re
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.panel import Panel

console = Console()
NUM_RUNS = 100

def run_experiment(scheduler_type, progress, task_id):
    total_tasks_completed = []
    total_throughput_mb = []
    average_latency_ms = []
    
    total_tasks_dropped = []
    scheduling_success_rate = []
    average_scheduling_overhead_ms = []
    total_system_offloaded = []
    
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
        
        if tasks_match and throughput_match and latency_match and dropped_match and success_rate_match and overhead_match and offload_match:
            total_tasks_completed.append(int(tasks_match.group(1)))
            total_throughput_mb.append(float(throughput_match.group(1)))
            average_latency_ms.append(float(latency_match.group(1)))
            total_tasks_dropped.append(int(dropped_match.group(1)))
            scheduling_success_rate.append(float(success_rate_match.group(1)))
            average_scheduling_overhead_ms.append(float(overhead_match.group(1)))
            total_system_offloaded.append(int(offload_match.group(1)))
        
        progress.advance(task_id)
    
    # Calculate averages
    if total_tasks_completed:
        return {
            'tasks': sum(total_tasks_completed) / len(total_tasks_completed),
            'throughput': sum(total_throughput_mb) / len(total_throughput_mb),
            'latency': sum(average_latency_ms) / len(average_latency_ms),
            'dropped': sum(total_tasks_dropped) / len(total_tasks_dropped),
            'succ_rate': sum(scheduling_success_rate) / len(scheduling_success_rate),
            'overhead': sum(average_scheduling_overhead_ms) / len(average_scheduling_overhead_ms),
            'offloads': sum(total_system_offloaded) / len(total_system_offloaded)
        }
    return None

def main():
    console.print(Panel("[bold green]iFogSim FANET Scheduler Benchmarking[/bold green]\nStarting comparative analysis of task scheduling algorithms.", expand=False))
    
    with console.status("[bold yellow]Compiling Java source...") as status:
        subprocess.run(["javac", "--release", "25", "-d", "out/production/iFogSim", "-cp", "jars/*:jars/commons-math3-3.5/*", "-sourcepath", "src", "src/org/fog/test/perfeval/FANETSimulation.java"], check=True)
        status.update("[bold green]Compilation successful!")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        
        task_mogs = progress.add_task("[cyan]Starting MOGS...", total=NUM_RUNS)
        task_dynamic = progress.add_task("[magenta]Starting DYNAMIC...", total=NUM_RUNS)
        
        mogs_results = run_experiment("MOGS", progress, task_mogs)
        progress.update(task_mogs, description="[green]MOGS Complete!")
        
        dynamic_results = run_experiment("DYNAMIC", progress, task_dynamic)
        progress.update(task_dynamic, description="[green]DYNAMIC Complete!")
        
    console.print()
    
    if mogs_results and dynamic_results:
        table = Table(title=f"FINAL BENCHMARK COMPARISON ({NUM_RUNS} RUNS EACH)", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="dim", width=25)
        table.add_column("MOGS (Baseline)", justify="right")
        table.add_column("DYNAMIC REPAIR (New)", justify="right")
        table.add_column("Diff", justify="right")
        
        metrics = [
            ('Success Rate (%)', 'succ_rate', '{:.2f}'),
            ('Sched Overhead (ms)', 'overhead', '{:.4f}'),
            ('Average Latency (ms)', 'latency', '{:.2f}'),
            ('Tasks Completed', 'tasks', '{:.2f}'),
            ('Throughput (MB)', 'throughput', '{:.4f}'),
            ('Tasks Dropped', 'dropped', '{:.2f}'),
            ('P2P Offloads', 'offloads', '{:.2f}')
        ]
        
        for label, key, format_str in metrics:
            m_val = mogs_results[key]
            d_val = dynamic_results[key]
            diff = d_val - m_val
            
            diff_str = f"{diff:+.4f}" if "overhead" in key or "throughput" in key else f"{diff:+.2f}"
            
            # Color coding for Diff
            if "overhead" in key or "latency" in key or "dropped" in key:
                # Lower is better
                color = "green" if diff < 0 else "red" if diff > 0 else "white"
            else:
                # Higher is better
                color = "green" if diff > 0 else "red" if diff < 0 else "white"
                
            diff_str = f"[{color}]{diff_str}[/{color}]"
            
            table.add_row(label, format_str.format(m_val), format_str.format(d_val), diff_str)
            
        console.print(table)
        
        speedup = mogs_results['overhead'] / dynamic_results['overhead']
        console.print(f"\n[bold]Speedup Factor (Overhead):[/bold] [bold green]{speedup:.2f}x[/bold green]")
        
    else:
        console.print("[bold red]Failed to gather complete benchmark data.[/bold red]")

if __name__ == "__main__":
    main()
