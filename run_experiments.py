import subprocess
import re

NUM_RUNS = 20

total_tasks_completed = []
total_throughput_mb = []
average_latency_ms = []

total_tasks_dropped = []
scheduling_success_rate = []
average_scheduling_overhead_ms = []
total_system_offloaded = []

print(f"Starting {NUM_RUNS} runs of the iFogSim FANET simulation...")
print("=========================================================")

# Ensure java is compiled before starting runs
print("Compiling Java source...")
subprocess.run(["javac", "--release", "25", "-d", "out/production/iFogSim", "-cp", "jars/*:jars/commons-math3-3.5/*", "-sourcepath", "src", "src/org/fog/test/perfeval/FANETSimulation.java"], check=True)

for i in range(1, NUM_RUNS + 1):
    print(f"Running simulation {i}/{NUM_RUNS}...")
    
    # Run the simulation
    result = subprocess.run(["./run_single_sim.sh"], capture_output=True, text=True)
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
        tasks = int(tasks_match.group(1))
        throughput = float(throughput_match.group(1))
        latency = float(latency_match.group(1))
        dropped = int(dropped_match.group(1))
        succ_rate = float(success_rate_match.group(1))
        overhead = float(overhead_match.group(1))
        offload = int(offload_match.group(1))
        
        total_tasks_completed.append(tasks)
        total_throughput_mb.append(throughput)
        average_latency_ms.append(latency)
        total_tasks_dropped.append(dropped)
        scheduling_success_rate.append(succ_rate)
        average_scheduling_overhead_ms.append(overhead)
        total_system_offloaded.append(offload)
        
        print(f"  -> Tasks: {tasks} | SuccRate: {succ_rate:.1f}% | Drop: {dropped} | Overhead: {overhead:.2f} ms")
    else:
        print(f"  -> Failed to parse output for run {i}. Please check logs.")
        # If there's an error, maybe print a snippet
        # print(output[-500:])

print("\n=========================================================")
print(f"FINAL SUMMARY AVERAGES OVER {NUM_RUNS} RUNS")
print("=========================================================")

if total_tasks_completed:
    avg_tasks = sum(total_tasks_completed) / len(total_tasks_completed)
    avg_throughput = sum(total_throughput_mb) / len(total_throughput_mb)
    avg_latency = sum(average_latency_ms) / len(average_latency_ms)
    
    avg_dropped = sum(total_tasks_dropped) / len(total_tasks_dropped)
    avg_succ_rate = sum(scheduling_success_rate) / len(scheduling_success_rate)
    avg_overhead = sum(average_scheduling_overhead_ms) / len(average_scheduling_overhead_ms)
    avg_offloads = sum(total_system_offloaded) / len(total_system_offloaded)

    print(f"Average Tasks Completed : {avg_tasks:.2f}")
    print(f"Average Throughput      : {avg_throughput:.4f} MB")
    print(f"Average Latency         : {avg_latency:.2f} ms")
    print(f"Average Tasks Dropped   : {avg_dropped:.2f}")
    print(f"Average Success Rate    : {avg_succ_rate:.2f} %")
    print(f"Average Sched Overhead  : {avg_overhead:.4f} ms")
    print(f"Average P2P Offloaded   : {avg_offloads:.2f}")
else:
    print("No valid data collected.")
