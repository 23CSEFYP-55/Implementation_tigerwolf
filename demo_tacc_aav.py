"""
Demonstration of TACC-AAV:
Task Allocation with Communication Coordination for AAV-Assisted Edge Computing
"""

import math
import os
import random
import matplotlib.pyplot as plt
import numpy as np

from ppo_agent.tacc_aav import (
    AAVState,
    TACCAAVCoordinator,
    VehicleTask,
)

def run_demo():
    print("================================================================================")
    print("           TACC-AAV: Task Allocation & Communication Coordination Demo          ")
    print("================================================================================")
    
    random.seed(42)
    np.random.seed(42)
    
    num_aavs = 6
    num_tasks = 40
    num_steps = 30
    
    # Initialize AAV swarm across 2000m x 2000m airspace
    agents = [
        AAVState(
            agent_id=i,
            x=random.uniform(400, 1600),
            y=random.uniform(400, 1600),
            energy=80000.0,
            speed=14.0,
            heading=random.uniform(0, 2 * math.pi),
            comm_range=500.0
        )
        for i in range(num_aavs)
    ]
    
    # Initialize dynamic vehicle tasks
    tasks = [
        VehicleTask(
            task_id=k,
            x=random.uniform(200, 1800),
            y=random.uniform(200, 1800),
            comp_size_mi=random.uniform(1000, 4000),
            data_size_mb=random.uniform(5, 30),
            deadline_ms=random.uniform(400, 1200),
            priority_omega=random.uniform(0.5, 1.5),
            arrival_time=float(random.randint(0, 10))
        )
        for k in range(num_tasks)
    ]
    
    coordinator = TACCAAVCoordinator(num_aavs=num_aavs, max_tasks=num_tasks)
    
    aav_paths = {i: [(agents[i].x, agents[i].y)] for i in range(num_aavs)}
    total_gating_attempts = 0
    total_broadcasts = 0
    total_assigned_tasks = 0
    energy_history = {i: [] for i in range(num_aavs)}
    
    print(f"\n[Simulating {num_steps} Coordination Steps across {num_aavs} AAVs and {num_tasks} Tasks...]")
    for step in range(num_steps):
        # Filter tasks arrived by current step
        active_tasks = [t for t in tasks if t.arrival_time <= step]
        
        result = coordinator.step_simulation(agents, active_tasks, current_time=float(step * 10))
        trajectories = result["trajectories"]
        gating = result["gating_decisions"]
        assignments = result["assignments"]
        
        total_gating_attempts += len(agents)
        broadcasts_this_step = sum(gating.values())
        total_broadcasts += broadcasts_this_step
        unique_tasks = len(set(a["task_id"] for a in assignments))
        total_assigned_tasks = max(total_assigned_tasks, unique_tasks)
        
        # Advance AAV kinematics along continuous trajectory [v, theta]
        dt = 2.0  # 2 seconds per step
        for a in agents:
            traj = trajectories[a.agent_id]
            v = traj["velocity"]
            heading = traj["heading"]
            a.speed = v
            a.heading = heading
            a.x = max(50.0, min(1950.0, a.x + v * math.cos(heading) * dt))
            a.y = max(50.0, min(1950.0, a.y + v * math.sin(heading) * dt))
            aav_paths[a.agent_id].append((a.x, a.y))
            energy_history[a.agent_id].append(a.energy)
            
        # Periodic training update (CTDE)
        if step % 5 == 0:
            coordinator.train_step(batch_size=16)

    comm_reduction_pct = 100.0 * (1.0 - (total_broadcasts / total_gating_attempts))
    avg_final_energy = np.mean([a.energy for a in agents])
    
    print("\n======================= BENCHMARK PERFORMANCE =======================")
    print(f"Total Tasks Serviceable:          {total_assigned_tasks} / {num_tasks} ({total_assigned_tasks/num_tasks*100:.1f}%)")
    print(f"Total TDMA Comm Opportunities:   {total_gating_attempts}")
    print(f"Actual Transmissions Broadcast:  {total_broadcasts}")
    print(f"Communication Bandwidth Saved:   {comm_reduction_pct:.2f}%")
    print(f"Average Final AAV Battery Energy:{avg_final_energy:.1f} J (out of 80,000 J)")
    print(f"Dual Multiplier lambda_comm:     {coordinator.lambda_comm:.4f}")
    print(f"Dual Multiplier lambda_energy:   {coordinator.lambda_energy:.4f}")
    print("=====================================================================\n")

    # ==============================================================================
    # GENERATE VISUALIZATION PLOT (tacc_aav_trajectories.png)
    # ==============================================================================
    fig, axs = plt.subplots(1, 2, figsize=(16, 7))
    
    # Subplot 1: Spatial 2D Trajectories and Task Assignments
    ax1 = axs[0]
    ax1.set_title("TACC-AAV Continuous Flight Trajectories & Vehicle Tasks", fontsize=13, fontweight='bold')
    ax1.set_xlim(0, 2000)
    ax1.set_ylim(0, 2000)
    ax1.set_xlabel("X Coordinate (m)", fontsize=11)
    ax1.set_ylabel("Y Coordinate (m)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    
    # Plot tasks
    task_xs = [t.x for t in tasks]
    task_ys = [t.y for t in tasks]
    ax1.scatter(task_xs, task_ys, c='gray', marker='x', s=50, alpha=0.7, label='Vehicle Tasks')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for i in range(num_aavs):
        xs, ys = zip(*aav_paths[i])
        c = colors[i % len(colors)]
        ax1.plot(xs, ys, color=c, linewidth=2.0, label=f"AAV {i} Path")
        ax1.scatter([xs[0]], [ys[0]], color=c, marker='o', s=80, edgecolors='black')
        ax1.scatter([xs[-1]], [ys[-1]], color=c, marker='^', s=100, edgecolors='black')
        
    ax1.legend(loc='upper right', fontsize=9)
    
    # Subplot 2: Bandwidth Savings & Energy Depletion
    ax2 = axs[1]
    ax2.set_title("Communication Gating Savings & Battery Headroom", fontsize=13, fontweight='bold')
    
    categories = ['Unconstrained\nBroadcasts', 'TACC-AAV\nGated Broadcasts']
    values = [total_gating_attempts, total_broadcasts]
    bar_colors = ['#d9534f', '#5cb85c']
    
    bars = ax2.bar(categories, values, color=bar_colors, width=0.45, edgecolor='black')
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                 f"{val} ({val/total_gating_attempts*100:.1f}%)",
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
        
    ax2.set_ylabel("Total Transmission Events", fontsize=11)
    ax2.set_ylim(0, total_gating_attempts * 1.2)
    ax2.grid(True, linestyle="--", alpha=0.6, axis='y')
    
    # Inset annotation of bandwidth saved
    ax2.text(0.5, 0.75, f"Communication Overhead Reduced by:\n{comm_reduction_pct:.1f}%\nVia MOCPO Gating Policy",
             transform=ax2.transAxes, ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f8ff", edgecolor="#337ab7", linewidth=1.5),
             fontsize=12, fontweight='bold', color="#1d4e89")
    
    plt.tight_layout()
    plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "trajectories")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, "tacc_aav_trajectories.png")
    plt.savefig(plot_path, dpi=200)
    print(f"Saved trajectory and gating visualization to {plot_path}")

if __name__ == "__main__":
    run_demo()
