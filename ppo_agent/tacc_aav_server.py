"""
TCP Socket Server for TACC-AAV Trajectory Planning & Task Coordination
Port: 5530
"""

import json
import math
import os
import socket
import sys
from typing import Dict, List

import numpy as np

# Ensure root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ppo_agent.tacc_aav import (
    AAVState,
    TACCAAVCoordinator,
    VehicleTask,
)

HOST, PORT = "localhost", 5530

def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind((HOST, PORT))
    except OSError as e:
        print(f"Error binding to {HOST}:{PORT}: {e}")
        sys.exit(1)

    server.listen(1)
    print(f"TACC-AAV Trajectory Server listening on port {PORT}...")

    coordinator = TACCAAVCoordinator(num_aavs=30, max_tasks=200)
    step_counter = 0

    while True:
        conn, addr = server.accept()
        with conn:
            buffer = ""
            while True:
                data = conn.recv(32768)
                if not data:
                    break
                buffer += data.decode("utf-8")

                if "\n" in buffer:
                    messages = buffer.split("\n")
                    for msg in messages[:-1]:
                        msg = msg.strip()
                        if not msg:
                            continue
                        if msg == "CLOSE":
                            break

                        try:
                            req = json.loads(msg)
                            raw_uavs = req.get("uavs", [])
                            raw_mds = req.get("mds", [])
                            sim_time = req.get("time", float(step_counter * 10.0))

                            agents = []
                            for idx, u in enumerate(raw_uavs):
                                u_id = u.get("id", f"uav_{idx}")
                                try:
                                    num_id = int("".join(filter(str.isdigit, str(u_id))))
                                except ValueError:
                                    num_id = idx

                                x = float(u.get("x", 1000.0))
                                y = float(u.get("y", 1000.0))
                                energy = float(u.get("batt", 80000.0))
                                speed = float(u.get("speed", 14.0))
                                heading = float(u.get("heading", 0.0))
                                agents.append(AAVState(
                                    agent_id=num_id,
                                    x=x, y=y,
                                    energy=energy,
                                    speed=speed,
                                    heading=heading
                                ))

                            tasks = []
                            for m in raw_mds:
                                m_id = int(m.get("id", 0))
                                mx = float(m.get("x", 0.0))
                                my = float(m.get("y", 0.0))
                                comp = float(m.get("comp", 2000.0))
                                size = float(m.get("size", 10.0))
                                deadline = float(m.get("lat", m.get("deadline", 500.0)))
                                tasks.append(VehicleTask(
                                    task_id=m_id,
                                    x=mx, y=my,
                                    comp_size_mi=comp,
                                    data_size_mb=size,
                                    deadline_ms=deadline,
                                    priority_omega=1.0,
                                    source_md_id=m_id
                                ))

                            # Execute TACC-AAV cycle
                            res = coordinator.step_simulation(agents, tasks, current_time=sim_time)
                            trajectories = res["trajectories"]
                            gating = res["gating_decisions"]
                            consensus = res.get("consensus")

                            step_counter += 1
                            if step_counter % 20 == 0:
                                coordinator.train_step(batch_size=16)

                            uav_actions = []
                            for a in agents:
                                traj = trajectories.get(a.agent_id, {"velocity": 12.0, "heading": 0.0})
                                v = traj["velocity"]
                                heading = traj["heading"]

                                # If AAV has tasks in bundle, guide trajectory toward target task
                                if consensus is not None and a.agent_id < len(consensus.bundles):
                                    bundle = consensus.bundles[a.agent_id]
                                    if bundle and len(tasks) > 0:
                                        target_idx = bundle[0]
                                        if target_idx < len(tasks):
                                            target_t = tasks[target_idx]
                                            dx = target_t.x - a.x
                                            dy = target_t.y - a.y
                                            dist_to_task = math.hypot(dx, dy)
                                            if dist_to_task > 20.0:
                                                heading = math.atan2(dy, dx)

                                # Continuous displacement per 10ms sync step
                                dt = 0.1  # seconds
                                dist = min(3.0, v * dt)
                                is_gated = gating.get(a.agent_id, 1)

                                uav_actions.append({
                                    "id": f"uav_{a.agent_id}",
                                    "velocity": v,
                                    "heading": heading,
                                    "distance": dist,
                                    "angle": heading,
                                    "comm_gated": is_gated
                                })

                            resp = {
                                "uav_actions": uav_actions,
                                "total_tasks": len(tasks),
                                "lambda_comm": coordinator.lambda_comm,
                                "lambda_energy": coordinator.lambda_energy
                            }
                            conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))

                        except Exception as e:
                            print(f"Error handling TACC-AAV message: {e}")
                            conn.sendall(b'{"error": "processing error"}\n')

                    buffer = messages[-1]
                    if "CLOSE" in buffer:
                        break

if __name__ == "__main__":
    start_server()
