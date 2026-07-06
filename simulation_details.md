# FANET Simulation Explanation

This document provides a detailed breakdown of the FANET (Flying Ad-Hoc Network) simulation based on the `iFogSim` environment and the integrated PPO (Proximal Policy Optimization) Python agent. The architecture represents a terrestrial Mobile Edge Computing (MEC) simulation to benchmark the system proposed in "A Multi-UAV Cooperative Task Scheduling in Dynamic Environments: Throughput Maximization" (Zhao et al., 2025).

## 1. Sensors and What They Collect
- **The Sensor**: There are 800 sensors, each attached to a mobile ground device (`ground_device_0` to `ground_device_799`). The devices are distributed randomly across a $2000m \times 2000m$ grid.
- **What it Collects**: The sensors generate a continuous stream of raw telemetry or environmental data labeled `SENSOR_DATA`.
- **Emission Rate**: Tasks are generated following a Poisson Point Process with an average inter-arrival time (lambda) of 1000.0 milliseconds per device.

## 2. The Models and Roles

The simulation utilizes a strict role separation architecture between the Java environment and the Python AI server.

### Java Specifications (iFogSim & MOGS Scheduling)
- **Role**: Exclusively handles task generation, data transmission, and the **Many-to-One Gale-Shapley (MOGS)** task scheduling algorithm.
- **The Environment**: A strict terrestrial MEC model with absolutely no centralized Cloud infrastructure.
- **The MOGS Algorithm**:
  1. **Active Party (Ground Devices)**: Calculate a score for all UAVs in range (e.g., $d \le 130m$) using a function of uplink distance, downlink distance, and current UAV load penalty. The MD creates a ranked preference list of UAVs.
  2. **Passive Party (UAVs)**: Either accept the task if their queue is below capacity, or attempt to offload the task to a peer UAV (within 150m). If peer offloading fails, the task is rejected and the MD proposes to its next preferred UAV.

### Python Specifications (TF-PPO RL Agent)
- **Role**: Exclusively for UAV flight trajectory and formation control.
- **The Architecture**: A PyTorch-based Proximal Policy Optimization (PPO) agent following a Centralized Training with Decentralized Execution (CTDE) framework.
- **Operation**: Periodically receives the global state (UAV positions, queues, energy) from Java over a TCP socket (port 5500), computes optimal trajectories, and sends the updated $(X, Y, Z, V)$ target states back to the simulation. The Python server **does not** dictate task assignments or evaluate task routing costs.

## 3. Data Being Sent (Trace)
The data flows through a strict processing pipeline traversing the Edge nodes. Since there is no Cloud, all computation happens locally or on the UAV swarm.

| Edge Flow | Source Node | Destination Node | Data Type | CPU Length (MI) |
| :--- | :--- | :--- | :--- | :--- |
| **1** | Sensor (`sensor_X`) | Ground Device (`sensor_module`) | `SENSOR_DATA` | 100 |
| **2** | Ground Device | UAV Edge (`processing_module`) | `PREPROCESSED_DATA`| 100 |
| **3** | UAV Edge | Actuator (`actuator_X`) | `ACTUATOR_CMD` | 50 |

## 4. Peer-to-Peer Task Offloading
To maximize system throughput, UAVs employ Peer-to-Peer (P2P) offloading. When a UAV reaches its task capacity threshold, rather than immediately dropping an incoming task, it evaluates the proximity of neighboring UAVs. If a peer UAV is within communication range ($< 150m$) and has available capacity, the task is seamlessly offloaded to the peer. This mechanism is critical for balancing load across the swarm.

## 5. System Execution Flow
1. **Initialization**: Java dynamically spawns 2-12 UAVs and 800 MDs.
2. **Socket Communication**: Java establishes a bidirectional JSON over TCP socket connection with the Python TF-PPO server.
3. **Trajectory Update Phase**: Python computes the next geographic targets for the UAVs and sends them to Java.
4. **Task Generation**: MDs generate `PREPROCESSED_DATA` tuples which are intercepted by the ground devices and pushed into a global `taskWaitingPool`.
5. **MOGS Scheduling Loop**: When the task batch reaches a threshold, Java executes the MOGS matching algorithm to route tasks to specific UAVs.
6. **Task Processing**: Tasks are sent to the assigned UAVs via `FogEvents.TUPLE_ARRIVAL`, processed via the UAVs' CloudletScheduler, and removed from the active load once completed.
7. **Benchmarking**: Upon simulation termination, a comprehensive ASCII dashboard is printed detailing system throughput, latency, energy consumption, and total P2P tasks offloaded per UAV.
