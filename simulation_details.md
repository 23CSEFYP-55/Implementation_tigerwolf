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

## 6. PPO Global and Local Models (Deep Dive)

The Python RL agent implements a **Centralized Training with Decentralized Execution (CTDE)** architecture using PyTorch (`tf_ppo_model.py`). This architecture aims to solve multi-agent coordination by having a global perspective during training, but allowing UAVs to act independently based on their local observations during execution.

### The Local Models (Decentralized Actors)
- **Architecture**: Each UAV is assigned its own independent Actor network (`UAVActor`).
- **Input (Local State)**: The network takes a 5-dimensional local observation vector containing only the UAV's specific data: `[x, y, z, load, mds_in_range]`.
- **Output (Action)**: It outputs the mean and standard deviation for a continuous action space (dx, dy, dz), which is scaled to a maximum movement distance of 50.0m per step.
- **Execution Role**: During the simulation, the local models are actively used to dictate the individual trajectory (flight path) of each UAV independently without needing to know the state of the entire swarm.

### The Global Model (Centralized Critic)
- **Architecture**: A single, shared Critic network (`GlobalCritic`).
- **Input (Global State)**: The network takes the flattened, concatenated states of *all* UAVs in the swarm (`state_dim * num_uavs`).
- **Output (Value)**: It outputs a single scalar value representing the estimated "goodness" (expected discounted return) of the entire swarm's current joint state.
- **Training Role**: The Critic is used *strictly* for training. It evaluates the actions taken by the local actors and calculates the Advantage (how much better or worse the actions were compared to the expected baseline). This advantage is used to calculate the loss and update the weights of the decentralized actors.

### Where and How it is Trained
- **Location**: The models are instantiated and trained entirely on the Python server (`fanet_ppo_server.py`).
- **The Loop**: 
  1. The Java iFogSim simulation acts as the environment, calculating the state of the simulation and a joint `reward` float.
  2. Java sends the state and reward to Python over the TCP socket as a JSON string.
  3. Python stores the transitions (state, action, log probability, reward) in an experience buffer.
  4. **Online Training**: Once the buffer accumulates 64 steps, the PPO `update()` function is triggered. The Global Critic is updated to minimize the mean squared error (MSE) of its value predictions, while the Local Actors are updated using the PPO clipped surrogate objective function to maximize the expected reward based on the Critic's advantage estimates. The buffer is then cleared.

### What is Lacking in the Current Implementation
While the CTDE setup is standard for Multi-Agent Reinforcement Learning (MARL), the current repository implementation has several significant limitations:

1. **Decoupled from Task Scheduling**: The PPO model *only* controls UAV movement. It is entirely blind to the MOGS task scheduling algorithm running in Java. True joint optimization (learning to move *to* optimize offloading/MOGS matching) is difficult because the RL agent cannot control or explicitly observe the network topology decisions being made.
2. **Credit Assignment Problem**: The Java simulation sends a single global `reward` scalar for the entire swarm. The Actors receive this shared reward, meaning a UAV that made a poor movement decision might be positively reinforced if the rest of the swarm performed well. It lacks localized, per-UAV reward shaping.
3. **No Inter-Agent Communication or Observation**: The Actor networks only observe their own `[x, y, z, load, mds_in_range]`. They do not observe the positions of neighboring UAVs. This makes learning cooperative behaviors (like collision avoidance or spatial load distribution) extremely difficult, as they must infer swarm dynamics solely from the global reward signal.
4. **Online Training Instability**: The script trains the model *online* from scratch upon connection. Deep RL typically requires millions of episodes to converge. Running PPO online from random initialization during a discrete-event simulation run will likely result in unstable, random-walk trajectories unless pre-trained weights are loaded (which the script currently does not support).
