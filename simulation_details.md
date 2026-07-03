# FANET Simulation Explanation

This document provides a detailed breakdown of the FANET (Flying Ad-Hoc Network) simulation based on the `iFogSim` environment and the integrated PPO (Proximal Policy Optimization) Python agent.

## 1. Sensors and What They Collect
- **The Sensor**: There is a single sensor named `sensor_0` attached to a mobile ground gateway (`ground_device_0`).
- **What it Collects**: The sensor generates a continuous stream of raw telemetry or environmental data labeled `SENSOR_DATA`. 
- **Emission Rate**: Tasks are generated periodically using a `UniformDistribution` with an average interval of 10.0 milliseconds ($\pm$ 3.0 ms), simulating a high-frequency but realistically jittered continuous data stream.

## 2. The Models and Formulas

The simulation relies on a hybrid model that uses **Physics-based heuristics** alongside an **AI RL Agent** to rank and assign tasks to UAVs. 

### Physics and Network Model
When a task needs to be offloaded to a UAV, the Python server calculates costs based on the physical properties of the task and the UAV:

- **Distance ($D$)**: Calculated from the UAV's Cartesian coordinates.
  $$D = \sqrt{X^2 + Y^2}$$
- **Data Rate ($R$)**: Degrades as distance increases.
  $$R = \max\left(\frac{20.0}{1 + D \cdot 0.005}, 0.1\right)$$
- **Transmission Time ($T_{trans}$)**:
  $$T_{trans} = \frac{\text{Task Data Size}}{R}$$
- **Computation Time ($T_{comp}$)**:
  $$T_{comp} = \frac{\text{Task CPU Cycles}}{\text{UAV MIPS}}$$
- **Energy Consumption ($E$)**: Based on the UAV's MIPS capacity ($\kappa = 10^{-10}$).
  $$E = \kappa \cdot \text{Task CPU Cycles} \cdot \text{UAV MIPS}^2$$

### Composite Score Formula
The total cost of assigning a task to a UAV combines the transmission delay, computation delay, energy, queue penalties, and the AI probability score:

$$\text{Cost} = T_{trans} + (T_{comp} \cdot \text{Priority}) + (\text{Queue} \cdot 50.0 \cdot \text{Priority}) - (P_{AI} \cdot 10.0)$$

*Lower cost yields a higher placement score.*

### The AI Model
- The AI model is a **PPO (Proximal Policy Optimization)** agent utilizing an MLP (Multi-Layer Perceptron) policy.
- Its role is to act as an intelligent tie-breaker. It outputs a probability distribution $P_{AI}$ predicting the optimal UAV, which reduces the total physical cost in the composite formula.

## 3. Data Being Sent (Trace)
The data flows through a strict processing pipeline (AppLoop) traversing the Edge and Cloud.

| Edge Flow | Source Node | Destination Node | Data Type | Size (bits) | Required CPU (MI) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | Sensor (`sensor_0`) | Ground Device (`sensor_module`) | `SENSOR_DATA` | 1000 | 500 |
| **2** | Ground Device | UAV Edge (`processing_module`) | `PREPROCESSED_DATA`| 2000 | 1000 |
| **3** | UAV Edge | Cloud (`storage_module`) | `RESULT_DATA` | 500 | 200 |
| **4** | UAV Edge | Actuator (`actuator_0`) | `ACTUATOR_CMD` | 100 | 50 |

## 4. Training Data
The PPO Agent is trained in an isolated Gymnasium environment (`FANETEnv`) for 50,000 timesteps before handling live simulation data.

The observation state (training data) consists of **15 synthesized features**:
- **Task Features (3):** Priority [1 to 3], Data Size [200 to 8000], CPU Cycles [500 to 8000].
- **UAV Features (12):** For each of the 3 UAVs, it observes its MIPS capacity, Current Queue Size (0 to 10), X position, and Y position.

The agent trains by randomly generating task demands and UAV queue states, calculating the reward using the physical formulas, and heavily penalizing queues $\ge 10$.

## 5. Task Generation and Homogeneity
- **Where are tasks generated?** Tasks are generated exclusively by `sensor_0` at the ground level.
- **Are all tasks the same?** 
  - **In Java (Data Level):** No. While the baseline requirements are 1000 bits and 500 MI, the `PreferenceBuilder` injects significant randomness into each tuple before sending it to the RL agent. Task Data Sizes and CPU Cycles are dynamically scaled by a random factor between $0.2\times$ and $1.5\times$ to simulate varied real-world loads.
  - **In Python (Priority Level):** The Python server also intercepts tasks and dynamically assigns them alternating priorities based on their ID: `Priority = (Task ID % 3) + 1`. This tests the system's ability to prioritize critical data during routing.

## 6. Reaching the UAVs
1. **Network Topology**: The ground device has an uplink connection directly to `uav_0`.
2. **Placement Interception**: When the ground module generates `PREPROCESSED_DATA` destined for a UAV `processing_module`, the MOGS Controller (Java) intercepts the placement request.
3. **Socket Communication**: Java serializes the task requirements and the live status (MIPS, exact queue size, positions) of all 3 UAVs into a JSON payload and sends it to `localhost:5500`.
4. **Python Evaluation**: The Python PPO server decodes the JSON, computes the composite score for each UAV using the AI+Physics engine, and ranks them from best to worst.
5. **Final Routing**: Python replies with a JSON list containing the ordered UAV IDs. Java assigns the incoming task to the highest-ranked UAV on that list.
