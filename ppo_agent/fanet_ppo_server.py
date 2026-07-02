"""
FANET PPO Server — Final Clean Architecture
-------------------------------------------
- Listens on Port 5000 for Java iFogSim JSON requests.
- Uses exact, real-time queue sizes reported by Java.
- Physics-based ranking (Latency + Energy + Queue Penalty).
- PPO Agent acts as the intelligent tie-breaker.
"""

import socket
import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import os
import torch

# ─────────────────────────────────────────────
# 1. Gymnasium Environment (For Training)
# ─────────────────────────────────────────────
class FANETEnv(gym.Env):
    NUM_UAVS = 3
    # Obs: [TaskPriority, TaskData, TaskCPU] + 3 * [MIPS, Queue, X, Y] = 15 features
    OBS_SIZE  = 3 + NUM_UAVS * 4
    UAV_POSITIONS = [(100.0, 50.0), (200.0, 100.0), (300.0, 150.0)]

    def __init__(self):
        super(FANETEnv, self).__init__()
        # Define the min/max limits for the observation space
        self.observation_space = spaces.Box(
            low=np.zeros(self.OBS_SIZE, dtype=np.float32),
            high=np.array([3.0, 10000.0, 10000.0, 
                           5000.0, 15.0, 500.0, 500.0,
                           5000.0, 15.0, 500.0, 500.0, 
                           5000.0, 15.0, 500.0, 500.0], dtype=np.float32),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.NUM_UAVS)
        self.state    = None
        self.uav_data = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        priority   = float(np.random.randint(1, 4))
        data_size  = float(np.random.uniform(200, 8000))
        cpu_cycles = float(np.random.uniform(500, 8000))
        
        obs = [priority, data_size, cpu_cycles]
        self.uav_data = []
        for i in range(self.NUM_UAVS):
            mips  = float(np.random.choice([1000, 2000, 2800, 4000]))
            queue = float(np.random.randint(0, 10))
            x, y  = self.UAV_POSITIONS[i]
            obs  += [mips, queue, x, y]
            self.uav_data.append({"mips": mips, "queue": queue, "x": x, "y": y})
            
        self.state = np.array(obs, dtype=np.float32)
        return self.state, {}

    def step(self, action):
        task_priority   = self.state[0]
        task_data_size  = self.state[1]
        task_cpu_cycles = self.state[2]
        chosen          = self.uav_data[action]

        # Physics Math
        distance  = np.sqrt(chosen["x"]**2 + chosen["y"]**2)
        data_rate = max(20.0 / (1 + distance * 0.005), 0.1)
        t_trans   = task_data_size / data_rate
        t_comp    = task_cpu_cycles / max(chosen["mips"], 1.0)
        kappa     = 1e-10
        energy    = kappa * task_cpu_cycles * (chosen["mips"] ** 2)

        # Base Reward
        reward = -(0.5 * t_trans + 0.3 * t_comp + 0.2 * energy)

        # Queue Penalties (Avoid overloading)
        if chosen["queue"] >= 10:
            reward -= 1000.0
        elif chosen["queue"] >= 7:
            reward -= 300.0

        # Reward smart matching
        min_queue = min(u["queue"] for u in self.uav_data)
        if chosen["queue"] == min_queue:
            reward += 100.0

        return self.state, reward, True, False, {}

# ─────────────────────────────────────────────
# 2. Train or Load PPO Model
# ─────────────────────────────────────────────
MODEL_PATH     = "fanet_ppo_model"
TRAINING_STEPS = 50_000

def get_model():
    env = FANETEnv()
    if os.path.exists(MODEL_PATH + ".zip"):
        print("[PPO] Loading existing trained model...")
        model = PPO.load(MODEL_PATH, env=env)
    else:
        print(f"[PPO] Training new model for {TRAINING_STEPS:,} steps (~1 min)...")
        model = PPO("MlpPolicy", env, verbose=1,
                    learning_rate=3e-4, n_steps=2048, batch_size=64,
                    n_epochs=10, gamma=0.99)
        model.learn(total_timesteps=TRAINING_STEPS)
        model.save(MODEL_PATH)
        print(f"[PPO] Training complete. Model saved.")
    return model, env

# ─────────────────────────────────────────────
# 3. Convert JSON → Observation Vector
# ─────────────────────────────────────────────
def json_to_obs(data):
    # Ensure values stay within the observation space bounds
    obs = [
        min(float(data.get("task_priority",  1)), 3.0),
        min(float(data.get("task_data_size", 1000)), 10000.0),
        min(float(data.get("task_cpu_cycles",2000)), 10000.0),
    ]
    for i, uav in enumerate(data["uavs"][:3]):
        px, py = FANETEnv.UAV_POSITIONS[i] if i < 3 else (0.0, 0.0)
        obs += [
            min(float(uav.get("mips", 2800)), 5000.0),
            min(float(uav.get("queue_size", 0)), 15.0), # Live exact queue from Java
            float(uav.get("x", px)),
            float(uav.get("y", py)),
        ]
    # Pad if fewer than 3 UAVs
    while len(obs) < FANETEnv.OBS_SIZE:
        obs.append(0.0)
    return np.array(obs, dtype=np.float32)

# ─────────────────────────────────────────────
# 4. Rank UAVs (Physics + AI)
# ─────────────────────────────────────────────
def rank_uavs(model, obs, uav_list, task_data_size, task_cpu_cycles, task_priority):
    # 1. Get AI probabilities
    obs_tensor = model.policy.obs_to_tensor(obs.reshape(1, -1))[0]
    with torch.no_grad():
        dist  = model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs.cpu().numpy().flatten()

    scores = {}
    for i, uav in enumerate(uav_list):
        uid   = uav["id"] # e.g., "uav_0"
        mips  = max(float(uav.get("mips", 2800)), 1.0)
        queue = float(uav.get("queue_size", 0)) # Exact queue from Java
        x     = float(uav.get("x", FANETEnv.UAV_POSITIONS[i % FANETEnv.NUM_UAVS][0]))
        y     = float(uav.get("y", FANETEnv.UAV_POSITIONS[i % FANETEnv.NUM_UAVS][1]))
        prob  = float(probs[i]) if i < len(probs) else 0.0

        # 2. Physics Cost Calculation
        distance   = np.sqrt(x**2 + y**2)
        data_rate  = max(20.0 / (1 + distance * 0.005), 0.1)
        t_trans    = task_data_size / data_rate
        t_comp     = task_cpu_cycles / mips
        queue_cost = queue * 50.0

        # 3. Composite Score: Lower cost = Better Score
        cost = t_trans + (t_comp * task_priority) + (queue_cost * task_priority) - (prob * 10.0)
        scores[uid] = -cost

    # Return list of UAV IDs sorted from best (highest score) to worst
    return sorted(scores.keys(), key=lambda uid: scores[uid], reverse=True)

# ─────────────────────────────────────────────
# 5. Socket Server Engine
# ─────────────────────────────────────────────
def run_server():
    model, env = get_model()
    HOST, PORT = "localhost", 5500
    
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)
    
    print(f"\n[PPO Server] Listening on {HOST}:{PORT}")
    print("[PPO Server] Waiting for Java iFogSim to connect...\n")

    while True:
        try:
            conn, addr = server.accept()
            print(f"[PPO Server] SUCCESS: Connected to client at {addr}\n")
        except KeyboardInterrupt:
            break

        buffer = ""
        count  = 0
        should_exit = False
        
        while True:
            try:
                chunk = conn.recv(4096).decode("utf-8")
                if not chunk:
                    break
                buffer += chunk
                
                # Process complete JSON payloads
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                        
                    if line == "CLOSE":
                        print("\n[PPO Server] Java simulation ended. Closing.")
                        should_exit = True
                        break
    
                    # Parse JSON sent by Java
                    data     = json.loads(line)
                    uav_list = data["uavs"]
    
                    task_data = float(data.get("task_data_size",  1000))
                    task_cpu  = float(data.get("task_cpu_cycles", 2000))
                    task_id   = data.get("task_id", count)
                    
                    # Cycle priority if not provided by Java
                    task_priority = float((int(task_id) % 3) + 1)
                    data["task_priority"] = task_priority  
    
                    # Generate Obs and Rank
                    obs    = json_to_obs(data)
                    ranked = rank_uavs(model, obs, uav_list, task_data, task_cpu, task_priority)
    
                    # Send ranked list back to Java
                    response = json.dumps({"ranked_uavs": ranked}) + "\n"
                    conn.sendall(response.encode("utf-8"))
                    
                    count += 1
    
                    # Clean Terminal Printout
                    queues = [u.get("queue_size", 0) for u in uav_list]
                    print(f"[Task {task_id:>4}] Prio: {int(task_priority)} | Queues: {queues} | Assigned: {ranked[0]}")
    
            except Exception as e:
                print(f"[PPO Server] Error: {e}")
                break
            if should_exit:
                break
        
        conn.close()
        print(f"[PPO] Session finished. Ranked {count} tasks.")
        if should_exit:
            break
            
    server.close()

if __name__ == "__main__":
    run_server()