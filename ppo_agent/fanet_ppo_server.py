import socket
import json
import numpy as np
from tf_ppo_model import TFPPO

HOST, PORT = "localhost", 5500

def run_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)
    
    print(f"\n[TF-PPO Server] Listening on {HOST}:{PORT}")
    print("[TF-PPO Server] Waiting for Java iFogSim to connect...\n")

    while True:
        try:
            conn, addr = server.accept()
            print(f"[TF-PPO Server] Connected to {addr}")
        except KeyboardInterrupt:
            break

        buffer = ""
        agent = None
        max_dist = 50.0 # max distance a UAV can move per step
        
        while True:
            try:
                chunk = conn.recv(8192).decode("utf-8")
                if not chunk:
                    break
                buffer += chunk
                
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                        
                    if line == "CLOSE":
                        if agent is not None:
                            agent.update() # final update
                        print("[TF-PPO Server] Simulation ended.")
                        break
                        
                    # State JSON format:
                    # { "reward": 0.0, "uavs": [ {"id": "uav_0", "x": 100, "y": 100, "z": 50, "load": 2, "mds_in_range": 5}, ... ] }
                    data = json.loads(line)
                    uavs = data["uavs"]
                    reward = float(data.get("reward", 0.0))
                    
                    if agent is None:
                        # Initialize agent based on the number of UAVs
                        # state_dim = 5 (x, y, z, load, mds_in_range)
                        # action_dim = 3 (dx, dy, dz)
                        agent = TFPPO(num_uavs=len(uavs), state_dim=5, action_dim=3)
                        last_global_state = None
                        last_actions = None
                        last_log_probs = None
                    
                    # Store previous transition if available
                    if last_global_state is not None:
                        agent.store_transition(last_global_state, last_actions, last_log_probs, reward)
                    
                    # Extract current global state
                    current_global_state = []
                    uav_ids = []
                    for uav in uavs:
                        uav_ids.append(uav["id"])
                        state_vec = [
                            float(uav["x"]), 
                            float(uav["y"]), 
                            float(uav["z"]), 
                            float(uav["load"]), 
                            float(uav["mds_in_range"])
                        ]
                        current_global_state.append(state_vec)
                        
                    # Select actions
                    actions, log_probs = agent.select_action(current_global_state)
                    
                    # Save for next iteration to store transition
                    last_global_state = current_global_state
                    last_actions = actions
                    last_log_probs = log_probs
                    
                    # Build response
                    response_data = {"uav_actions": []}
                    for i, action in enumerate(actions):
                        # Action is continuous [-1, 1], scale to max_dist
                        dx = float(action[0]) * max_dist
                        dy = float(action[1]) * max_dist
                        dz = float(action[2]) * max_dist
                        response_data["uav_actions"].append({
                            "id": uav_ids[i],
                            "dx": dx,
                            "dy": dy,
                            "dz": dz
                        })
                        
                    response = json.dumps(response_data) + "\n"
                    conn.sendall(response.encode("utf-8"))
                    
                    # Periodically update model
                    if len(agent.buffer['rewards']) >= 64:
                        agent.update()
                        
            except Exception as e:
                print(f"[TF-PPO Server] Error: {e}")
                break
                
        conn.close()

if __name__ == "__main__":
    run_server()