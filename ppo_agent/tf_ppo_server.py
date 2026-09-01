import socket
import json
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
from models import UAVActor, UAVCritic, GlobalTaskFactorizer

HOST, PORT = "localhost", 5500
MAX_DIST = 50.0 # max distance a UAV can move per step

class TFPPO_Agent:
    def __init__(self, num_uavs=12, state_dim=12, action_dim=2, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4):
        self.num_uavs = num_uavs
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        # CTDE Setup
        self.actors = [UAVActor(state_dim, action_dim) for _ in range(num_uavs)]
        self.critics = [UAVCritic(state_dim) for _ in range(num_uavs)]
        self.factorizer = GlobalTaskFactorizer(num_uavs, state_dim * num_uavs)
        
        # Optimizers
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=lr) for actor in self.actors]
        self.critic_optimizers = [optim.Adam(critic.parameters(), lr=lr) for critic in self.critics]
        self.factorizer_optimizer = optim.Adam(self.factorizer.parameters(), lr=lr)
        
        self.buffer = {'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'global_states': []}
        
    def select_action(self, global_state_list):
        actions = []
        log_probs = []
        for i, local_state in enumerate(global_state_list):
            state_tensor = torch.FloatTensor(local_state).unsqueeze(0)
            action, log_prob = self.actors[i].get_action(state_tensor)
            actions.append(action.numpy()[0])
            log_probs.append(log_prob.item())
        return actions, log_probs
        
    def store_transition(self, global_state, actions, log_probs, reward):
        flat_global_state = np.concatenate(global_state)
        self.buffer['states'].append(global_state)
        self.buffer['global_states'].append(flat_global_state)
        self.buffer['actions'].append(actions)
        self.buffer['log_probs'].append(log_probs)
        self.buffer['rewards'].append(reward)
        
    def update(self):
        if len(self.buffer['rewards']) == 0:
            return
            
        rewards = []
        discounted_reward = 0
        for reward in reversed(self.buffer['rewards']):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        global_states = torch.FloatTensor(np.array(self.buffer['global_states']))
        
        for _ in range(self.k_epochs):
            # Calculate Global Baseline Value
            local_vs = []
            for i in range(self.num_uavs):
                local_states = torch.FloatTensor(np.array([s[i] for s in self.buffer['states']]))
                v_local = self.critics[i](local_states)
                local_vs.append(v_local)
                
            local_vs_tensor = torch.cat(local_vs, dim=1) # (batch_size, num_agents)
            
            # Factorize to Global Value
            values = self.factorizer(local_vs_tensor, global_states)
            advantages = rewards - values.detach()
            
            # Update Critic and Factorizer
            critic_loss = nn.MSELoss()(values, rewards)
            
            for opt in self.critic_optimizers:
                opt.zero_grad()
            self.factorizer_optimizer.zero_grad()
            
            critic_loss.backward()
            
            for opt in self.critic_optimizers:
                opt.step()
            self.factorizer_optimizer.step()
            
            # Update Actors
            for i in range(self.num_uavs):
                local_states = torch.FloatTensor(np.array([s[i] for s in self.buffer['states']]))
                old_actions = torch.FloatTensor(np.array([a[i] for a in self.buffer['actions']]))
                old_log_probs = torch.FloatTensor(np.array([lp[i] for lp in self.buffer['log_probs']]))
                
                log_probs, entropy = self.actors[i].evaluate(local_states, old_actions)
                
                ratios = torch.exp(log_probs - old_log_probs)
                surr1 = ratios * advantages.squeeze()
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages.squeeze()
                
                actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()
                
                self.actor_optimizers[i].zero_grad()
                actor_loss.backward()
                self.actor_optimizers[i].step()
                
        # Clear buffer
        self.buffer = {'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'global_states': []}

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
        
        last_global_state = None
        last_actions = None
        last_log_probs = None
        
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
                        
                    data = json.loads(line)
                    uavs = data["uavs"]
                    reward = float(data.get("reward", 0.0))
                    
                    if agent is None:
                        agent = TFPPO_Agent(num_uavs=len(uavs), state_dim=12, action_dim=2)
                    
                    if last_global_state is not None:
                        agent.store_transition(last_global_state, last_actions, last_log_probs, reward)
                    
                    current_global_state = []
                    uav_ids = []
                    for uav in uavs:
                        uav_ids.append(uav["id"])
                        # Extract the 12-dim state
                        state_vec = [
                            float(uav["x"]), float(uav["y"]), float(uav["z"]), 
                            float(uav["load"]),
                            float(uav.get("mds_n", 0)), float(uav.get("mds_ne", 0)), 
                            float(uav.get("mds_e", 0)), float(uav.get("mds_se", 0)),
                            float(uav.get("mds_s", 0)), float(uav.get("mds_sw", 0)), 
                            float(uav.get("mds_w", 0)), float(uav.get("mds_nw", 0))
                        ]
                        current_global_state.append(state_vec)
                        
                    actions, log_probs = agent.select_action(current_global_state)
                    
                    last_global_state = current_global_state
                    last_actions = actions
                    last_log_probs = log_probs
                    
                    response_data = {"uav_actions": []}
                    for i, action in enumerate(actions):
                        # action is in [-1, 1] from tanh
                        raw_angle = float(action[0])
                        raw_dist = float(action[1])
                        
                        # Scale to actual bounds for Java
                        angle = (raw_angle + 1.0) * math.pi # [0, 2pi]
                        distance = ((raw_dist + 1.0) / 2.0) * MAX_DIST # [0, MAX_DIST]
                        
                        response_data["uav_actions"].append({
                            "id": uav_ids[i],
                            "angle": angle,
                            "distance": distance
                        })
                        
                    response = json.dumps(response_data) + "\n"
                    conn.sendall(response.encode("utf-8"))
                    
                    # Update every 100 steps
                    if len(agent.buffer['rewards']) >= 100:
                        agent.update()
                        
            except Exception as e:
                import traceback
                print(f"[TF-PPO Server] Error: {e}")
                traceback.print_exc()
                break
                
        conn.close()

if __name__ == "__main__":
    run_server()
