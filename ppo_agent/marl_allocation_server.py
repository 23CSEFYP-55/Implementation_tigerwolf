import socket
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import math
import sys
import os

HOST, PORT = "localhost", 5501

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

class RequestNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(RequestNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.lstm = nn.LSTM(64, 64, batch_first=True, bidirectional=True)
        self.fc_actor = nn.Linear(128, action_dim)
        self.fc_critic = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        action_probs = F.softmax(self.fc_actor(out), dim=-1)
        state_value = self.fc_critic(out)
        return action_probs, state_value

class ResponseNet(nn.Module):
    def __init__(self, state_dim):
        super(ResponseNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.q_value = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q_value(x)

class PrioritizedReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.epsilon = 1e-5

    def push(self, transition, priority):
        self.buffer.append(transition)
        self.priorities.append(priority + self.epsilon)

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return [], []
        prios = np.array(self.priorities)
        probs = prios / prios.sum()
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), p=probs, replace=False)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, weights

class MARLServer:
    def __init__(self, num_uavs=30):
        self.num_uavs = num_uavs
        self.req_state_dim = 3 + num_uavs * 3
        self.req_action_dim = num_uavs
        
        self.request_net = RequestNet(self.req_state_dim, self.req_action_dim)
        self.req_optimizer = optim.Adam(self.request_net.parameters(), lr=0.001)
        
        self.res_state_dim = 7
        self.response_net = ResponseNet(self.res_state_dim)
        self.res_optimizer = optim.Adam(self.response_net.parameters(), lr=0.001)
        
        self.replay_buffer = PrioritizedReplayBuffer(1000)
        
        self.req_weights_file = "request_net.pth"
        self.res_weights_file = "response_net.pth"
        if os.path.exists(self.req_weights_file) and os.path.exists(self.res_weights_file):
            print("Loading pre-trained weights for MARL Server...")
            self.request_net.load_state_dict(torch.load(self.req_weights_file, weights_only=True))
            self.response_net.load_state_dict(torch.load(self.res_weights_file, weights_only=True))
        else:
            print("No pre-trained weights found. Starting MARL from scratch...")

    def save_weights(self):
        print("Saving MARL weights to disk...")
        torch.save(self.request_net.state_dict(), self.req_weights_file)
        torch.save(self.response_net.state_dict(), self.res_weights_file)
        
    def _get_req_state(self, task, uavs):
        state = [task['comp'], task['lat'], 1.0]
        # Ensure we always have num_uavs elements
        for i in range(self.num_uavs):
            if i < len(uavs):
                u = uavs[i]
                if u['active']:
                    dist = calculate_distance(task['x'], task['y'], u['x'], u['y'])
                    state.extend([float(u['rem_cap']), u['batt'], dist])
                else:
                    state.extend([0.0, 0.0, 1000.0])
            else:
                state.extend([0.0, 0.0, 1000.0])
        return torch.tensor(state, dtype=torch.float32)

    def _get_res_state(self, task, uav):
        dist = calculate_distance(task['x'], task['y'], uav['x'], uav['y'])
        # simplified avg comp
        avg_comp = 0.0 # Java doesn't send assigned tasks, approximate to 0
        state = [
            task['comp'], task['lat'],
            avg_comp, float(uav['rem_cap']), uav['batt'],
            uav['speed'], dist
        ]
        return torch.tensor(state, dtype=torch.float32)

    def schedule(self, req_data):
        tasks = req_data.get('tasks', [])
        uavs = req_data.get('uavs', [])
        
        assignments = []
        remaining_queue = []
        
        uav_map = {u['id']: u for u in uavs}
        uav_list = list(uav_map.values())
        
        for task in tasks:
            req_state = self._get_req_state(task, uav_list)
            action_probs, state_value = self.request_net(req_state.unsqueeze(0))
            
            K = min(5, len(uav_list))
            top_k_indices = torch.topk(action_probs[0], K).indices.tolist()
            
            responded_uavs = []
            res_transitions = []
            
            for idx in top_k_indices:
                if idx >= len(uav_list): continue
                uav = uav_list[idx]
                if not uav['active'] or uav['batt'] <= 5.0 or uav['rem_cap'] <= 0:
                    continue
                
                dist = calculate_distance(task['x'], task['y'], uav['x'], uav['y'])
                if dist > uav['radius']:
                    continue
                    
                res_state = self._get_res_state(task, uav)
                q_values = self.response_net(res_state.unsqueeze(0))
                
                if random.random() < 0.1:
                    action = random.randint(0, 1)
                else:
                    action = torch.argmax(q_values[0]).item()
                    
                if action == 1:
                    responded_uavs.append(uav)
                    
                res_transitions.append((res_state, action))
                
            best_uav = None
            if responded_uavs:
                best_uav = min(responded_uavs, key=lambda u: calculate_distance(task['x'], task['y'], u['x'], u['y']))
                
            if best_uav is not None:
                dist = calculate_distance(task['x'], task['y'], best_uav['x'], best_uav['y'])
                req_reward = 10.0 - dist * 0.1
                best_uav['rem_cap'] -= 1
                assignments.append({'task_id': task['id'], 'uav_id': best_uav['id']})
            else:
                req_reward = -10.0
                
            td_error_req = abs(req_reward - state_value.item())
            self.replay_buffer.push(('req', req_state, top_k_indices, req_reward), td_error_req)
            
            for res_state, action in res_transitions:
                res_reward = 5.0 if best_uav is not None else -1.0
                with torch.no_grad():
                    q_val = self.response_net(res_state.unsqueeze(0))[0][action].item()
                td_error_res = abs(res_reward - q_val)
                self.replay_buffer.push(('res', res_state, action, res_reward), td_error_res)
                
        # Train
        batch, weights = self.replay_buffer.sample(32)
        if batch:
            weights_t = torch.tensor(weights, dtype=torch.float32)
            for idx, transition in enumerate(batch):
                w = weights_t[idx]
                type_, state, action, reward = transition
                
                if type_ == 'req':
                    action_probs, state_value = self.request_net(state.unsqueeze(0))
                    advantage = reward - state_value.item()
                    log_probs = torch.log(action_probs[0][action[0]] + 1e-10) # simplified to first action
                    actor_loss = -(log_probs * advantage * w)
                    critic_loss = F.mse_loss(state_value[0], torch.tensor([reward], dtype=torch.float32)) * w
                    loss = actor_loss + critic_loss
                    
                    self.req_optimizer.zero_grad()
                    loss.backward()
                    self.req_optimizer.step()
                elif type_ == 'res':
                    q_values = self.response_net(state.unsqueeze(0))
                    q_val = q_values[0][action]
                    loss = F.mse_loss(q_val, torch.tensor(reward, dtype=torch.float32)) * w
                    
                    self.res_optimizer.zero_grad()
                    loss.backward()
                    self.res_optimizer.step()

        return {'assignments': assignments}

def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind((HOST, PORT))
    except OSError as e:
        print(f"Error binding to {HOST}:{PORT}: {e}")
        sys.exit(1)
        
    server.listen(1)
    print(f"MARL Task Allocation Server listening on port {PORT}...")
    
    agent = MARLServer(num_uavs=30) # Hardcoded to max 30 uavs
    
    while True:
        conn, addr = server.accept()
        with conn:
            buffer = ""
            while True:
                data = conn.recv(8192)
                if not data:
                    break
                buffer += data.decode('utf-8')
                
                if '\n' in buffer:
                    messages = buffer.split('\n')
                    for msg in messages[:-1]:
                        msg = msg.strip()
                        if not msg: continue
                        if msg == "CLOSE":
                            agent.save_weights()
                            break
                        try:
                            req_data = json.loads(msg)
                            response = agent.schedule(req_data)
                            conn.sendall((json.dumps(response) + '\n').encode('utf-8'))
                        except Exception as e:
                            print(f"Error processing message: {e}")
                            conn.sendall(b'{"error": "processing error"}\n')
                    buffer = messages[-1]
                    if "CLOSE" in buffer:
                        agent.save_weights()
                        break

if __name__ == "__main__":
    start_server()
