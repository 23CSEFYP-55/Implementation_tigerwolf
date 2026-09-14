"""
TACC-AAV: Task Allocation with Communication Coordination for AAV-Assisted Edge Computing
Combines:
1. MOCPO (Multi-Objective Constrained Policy Optimization) for communication gating
2. ACBBA (Asynchronous Consensus-Based Bundle Algorithm) with dynamic energy-derived capacities
3. MHSA-MAD3PG (Multi-Head Self-Attention Actor) for continuous trajectory & offloading control
4. CTDE (Centralized Training with Decentralized Execution) with dual Lagrangian updates
"""

import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ==============================================================================
# SECTION 1: AERODYNAMIC & ENERGY HEADROOM MODEL (Zhang et al. / Wu et al.)
# ==============================================================================

@dataclass
class AAVPhysicalParams:
    mass: float = 2.5             # kg (M)
    drag_coeff: float = 0.03      # C
    frontal_area: float = 0.05    # m^2 (S)
    wingspan: float = 0.8         # m (b)
    air_density: float = 1.225    # kg/m^3 (rho)
    v_max: float = 35.0           # m/s max flight speed
    f_max: float = 2.0e9          # 2.0 GHz CPU frequency
    tx_power: float = 0.5         # Watts transmission power
    battery_capacity: float = 100000.0  # Joules (100 kJ)
    reserve_energy: float = 10000.0     # Reserve energy threshold (10 kJ)
    avg_comp_energy_per_task: float = 250.0  # Joules
    avg_comm_energy_per_task: float = 100.0  # Joules

class AerodynamicEnergyModel:
    def __init__(self, params: Optional[AAVPhysicalParams] = None):
        self.p = params or AAVPhysicalParams()
        # Gross weight W = M * g
        g = 9.8
        w = self.p.mass * g
        
        # Compute optimal horizontal velocity v_opt (Wu et al. Eq. 4)
        term1 = 2.0 * (w ** 2)
        term2 = 3.0 * self.p.drag_coeff * self.p.frontal_area * ((self.p.air_density * self.p.wingspan) ** 2)
        self.v_opt = (term1 / max(1e-6, term2)) ** 0.25
        
        # Optimal horizontal flight power P_H*
        self.p_h_opt = (4.0 / 3.0) * (w ** 2) / (self.p.air_density * (self.p.wingspan ** 2) * max(1e-3, self.v_opt))
        
        # Hovering power P_V
        self.p_v = (w ** 1.5) / math.sqrt(2.0 * self.p.frontal_area * self.p.air_density)

    def flight_power(self, speed: float) -> float:
        """Horizontal propulsive power as a function of forward speed v."""
        v = max(0.1, min(self.p.v_max, speed))
        g = 9.8
        w = self.p.mass * g
        parasitic = 0.5 * self.p.drag_coeff * self.p.frontal_area * self.p.air_density * (v ** 3)
        induced = (w ** 2) / (self.p.air_density * (self.p.wingspan ** 2) * v)
        return parasitic + induced

    def dynamic_capacity(self, remaining_energy: float) -> int:
        """
        Derives remaining task capacity N_j(E_j) from energy headroom (Phase 0 / Eq. 22-24).
        Rather than a fixed integer cap, N_j dynamically scales with E_j.
        """
        headroom = remaining_energy - self.p.reserve_energy
        if headroom <= 0:
            return 1  # Minimum emergency headroom
        unit_task_energy = self.p.avg_comp_energy_per_task + self.p.avg_comm_energy_per_task
        cap = int(headroom / unit_task_energy)
        return max(1, cap)

# ==============================================================================
# SECTION 2: TASK & ENVIRONMENT REPRESENTATIONS
# ==============================================================================

@dataclass
class VehicleTask:
    task_id: int
    x: float
    y: float
    comp_size_mi: float      # Computation size in Mega-Instructions (MI)
    data_size_mb: float      # Data payload in Megabytes (MB)
    deadline_ms: float       # Tolerant delay in ms
    priority_omega: float    # Revenue / importance weight omega_k
    source_md_id: int = -1
    arrival_time: float = 0.0

@dataclass
class AAVState:
    agent_id: int
    x: float
    y: float
    energy: float
    speed: float = 12.0
    heading: float = 0.0
    comm_range: float = 350.0  # 1-hop TDMA broadcast reach in meters

# ==============================================================================
# SECTION 3: MOCPO NETWORKS (COMMUNICATION GATING)
# ==============================================================================

class MOCPOActor(nn.Module):
    """Gated communication policy network: pi(theta_actor)(o_{j,t}) -> {0, 1}."""
    def __init__(self, obs_dim: int = 8, hidden_dim: int = 64):
        super(MOCPOActor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.gate_head = nn.Linear(hidden_dim, 1)

    def forward(self, o: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(o))
        x = F.relu(self.fc2(x))
        prob = torch.sigmoid(self.gate_head(x))
        return prob

class MOCPOCritic(nn.Module):
    """Value network estimating expected task return."""
    def __init__(self, obs_dim: int = 8, hidden_dim: int = 64):
        super(MOCPOCritic, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.v_head = nn.Linear(hidden_dim, 1)

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(o))
        x = F.relu(self.fc2(x))
        return self.v_head(x)

class MOCPOPenaltyCritic(nn.Module):
    """Penalty critic estimating expected constraint cost (comm bandwidth or energy)."""
    def __init__(self, obs_dim: int = 8, hidden_dim: int = 64):
        super(MOCPOPenaltyCritic, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.p_head = nn.Linear(hidden_dim, 1)

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(o))
        x = F.relu(self.fc2(x))
        return F.softplus(self.p_head(x))

# ==============================================================================
# SECTION 4: MHSA-MAD3PG ACTOR (ATTENTION TRAJECTORY & OFFLOADING)
# ==============================================================================

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int = 64, num_heads: int = 4):
        super(MultiHeadAttentionBlock, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-Attention + Residual
        attn_out, _ = self.mha(x, x, x)
        x = self.ln1(x + attn_out)
        # Feed-forward + Residual
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)
        return x

class MHSATrajectoryActor(nn.Module):
    """
    Multi-Head Self-Attention Actor (pi_MHSA, Zhang et al. Eq. 33).
    Jointly produces:
    1. Node selection: sigmoid(gamma_hat)
    2. Data split ratios: Dirichlet(o_k)
    3. CPU allocation: f_{k,m} <= f_max
    4. Continuous trajectory actions: [v_j(t), vartheta_j(t)]
    """
    def __init__(self, aav_feat_dim: int = 6, task_feat_dim: int = 6, embed_dim: int = 64):
        super(MHSATrajectoryActor, self).__init__()
        self.aav_proj = nn.Linear(aav_feat_dim, embed_dim)
        self.task_proj = nn.Linear(task_feat_dim, embed_dim)
        
        self.attn = MultiHeadAttentionBlock(embed_dim=embed_dim, num_heads=4)
        
        # Trajectory Head: continuous [speed, heading]
        self.traj_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # [norm_v in (0,1), norm_heading in (0,1)]
        )
        
        # Offloading Heads
        self.node_select_head = nn.Linear(embed_dim, 1)  # sigmoid probability of self vs peer/BS
        self.split_head = nn.Linear(embed_dim, 3)        # Dirichlet concentration (local, peer, BS)
        self.cpu_head = nn.Linear(embed_dim, 1)          # normalized compute allocation

    def forward(self, aav_feat: torch.Tensor, task_feats: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        aav_feat: (batch_size, 1, aav_feat_dim)
        task_feats: (batch_size, num_tasks, task_feat_dim) or None
        """
        h_aav = self.aav_proj(aav_feat)
        if task_feats is not None and task_feats.size(1) > 0:
            h_tasks = self.task_proj(task_feats)
            seq = torch.cat([h_aav, h_tasks], dim=1)
        else:
            seq = h_aav

        encoded = self.attn(seq)
        aav_repr = encoded[:, 0, :]  # AAV pooled token

        # Trajectory outputs
        traj_raw = torch.sigmoid(self.traj_head(aav_repr))
        v = traj_raw[:, 0] * 20.0            # 0 to 20 m/s
        heading = traj_raw[:, 1] * 2.0 * math.pi  # 0 to 2pi radians

        # Task offload outputs (if tasks present)
        out = {"velocity": v, "heading": heading}
        if task_feats is not None and task_feats.size(1) > 0:
            task_repr = encoded[:, 1:, :]
            out["node_select"] = torch.sigmoid(self.node_select_head(task_repr))
            out["dirichlet_alpha"] = F.softplus(self.split_head(task_repr)) + 1.0
            out["cpu_ratio"] = torch.sigmoid(self.cpu_head(task_repr))

        return out

# ==============================================================================
# SECTION 5: ACBBA CONSENSUS WITH DYNAMIC ENERGY-DERIVED CAPACITIES
# ==============================================================================

class ACBBAConsensus:
    """
    Asynchronous Consensus-Based Bundle Algorithm with Marginal-Value Bids & Table B.6 Deconfliction.
    """
    def __init__(self, num_agents: int, num_tasks: int, beta1: float = 1.0, beta2: float = 0.5, beta3: float = 0.2):
        self.num_agents = num_agents
        self.num_tasks = num_tasks
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        
        # State vectors per agent j: winning agent z, winning bid y, timestamp s
        self.z: List[List[int]] = [[-1] * num_tasks for _ in range(num_agents)]
        self.y: List[List[float]] = [[0.0] * num_tasks for _ in range(num_agents)]
        self.s: List[List[float]] = [[0.0] * num_tasks for _ in range(num_agents)]
        
        # Bundles p_j per agent
        self.bundles: List[List[int]] = [[] for _ in range(num_agents)]

    def compute_marginal_bid(self, agent: AAVState, task: VehicleTask, aero: AerodynamicEnergyModel) -> float:
        """
        Computes marginal-value bid c_{j,k} <- Zhang et al. reward:
        c_{j,k} = beta1 * omega_k(t) - beta2 * T_total - beta3 * E_total
        """
        dist = math.hypot(agent.x - task.x, agent.y - task.y)
        v = aero.v_opt
        flight_time_s = dist / max(1.0, v)
        comm_time_s = (task.data_size_mb * 8.0) / 50.0  # 50 Mbps link approx
        comp_time_s = (task.comp_size_mi) / (agent.speed * 100.0)
        t_total = flight_time_s + comm_time_s + comp_time_s
        
        flight_energy = aero.flight_power(v) * flight_time_s
        comp_energy = 1e-27 * ((aero.p.f_max) ** 2) * (task.comp_size_mi * 1e6)
        comm_energy = aero.p.tx_power * comm_time_s
        e_total = flight_energy + comp_energy + comm_energy

        # Marginal value (higher is better)
        revenue = self.beta1 * task.priority_omega * 10.0
        delay_cost = self.beta2 * (t_total)
        energy_cost = self.beta3 * (e_total / 1000.0)
        bid = revenue - delay_cost - energy_cost
        return max(0.01, bid)

    def bundle_construction(self, agent_idx: int, agent: AAVState, tasks: List[VehicleTask],
                            aero: AerodynamicEnergyModel, current_time: float):
        """
        Phase 1: Local bidding with dynamic energy-derived capacity check |p_j| < N_j(E_j).
        """
        n_cap = aero.dynamic_capacity(agent.energy)
        local_bundle = self.bundles[agent_idx]

        for k, task in enumerate(tasks):
            if k in local_bundle:
                continue
            if len(local_bundle) >= n_cap:
                break

            bid = self.compute_marginal_bid(agent, task, aero)
            # Check capacity and winning bid condition
            if bid > self.y[agent_idx][k]:
                local_bundle.append(k)
                self.z[agent_idx][k] = agent_idx
                self.y[agent_idx][k] = bid
                self.s[agent_idx][k] = current_time

    def deconflict(self, receiver_idx: int, sender_idx: int,
                   sender_z: List[int], sender_y: List[float], sender_s: List[float]):
        """
        Phase 2: Table B.6 consensus deconfliction rules.
        Resolves conflicts between receiver and sender vectors.
        """
        for k in range(self.num_tasks):
            z_recv = self.z[receiver_idx][k]
            y_recv = self.y[receiver_idx][k]
            s_recv = self.s[receiver_idx][k]
            
            z_send = sender_z[k]
            y_send = sender_y[k]
            s_send = sender_s[k]

            # Rule 1: Sender believes it is winner
            if z_send == sender_idx:
                if z_recv == receiver_idx:
                    # Both believe they are winners: higher bid wins
                    if y_send > y_recv or (abs(y_send - y_recv) < 1e-6 and sender_idx < receiver_idx):
                        self._yield_task(receiver_idx, k, z_send, y_send, s_send)
                elif z_recv == sender_idx:
                    # Update info from the actual winner
                    self.y[receiver_idx][k] = y_send
                    self.s[receiver_idx][k] = s_send
                elif z_recv == -1:
                    # Unassigned locally: adopt sender's assignment
                    self.z[receiver_idx][k] = z_send
                    self.y[receiver_idx][k] = y_send
                    self.s[receiver_idx][k] = s_send
                else:
                    # Receiver thinks another agent m won: compare timestamps
                    if s_send > s_recv or y_send > y_recv:
                        self.z[receiver_idx][k] = z_send
                        self.y[receiver_idx][k] = y_send
                        self.s[receiver_idx][k] = s_send

            # Rule 2: Sender believes receiver is winner
            elif z_send == receiver_idx:
                if z_recv == receiver_idx:
                    self.s[receiver_idx][k] = max(s_recv, s_send)
                else:
                    # Receiver already gave it up
                    self.y[receiver_idx][k] = 0.0

            # Rule 3: Sender thinks task is unassigned
            elif z_send == -1:
                if z_recv == sender_idx:
                    # Sender dropped its assignment: clear locally
                    self._yield_task(receiver_idx, k, -1, 0.0, s_send)
                elif s_send > s_recv and z_recv != receiver_idx:
                    self.z[receiver_idx][k] = -1
                    self.y[receiver_idx][k] = 0.0
                    self.s[receiver_idx][k] = s_send

            # Rule 4: Sender believes third agent m won
            else:
                if s_send > s_recv:
                    if z_recv == receiver_idx and y_send > y_recv:
                        self._yield_task(receiver_idx, k, z_send, y_send, s_send)
                    else:
                        self.z[receiver_idx][k] = z_send
                        self.y[receiver_idx][k] = y_send
                        self.s[receiver_idx][k] = s_send

    def _yield_task(self, agent_idx: int, task_idx: int, new_z: int, new_y: float, new_s: float):
        """Cascades removal of task from bundle and reclaims capacity."""
        bundle = self.bundles[agent_idx]
        if task_idx in bundle:
            pos = bundle.index(task_idx)
            # Truncate bundle from contested task onward (ACBBA cascade release)
            for removed_task in bundle[pos:]:
                self.z[agent_idx][removed_task] = -1
                self.y[agent_idx][removed_task] = 0.0
            self.bundles[agent_idx] = bundle[:pos]

        self.z[agent_idx][task_idx] = new_z
        self.y[agent_idx][task_idx] = new_y
        self.s[agent_idx][task_idx] = new_s

# ==============================================================================
# SECTION 6: TACC-AAV SWARM COORDINATOR & CTDE TRAINING ENGINE
# ==============================================================================

class TACCAAVCoordinator:
    def __init__(self, num_aavs: int = 30, max_tasks: int = 100,
                 c_sup: float = 0.5, c_energy: float = 50000.0,
                 lr: float = 0.001):
        self.num_aavs = num_aavs
        self.max_tasks = max_tasks
        self.c_sup = c_sup          # Bandwidth transmission budget
        self.c_energy = c_energy    # Total energy budget constraint
        
        self.aero = AerodynamicEnergyModel()
        
        # MOCPO Networks (Gating Policy)
        self.actor = MOCPOActor(obs_dim=8)
        self.critic = MOCPOCritic(obs_dim=8)
        self.penalty_comm = MOCPOPenaltyCritic(obs_dim=8)
        self.penalty_energy = MOCPOPenaltyCritic(obs_dim=8)
        
        # Dual Lagrangian Multipliers
        self.lambda_comm = 0.1
        self.lambda_energy = 0.05
        self.alpha_lambda = 0.01
        
        # MHSA Trajectory Actor
        self.mhsa_actor = MHSATrajectoryActor()
        
        # Optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)
        self.penalty_comm_opt = optim.Adam(self.penalty_comm.parameters(), lr=lr)
        self.penalty_energy_opt = optim.Adam(self.penalty_energy.parameters(), lr=lr)
        self.mhsa_opt = optim.Adam(self.mhsa_actor.parameters(), lr=lr)
        
        self.replay_buffer = deque(maxlen=2000)

    def extract_observation(self, agent_idx: int, agent: AAVState, consensus: ACBBAConsensus,
                            neighbors: List[AAVState], t_norm: float) -> torch.Tensor:
        """Constructs o_{j,t}: BK feature, VoM feature, normalized step, channel access."""
        n_cap = self.aero.dynamic_capacity(agent.energy)
        bundle = consensus.bundles[agent_idx]
        bk_ratio = len(bundle) / max(1, n_cap)
        avg_bid = np.mean([consensus.y[agent_idx][k] for k in bundle]) if bundle else 0.0
        
        # VoM: highest available unallocated task bid
        unallocated_bids = [consensus.y[agent_idx][k] for k in range(consensus.num_tasks) if k not in bundle]
        vom = max(unallocated_bids) if unallocated_bids else 0.0
        
        # Channel access / neighbor density
        neighbor_density = len(neighbors) / max(1.0, float(self.num_aavs))
        avg_dist = np.mean([math.hypot(agent.x - n.x, agent.y - n.y) for n in neighbors]) if neighbors else 500.0
        norm_dist = min(1.0, avg_dist / 1000.0)
        norm_energy = agent.energy / self.aero.p.battery_capacity

        obs = np.array([
            bk_ratio, avg_bid / 50.0, vom / 50.0, t_norm,
            neighbor_density, norm_dist, norm_energy, agent.speed / 20.0
        ], dtype=np.float32)
        return torch.tensor(obs)

    def step_simulation(self, agents: List[AAVState], tasks: List[VehicleTask], current_time: float) -> Dict:
        """
        Executes one complete TACC-AAV cycle:
        Phase 1: Bundle bidding + MOCPO gating
        Phase 2: Local consensus broadcast
        Phase 3: MHSA continuous trajectory + offload execution
        """
        num_tasks = len(tasks)
        if num_tasks == 0:
            return {"trajectories": {}, "gating_decisions": {}, "assignments": []}

        consensus = ACBBAConsensus(len(agents), num_tasks)
        t_norm = (current_time % 1000.0) / 1000.0

        # Phase 1: Local Bidding
        for i, agent in enumerate(agents):
            consensus.bundle_construction(i, agent, tasks, self.aero, current_time)

        # Communication gating via MOCPO
        gating_decisions = {}
        observations = {}
        for i, agent in enumerate(agents):
            neighbors = [other for j, other in enumerate(agents) if i != j and math.hypot(agent.x - other.x, agent.y - other.y) <= agent.comm_range]
            obs = self.extract_observation(i, agent, consensus, neighbors, t_norm)
            observations[i] = obs
            with torch.no_grad():
                prob = self.actor(obs.unsqueeze(0)).item()
            # Gated binary decision via Bernoulli policy
            action = 1 if random.random() < prob else 0
            gating_decisions[i] = action

        # Phase 2: Consensus Stage (broadcast only if gating decision == 1)
        for i, agent in enumerate(agents):
            if gating_decisions[i] == 1:
                # Transmit to 1-hop neighbors
                for j, other in enumerate(agents):
                    if i != j and math.hypot(agent.x - other.x, agent.y - other.y) <= agent.comm_range:
                        consensus.deconflict(j, i, consensus.z[i], consensus.y[i], consensus.s[i])

        # Phase 3: Trajectory & Offload Execution via MHSA
        trajectories = {}
        assignments = []
        for i, agent in enumerate(agents):
            bundle = consensus.bundles[i]
            aav_feat = torch.tensor([[
                agent.x / 2000.0, agent.y / 2000.0, agent.energy / self.aero.p.battery_capacity,
                agent.speed / 20.0, math.cos(agent.heading), math.sin(agent.heading)
            ]], dtype=torch.float32).unsqueeze(0)  # (1, 1, 6)

            task_feats = []
            for tid in bundle:
                t = tasks[tid]
                task_feats.append([
                    t.x / 2000.0, t.y / 2000.0, t.comp_size_mi / 5000.0,
                    t.data_size_mb / 100.0, t.deadline_ms / 2000.0, t.priority_omega
                ])
                assignments.append({"task_id": t.task_id, "uav_id": agent.agent_id})

            if task_feats:
                t_tensor = torch.tensor(task_feats, dtype=torch.float32).unsqueeze(0)
            else:
                t_tensor = None

            with torch.no_grad():
                act = self.mhsa_actor(aav_feat, t_tensor)
                v = act["velocity"].item()
                heading = act["heading"].item()

            trajectories[agent.agent_id] = {
                "velocity": v,
                "heading": heading,
                "bundle_size": len(bundle)
            }

            # Deduct energy
            flight_p = self.aero.flight_power(v)
            agent.energy = max(0.0, agent.energy - (flight_p * 1.0 + len(bundle) * 5.0))

            # Store transition for training
            self.replay_buffer.append((
                observations[i],
                gating_decisions[i],
                len(bundle) * 5.0,  # reward
                1.0 if gating_decisions[i] == 1 else 0.0,  # comm cost
                flight_p / 100.0  # energy cost
            ))

        return {
            "trajectories": trajectories,
            "gating_decisions": gating_decisions,
            "assignments": assignments,
            "consensus": consensus
        }

    def train_step(self, batch_size: int = 32):
        """
        Phase 4: CTDE Training update with dual Lagrangian multipliers.
        """
        if len(self.replay_buffer) < batch_size:
            return

        batch = random.sample(self.replay_buffer, batch_size)
        obs_t = torch.stack([b[0] for b in batch])
        actions_t = torch.tensor([b[1] for b in batch], dtype=torch.float32)
        rewards_t = torch.tensor([b[2] for b in batch], dtype=torch.float32).unsqueeze(1)
        comm_costs_t = torch.tensor([b[3] for b in batch], dtype=torch.float32).unsqueeze(1)
        energy_costs_t = torch.tensor([b[4] for b in batch], dtype=torch.float32).unsqueeze(1)

        # Value & Penalty updates
        v_pred = self.critic(obs_t)
        p_comm_pred = self.penalty_comm(obs_t)
        p_energy_pred = self.penalty_energy(obs_t)

        loss_v = F.mse_loss(v_pred, rewards_t)
        loss_p_comm = F.mse_loss(p_comm_pred, comm_costs_t)
        loss_p_energy = F.mse_loss(p_energy_pred, energy_costs_t)

        self.critic_opt.zero_grad()
        loss_v.backward()
        self.critic_opt.step()

        self.penalty_comm_opt.zero_grad()
        loss_p_comm.backward()
        self.penalty_comm_opt.step()

        self.penalty_energy_opt.zero_grad()
        loss_p_energy.backward()
        self.penalty_energy_opt.step()

        # Actor Lagrangian optimization
        probs = self.actor(obs_t).squeeze()
        log_probs = torch.where(actions_t == 1, torch.log(probs + 1e-8), torch.log(1.0 - probs + 1e-8))
        adv = (rewards_t - v_pred.detach()).squeeze()
        adv_comm = (comm_costs_t - p_comm_pred.detach()).squeeze()
        adv_energy = (energy_costs_t - p_energy_pred.detach()).squeeze()

        # Lagrangian surrogate: A - lambda_comm * P_comm - lambda_energy * P_energy
        lagrangian_adv = adv - (self.lambda_comm * adv_comm) - (self.lambda_energy * adv_energy)
        actor_loss = -(log_probs * lagrangian_adv).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Dual ascent multiplier update: lambda <- max(0, lambda + alpha * (C - c_sup))
        mean_comm = comm_costs_t.mean().item()
        mean_energy = energy_costs_t.mean().item()
        self.lambda_comm = max(0.0, self.lambda_comm + self.alpha_lambda * (mean_comm - self.c_sup))
        self.lambda_energy = max(0.0, self.lambda_energy + self.alpha_lambda * (mean_energy - (self.c_energy / 1000.0)))
