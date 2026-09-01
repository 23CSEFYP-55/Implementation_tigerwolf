"""
Pointer-network based trajectory planner for the CAR algorithm.

Implements the pointer network of:

    "UAV Trajectory Optimization Based on Pointer Networks and Adaptive
     Region Partitioning" (Guo, Tang, Tan, Luo, Zhao - IEEE IoT Journal).

Architecture (paper Sec. IV-C):
- Encoder: LSTM over the embedding eC = Eb * C of the input set
  C = {A0, A1, ..., An} (Eq. 19).
- Decoder: LSTM; at each step t the attention mechanism scores the
  candidate regions with
      u^t_i = v^T tanh(W1 e_i + W2 h_t)     (valid regions)
      u^t_i = -inf                          (already visited)            (Eq. 20)
  and the next region is drawn from P(pi_t | ...) = softmax(u^t)        (Eq. 21).
- Training: actor-critic (policy gradient with a critic baseline, Eq. 22-24).
  Reward R = -(sigma * SUM_i P(Ui) + max_i T(Ui)) from the paper's cost
  model (Eqs. 7-13); convergence is reported after ~2000 steps (paper Fig. 5).

The network is trained on small synthetic cluster instances and generalizes
to larger ones (paper Sec. VI).
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointerNetwork(nn.Module):
    """Actor network: encoder(LSTM) + decoder(LSTM) with pointer attention."""

    def __init__(self, input_dim=2, emb_dim=128, hidden_dim=128):
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        # Embedding matrix Eb (Eq. 19): coordinate -> feature vector.
        self.embed = nn.Linear(input_dim, emb_dim)

        # Recurrent encoder.
        self.encoder = nn.LSTM(emb_dim, hidden_dim, batch_first=True)

        # Recurrent decoder (reads the embedding of the last selected node).
        self.decoder_input = nn.Linear(input_dim, emb_dim)
        self.decoder = nn.LSTM(emb_dim, hidden_dim, batch_first=True)

        # Attention parameters (v, W1, W2) of Eq. 20.
        self.proj_enc = nn.Linear(hidden_dim, hidden_dim)  # W1 e_i
        self.proj_dec = nn.Linear(hidden_dim, hidden_dim)  # W2 h_t
        self.v = nn.Linear(hidden_dim, 1, bias=False)      # v^T ()

    def forward(self, coords, mask=None, greedy=False, lens=None):
        """
        coords: (batch, seq, 2)  - [origin, region0, region1, ...]
        lens:   (batch,) - number of real nodes (origin + regions) per instance;
                used to mask padding when instances in a batch differ in length.
        Returns the permutation indices of the regions ONLY (origin excluded).
        """
        batch, seq, _ = coords.shape

        # Encode the whole input sequence (Eq. 19).
        emb = torch.tanh(self.embed(coords))
        enc_out, (h, c) = self.encoder(emb)
        # seed decoder hidden with the encoder's final state
        h_dec = (h, c)

        dec_in = emb[:, 0:1]  # decoder starts at the origin node
        selected = []
        visited_mask = torch.zeros(batch, seq, dtype=torch.bool, device=coords.device)
        visited_mask[:, 0] = True  # origin is always first (pi_0 = A0)

        enc_keys = self.proj_enc(enc_out)  # (batch, seq, hidden)

        if lens is not None:
            pad_mask = torch.arange(seq, device=coords.device) >= lens.unsqueeze(1)
        else:
            pad_mask = torch.zeros(batch, seq, dtype=torch.bool, device=coords.device)

        for _ in range(seq - 1):
            dec_out, h_dec = self.decoder(dec_in, h_dec)  # (batch,1,hidden)

            # Attention relevance u^t_i (Eq. 20).
            score = self.v(torch.tanh(enc_keys + self.proj_dec(dec_out)))  # (batch,seq,1)
            score = score.squeeze(-1)  # (batch, seq)

            # Mask already-visited and padding regions.
            mask_ = visited_mask | pad_mask
            score = score.masked_fill(mask_, float("-inf"))

            # Softmax over valid regions (Eq. 21).
            probs = F.softmax(score, dim=-1)

            if greedy:
                choice = torch.argmax(probs, dim=-1)
            else:
                probs_safe = probs + 1e-9
                probs_safe = probs_safe / probs_safe.sum(dim=-1, keepdim=True)
                choice = torch.multinomial(probs_safe, 1).squeeze(-1)

            selected.append(choice)
            visited_mask = visited_mask.scatter(1, choice.unsqueeze(-1).long(), True)

            # Feed the embedding of the chosen node into the decoder.
            chosen_emb = emb.gather(1, choice.long().unsqueeze(-1).unsqueeze(-1)
                                    .expand(-1, -1, self.emb_dim))
            dec_in = chosen_emb

        tour = torch.stack(selected, dim=1)  # (batch, seq-1)
        # Shift indices so they point into the REGION array (index 0 = region0).
        tour = tour - 1
        return tour

    def log_probs(self, coords, tour, lens=None):
        """Log-probability of a pre-sampled region tour (for policy gradient)."""
        batch, seq, _ = coords.shape
        emb = torch.tanh(self.embed(coords))
        enc_out, (h, c) = self.encoder(emb)
        h_dec = (h, c)

        dec_in = emb[:, 0:1]
        enc_keys = self.proj_enc(enc_out)

        log_probs = []
        visited_mask = torch.zeros(batch, seq, dtype=torch.bool, device=coords.device)
        visited_mask[:, 0] = True

        if lens is not None:
            pad_mask = torch.arange(seq, device=coords.device) >= lens.unsqueeze(1)
        else:
            pad_mask = torch.zeros(batch, seq, dtype=torch.bool, device=coords.device)

        # tour entries index into the region array; convert back to node index.
        node_seq = tour + 1
        for t in range(seq - 1):
            dec_out, h_dec = self.decoder(dec_in, h_dec)
            score = self.v(torch.tanh(enc_keys + self.proj_dec(dec_out))).squeeze(-1)
            score = score.masked_fill(visited_mask | pad_mask, float("-inf"))
            probs = F.softmax(score, dim=-1)
            node = node_seq[:, t]
            lp = torch.log(probs.gather(1, node.unsqueeze(-1)).squeeze(-1) + 1e-9)
            log_probs.append(lp)
            visited_mask = visited_mask.scatter(1, node.unsqueeze(-1).long(), True)
            chosen_emb = emb.gather(1, node.unsqueeze(-1).long().unsqueeze(-1)
                                    .expand(-1, -1, self.emb_dim))
            dec_in = chosen_emb

        return torch.stack(log_probs, dim=1).sum(dim=1)  # (batch,)


class Critic(nn.Module):
    """Critic network: LSTM over the input set predicts the expected reward."""

    def __init__(self, input_dim=2, emb_dim=128, hidden_dim=128):
        super().__init__()
        self.embed = nn.Linear(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, coords):
        emb = torch.tanh(self.embed(coords))
        out, (h, _) = self.lstm(emb)
        return self.fc_out(out[:, -1])  # value of the full instance


class CostModel:
    """Paper cost model (Eqs. 7-13) for a single synthetic UAV."""

    def __init__(self, speed=16.0, scan_width=130.0, power=1.0, sigma=1.0, scale=1.0):
        self.speed = speed            # v_i
        self.scan_width = scan_width  # R_i
        self.power = power            # q_i
        self.sigma = sigma            # energy vs time weighting
        self.scan_speed = speed       # v^s_i = mu * v_i, mu = 1
        self.scale = scale            # 1.0 by default (distances are in meters)

    def scan_time(self, area):
        return area / (self.scan_width * self.scan_speed)  # Eq. 7

    def flight_time(self, dist):
        return dist * self.scale / self.speed              # Eq. 9

    def evaluate(self, coords, areas, tour):
        """
        coords: (seq, 2) - [origin, r0, r1, ...]
        areas:  (seq-1,) - region scan areas S_j
        tour:   (seq-1,) - region order (indices into the regions)
        Returns reward = -(sigma * P + T).
        """
        T = 0.0
        P = 0.0
        prev = coords[0]
        for i, r in enumerate(tour):
            cur = coords[r + 1]
            tF = self.flight_time(np.linalg.norm(cur - prev))
            tS = self.scan_time(areas[r])
            T += tF + tS
            P += self.power * (tF + tS)  # Eqs. 8 and 10: energy = q * time
            prev = cur
        return -(self.sigma * P + T)


def make_random_instance(rng, num_regions, area_max=2000.0):
    n = num_regions
    coords = rng.uniform(0.0, area_max, size=(n + 1, 2)).astype(np.float32)  # +origin
    areas = rng.uniform(50.0, 100.0, size=n).astype(np.float32)             # S_j in [50,100]
    return coords, areas


def train(model, critic, opt_a, opt_c, cost, steps=2000, batch_size=64,
          min_regions=5, max_regions=12, seed=0):
    """REINFORCE with actor-critic baseline (Eqs. 23-24), Adam (lr 1e-3).

    Instances within one batch share the same region count (classic pointer
    network training, Bello et al.); the region count varies across steps so
    the model learns to generalize to different cluster sizes.
    """
    rng = np.random.RandomState(seed)
    model.train()
    critic.train()
    device = next(model.parameters()).device

    for step in range(1, steps + 1):
        n = rng.randint(min_regions, max_regions + 1)
        coords_list = []
        areas_list = []
        for _ in range(batch_size):
            c, a = make_random_instance(rng, n)
            coords_list.append(c)
            areas_list.append(a)

        coords = torch.from_numpy(np.stack(coords_list)).to(device)

        # Sample a tour from the current policy and score it under the cost model.
        tour = model(coords, greedy=False)
        rewards = torch.tensor(
            [cost.evaluate(coords_list[i].astype(np.float64), areas_list[i].astype(np.float64),
                           [int(x) for x in tour[i].cpu().tolist()])
             for i in range(batch_size)],
            dtype=torch.float32, device=device)

        # Critic baseline V_psi(C) (Eq. 23).
        values = critic(coords).squeeze(-1)
        advantages = rewards - values.detach()

        # Actor: max J(theta) ~= mean (R - V) log p_theta            (Eq. 22-23)
        log_p = model.log_probs(coords, tour)
        actor_loss = -(advantages * log_p).mean()

        # Critic: minimize MSE between V_psi(C) and R               (Eq. 24)
        critic_loss = F.mse_loss(values, rewards)

        opt_a.zero_grad()
        actor_loss.backward()
        opt_a.step()

        opt_c.zero_grad()
        critic_loss.backward()
        opt_c.step()

        if step % 500 == 0 or step == steps:
            print(f"[CAR-PtrNet] step {step:5d} | actor_loss {actor_loss.item():.4f} | "
                  f"critic_loss {critic_loss.item():.6f} | mean_reward -{(-rewards).mean().item():.2f}")


def decode(model, origin, regions, cost=None, device="cpu"):
    """
    Greedy decode of a tour for one UAV cluster (paper inference).
    origin:  [x, y]; regions: list of [x, y]
    Returns the ordered list of region indices.
    """
    model.eval()
    if len(regions) == 0:
        return []
    seq = np.stack([origin] + regions).astype(np.float32)
    coords = torch.from_numpy(seq[None, ...]).to(device)
    with torch.no_grad():
        tour = model(coords, greedy=True)  # (1, n) region indices
    return tour[0].cpu().tolist()