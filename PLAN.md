# Implementation Plan: CAR Trajectory Algorithm

## Paper
"UAV Trajectory Optimization Based on Pointer Networks and Adaptive Region Partitioning"
(Zhiqi Guo, Fengxiao Tang, Tiao Tan, Linfeng Luo, Ming Zhao — IEEE IoT Journal)

## Algorithm summary (CAR — Capability-aware Adaptive Route optimization)

The paper is a 3-stage framework:

### 1. UAV capability assessment (Eq. 15)
Scanning width (Eq. 1): `R_i = 2 tan(alpha/2) * H_i / cos(beta)`
Scan time in region j (Eq. 7): `T^Sj_i = S_j / (R_i * v^s_i)`, scan energy (Eq. 8): `p^Sj_i = q_i * T^Sj_i`
Flight time to region k (Eq. 9): `T^Ui_jk = D_jk / v_i`, flight energy (Eq. 10): `p^Ui_jk = q_i * T^Ui_jk`
Capability (Eq. 15): `F(Ui) = d * SUM_j (T^Sj_i + p^Sj_i) + (1-d) * SUM_jk (T^Ui_jk + p^Ui_jk)` with d = 0.5
Lower F => more capable.

### 2. Adaptive density/distance region clustering (Eqs. 16-17)
Local density: `phi_j = SUM_k chi(D_jk, dc) * (T^Sk_i + T^Sj_i + lam * T^Ui_jk)`
`chi = 1 if D_jk <= dc else 0`; `lambda = 1`.
`dmin_j = min(D_jk) over k with phi_k > phi_j`
`omega_j = phi_j * dmin_j`
Cluster centers = largest jumps in the sorted-descending `omega_j` sequence, detected automatically with Otsu adaptive thresholding on the adjacent-difference series.
Remaining regions assigned to the nearest center.
Sequential matching: higher-capability UAVs (lower F) assigned to larger/more complex clusters (higher omega).

### 3. Pointer network trajectory planning (Eqs. 18-23)
Input `C = {A0, A1, ..., An}` (origin + region coordinates), output = visiting permutation Y.
Encoder: LSTM over embeddings `eC = Eb*C` (Eq. 19).
Decoder: LSTM; at step t, attention relevance `u^t_i = v^T tanh(W1 e_i + W2 h_t)` for valid (unvisited/energy-feasible) regions, `-inf` otherwise (Eq. 20).
`P(pi_t | ...) = softmax(u^t)` (Eq. 21).
Trained with actor-critic (policy gradient + critic baseline):
`grad_J ~= (1/B) SUM (R_i - V_psi(C_i)) grad log p_theta(Y_i|C_i)` (Eq. 23)
Critic loss `L(psi) = (1/B) SUM (V_psi(C_i) - R_i)^2` (Eq. 24), Adam lr=1e-3, gamma=0.96, batch=64.
Reward `R = -(sigma * SUM P(Ui) + max T(Ui))` (sigma weight).
Trained on small instances, generalizes to larger ones.

## Mapping to this codebase

| Paper concept | Codebase mapping |
|---|---|
| UAV app U_i | `FogDevice` named `uav_*` |
| hard-coded attributes (v_i, q_i, R_i) | deterministic CAR attributes derived from UAV id (heterogeneity) |
| Region A_j | ground-device (`ground_device_*`) coordinates as region waypoints + per-region scan area S_j |
| Cluster G_k | subset of MD waypoints |
| A0 origin | UAV's current (x, y) position |
| Trajectory | ordered waypoint sequence per UAV, followed by moving `<= MAX_DIST`/step |

## Files

1. `plans/PLAN.md` — this plan.
2. `src/org/fog/fanet/CARRegionClustering.java` — density (phi), dmin, omega, Otsu center
   selection, region-to-center assignment, UAV capability scoring (Eq. 15), capability-based
   cluster matching, cluster balancing to numUAVs.
3. `src/org/fog/fanet/CarTrajectoryModel.java` — `TrajectoryModel` impl:
   - `start()`: open socket to CAR pointer-network server (port 5510), cluster MDs, request tours.
   - `updateTrajectories()`: advance each UAV toward its current waypoint (angle/distance), advance
     on reach, recompute plans periodically.
   - `stop()`: close socket.
4. `ppo_agent/car_pointer_net.py` — PointerNetwork (encoder LSTM + decoder LSTM + attention),
   LSTM critic, REINFORCE-with-baseline training loop on synthetic cluster instances.
5. `ppo_agent/car_ppo_server.py` — TCP server (port 5510); receives per-UAV `{id,origin,regions}`,
   returns greedy-decoded tours `{id, order}`.
6. `src/org/fog/test/perfeval/FANETSimulation.java` — accept `args[1]` trajectory type
   (`PPO` default | `CAR`), instantiate the matching `TrajectoryModel`.
7. `run_car_sim.sh` — launch the CAR Python server then run the Java sim with `CAR` trajectory.

## Scope

Only the CAR trajectory algorithm (paper's framework applied to this FANET sim).
No changes to task schedulers, no benchmarking-harness generalization.