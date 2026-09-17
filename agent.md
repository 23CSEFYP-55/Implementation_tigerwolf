# AGENT INSTRUCTIONS & REPOSITORY HANDBOOK: FANET-MEC TESTBED

> **CRITICAL AGENT RULES:**
> 1. **GIT COMMIT REGULARLY:** After completing any major milestone, bugfix, algorithm implementation, or metric addition, make a clean, informative git commit (`git add <files> && git commit -m "..."`). Never let uncommitted work pile up.
> 2. **UPDATE `agent.md` AFTER CHANGES:** Whenever you alter the architecture, implement a new scheduler or trajectory model, add or modify metrics, change port mappings, or add new scripts, you MUST update this `agent.md` file to reflect the new state.

---

## 1. System Overview & Problem Formulation

This repository hosts a hybrid Java/Python simulation and benchmarking testbed for **Uncrewed Aerial Vehicle (UAV / AAV) Assisted Mobile Edge Computing (FANET-MEC) Networks**.

* **Scenario:** 800 mobile ground devices (MDs) distributed across a $2000\text{m} \times 2000\text{m}$ area generating computation tasks via a Poisson process ($\lambda = 1000\text{ ms}$).
* **Aerial Swarm:** Heterogeneous UAV fleet ($N = 30$ drones) equipped with onboard edge processors (e.g., $2800\text{--}26000$ MIPS, $150$-task buffer capacity, $20\text{m}\text{--}140\text{m}$ coverage radius).
* **Core Problem:** Jointly optimize:
  1. **Task Allocation:** Mapping ground tasks to aerial edge servers under coverage, buffer, battery, and latency constraints.
  2. **Trajectory Planning:** Guiding continuous UAV flight paths to balance energy consumption, coverage, and deadline satisfaction.

---

## 2. Implemented Schedulers (Task Allocation Algorithms)

Schedulers implement `src/org/fog/fanet/TaskScheduler.java` (`void scheduleTasks(List<Tuple> tasks, List<FogDevice> uavs)`):

| Scheduler Name | Implementation Class | Paradigm | Key Characteristics |
| :--- | :--- | :--- | :--- |
| **`DYNAMIC`** | [`DynamicRepairScheduler.java`](file:///Users/totallynotmanas/Documents/GitHub/Implementation_tigerwolf/src/org/fog/fanet/DynamicRepairScheduler.java) | **Proposed Dynamic Matching Maintenance** | Caches active pairings in `AssignmentState`. Selectively invalidates broken links ($d > R_{\text{cov}}$), routes displaced/new tasks through an incremental repair queue, and supports cooperative P2P offloading ($d \le 500\text{m}$). Overhead $< 0.3\text{ ms}$, throughput $> 41\%$. |
| **`MOGS`** | [`MOGSScheduler.java`](file:///Users/totallynotmanas/Documents/GitHub/Implementation_tigerwolf/src/org/fog/fanet/MOGSScheduler.java) | Many-to-One Gale-Shapley Matching | Baseline algorithm. Rebuilds preference lists and computes deferred-acceptance matching from scratch every epoch. Overhead $\approx 0.5\text{--}0.7\text{ ms}$. |
| **`LEXICOGRAPHIC`** | [`LexicographicScheduler.java`](file:///Users/totallynotmanas/Documents/GitHub/Implementation_tigerwolf/src/org/fog/fanet/LexicographicScheduler.java) | Priority Queue Matching | Prioritizes tasks based on deadline slack and matches them with available UAVs greedily. Overhead $\approx 0.8\text{--}1.5\text{ ms}$. |
| **`BIDDING`** | [`BiddingScheduler.java`](file:///Users/totallynotmanas/Documents/GitHub/Implementation_tigerwolf/src/org/fog/fanet/BiddingScheduler.java) | Auction / Contract-Net Protocol | Drones submit bids based on queue load and distance. Slower due to iterative round exchange ($\approx 7\text{--}10\text{ ms}$). |
| **`MARL`** | [`MARLScheduler.java`](file:///Users/totallynotmanas/Documents/GitHub/Implementation_tigerwolf/src/org/fog/fanet/MARLScheduler.java) + `ppo_agent/marl_allocation_server.py` | Multi-Agent Deep RL (Port 5501) | Two-stage `RequestNet` (actor-critic) + `ResponseNet` (DQN). Uses **Geometric Candidate Masking** ($d \le R_{\text{cov}}$) and remaining battery headroom tracking. Overhead $\approx 10\text{--}12\text{ ms}$, throughput $\approx 13\%\text{--}18\%$. |

---

## 3. Implemented Trajectory Models

Trajectory models implement `src/org/fog/fanet/TrajectoryModel.java` (`start()`, `updateTrajectories(...)`, `stop()`):

| Model Name | Implementation Class & Python Server | Port | Paradigm | Key Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| **`PPO`** | [`PPOTrajectoryModel.java`](file:///Users/totallynotmanas/Documents/GitHub/Implementation_tigerwolf/src/org/fog/fanet/PPOTrajectoryModel.java) + `ppo_agent/tf_ppo_server.py` | **5500** | Trajectory-Feedback PPO | Maps MD densities in 8 cardinal/intercardinal sectors into continuous velocity and heading angle actions. |
| **`CAR`** | [`CarTrajectoryModel.java`](file:///Users/totallynotmanas/Documents/GitHub/Implementation_tigerwolf/src/org/fog/fanet/CarTrajectoryModel.java) + `ppo_agent/car_ppo_server.py` | **5510** | Capability-Aware Adaptive Route | Density-based Otsu region partitioning, capability matching, and Pointer-Network TSP tour visiting orders. |
| **`TACC`** / **`TACC_AAV`** | [`TACCAAVTrajectoryModel.java`](file:///Users/totallynotmanas/Documents/GitHub/Implementation_tigerwolf/src/org/fog/fanet/TACCAAVTrajectoryModel.java) + `ppo_agent/tacc_aav_server.py` | **5530** | Task Allocation with Comm Coordination | MOCPO communication gating policy ($a_{j,t} \in \{0, 1\}$), ACBBA 18-rule consensus (Table B.6) with dynamic energy capacities $N_j(E_j)$, MHSA continuous velocity/heading $[v_j, \vartheta_j]$, Dirichlet task splitting, and CTDE dual Lagrangian ascent. Saves $\approx 50\%$ comm overhead. |

---

## 4. Evaluation Metrics System

All simulations output real-time per-batch telemetry lines formatted as:
```
[METRIC_EPOCH] Epoch:e, Generated:g, Scheduled:s, Dropped:d, RuntimeMs:r, UavUtil:u, CapUtil:c, RepairQ:rq, Invalid:inv, Reassigned:re, Fairness:f, Churn:ch, Slack:sl
```

### Specialized Allocation Metrics:
1. **Jain's Fairness Index ($\mathcal{J}_{\text{load}}$):**
   $$\mathcal{J} = \frac{\left(\sum_{j=1}^N L_j\right)^2}{N \sum_{j=1}^N L_j^2}, \quad L_j = \frac{|\mathcal{Q}_j|}{C_j}$$
   Quantifies workload distribution balance across the swarm ($1.0$ is perfect equality).
2. **Allocation Churn Rate ($\chi$):**
   $$\chi = \frac{1}{|\mathcal{M}_{\text{eval}}|} \sum_{i \in \mathcal{M}} \mathbb{I}\left(u_i^{(t)} \ne u_i^{(t-1)} \wedge d(i, u_i^{(t-1)}) \le R_{\text{cov}} \wedge |\mathcal{Q}_{u_i^{(t-1)}}| < C\right)$$
   Measures unnecessary task reallocations away from a UAV that was still reachable and had available queue buffer.
3. **Deadline Slack Margin ($\bar{S}_{\text{slack}}$):**
   $$\bar{S} = \frac{1}{|K|} \sum_{k \in K} \max\left(0, \min\left(1, \frac{T_k^{\text{tol}} - (T_{\text{tx}} + T_{\text{queue}})}{T_k^{\text{tol}}}\right)\right)$$
   Measures the safety margin before tasks violate their latency deadline.

---

## 5. Development Environment & Command Cheatsheet

### Environment Requirements
* **Python:** 3.12+ in `.venv/` with PyTorch, NumPy, Matplotlib, Rich, Pytest.
* **Java:** OpenJDK 21 at `/opt/homebrew/opt/openjdk@21/bin/java`.

### Compiling Java Code
Always recompile Java after editing any `.java` files:
```bash
javac --release 21 -d out/production/iFogSim -cp "jars/*:jars/commons-math3-3.5/*" -sourcepath src src/org/fog/test/perfeval/FANETSimulation.java
```

### Running Automated Mix-and-Match Benchmarking
Use `run_compare.py` to run combinations of schedulers and trajectories:
```bash
# Automated run across all 15 combinations (3 trajectories x 5 schedulers)
.venv/bin/python run_compare.py --sched DYNAMIC,MOGS,LEXICOGRAPHIC,BIDDING,MARL --traj PPO,CAR,TACC --runs 1

# Interactive menu
.venv/bin/python run_compare.py
```
Outputs are canonicalized in:
`results/by_trajectory/<TRAJECTORY>/<SCHEDULER>/run-<n>.log`
as well as timestamped archive directories under `results/compare_<timestamp>/`.

### Running Cross-Multiplied Graph Generation
Cross-multiplies all 3 trajectory algorithms (`PPO`, `CAR`, `TACC`) with all 5 task schedulers (`DYNAMIC`, `MOGS`, `LEXICOGRAPHIC`, `BIDDING`, `MARL`):
```bash
# Render from stored logs in results/by_trajectory/ (instant)
.venv/bin/python graph_generator.py

# Force re-running simulations
.venv/bin/python graph_generator.py --run
```
Generated 3-panel cross-trajectory plots in `images/benchmarks/` ($1 \times 3$ subplots: PPO | CAR | TACC):
* `images/benchmarks/throughput_comparison.png`
* `images/benchmarks/runtime_comparison.png`
* `images/benchmarks/utilization_comparison.png`
* `images/benchmarks/capacity_utilization_comparison.png`
* `images/benchmarks/queue_length_comparison.png`
* `images/benchmarks/repair_queue_comparison.png`
* `images/benchmarks/fairness_comparison.png`
* `images/benchmarks/churn_comparison.png`
* `images/benchmarks/slack_margin_comparison.png`
Macro cross-product summary:
* `images/benchmarks/trajectory_cross_summary.png` (4-panel grouped bar chart comparing all 15 combinations across Success Rate, Overhead, Fairness, and Slack)

### Running 2D Spatial Trajectory Map Visualizer
Visualizes continuous 2D flight paths across the $2000\text{m} \times 2000\text{m}$ area for all trajectory algorithms:
```bash
.venv/bin/python trajectory_map_generator.py
```
Generated trajectory map figures in `images/trajectories/`:
* `images/trajectories/trajectory_maps_all.png` (3-panel publication-quality comparative figure: PPO vector flow | CAR Otsu TSP tours | TACC-AAV comm gating & ACBBA bundles)
* `images/trajectories/trajectory_map_ppo.png` (high-res PPO flight paths & MD density flow)
* `images/trajectories/trajectory_map_car.png` (high-res CAR region partitions, cluster centroids, and pointer-net TSP tours)
* `images/trajectories/trajectory_map_tacc.png` (high-res TACC-AAV continuous heading flight paths with communication gating segments)

### IEEE Transactions Research Paper
A complete, self-contained scholarly research article analyzing all 15 trajectory $\times$ scheduler combinations:
* **LaTeX Source:** `papers/inference_paper.tex`
* **BibTeX References:** `papers/references.bib`
* **Document Class:** `papers/IEEEtran.cls` (IEEEtran v1.8b)

To compile to PDF (on any system with LaTeX / MacTeX / Overleaf):
```bash
cd papers
pdflatex inference_paper.tex
bibtex inference_paper
pdflatex inference_paper.tex
pdflatex inference_paper.tex
```

### Running Single Simulation Directly
```bash
# Schedulers: DYNAMIC, MOGS, LEXICOGRAPHIC, BIDDING, MARL
# Trajectories: PPO, CAR, TACC
bash run_single_sim.sh DYNAMIC PPO
bash run_single_sim.sh DYNAMIC CAR
bash run_single_sim.sh DYNAMIC TACC
```

### Running Unit Tests & Visual Demonstrations
```bash
# Pytest unit tests
.venv/bin/pytest tests/ -v

# TACC-AAV visual simulation demo (produces images/trajectories/tacc_aav_trajectories.png)
.venv/bin/python demo_tacc_aav.py
```

---

## 6. Port Allocations

* **Port 5500:** `tf_ppo_server.py` (`PPO` Trajectory Model)
* **Port 5501:** `marl_allocation_server.py` (`MARL` Task Scheduler)
* **Port 5510:** `car_ppo_server.py` (`CAR` Trajectory Model)
* **Port 5520:** Reserved for D-SOSTRA / Wu et al. trajectory server
* **Port 5530:** `tacc_aav_server.py` (`TACC` / `TACC_AAV` Trajectory Model)

---

## 7. Change Log & Maintenance Rules
* **2026-09:** Refined research paper (`papers/inference_paper.tex` and `papers/inference_paper.pdf`): removed journal headers (`\markboth`) and all citations/references section; verified clean 7-page compilation containing complete parameters formulation and all embedded benchmark/trajectory graphs.
* **2026-09:** Updated IEEE Transactions paper (`papers/inference_paper.tex` and `papers/inference_paper.pdf`): set affiliation to Amrita Vishwa Vidyapeetham, Coimbatore Campus; streamlined structure by removing generic introductory and conclusion boilerplate; implemented parameter-first technical formulation followed by data-driven empirical inferences; embedded all primary benchmark graphs (throughput, runtime overhead, Jain's fairness, deadline slack margin, allocation churn, queue length backlogs, capacity utilization, 2D spatial trajectory maps, and macro cross-product summary); compiled cleanly to PDF via `tectonic`.
* **2026-09:** Cleaned up repository root and organized all visual image assets into structured subdirectories (`images/benchmarks/`, `images/trajectories/`, `images/legacy/`). Updated generation scripts to output directly to these targets.
* **2026-09:** Authored complete research paper in IEEE Transactions format (`papers/inference_paper.tex`, `papers/references.bib`, `papers/IEEEtran.cls`) presenting empirical inferences and architectural guidelines for all 15 trajectory $\times$ scheduler combinations.
* **2026-09:** Organized simulation results into trajectory-grouped canonical directories (`results/by_trajectory/<TRAJ>/<SCHED>/run-<n>.log`).
* **2026-09:** Upgraded `graph_generator.py` to perform full Cartesian cross-multiplication ($3\text{ Trajectories} \times 5\text{ Schedulers} = 15\text{ combinations}$) with 3-panel comparative figures and macro summary bar charts (`images/benchmarks/trajectory_cross_summary.png`).
* **2026-09:** Implemented 2D spatial trajectory map visualizer (`trajectory_map_generator.py`) producing `trajectory_maps_all.png`, `trajectory_map_ppo.png`, `trajectory_map_car.png`, and `trajectory_map_tacc.png`.
* **2026-09:** Upgraded `run_single_sim.sh` to accept both scheduler and trajectory arguments (`./run_single_sim.sh <SCHED> <TRAJ>`), dynamically managing server ports.
* **2026-09:** Added `[TRAJECTORY_EPOCH]` real-time coordinate telemetry to `Controller.java`.
* **2026-09:** Implemented and benchmarked TACC-AAV algorithm (`tacc_aav.py`, `tacc_aav_server.py`, `TACCAAVTrajectoryModel.java`, `test_tacc_aav.py`, `demo_tacc_aav.py`).
* **2026-09:** Resolved MARL throughput collapse via geometric candidate masking ($d \le R_{\text{cov}}$) and battery headroom calculation in `MarlClient.java` / `marl_allocation_server.py`.
* **2026-09:** Extended simulation to 100 epochs; regenerated all time-series plots.
