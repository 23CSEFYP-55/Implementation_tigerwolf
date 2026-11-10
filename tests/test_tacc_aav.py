import math
import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ppo_agent.tacc_aav import (
    AAVPhysicalParams,
    AAVState,
    ACBBAConsensus,
    AerodynamicEnergyModel,
    MHSATrajectoryActor,
    MOCPOActor,
    MOCPOCritic,
    MOCPOPenaltyCritic,
    TACCAAVCoordinator,
    VehicleTask,
)

def test_aerodynamic_and_dynamic_capacity():
    params = AAVPhysicalParams(mass=2.5, battery_capacity=100000.0, reserve_energy=10000.0)
    aero = AerodynamicEnergyModel(params)
    
    # Check physical values are strictly positive and plausible
    assert aero.v_opt > 5.0 and aero.v_opt < 25.0
    assert aero.p_h_opt > 30.0 and aero.p_h_opt < 500.0
    assert aero.p_v > 50.0 and aero.p_v < 500.0
    
    # Check that v_opt achieves minimal power relative to high parasitic drag at 30 m/s
    p_opt = aero.flight_power(aero.v_opt)
    p_fast = aero.flight_power(32.0)
    assert p_fast > p_opt
    
    # Dynamic capacity derived from energy headroom
    cap_full = aero.dynamic_capacity(100000.0)
    cap_half = aero.dynamic_capacity(50000.0)
    cap_reserve = aero.dynamic_capacity(10000.0)
    
    assert cap_full > cap_half
    assert cap_half > cap_reserve
    assert cap_reserve >= 1

def test_mocpo_networks():
    obs_dim = 8
    actor = MOCPOActor(obs_dim=obs_dim)
    critic = MOCPOCritic(obs_dim=obs_dim)
    penalty_comm = MOCPOPenaltyCritic(obs_dim=obs_dim)
    penalty_energy = MOCPOPenaltyCritic(obs_dim=obs_dim)
    
    obs = torch.randn(4, obs_dim)
    
    prob = actor(obs)
    v = critic(obs)
    p_c = penalty_comm(obs)
    p_e = penalty_energy(obs)
    
    assert prob.shape == (4, 1)
    assert (prob >= 0.0).all() and (prob <= 1.0).all()
    assert v.shape == (4, 1)
    assert p_c.shape == (4, 1) and (p_c >= 0.0).all()
    assert p_e.shape == (4, 1) and (p_e >= 0.0).all()

def test_mhsa_trajectory_actor():
    actor = MHSATrajectoryActor(aav_feat_dim=6, task_feat_dim=6, embed_dim=64)
    
    aav_feat = torch.randn(2, 1, 6)
    task_feats = torch.randn(2, 5, 6)
    
    out = actor(aav_feat, task_feats)
    
    assert "velocity" in out and "heading" in out
    assert out["velocity"].shape == (2,)
    assert out["heading"].shape == (2,)
    assert (out["velocity"] >= 0.0).all() and (out["velocity"] <= 20.0).all()
    assert (out["heading"] >= 0.0).all() and (out["heading"] <= 2.0 * math.pi).all()
    
    # Offloading heads
    assert "node_select" in out
    assert out["node_select"].shape == (2, 5, 1)
    assert "dirichlet_alpha" in out
    assert out["dirichlet_alpha"].shape == (2, 5, 3)
    assert (out["dirichlet_alpha"] >= 1.0).all()

def test_acbba_consensus_and_deconfliction():
    num_agents = 3
    num_tasks = 4
    consensus = ACBBAConsensus(num_agents, num_tasks)
    aero = AerodynamicEnergyModel()
    
    agents = [
        AAVState(0, x=100.0, y=100.0, energy=80000.0),
        AAVState(1, x=200.0, y=200.0, energy=80000.0),
        AAVState(2, x=300.0, y=300.0, energy=80000.0),
    ]
    tasks = [
        VehicleTask(0, x=110.0, y=105.0, comp_size_mi=1000.0, data_size_mb=10.0, deadline_ms=500.0, priority_omega=1.0),
        VehicleTask(1, x=210.0, y=195.0, comp_size_mi=1500.0, data_size_mb=15.0, deadline_ms=600.0, priority_omega=0.8),
        VehicleTask(2, x=290.0, y=310.0, comp_size_mi=2000.0, data_size_mb=20.0, deadline_ms=700.0, priority_omega=0.9),
        VehicleTask(3, x=500.0, y=500.0, comp_size_mi=800.0, data_size_mb=8.0, deadline_ms=400.0, priority_omega=0.5),
    ]
    
    # Phase 1: Local bidding
    for i, a in enumerate(agents):
        consensus.bundle_construction(i, a, tasks, aero, current_time=1.0)
        assert len(consensus.bundles[i]) > 0
    
    # Agent 0 should bid highest on task 0 (closest)
    assert consensus.z[0][0] == 0
    
    # Phase 2: Deconfliction between Agent 0 and Agent 1
    consensus.deconflict(1, 0, consensus.z[0], consensus.y[0], consensus.s[0])
    # Agent 1 should recognize Agent 0's win on task 0
    assert consensus.z[1][0] == 0

def test_tacc_aav_step_and_training():
    coordinator = TACCAAVCoordinator(num_aavs=4, max_tasks=10)
    
    agents = [
        AAVState(0, x=100.0, y=100.0, energy=80000.0),
        AAVState(1, x=250.0, y=250.0, energy=80000.0),
        AAVState(2, x=400.0, y=400.0, energy=80000.0),
        AAVState(3, x=550.0, y=550.0, energy=80000.0),
    ]
    tasks = [
        VehicleTask(i, x=100.0 + i * 50.0, y=100.0 + i * 40.0, comp_size_mi=1000.0,
                    data_size_mb=10.0, deadline_ms=500.0, priority_omega=1.0)
        for i in range(8)
    ]
    
    # Run multiple steps to generate transitions
    for step in range(10):
        result = coordinator.step_simulation(agents, tasks, current_time=float(step))
        assert "trajectories" in result
        assert "gating_decisions" in result
        assert "assignments" in result
        assert len(result["trajectories"]) == 4
    
    # Test CTDE training step
    initial_lambda_comm = coordinator.lambda_comm
    coordinator.train_step(batch_size=16)
    # Check parameters and replay buffer
    assert len(coordinator.replay_buffer) > 0
