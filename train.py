"""
Training Script: Train DQN and PPO agents for Parking Enforcement Patrol
========================================================================

This script:
1. Runs the Phase 0 data engineering pipeline
2. Builds the Phase 1 Congestion Intelligence engine
3. Creates the Phase 2 RL environment
4. Trains a DQN agent (primary)
5. Trains a PPO agent (secondary)
6. Evaluates all agents against baselines
7. Saves results for the dashboard

WHY DQN first:
- Simpler architecture, faster to train
- Easier to debug if the environment has issues
- Good enough for discrete action spaces
- The research doc (Phase 3) recommends DQN as "initial training verification"

WHY PPO second:
- Better sample efficiency in stochastic environments
- Handles the exploration-exploitation tradeoff more gracefully
- Actor-critic architecture naturally handles our composite reward
"""

import os
import sys
import io
import json
import pickle
import numpy as np
import pandas as pd

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_engineering import run_pipeline
from src.congestion_intelligence import CongestionIntelligence
from src.rl_environment import (
    ParkingEnforcementEnv, RandomAgent, GreedyAgent, 
    FixedRouteAgent, evaluate_agent
)


def train_and_evaluate(csv_path: str, output_dir: str = 'results'):
    """
    Full training pipeline: data → intelligence → environment → agents → evaluation
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # =========================================================================
    # STEP 1: Data Engineering (Phase 0)
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 1: DATA ENGINEERING")
    print("="*70)
    
    df, zone_stats, adjacency = run_pipeline(csv_path)
    
    # Save processed data
    df.to_parquet('data/processed/violations_processed.parquet', index=False)
    zone_stats.to_parquet('data/processed/zone_stats.parquet', index=False)
    with open('data/processed/adjacency.pkl', 'wb') as f:
        pickle.dump(adjacency, f)
    
    print(f"\n  [OK] Processed data saved to data/processed/")
    
    # =========================================================================
    # STEP 2: Build Congestion Intelligence (Phase 1)
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 2: CONGESTION INTELLIGENCE")
    print("="*70)
    
    intel = CongestionIntelligence(zone_stats, adjacency)
    
    # Show top hotspots for a sample time
    print("\n  Top 10 enforcement hotspots (Monday 10 AM):")
    hotspots = intel.get_top_hotspots(hour=10, dow=0, n=10)
    for i, h in enumerate(hotspots):
        print(f"    {i+1}. Zone {h['geohash']} | "
              f"Score: {h['composite_score']:.2f} | "
              f"P(viol): {h['p_violation']:.2f} | "
              f"CIS: {h['expected_cis']:.2f}")
    
    intel.save('data/processed/congestion_intel.pkl')
    print(f"\n  [OK] Intelligence engine saved")
    
    # =========================================================================
    # STEP 3: Create RL Environment (Phase 2)
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 3: RL ENVIRONMENT")
    print("="*70)
    
    env = ParkingEnforcementEnv(
        congestion_intel=intel,
        adjacency=adjacency,
        zone_stats=zone_stats,
        shift_length=8,
        reward_base=10.0,
        cruising_penalty=-0.5,
        revisit_penalty=-2.0,
        explore_bonus=1.0,
    )
    
    # Quick sanity check
    obs, info = env.reset()
    print(f"\n  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.n}")
    print(f"  Number of zones: {env.n_zones}")
    print(f"  Sample obs shape: {obs.shape}")
    
    # Test a few random steps
    total_reward = 0
    for _ in range(8):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print(f"  Random 8-step test reward: {total_reward:.2f}")
    print(f"  Citations in test: {info['total_citations']}")
    
    # =========================================================================
    # STEP 4: Train DQN Agent
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 4: TRAINING DQN AGENT")
    print("="*70)
    
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import EvalCallback
    
    # Create training environment
    train_env = ParkingEnforcementEnv(
        congestion_intel=intel,
        adjacency=adjacency,
        zone_stats=zone_stats,
        shift_length=8,
    )
    
    # Create eval environment
    eval_env = ParkingEnforcementEnv(
        congestion_intel=intel,
        adjacency=adjacency,
        zone_stats=zone_stats,
        shift_length=8,
    )
    
    # DQN hyperparameters
    # WHY these values:
    # - learning_rate=1e-3: aggressive enough for quick convergence in hackathon
    # - buffer_size=50000: enough replay memory for our small state space
    # - exploration_fraction=0.3: explore for 30% of training, then exploit
    # - gamma=0.95: moderately forward-looking (values future citations)
    # - batch_size=64: standard for DQN
    # - target_update_interval=500: stable target network updates
    dqn_model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=1e-3,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.95,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        target_update_interval=500,
        train_freq=4,
        verbose=1,
        tensorboard_log=os.path.join(output_dir, 'tb_logs'),
    )
    
    # Train — 50K steps is a reasonable prototype training run
    # For production, you'd want 500K-1M steps
    TOTAL_TIMESTEPS = 50_000
    print(f"\n  Training DQN for {TOTAL_TIMESTEPS:,} timesteps...")
    print(f"  (This should take 2-5 minutes)\n")
    
    dqn_model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        progress_bar=True,
    )
    
    dqn_model.save(os.path.join(output_dir, 'dqn_parking_agent'))
    print(f"\n  [OK] DQN model saved")
    
    # =========================================================================
    # STEP 5: Train PPO Agent
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 5: TRAINING PPO AGENT")
    print("="*70)
    
    from stable_baselines3 import PPO
    
    ppo_env = ParkingEnforcementEnv(
        congestion_intel=intel,
        adjacency=adjacency,
        zone_stats=zone_stats,
        shift_length=8,
    )
    
    # PPO hyperparameters
    # WHY PPO:
    # - Better for stochastic environments (clipped objective prevents large updates)
    # - Actor-critic naturally balances exploration vs exploitation
    # - n_steps=256: collect 256 steps before each update (enough for 32 episodes)
    ppo_model = PPO(
        "MlpPolicy",
        ppo_env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.95,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=os.path.join(output_dir, 'tb_logs'),
    )
    
    print(f"\n  Training PPO for {TOTAL_TIMESTEPS:,} timesteps...")
    print(f"  (This should take 2-5 minutes)\n")
    
    ppo_model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        progress_bar=True,
    )
    
    ppo_model.save(os.path.join(output_dir, 'ppo_parking_agent'))
    print(f"\n  [OK] PPO model saved")
    
    # =========================================================================
    # STEP 6: Evaluate All Agents
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 6: EVALUATION & COMPARISON")
    print("="*70)
    
    eval_env = ParkingEnforcementEnv(
        congestion_intel=intel,
        adjacency=adjacency,
        zone_stats=zone_stats,
        shift_length=8,
    )
    
    N_EVAL = 100  # episodes for evaluation
    
    agents = {
        'DQN (Ours)': dqn_model,
        'PPO (Ours)': ppo_model,
        'Random Patrol': RandomAgent(eval_env.action_space.n),
        'Greedy Nearest': GreedyAgent(eval_env.n_zones),
        'Fixed Route': FixedRouteAgent(eval_env.action_space.n),
    }
    
    all_results = {}
    for name, agent in agents.items():
        print(f"\n  Evaluating {name} ({N_EVAL} episodes)...")
        results = evaluate_agent(eval_env, agent, n_episodes=N_EVAL)
        all_results[name] = results
        
        print(f"    Citations/shift:  {results['citations_mean']:.1f} ± {results['citations_std']:.1f}")
        print(f"    CIS collected:    {results['cis_collected_mean']:.1f} ± {results['cis_collected_std']:.1f}")
        print(f"    Distance:         {results['distance_mean']:.1f} ± {results['distance_std']:.1f}")
        print(f"    Coverage:         {results['coverage_mean']:.1%}")
        print(f"    Efficiency:       {results['efficiency']:.3f} citations/km")
        print(f"    Total reward:     {results['reward_mean']:.1f}")
    
    # Save results
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print comparison table
    print("\n" + "="*70)
    print("RESULTS COMPARISON TABLE")
    print("="*70)
    print(f"{'Agent':<20} {'Citations':>10} {'CIS':>10} {'Distance':>10} {'Coverage':>10} {'Efficiency':>12}")
    print("-" * 72)
    for name, r in all_results.items():
        print(f"{name:<20} {r['citations_mean']:>10.1f} {r['cis_collected_mean']:>10.1f} "
              f"{r['distance_mean']:>10.1f} {r['coverage_mean']:>9.1%} {r['efficiency']:>12.3f}")
    
    print(f"\n  [OK] All results saved to {output_dir}/")
    
    return all_results


if __name__ == "__main__":
    csv_path = 'data/raw/jan to may police violation_anonymized791b166.csv'
    
    if not os.path.exists(csv_path):
        print(f"ERROR: Dataset not found at {csv_path}")
        sys.exit(1)
    
    results = train_and_evaluate(csv_path)
