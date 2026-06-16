"""
Phase 2: Reinforcement Learning Environment
============================================
A custom Gymnasium environment for the Parking Enforcement Patrol problem.

This implements a simplified version of the Traveling Officer Problem (TOP)
described in the research doc. Key simplifications:

  1. SMDP → standard MDP: Instead of continuous-time macro-actions, we use
     discrete time steps where each step = 1 hour. The agent chooses which
     adjacent zone to move to, and "arrives" in that zone at the next step.
     
  2. Stochastic violations: At each timestep, violations spawn/despawn in
     each zone based on historical probabilities from the CongestionIntelligence.
     
  3. Single agent: We start with one officer. Multi-agent can be added later
     with proximity penalties (Phase 4 of the implementation plan).

State Space:
  - Agent's current zone index
  - Current hour (0-23)
  - Current day of week (0-6)
  - Remaining shift time (0-shift_length)
  - For each zone: [violation_active, expected_CIS, distance_from_agent]
  
Action Space:
  - Discrete: 0 = stay, 1..N = move to the i-th neighbor or high-priority zone
  
Reward Function (from research doc, Section "Advanced Reward Shaping"):
  +R_base × CIS    : successful citation (violation was active when agent arrived)
  -cruising_penalty : cost of movement (fuel, time)
  +explore_bonus    : visiting an unpatrolled zone
  -revisit_penalty  : returning to recently patrolled zone
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional
import random


class ParkingEnforcementEnv(gym.Env):
    """
    Gymnasium environment for optimal parking enforcement patrol routing.
    
    The agent is a traffic officer who must patrol a grid of zones (Geohash cells),
    intercepting illegal parking violations that dynamically appear and disappear.
    The goal is to maximize citations weighted by Congestion Impact Score (CIS).
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(
        self,
        congestion_intel,
        adjacency: dict,
        zone_stats: pd.DataFrame = None,
        shift_length: int = 8,      # 8-hour shift
        max_actions: int = 9,       # max possible actions (stay + 8 neighbors)
        reward_base: float = 10.0,  # base reward for successful citation
        cruising_penalty: float = -0.5,  # penalty per movement step
        revisit_penalty: float = -2.0,   # penalty for revisiting within cooldown
        explore_bonus: float = 1.0,      # bonus for visiting unpatrolled zone
        cooldown_steps: int = 3,         # how many steps before revisit penalty expires
        start_hour: int = 8,             # shift start hour (8 AM IST)
        start_dow: int = 0,              # Monday
        render_mode: Optional[str] = None,
    ):
        """
        Args:
            congestion_intel: CongestionIntelligence instance
            adjacency: dict mapping geohash → list of neighbor geohashes
            zone_stats: DataFrame with zone statistics (for metadata)
            shift_length: number of hours in a patrol shift
            max_actions: maximum number of actions (padded for consistent space)
            reward_base: base reward multiplier for successful citations
            cruising_penalty: per-step movement cost
            revisit_penalty: penalty for going back to recently-patrolled zone
            explore_bonus: reward for visiting a new zone
            cooldown_steps: steps before a zone can be revisited without penalty
            start_hour: hour of day when shift starts
            start_dow: day of week when shift starts (0=Monday)
        """
        super().__init__()
        
        self.intel = congestion_intel
        self.adjacency = adjacency
        self.zone_list = sorted(adjacency.keys())
        self.n_zones = len(self.zone_list)
        self.zone_to_idx = {z: i for i, z in enumerate(self.zone_list)}
        
        # Environment parameters
        self.shift_length = shift_length
        self.max_actions = max_actions
        self.reward_base = reward_base
        self.cruising_penalty = cruising_penalty
        self.revisit_penalty = revisit_penalty
        self.explore_bonus = explore_bonus
        self.cooldown_steps = cooldown_steps
        self.start_hour = start_hour
        self.start_dow = start_dow
        self.render_mode = render_mode
        
        # Zone coordinates for rendering
        self.zone_coords = {}
        if zone_stats is not None:
            for _, row in zone_stats.groupby('geohash').agg(
                lat=('lat', 'mean'), lon=('lon', 'mean')
            ).iterrows():
                self.zone_coords[_] = (row['lat'], row['lon'])
        
        # =====================================================================
        # Define observation and action spaces
        # =====================================================================
        # Observation: [agent_zone_idx, hour, dow, remaining_time,
        #               zone_0_violation, zone_0_cis, zone_0_dist,
        #               zone_1_violation, zone_1_cis, zone_1_dist, ...]
        # 
        # Total obs size = 4 + (n_zones * 3)
        obs_size = 4 + (self.n_zones * 3)
        self.observation_space = spaces.Box(
            low=-1.0, high=100.0, shape=(obs_size,), dtype=np.float32
        )
        
        # Action: index into available moves (0=stay, 1..max_actions-1=neighbors/targets)
        self.action_space = spaces.Discrete(max_actions)
        
        # =====================================================================
        # Internal state (initialized in reset())
        # =====================================================================
        self.agent_zone_idx = 0
        self.current_hour = start_hour
        self.current_dow = start_dow
        self.remaining_time = shift_length
        self.violations_active = np.zeros(self.n_zones, dtype=np.float32)
        self.violation_cis = np.zeros(self.n_zones, dtype=np.float32)
        self.visit_history = np.zeros(self.n_zones, dtype=np.int32)  # steps since last visit
        self.total_citations = 0
        self.total_cis_collected = 0.0
        self.total_distance = 0
        self.zones_visited = set()
        self.episode_rewards = []
        
        # Track available actions for current position
        self._current_actions = []
    
    def _get_available_actions(self) -> list:
        """
        Get the list of zones the agent can move to from current position.
        
        Returns list of zone indices. First entry is always "stay" (current zone).
        Remaining entries are neighbors + top priority zones within reach.
        """
        current_zone = self.zone_list[self.agent_zone_idx]
        neighbors = self.adjacency.get(current_zone, [])
        
        # Actions: [stay, neighbor_0, neighbor_1, ..., neighbor_7]
        actions = [self.agent_zone_idx]  # 0 = stay
        for nbr in neighbors:
            if nbr in self.zone_to_idx:
                actions.append(self.zone_to_idx[nbr])
        
        # Pad with -1 if fewer than max_actions
        while len(actions) < self.max_actions:
            actions.append(-1)  # invalid action
        
        return actions[:self.max_actions]
    
    def _spawn_violations(self):
        """
        Stochastically generate violations across the grid.
        
        WHY stochastic:
        - In reality, violations appear and disappear randomly
        - The RL agent must learn to chase violations that are likely
          to still be there when it arrives (risk-reward tradeoff)
        - We use the CongestionIntelligence probabilities calibrated
          from historical data
        """
        for i, zone in enumerate(self.zone_list):
            p = self.intel.get_violation_probability(
                zone, self.current_hour, self.current_dow
            )
            if random.random() < p:
                self.violations_active[i] = 1.0
                self.violation_cis[i] = self.intel.get_expected_cis(
                    zone, self.current_hour, self.current_dow
                )
            else:
                # Some existing violations may despawn (vehicles leave)
                # Despawn rate: 30% per timestep for active violations
                if self.violations_active[i] > 0 and random.random() < 0.3:
                    self.violations_active[i] = 0.0
                    self.violation_cis[i] = 0.0
    
    def _compute_distances(self) -> np.ndarray:
        """
        Compute graph distance from agent's current zone to all other zones.
        
        WHY graph distance over Euclidean:
        - Two zones might be geographically close but separated by a one-way
          system, river, or railway (as noted in the research doc)
        - BFS on the adjacency graph gives realistic traversal cost
        - We cap at 10 to keep values bounded for the neural network
        """
        distances = np.full(self.n_zones, 10.0, dtype=np.float32)  # max distance
        
        # BFS from current position
        visited = {self.agent_zone_idx}
        queue = [(self.agent_zone_idx, 0)]
        
        while queue:
            zone_idx, dist = queue.pop(0)
            distances[zone_idx] = min(distances[zone_idx], float(dist))
            
            if dist >= 10:  # cap search depth
                continue
            
            zone_hash = self.zone_list[zone_idx]
            for nbr in self.adjacency.get(zone_hash, []):
                if nbr in self.zone_to_idx:
                    nbr_idx = self.zone_to_idx[nbr]
                    if nbr_idx not in visited:
                        visited.add(nbr_idx)
                        queue.append((nbr_idx, dist + 1))
        
        return distances
    
    def _get_observation(self) -> np.ndarray:
        """
        Construct the observation vector.
        
        Structure: [agent_zone, hour_normalized, dow_normalized, remaining_time_normalized,
                     zone_0_violation, zone_0_cis_normalized, zone_0_distance_normalized,
                     ...]
        
        WHY normalize:
        - Neural networks work best with inputs in [0, 1] or [-1, 1] range
        - Raw values (hour=0-23, CIS=0-20+, distance=0-10) have different scales
        """
        distances = self._compute_distances()
        
        obs = np.zeros(4 + self.n_zones * 3, dtype=np.float32)
        
        # Agent state (normalized to [0, 1])
        obs[0] = self.agent_zone_idx / max(1, self.n_zones - 1)
        obs[1] = self.current_hour / 23.0
        obs[2] = self.current_dow / 6.0
        obs[3] = self.remaining_time / self.shift_length
        
        # Per-zone features
        for i in range(self.n_zones):
            base = 4 + i * 3
            obs[base] = self.violations_active[i]
            obs[base + 1] = min(self.violation_cis[i] / 10.0, 1.0)  # normalize CIS
            obs[base + 2] = distances[i] / 10.0  # normalize distance
        
        return obs
    
    def reset(self, *, seed=None, options=None):
        """Reset the environment for a new episode (new patrol shift)."""
        super().reset(seed=seed)
        
        # Randomize start conditions for training diversity
        if options and 'start_zone' in options:
            self.agent_zone_idx = options['start_zone']
        else:
            self.agent_zone_idx = random.randint(0, self.n_zones - 1)
        
        if options and 'start_hour' in options:
            self.current_hour = options['start_hour']
        else:
            # Random start hour weighted toward common patrol times
            self.current_hour = random.choice([6, 7, 8, 9, 10, 14, 15, 16, 17])
        
        if options and 'start_dow' in options:
            self.current_dow = options['start_dow']
        else:
            self.current_dow = random.randint(0, 6)
        
        self.remaining_time = self.shift_length
        self.violations_active = np.zeros(self.n_zones, dtype=np.float32)
        self.violation_cis = np.zeros(self.n_zones, dtype=np.float32)
        self.visit_history = np.full(self.n_zones, self.cooldown_steps + 1, dtype=np.int32)
        self.total_citations = 0
        self.total_cis_collected = 0.0
        self.total_distance = 0
        self.zones_visited = set()
        self.episode_rewards = []
        
        # Initial violation spawn
        self._spawn_violations()
        
        # Update available actions
        self._current_actions = self._get_available_actions()
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: int):
        """
        Execute one patrol step.
        
        The agent selects an action (stay or move to a neighbor zone).
        Time advances by 1 hour. Violations spawn/despawn stochastically.
        
        Reward structure (from research doc Section "Advanced Reward Shaping"):
        1. Citation reward: +base × CIS if violation found at destination
        2. Movement cost: -penalty for each move (encourages efficiency)
        3. Exploration bonus: +bonus for visiting new zones
        4. Revisit penalty: -penalty for returning to recently-patrolled zones
        """
        reward = 0.0
        
        # Resolve action to target zone
        available = self._current_actions
        if action >= len(available) or available[action] == -1:
            # Invalid action → stay in place with small penalty
            target_idx = self.agent_zone_idx
            reward += -0.1  # small penalty for invalid action
        else:
            target_idx = available[action]
        
        moved = (target_idx != self.agent_zone_idx)
        
        # === MOVEMENT ===
        if moved:
            self.agent_zone_idx = target_idx
            self.total_distance += 1
            reward += self.cruising_penalty  # movement cost
        
        # === CITATION CHECK ===
        # If there's an active violation at the agent's location, cite it!
        if self.violations_active[self.agent_zone_idx] > 0:
            cis = self.violation_cis[self.agent_zone_idx]
            reward += self.reward_base * cis  # CIS-weighted citation reward
            self.total_citations += 1
            self.total_cis_collected += cis
            
            # Clear the violation (vehicle is ticketed/towed)
            self.violations_active[self.agent_zone_idx] = 0.0
            self.violation_cis[self.agent_zone_idx] = 0.0
        
        # === EXPLORATION ===
        current_zone = self.zone_list[self.agent_zone_idx]
        if current_zone not in self.zones_visited:
            reward += self.explore_bonus
            self.zones_visited.add(current_zone)
        
        # === REVISIT PENALTY ===
        if self.visit_history[self.agent_zone_idx] < self.cooldown_steps:
            reward += self.revisit_penalty
        
        # Update visit history
        self.visit_history += 1  # increment all
        self.visit_history[self.agent_zone_idx] = 0  # reset current
        
        # === TIME ADVANCEMENT ===
        self.remaining_time -= 1
        self.current_hour = (self.current_hour + 1) % 24
        
        # Spawn/despawn violations for the new timestep
        self._spawn_violations()
        
        # Update available actions for new position
        self._current_actions = self._get_available_actions()
        
        # === TERMINATION ===
        terminated = self.remaining_time <= 0
        truncated = False
        
        # End-of-shift bonus based on overall performance
        if terminated:
            # Bonus for high citation yield
            if self.total_citations > 0:
                reward += self.total_citations * 2.0
            # Bonus for coverage (visiting many unique zones)
            coverage = len(self.zones_visited) / max(1, self.n_zones)
            reward += coverage * 5.0
        
        self.episode_rewards.append(reward)
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _get_info(self) -> dict:
        """Return episode metrics for logging and evaluation."""
        return {
            'total_citations': self.total_citations,
            'total_cis_collected': self.total_cis_collected,
            'total_distance': self.total_distance,
            'zones_visited': len(self.zones_visited),
            'coverage': len(self.zones_visited) / max(1, self.n_zones),
            'remaining_time': self.remaining_time,
            'current_hour': self.current_hour,
            'current_zone': self.zone_list[self.agent_zone_idx],
            'episode_reward': sum(self.episode_rewards),
        }
    
    def render(self):
        """Render the environment (for debugging)."""
        if self.render_mode == "human":
            zone = self.zone_list[self.agent_zone_idx]
            active = int(self.violations_active.sum())
            print(f"  Hour: {self.current_hour:02d}:00 | "
                  f"Zone: {zone} | "
                  f"Active violations: {active} | "
                  f"Citations: {self.total_citations} | "
                  f"CIS collected: {self.total_cis_collected:.1f} | "
                  f"Remaining: {self.remaining_time}h")



# =============================================================================
# Baseline Agents for Comparison
# =============================================================================

class RandomAgent:
    """
    Baseline 1: Random walker.
    Chooses random valid actions. Represents completely uninformed patrol.
    """
    def __init__(self, action_space_n):
        self.action_space_n = action_space_n
    
    def predict(self, obs, deterministic=True):
        return random.randint(0, self.action_space_n - 1), None


class GreedyAgent:
    """
    Baseline 2: Greedy nearest-violation chaser.
    Always moves toward the nearest zone with an active violation.
    If no violations visible, stays in place.
    
    WHY this baseline:
    - Represents the "naive AI" approach — always chase the closest target
    - Fails because it doesn't consider future positions or violation lifespans
    - The research doc calls this out in Section "Spatial-Aware DRL"
    """
    def __init__(self, n_zones):
        self.n_zones = n_zones
    
    def predict(self, obs, deterministic=True):
        # Extract per-zone violation flags from observation
        # obs layout: [agent_zone, hour, dow, remaining, z0_viol, z0_cis, z0_dist, ...]
        best_action = 0  # default: stay
        min_dist = float('inf')
        
        for i in range(self.n_zones):
            base = 4 + i * 3
            if base + 2 >= len(obs):
                break
            violation = obs[base]
            dist = obs[base + 2]
            
            if violation > 0.5 and dist < min_dist:
                min_dist = dist
                # Action 0 = stay, others = neighbors
                # If the nearest violation is at our location (dist=0), stay
                if dist < 0.01:
                    best_action = 0
                else:
                    # Move toward it (pick action 1 which is first neighbor)
                    # This is simplified — ideally we'd BFS to find the right direction
                    best_action = 1
        
        return best_action, None


class FixedRouteAgent:
    """
    Baseline 3: Fixed sweeping pattern.
    Cycles through zones in order, simulating a traditional patrol route.
    
    WHY this baseline:
    - Represents current human patrol behavior — fixed routes regardless
      of where violations actually occur
    - The research doc criticizes this in Section "Operational Challenge"
    """
    def __init__(self, action_space_n):
        self.action_space_n = action_space_n
        self.step_count = 0
    
    def predict(self, obs, deterministic=True):
        # Always move to the next neighbor (action 1, 2, 3... cycling)
        self.step_count += 1
        action = (self.step_count % (self.action_space_n - 1)) + 1
        return action, None


def evaluate_agent(env, agent, n_episodes: int = 50) -> dict:
    """
    Evaluate an agent over multiple episodes and return averaged KPIs.
    
    KPIs (from research doc Section "Evaluation and Validation Benchmarks"):
    - Citation Yield: total violations caught per shift
    - CIS Yield: total congestion impact score collected per shift
    - Distance Efficiency: citations per unit distance traveled
    - Zone Coverage: fraction of total zones visited
    """
    metrics = {
        'citations': [],
        'cis_collected': [],
        'distance': [],
        'coverage': [],
        'reward': [],
    }
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        metrics['citations'].append(info['total_citations'])
        metrics['cis_collected'].append(info['total_cis_collected'])
        metrics['distance'].append(info['total_distance'])
        metrics['coverage'].append(info['coverage'])
        metrics['reward'].append(info['episode_reward'])
    
    # Aggregate
    results = {}
    for key, values in metrics.items():
        results[f'{key}_mean'] = np.mean(values)
        results[f'{key}_std'] = np.std(values)
    
    # Derived metrics
    results['efficiency'] = (
        results['citations_mean'] / max(1, results['distance_mean'])
    )
    
    return results
