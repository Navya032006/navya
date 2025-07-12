import gym
from gym import spaces
import numpy as np
import time
import pandas as pd
from datetime import datetime
import json

class MultiUAVEnv(gym.Env):
    def __init__(self, grid_size=10, num_uavs=3, max_steps=200):
        super(MultiUAVEnv, self).__init__()
        self.grid_size = grid_size
        self.num_uavs = num_uavs
        self.max_steps = max_steps
        self.current_step = 0
        
        # Action space: 0=up, 1=right, 2=down, 3=left, 4=stay
        self.action_space = spaces.MultiDiscrete([5] * num_uavs)
        
        # Observation space: [uav_positions, visited_cells, battery_levels, target_locations]
        self.observation_space = spaces.Box(
            low=0, high=grid_size-1, 
            shape=(num_uavs * 4 + grid_size * grid_size,), 
            dtype=np.float32
        )
        
        # Initialize metrics
        self.metrics = {
            'total_coverage': 0,
            'energy_consumed': 0,
            'collision_count': 0,
            'mission_time': 0,
            'efficiency_score': 0,
            'aoi_coverage': 0,
            'path_length': 0
        }
        
        # Data collection points (from CSV)
        self.data_points = []
        self.collected_data = []
        
        self.reset()
    
    def reset(self):
        self.current_step = 0
        self.uav_positions = np.random.randint(0, self.grid_size, size=(self.num_uavs, 2))
        self.visited = np.zeros((self.grid_size, self.grid_size))
        self.battery_levels = np.full(self.num_uavs, 100.0)
        self.uav_paths = [[] for _ in range(self.num_uavs)]
        
        # Generate random target locations
        self.target_locations = np.random.randint(0, self.grid_size, size=(5, 2))
        
        # Reset metrics
        self.metrics = {key: 0 for key in self.metrics.keys()}
        self.collected_data = []
        
        return self._get_observation()
    
    def _get_observation(self):
        # Flatten observation
        obs = np.concatenate([
            self.uav_positions.flatten(),
            self.battery_levels,
            self.target_locations.flatten(),
            self.visited.flatten()
        ])
        return obs.astype(np.float32)
    
    def step(self, actions):
        self.current_step += 1
        rewards = 0
        collisions = 0
        
        # Move UAVs
        for i, action in enumerate(actions):
            if self.battery_levels[i] > 0:
                old_pos = self.uav_positions[i].copy()
                new_pos = self._move_uav(i, action)
                
                # Update path
                self.uav_paths[i].append(new_pos.copy())
                
                # Calculate energy consumption
                energy_cost = self._calculate_energy_cost(old_pos, new_pos)
                self.battery_levels[i] -= energy_cost
                self.metrics['energy_consumed'] += energy_cost
                
                # Check for data collection
                if self.visited[new_pos[0], new_pos[1]] == 0:
                    rewards += 10  # Data collection reward
                    self.visited[new_pos[0], new_pos[1]] = 1
                    self.metrics['total_coverage'] += 1
                    
                    # Simulate data collection
                    self._collect_data(i, new_pos)
                
                # Check target location coverage
                for target in self.target_locations:
                    if np.array_equal(new_pos, target):
                        rewards += 20  # Target location bonus
                        self.metrics['aoi_coverage'] += 1
        
        # Check for collisions
        for i in range(self.num_uavs):
            for j in range(i + 1, self.num_uavs):
                if np.array_equal(self.uav_positions[i], self.uav_positions[j]):
                    rewards -= 50  # Collision penalty
                    collisions += 1
        
        self.metrics['collision_count'] += collisions
        
        # Mission efficiency
        coverage_ratio = self.metrics['total_coverage'] / (self.grid_size ** 2)
        self.metrics['efficiency_score'] = coverage_ratio * 100
        
        # Path length calculation
        self.metrics['path_length'] = sum(len(path) for path in self.uav_paths)
        
        # Check if mission is complete
        done = (self.current_step >= self.max_steps or 
                coverage_ratio >= 0.95 or 
                all(battery <= 0 for battery in self.battery_levels))
        
        if done:
            self.metrics['mission_time'] = self.current_step
        
        return self._get_observation(), rewards, done, {'metrics': self.metrics}
    
    def _move_uav(self, uav_id, action):
        pos = self.uav_positions[uav_id]
        if action == 0 and pos[0] > 0:  # Up
            pos[0] -= 1
        elif action == 1 and pos[1] < self.grid_size - 1:  # Right
            pos[1] += 1
        elif action == 2 and pos[0] < self.grid_size - 1:  # Down
            pos[0] += 1
        elif action == 3 and pos[1] > 0:  # Left
            pos[1] -= 1
        # action == 4 is stay (no movement)
        
        return pos
    
    def _calculate_energy_cost(self, old_pos, new_pos):
        # Energy cost based on distance and hover time
        distance = np.linalg.norm(new_pos - old_pos)
        return max(0.5, distance * 2.0)  # Minimum hover cost
    
    def _collect_data(self, uav_id, position):
        # Simulate data collection with timestamp
        data_point = {
            'ID': f'UAV{uav_id}_{len(self.collected_data)}',
            'source_time': self.current_step,
            'lat': 52.815992 + np.random.normal(0, 0.001),
            'lon': -4.131736 + np.random.normal(0, 0.001),
            'source_spotted': 1 if np.random.random() > 0.3 else 0,
            'track': np.random.randint(-90, 91),
            'takeoff_landing_time': self.current_step,
            'battery': self.battery_levels[uav_id],
            'AoI': np.random.randint(0, 6),
            'FLA': np.random.randint(1, 6)
        }
        self.collected_data.append(data_point)
    
    def get_metrics(self):
        return self.metrics.copy()
    
    def get_collected_data(self):
        return self.collected_data.copy()
    
    def render(self, mode='human'):
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"UAV Positions: {self.uav_positions}")
            print(f"Battery Levels: {self.battery_levels}")
            print(f"Coverage: {self.metrics['efficiency_score']:.1f}%")
            print(f"Data Points Collected: {len(self.collected_data)}")
            print("-" * 50)
        
        return {
            'uav_positions': self.uav_positions.copy(),
            'visited': self.visited.copy(),
            'battery_levels': self.battery_levels.copy(),
            'metrics': self.metrics.copy(),
            'step': self.current_step
        }