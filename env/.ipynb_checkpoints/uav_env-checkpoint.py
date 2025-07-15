import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import time

class RealTimeMultiUAVEnv(gym.Env):
    def __init__(self, grid_size=10, num_uavs=3, real_coords=True):
        super(RealTimeMultiUAVEnv, self).__init__()
        self.grid_size = grid_size
        self.num_uavs = num_uavs
        self.real_coords = real_coords
        
        # Base coordinates (Wales, UK)
        self.base_lat = 52.815992
        self.base_lon = -4.131736
        self.coord_scale = 0.001  # Scale for grid to GPS conversion
        
        # Action space: 0=Up, 1=Right, 2=Down, 3=Left, 4=Stay
        self.action_space = spaces.MultiDiscrete([5] * num_uavs)
        
        # Observation space: UAV positions + battery + visited cells
        obs_dim = num_uavs * 4 + grid_size * grid_size  # pos(2) + battery(1) + spotted(1) + visited_grid
        self.observation_space = spaces.Box(low=0, high=100, shape=(obs_dim,), dtype=np.float32)
        
        # Initialize tracking variables
        self.reset()
        self.step_count = 0
        self.start_time = time.time()
        
    def seed(self, seed=None):
        np.random.seed(seed)
        import random
        random.seed(seed)

        
    def reset(self):
        # Initialize UAV positions randomly
        self.uav_positions = []
        self.uav_batteries = []
        self.uav_spotted = []
        self.uav_tracks = []
        
        for i in range(self.num_uavs):
            # Random start positions
            pos = np.array([
                np.random.randint(0, self.grid_size),
                np.random.randint(0, self.grid_size)
            ])
            self.uav_positions.append(pos)
            self.uav_batteries.append(100.0)  # Full battery
            self.uav_spotted.append(0)  # Not spotted initially
            self.uav_tracks.append(np.random.randint(-180, 180))  # Random track
        
        # Grid state
        self.visited = np.zeros((self.grid_size, self.grid_size))
        self.coverage_data = np.zeros((self.grid_size, self.grid_size))
        
        # Time tracking
        self.step_count = 0
        self.start_time = time.time()
        
        return self._get_observation()
    
    def _get_observation(self):
        obs = []
        
        # UAV states
        for i in range(self.num_uavs):
            obs.extend([
                self.uav_positions[i][0] / self.grid_size,  # Normalized position
                self.uav_positions[i][1] / self.grid_size,
                self.uav_batteries[i] / 100.0,  # Normalized battery
                self.uav_spotted[i]  # Spotted flag
            ])
        
        # Grid state (flattened)
        obs.extend(self.visited.flatten())
        
        return np.array(obs, dtype=np.float32)
    
    def step(self, actions):
        self.step_count += 1
        total_reward = 0
        
        # Move UAVs
        for i, action in enumerate(actions):
            if self.uav_batteries[i] <= 0:
                continue  # UAV out of battery
                
            pos = self.uav_positions[i].copy()
            
            # Execute action
            if action == 0 and pos[0] > 0:  # Up
                pos[0] -= 1
            elif action == 1 and pos[1] < self.grid_size - 1:  # Right
                pos[1] += 1
            elif action == 2 and pos[0] < self.grid_size - 1:  # Down
                pos[0] += 1
            elif action == 3 and pos[1] > 0:  # Left
                pos[1] -= 1
            # action == 4 is stay (no movement)
            
            self.uav_positions[i] = pos
            
            # Battery consumption
            battery_cost = 2.0 if action != 4 else 0.5  # Moving costs more
            self.uav_batteries[i] = max(0, self.uav_batteries[i] - battery_cost)
            
            # Reward for visiting new cells
            if self.visited[pos[0], pos[1]] == 0:
                self.visited[pos[0], pos[1]] = 1
                self.coverage_data[pos[0], pos[1]] = self.step_count
                total_reward += 10  # Discovery reward
                
                # Random spotting
                if np.random.random() < 0.1:  # 10% chance to spot something
                    self.uav_spotted[i] = 1
                    total_reward += 5  # Spotting bonus
            
            # Energy penalty
            total_reward -= 0.1
            
            # Update track (simple heading based on last action)
            if action != 4:
                self.uav_tracks[i] = action * 90  # 0°, 90°, 180°, 270°
        
        # Check if done
        coverage_ratio = np.sum(self.visited) / (self.grid_size ** 2)
        done = coverage_ratio >= 0.95 or self.step_count >= 500
        
        # Additional rewards
        if done and coverage_ratio >= 0.95:
            total_reward += 100  # Mission completion bonus
        
        info = {
            'coverage_ratio': coverage_ratio,
            'step_count': self.step_count,
            'active_uavs': sum(1 for b in self.uav_batteries if b > 0),
            'total_spots': sum(self.uav_spotted)
        }
        
        return self._get_observation(), total_reward, done, info
    
    def get_real_coordinates(self, grid_pos):
        """Convert grid position to real GPS coordinates"""
        if not self.real_coords:
            return grid_pos
        
        lat = self.base_lat + (grid_pos[0] * self.coord_scale)
        lon = self.base_lon + (grid_pos[1] * self.coord_scale)
        return [lat, lon]
    
    def generate_csv_data(self, filename="external_obs.csv"):
        """Generate CSV data with current UAV states"""
        data = []
        current_time = time.time() - self.start_time
        
        for i in range(self.num_uavs):
            real_coords = self.get_real_coordinates(self.uav_positions[i])
            
            row = {
                'ID': f'UAV{i}',
                'source_time': int(current_time),
                'lat': real_coords[0] if self.real_coords else self.uav_positions[i][0],
                'lon': real_coords[1] if self.real_coords else self.uav_positions[i][1],
                'source_spotted': self.uav_spotted[i],
                'track': self.uav_tracks[i],
                'takeoff_landing_time': int(current_time),
                'battery': int(self.uav_batteries[i]),
                'AoI': np.sum(self.visited),  # Area of Interest coverage
                'FLA': self.step_count  # Flight Log Actions
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        return df
    
    # def render(self, mode='console'):
    #     if mode == 'console':
    #         print(f"Step {self.step_count}:")
    #         for i, pos in enumerate(self.uav_positions):
    #             battery = self.uav_batteries[i]
    #             spotted = self.uav_spotted[i]
    #             print(f"  UAV{i}: Pos({pos[0]},{pos[1]}) Battery:{battery:.1f}% Spotted:{spotted}")
    #         print(f"Coverage: {np.sum(self.visited)}/{self.grid_size**2} ({100*np.sum(self.visited)/(self.grid_size**2):.1f}%)")
        
    #     return {
    #         'positions': self.uav_positions.copy(),
    #         'batteries': self.uav_batteries.copy(),
    #         'visited': self.visited.copy(),
    #         'spotted': self.uav_spotted.copy(),
    #         'step': self.step_count
    #     }
    def render(self, mode='console'):
        if mode == 'console':
            print(f"Step {self.step_count}:")
            for i, pos in enumerate(self.uav_positions):
                battery = self.uav_batteries[i]
                spotted = self.uav_spotted[i]
                print(f"  UAV{i}: Pos({pos[0]},{pos[1]}) Battery:{battery:.1f}% Spotted:{spotted}")
            print(f"Coverage: {np.sum(self.visited)}/{self.grid_size**2} ({100*np.sum(self.visited)/(self.grid_size**2):.1f}%)")
        
        return {
            'positions': self.uav_positions.copy(),
            'batteries': self.uav_batteries.copy(),
            'visited': self.visited.copy(),
            'spotted': self.uav_spotted.copy(),
            'step': self.step_count
        }

    def seed(self, seed=None):
        np.random.seed(seed)
        import random
        random.seed(seed)
