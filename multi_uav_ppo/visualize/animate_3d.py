import sys
sys.path.append('../')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import numpy as np
import pandas as pd
from env.uav_env import MultiUAVEnv
from stable_baselines3 import PPO

class UAV3DVisualizer:
    def __init__(self, grid_size=10, num_uavs=3):
        self.grid_size = grid_size
        self.num_uavs = num_uavs
        self.colors = ['red', 'blue', 'green', 'orange', 'purple']
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
    def create_3d_paths(self, env, model, num_episodes=1):
        """Generate 3D paths for UAVs"""
        print("Generating 3D UAV paths...")
        
        all_paths = []
        all_altitudes = []
        all_visited = []
        
        for episode in range(num_episodes):
            obs = env.reset()
            done = False
            step = 0
            
            paths = [[] for _ in range(self.num_uavs)]
            altitudes = [[] for _ in range(self.num_uavs)]
            visited_sequence = []
            
            while not done and step < 100:
                # Get model prediction
                actions, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(actions)
                
                # Get render data
                render_data = env.render(mode='dict')
                
                # Store positions with altitude variation
                for i, pos in enumerate(render_data['uav_positions']):
                    paths[i].append(pos)
                    # Vary altitude based on battery level and mission progress
                    battery_factor = render_data['battery_levels'][i] / 100.0
                    mission_factor = step / 100.0
                    altitude = 5 + 3 * battery_factor + 2 * mission_factor
                    altitudes[i].append(altitude)
                
                visited_sequence.append(render_data['visited'].copy())
                step += 1
            
            all_paths.append(paths)
            all_altitudes.append(altitudes)
            all_visited.append(visited_sequence)
        
        return all_paths, all_altitudes, all_visited
    
    def animate_3d(self, paths, altitudes, visited_sequence, save_gif=True):
        """Create 3D animation"""
        print("Creating 3D animation...")
        
        # Use first episode data
        episode_paths = paths[0]
        episode_altitudes = altitudes[0]
        episode_visited = visited_sequence[0]
        
        max_steps = min(len(episode_paths[0]), len(episode_altitudes[0]))
        
        def animate(frame):
            self.ax.clear()
            
            if frame < max_steps:
                # Set up 3D space
                self.ax.set_xlim(0, self.grid_size)
                self.ax.set_ylim(0, self.grid_size)
                self.ax.set_zlim(0, 15)
                
                # Draw ground grid
                x_grid, y_grid = np.meshgrid(range(self.grid_size + 1), range(self.grid_size + 1))
                z_grid = np.zeros_like(x_grid)
                self.ax.plot_wireframe(x_grid, y_grid, z_grid, alpha=0.3, color='gray')
                
                # Draw visited cells as elevated blocks
                if frame < len(episode_visited):
                    visited = episode_visited[frame]
                    for i in range(self.grid_size):
                        for j in range(self.grid_size):
                            if visited[i][j] == 1:
                                # Create a small block for visited cells
                                x = [j, j+1, j+1, j, j]
                                y = [i, i, i+1, i+1, i]
                                z = [0, 0, 0, 0, 0]
                                self.ax.plot(x, y, z, 'g-', alpha=0.5)
                                
                                # Add height to visited cells
                                for k in range(len(x)-1):
                                    self.ax.plot([x[k], x[k]], [y[k], y[k]], [0, 0.5], 'g-', alpha=0.7)
                
                # Draw UAV paths and current positions
                for i in range(self.num_uavs):
                    if frame < len(episode_paths[i]):
                        # Draw path trail
                        if frame > 0:
                            path_x = [pos[1] + 0.5 for pos in episode_paths[i][:frame]]
                            path_y = [pos[0] + 0.5 for pos in episode_paths[i][:frame]]
                            path_z = episode_altitudes[i][:frame]
                            
                            self.ax.plot(path_x, path_y, path_z, 
                                       color=self.colors[i], linewidth=2, alpha=0.7)
                        
                        # Draw current UAV position
                        current_pos = episode_paths[i][frame]
                        current_alt = episode_altitudes[i][frame]
                        
                        self.ax.scatter([current_pos[1] + 0.5], [current_pos[0] + 0.5], [current_alt],
                                      s=150, c=self.colors[i], marker='o', edgecolor='black', linewidth=2)
                        
                        # Add UAV label
                        self.ax.text(current_pos[1] + 0.5, current_pos[0] + 0.5, current_alt + 0.5,
                                   f'UAV{i}', fontsize=10, color=self.colors[i])
                
                # Set labels and title
                self.ax.set_xlabel('X Position')
                self.ax.set_ylabel('Y Position')
                self.ax.set_zlabel('Altitude')
                self.ax.set_title(f'Multi-UAV 3D Path Planning - Step {frame}')
                
                # Set viewing angle
                self.ax.view_init(elev=20, azim=45 + frame * 0.5)
        
        anim = animation.FuncAnimation(self.fig, animate, frames=max_steps, 
                                     interval=200, repeat=True, blit=False)
        
        if save_gif:
            try:
                anim.save('../data/uav_3d_animation.gif', writer='pillow', fps=5)
                print("3D animation saved as GIF")
            except:
                print("Could not save 3D GIF (pillow not available)")
        
        return anim
    
    def create_static_3d_plot(self, paths, altitudes, save_plot=True):
        """Create static 3D plot showing all paths"""
        print("Creating static 3D plot...")
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Use first episode data
        episode_paths = paths[0]
        episode_altitudes = altitudes[0]
        
        # Draw ground grid
        x_grid, y_grid = np.meshgrid(range(self.grid_size + 1), range(self.grid_size + 1))
        z_grid = np.zeros_like(x_grid)
        ax.plot_wireframe(x_grid, y_grid, z_grid, alpha=0.3, color='gray')
        
        # Draw all UAV paths
        for i in range(self.num_uavs):
            if episode_paths[i]:
                path_x = [pos[1] + 0.5 for pos in episode_paths[i]]
                path_y = [pos[0] + 0.5 for pos in episode_paths[i]]
                path_z = episode_altitudes[i]
                
                ax.plot(path_x, path_y, path_z, 
                       color=self.colors[i], linewidth=3, alpha=0.8, label=f'UAV {i}')
                
                # Mark start and end points
                ax.scatter([path_x[0]], [path_y[0]], [path_z[0]], 
                          s=200, c=self.colors[i], marker='^', edgecolor='black', linewidth=2)
                ax.scatter([path_x[-1]], [path_y[-1]], [path_z[-1]], 
                          s=200, c=self.colors[i], marker='v', edgecolor='black', linewidth=2)
        
        # Set labels and title
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Altitude')
        ax.set_title('Multi-UAV 3D Path Planning - Complete Paths')
        ax.legend()
        
        # Set viewing angle
        ax.view_init(elev=20, azim=45)
        
        if save_plot:
            plt.savefig('./data/uav_3d_paths.png', dpi=300, bbox_inches='tight')
            print("Static 3D plot saved as PNG")
        
        return fig
    
    def create_altitude_profile(self, altitudes, save_plot=True):
        """Create altitude profile plot"""
        print("Creating altitude profile...")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Use first episode data
        episode_altitudes = altitudes[0]
        
        for i in range(self.num_uavs):
            if episode_altitudes[i]:
                steps = range(len(episode_altitudes[i]))
                ax.plot(steps, episode_altitudes[i], 
                       color=self.colors[i], linewidth=2, label=f'UAV {i}')
        
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Altitude')
        ax.set_title('UAV Altitude Profiles Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_plot:
            plt.savefig('./data/uav_altitude_profile.png', dpi=300, bbox_inches='tight')
            print("Altitude profile saved as PNG")
        
        return fig

def main():
    # Initialize environment and model
    env = MultiUAVEnv(grid_size=10, num_uavs=3, max_steps=100)
    
    try:
        model = PPO.load("../models/ppo_MultiUAVEnv")
        print("Loaded trained model")
    except:
        print("No trained model found. Creating demo 3D visualization...")
        model = None
    
    # Create 3D visualizer
    visualizer = UAV3DVisualizer(grid_size=10, num_uavs=3)
    
    if model:
        # Generate real paths
        paths, altitudes, visited = visualizer.create_3d_paths(env, model, num_episodes=1)
        
        # Create 3D animation
        anim = visualizer.animate_3d(paths, altitudes, visited)
        
        # Create static plot
        static_fig = visualizer.create_static_3d_plot(paths, altitudes)
        
        # Create altitude profile
        altitude_fig = visualizer.create_altitude_profile(altitudes)
        
        print("\n3D visualization complete!")
        print("Check the following files:")
        print("- ../data/uav_3d_animation.gif")
        print("- ../data/uav_3d_paths.png")
        print("- ../data/uav_altitude_profile.png")
    else:
        # Generate demo data
        demo_paths = []
        demo_altitudes = []
        
        for i in range(3):  # 3 UAVs
            path = []
            altitude = []
            
            # Create spiral paths
            for step in range(50):
                angle = step * 0.3 + i * 2.1
                radius = 3 + step * 0.05
                x = 5 + radius * np.cos(angle)
                y = 5 + radius * np.sin(angle)
                z = 5 + 3 * np.sin(step * 0.1)
                
                path.append([int(max(0, min(9, y))), int(max(0, min(9, x)))])
                altitude.append(z)
            
            demo_paths.append(path)
            demo_altitudes.append(altitude)
        
        # Create demo visualization
        all_paths = [demo_paths]
        all_altitudes = [demo_altitudes]
        demo_visited = [np.zeros((10, 10)) for _ in range(50)]
        
        static_fig = visualizer.create_static_3d_plot(all_paths, all_altitudes)
        altitude_fig = visualizer.create_altitude_profile(all_altitudes)
        
        print("\nDemo 3D visualization created!")
        print("Check the following files:")
        print("- ../data/uav_3d_paths.png")
        print("- ../data/uav_altitude_profile.png")
    
    plt.show()

if __name__ == "__main__":
    main()