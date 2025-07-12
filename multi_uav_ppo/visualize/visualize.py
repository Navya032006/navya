import sys
sys.path.append('../')

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import json
from datetime import datetime
import time
import webbrowser
import os
from env.uav_env import MultiUAVEnv
from stable_baselines3 import PPO

class RealTimeVisualizer:
    def __init__(self, grid_size=10, num_uavs=3):
        self.grid_size = grid_size
        self.num_uavs = num_uavs
        self.colors = ['red', 'blue', 'green', 'orange', 'purple']
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.animation_data = []
        
    def create_html_visualization(self, simulation_data, output_file="./data/real_time_visualization.html"):
        """Create HTML-based real-time visualization"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Multi-UAV Real-Time Path Planning</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        .grid-container {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat({self.grid_size}, 40px);
            grid-template-rows: repeat({self.grid_size}, 40px);
            gap: 2px;
            background-color: #333;
            padding: 10px;
            border-radius: 5px;
        }}
        .cell {{
            width: 40px;
            height: 40px;
            background-color: #fff;
            border: 1px solid #ccc;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 12px;
            border-radius: 3px;
            transition: all 0.3s ease;
        }}
        .visited {{
            background-color: #90EE90;
        }}
        .uav {{
            background-color: #FF6B6B;
            color: white;
        }}
        .uav-0 {{ background-color: #FF6B6B; }}
        .uav-1 {{ background-color: #4ECDC4; }}
        .uav-2 {{ background-color: #45B7D1; }}
        .controls {{
            margin-left: 20px;
            width: 300px;
        }}
        .metrics {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .metric-item {{
            margin-bottom: 10px;
            padding: 5px;
            background-color: white;
            border-radius: 3px;
            display: flex;
            justify-content: space-between;
        }}
        .progress-bar {{
            width: 100%;
            height: 20px;
            background-color: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
            margin-top: 5px;
        }}
        .progress-fill {{
            height: 100%;
            background-color: #4CAF50;
            transition: width 0.3s ease;
        }}
        .button {{
            background-color: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin: 5px;
            font-size: 14px;
        }}
        .button:hover {{
            background-color: #0056b3;
        }}
        .episode-info {{
            background-color: #e9ecef;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }}
        .legend {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 20px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 3px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Multi-UAV Real-Time Path Planning Visualization</h1>
            <p>Real-time simulation of {self.num_uavs} UAVs navigating a {self.grid_size}x{self.grid_size} grid</p>
        </div>
        
        <div class="grid-container">
            <div>
                <div class="grid" id="grid"></div>
                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #FF6B6B;"></div>
                        <span>UAV 0</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #4ECDC4;"></div>
                        <span>UAV 1</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #45B7D1;"></div>
                        <span>UAV 2</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #90EE90;"></div>
                        <span>Visited</span>
                    </div>
                </div>
            </div>
            
            <div class="controls">
                <div class="episode-info">
                    <h3>Episode: <span id="episode">1</span></h3>
                    <h4>Step: <span id="step">0</span></h4>
                </div>
                
                <div class="metrics">
                    <h3>Real-Time Metrics</h3>
                    <div class="metric-item">
                        <span>Coverage:</span>
                        <span id="coverage">0%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" id="coverage-bar" style="width: 0%"></div>
                    </div>
                    
                    <div class="metric-item">
                        <span>Energy Used:</span>
                        <span id="energy">0</span>
                    </div>
                    
                    <div class="metric-item">
                        <span>Mission Time:</span>
                        <span id="mission-time">0</span>
                    </div>
                    
                    <div class="metric-item">
                        <span>Data Points:</span>
                        <span id="data-points">0</span>
                    </div>
                    
                    <div class="metric-item">
                        <span>Collisions:</span>
                        <span id="collisions">0</span>
                    </div>
                </div>
                
                <div>
                    <button class="button" onclick="startAnimation()">Start</button>
                    <button class="button" onclick="pauseAnimation()">Pause</button>
                    <button class="button" onclick="resetAnimation()">Reset</button>
                    <button class="button" onclick="speedUp()">Speed Up</button>
                    <button class="button" onclick="slowDown()">Slow Down</button>
                </div>
                
                <div class="metrics">
                    <h4>Battery Levels</h4>
                    <div id="battery-levels"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const simulationData = {json.dumps(simulation_data)};
        let currentStep = 0;
        let isPlaying = false;
        let animationSpeed = 500; // milliseconds
        let animationInterval;
        
        function initializeGrid() {{
            const grid = document.getElementById('grid');
            grid.innerHTML = '';
            
            for (let i = 0; i < {self.grid_size * self.grid_size}; i++) {{
                const cell = document.createElement('div');
                cell.className = 'cell';
                cell.id = `cell-${{i}}`;
                grid.appendChild(cell);
            }}
        }}
        
        function updateVisualization() {{
            if (currentStep >= simulationData.length) {{
                pauseAnimation();
                return;
            }}
            
            const data = simulationData[currentStep];
            
            // Update episode and step info
            document.getElementById('episode').textContent = data.episode + 1;
            document.getElementById('step').textContent = data.step;
            
            // Clear grid
            for (let i = 0; i < {self.grid_size * self.grid_size}; i++) {{
                const cell = document.getElementById(`cell-${{i}}`);
                cell.className = 'cell';
                cell.textContent = '';
            }}
            
            // Update visited cells
            const visited = data.visited_cells;
            for (let i = 0; i < {self.grid_size}; i++) {{
                for (let j = 0; j < {self.grid_size}; j++) {{
                    const cellIndex = i * {self.grid_size} + j;
                    const cell = document.getElementById(`cell-${{cellIndex}}`);
                    if (visited[i][j] === 1) {{
                        cell.classList.add('visited');
                    }}
                }}
            }}
            
            // Update UAV positions
            const uavPositions = data.uav_positions;
            for (let i = 0; i < uavPositions.length; i++) {{
                const [row, col] = uavPositions[i];
                const cellIndex = row * {self.grid_size} + col;
                const cell = document.getElementById(`cell-${{cellIndex}}`);
                cell.className = `cell uav uav-${{i}}`;
                cell.textContent = `U${{i}}`;
            }}
            
            // Update metrics
            const metrics = data.metrics;
            document.getElementById('coverage').textContent = `${{metrics.efficiency_score.toFixed(1)}}%`;
            document.getElementById('coverage-bar').style.width = `${{metrics.efficiency_score}}%`;
            document.getElementById('energy').textContent = metrics.energy_consumed.toFixed(1);
            document.getElementById('mission-time').textContent = metrics.mission_time;
            document.getElementById('collisions').textContent = metrics.collision_count;
            
            // Update battery levels
            const batteryDiv = document.getElementById('battery-levels');
            batteryDiv.innerHTML = '';
            data.battery_levels.forEach((level, index) => {{
                const batteryItem = document.createElement('div');
                batteryItem.className = 'metric-item';
                batteryItem.innerHTML = `
                    <span>UAV ${{index}}:</span>
                    <span>${{level.toFixed(1)}}%</span>
                `;
                batteryDiv.appendChild(batteryItem);
            }});
            
            currentStep++;
        }}
        
        function startAnimation() {{
            if (!isPlaying) {{
                isPlaying = true;
                animationInterval = setInterval(updateVisualization, animationSpeed);
            }}
        }}
        
        function pauseAnimation() {{
            isPlaying = false;
            clearInterval(animationInterval);
        }}
        
        function resetAnimation() {{
            pauseAnimation();
            currentStep = 0;
            updateVisualization();
        }}
        
        function speedUp() {{
            animationSpeed = Math.max(100, animationSpeed - 100);
            if (isPlaying) {{
                pauseAnimation();
                startAnimation();
            }}
        }}
        
        function slowDown() {{
            animationSpeed += 100;
            if (isPlaying) {{
                pauseAnimation();
                startAnimation();
            }}
        }}
        
        // Initialize
        initializeGrid();
        updateVisualization();
    </script>
</body>
</html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        print(f"HTML visualization saved to {output_file}")
        return output_file
    
    def animate_real_time(self, env, model, num_episodes=1, save_html=True):
        """Run real-time animation with model predictions"""
        print("Starting real-time animation...")
        
        simulation_data = []
        
        for episode in range(num_episodes):
            obs = env.reset()
            done = False
            step = 0
            
            while not done and step < 100:  # Limit steps for demo
                # Get model prediction
                actions, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(actions)
                
                # Get render data
                render_data = env.render(mode='dict')
                
                # Store step data
                step_data = {
                    'episode': episode,
                    'step': step,
                    'uav_positions': render_data['uav_positions'].tolist(),
                    'visited_cells': render_data['visited'].tolist(),
                    'battery_levels': render_data['battery_levels'].tolist(),
                    'metrics': render_data['metrics']
                }
                
                simulation_data.append(step_data)
                step += 1
                
                # Real-time display
                if step % 10 == 0:
                    print(f"Episode {episode + 1}, Step {step}: Coverage {render_data['metrics']['efficiency_score']:.1f}%")
        
        # Create HTML visualization
        if save_html:
            html_file = self.create_html_visualization(simulation_data)
            
            # Try to open in browser
            try:
                webbrowser.open(f'file://{os.path.abspath(html_file)}')
                print("HTML visualization opened in browser")
            except:
                print("Could not open browser automatically")
        
        return simulation_data
    
    def create_matplotlib_animation(self, simulation_data, save_gif=True):
        """Create matplotlib animation"""
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        
        def animate(frame):
            self.ax.clear()
            
            if frame < len(simulation_data):
                data = simulation_data[frame]
                
                # Set up grid
                self.ax.set_xlim(-0.5, self.grid_size - 0.5)
                self.ax.set_ylim(-0.5, self.grid_size - 0.5)
                self.ax.set_xticks(range(self.grid_size))
                self.ax.set_yticks(range(self.grid_size))
                self.ax.grid(True, alpha=0.3)
                self.ax.set_aspect('equal')
                
                # Draw visited cells
                visited = np.array(data['visited_cells'])
                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        if visited[i][j] == 1:
                            self.ax.add_patch(plt.Rectangle((j-0.4, i-0.4), 0.8, 0.8, 
                                                          facecolor='lightgreen', alpha=0.7))
                
                # Draw UAVs
                uav_positions = data['uav_positions']
                for idx, pos in enumerate(uav_positions):
                    color = self.colors[idx % len(self.colors)]
                    self.ax.plot(pos[1], pos[0], 'o', color=color, markersize=15, 
                               markeredgecolor='black', markeredgewidth=2)
                    self.ax.text(pos[1], pos[0], f'U{idx}', ha='center', va='center', 
                               fontweight='bold', color='white')
                
                # Add metrics
                metrics = data['metrics']
                self.ax.set_title(f"Episode {data['episode'] + 1}, Step {data['step']}\n"
                                f"Coverage: {metrics['efficiency_score']:.1f}% | "
                                f"Energy: {metrics['energy_consumed']:.1f} | "
                                f"Collisions: {metrics['collision_count']}")
                
                # Invert y-axis for proper grid representation
                self.ax.invert_yaxis()
        
        anim = animation.FuncAnimation(self.fig, animate, frames=len(simulation_data), 
                                     interval=200, repeat=True, blit=False)
        
        if save_gif:
            try:
                anim.save('./data/uav_animation.gif', writer='pillow', fps=5)
                print("Animation saved as GIF")
            except:
                print("Could not save GIF (pillow not available)")
        
        return anim

def main():
    # Initialize environment and model
    env = MultiUAVEnv(grid_size=10, num_uavs=3, max_steps=100)
    
    try:
        model = PPO.load("../models/ppo_MultiUAVEnv")
        print("Loaded trained model")
    except:
        print("No trained model found. Using random actions for demonstration.")
        model = None
    
    # Create visualizer
    visualizer = RealTimeVisualizer(grid_size=10, num_uavs=3)
    
    # Run real-time animation
    if model:
        simulation_data = visualizer.animate_real_time(env, model, num_episodes=1)
    else:
        # Generate sample data for demonstration
        simulation_data = []
        for step in range(50):
            step_data = {
                'episode': 0,
                'step': step,
                'uav_positions': np.random.randint(0, 10, (3, 2)).tolist(),
                'visited_cells': np.random.randint(0, 2, (10, 10)).tolist(),
                'battery_levels': np.random.uniform(20, 100, 3).tolist(),
                'metrics': {
                    'efficiency_score': min(100, step * 2),
                    'energy_consumed': step * 3.5,
                    'collision_count': 0,
                    'mission_time': step
                }
            }
            simulation_data.append(step_data)
        
        # Create HTML visualization
        html_file = visualizer.create_html_visualization(simulation_data)
        print(f"Demo HTML visualization created: {html_file}")
    
    # Create matplotlib animation
    anim = visualizer.create_matplotlib_animation(simulation_data)
    
    print("\nVisualization complete!")
    print("Check the following files:")
    print("- ./data/real_time_visualization.html (open in browser)")
    print("- ./data/uav_animation.gif (if available)")

if __name__ == "__main__":
    main()