import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from env.uav_env import RealTimeMultiUAVEnv
from stable_baselines3 import PPO
import json
import time

class RealTimeVisualizer:
    def __init__(self, env, model=None):
        self.env = env
        self.model = model
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 6))
        self.paths = [[] for _ in range(env.num_uavs)]
        self.metrics_data = {
            'coverage': [],
            'batteries': [[] for _ in range(env.num_uavs)],
            'steps': []
        }
        
        # Colors for UAVs
        self.colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        
    def update_frame(self, frame):
        # Get current state
        if self.model:
            obs = self.env._get_observation()
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
        else:
            # Random actions for demo
            action = self.env.action_space.sample()
            obs, reward, done, info = self.env.step(action)
        
        # Update paths
        for i, pos in enumerate(self.env.uav_positions):
            self.paths[i].append(pos.copy())
        
        # Update metrics
        coverage = np.sum(self.env.visited) / (self.env.grid_size ** 2)
        self.metrics_data['coverage'].append(coverage)
        self.metrics_data['steps'].append(self.env.step_count)
        
        for i, battery in enumerate(self.env.uav_batteries):
            self.metrics_data['batteries'][i].append(battery)
        
        # Clear axes
        self.ax1.clear()
        self.ax2.clear()
        
        # Plot grid and UAVs
        self.ax1.set_xlim(-0.5, self.env.grid_size - 0.5)
        self.ax1.set_ylim(-0.5, self.env.grid_size - 0.5)
        self.ax1.set_title(f'Multi-UAV Coverage - Step {self.env.step_count}')
        self.ax1.grid(True, alpha=0.3)
        
        # Draw visited cells
        for i in range(self.env.grid_size):
            for j in range(self.env.grid_size):
                if self.env.visited[i, j]:
                    self.ax1.add_patch(plt.Rectangle((j-0.4, i-0.4), 0.8, 0.8, 
                                                   color='lightgreen', alpha=0.7))
        
        # Draw UAV paths
        for i, path in enumerate(self.paths):
            if len(path) > 1:
                path_array = np.array(path)
                self.ax1.plot(path_array[:, 1], path_array[:, 0], 
                            color=self.colors[i], alpha=0.5, linewidth=2)
        
        # Draw UAVs
        for i, pos in enumerate(self.env.uav_positions):
            battery = self.env.uav_batteries[i]
            spotted = self.env.uav_spotted[i]
            
            # UAV marker
            marker_size = 100 if battery > 0 else 50
            self.ax1.scatter(pos[1], pos[0], 
                           c=self.colors[i], s=marker_size, 
                           marker='o' if battery > 0 else 'x',
                           edgecolors='black', linewidth=2)
            
            # UAV label
            label = f'UAV{i}\n{battery:.0f}%'
            if spotted:
                label += '\n●'  # Spotted indicator
            
            self.ax1.text(pos[1], pos[0]-0.7, label, 
                         ha='center', va='top', fontsize=8,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Plot metrics
        if len(self.metrics_data['coverage']) > 1:
            steps = self.metrics_data['steps']
            
            # Coverage plot
            self.ax2.plot(steps, self.metrics_data['coverage'], 
                         'g-', linewidth=2, label='Coverage')
            
            # Battery plots
            for i in range(self.env.num_uavs):
                if len(self.metrics_data['batteries'][i]) > 0:
                    batteries = [b/100 for b in self.metrics_data['batteries'][i]]  # Normalize
                    self.ax2.plot(steps, batteries, 
                                color=self.colors[i], alpha=0.7, 
                                label=f'UAV{i} Battery')
        
        self.ax2.set_xlabel('Steps')
        self.ax2.set_ylabel('Ratio')
        self.ax2.set_title('Performance Metrics')
        self.ax2.legend()
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_ylim(0, 1)
        
        # Stop animation if done
        if info.get('coverage_ratio', 0) >= 0.95 or self.env.step_count >= 300:
            print(f"Mission completed! Coverage: {info.get('coverage_ratio', 0):.2f}")
            return False
        
        return True
    
    def animate(self, interval=100):
        """Start real-time animation"""
        ani = animation.FuncAnimation(
            self.fig, self.update_frame, interval=interval, 
            blit=False, repeat=False
        )
        plt.tight_layout()
        plt.show()
        return ani

# def create_html_visualization(env, model=None, max_steps=200):
#     """Create HTML-based real-time visualization"""
    
#     # Reset environment
#     obs = env.reset()
    
#     # Collect data
#     data = {
#         'steps': [],
#         'positions': [],
#         'batteries': [],
#         'visited': [],
#         'spotted': []
#     }
    
#     for step in range(max_steps):
#         if model:
#             action, _ = model.predict(obs, deterministic=True)
#         else:
#             action = env.action_space.sample()
        
#         obs, reward, done, info = env.step(action)
        
#         # Store data
#         data['steps'].append(step)
#         data['positions'].append([pos.tolist() for pos in env.uav_positions])
#         data['batteries'].append(env.uav_batteries.copy())
#         data['visited'].append(env.visited.tolist())
#         data['spotted'].append(env.uav_spotted.copy())
        
#         if done:
#             break

def create_html_visualization(env, model=None, max_steps=200):
    """Create HTML-based real-time visualization"""
    import json

    obs = env.reset()
    data = {
        'steps': [],
        'positions': [],
        'batteries': [],
        'visited': [],
        'spotted': []
    }

    for step in range(max_steps):
        if model:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, reward, done, info = env.step(action)

        data['steps'].append(step)
        data['positions'].append([pos.tolist() for pos in env.uav_positions])
        data['batteries'].append(env.uav_batteries.copy())
        data['visited'].append(env.visited.tolist())
        data['spotted'].append(env.uav_spotted.copy())

        if done:
            break
    
    # Generate HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Real-Time Multi-UAV Visualization</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ display: flex; gap: 20px; }}
            .visualization {{ border: 1px solid #ccc; }}
            .controls {{ margin: 10px 0; }}
            .metrics {{ margin-top: 20px; }}
            .uav-info {{ margin: 5px 0; padding: 5px; border: 1px solid #ddd; }}
        </style>
    </head>
    <body>
        <h1>Real-Time Multi-UAV Path Planning</h1>
        
        <div class="controls">
            <button onclick="startAnimation()">Start</button>
            <button onclick="pauseAnimation()">Pause</button>
            <button onclick="resetAnimation()">Reset</button>
            <span>Speed: </span>
            <input type="range" id="speed" min="50" max="1000" value="200" onchange="updateSpeed()">
        </div>
        
        <div class="container">
            <div>
                <svg id="grid" width="400" height="400" class="visualization"></svg>
                <div id="step-info">Step: 0</div>
            </div>
            <div>
                <svg id="metrics" width="400" height="300" class="visualization"></svg>
                <div id="uav-info"></div>
            </div>
        </div>
        
        <script>
            const data = {json.dumps(data)};
            const gridSize = {env.grid_size};
            const numUAVs = {env.num_uavs};
            const colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown'];
            
            let currentStep = 0;
            let animationId = null;
            let animationSpeed = 200;
            
            function setupGrid() {{
                const svg = d3.select('#grid');
                const cellSize = 400 / gridSize;
                
                // Draw grid
                for (let i = 0; i <= gridSize; i++) {{
                    svg.append('line')
                        .attr('x1', 0).attr('y1', i * cellSize)
                        .attr('x2', 400).attr('y2', i * cellSize)
                        .attr('stroke', '#ccc');
                    svg.append('line')
                        .attr('x1', i * cellSize).attr('y1', 0)
                        .attr('x2', i * cellSize).attr('y2', 400)
                        .attr('stroke', '#ccc');
                }}
            }}
            
            function updateVisualization() {{
                if (currentStep >= data.steps.length) {{
                    pauseAnimation();
                    return;
                }}
                
                const svg = d3.select('#grid');
                const cellSize = 400 / gridSize;
                
                // Clear previous UAVs and visited cells
                svg.selectAll('.visited').remove();
                svg.selectAll('.uav').remove();
                svg.selectAll('.path').remove();
                
                // Draw visited cells
                const visited = data.visited[currentStep];
                for (let i = 0; i < gridSize; i++) {{
                    for (let j = 0; j < gridSize; j++) {{
                        if (visited[i][j]) {{
                            svg.append('rect')
                                .attr('class', 'visited')
                                .attr('x', j * cellSize + 2)
                                .attr('y', i * cellSize + 2)
                                .attr('width', cellSize - 4)
                                .attr('height', cellSize - 4)
                                .attr('fill', 'lightgreen')
                                .attr('opacity', 0.7);
                        }}
                    }}
                }}
                
                // Draw UAV paths
                for (let uav = 0; uav < numUAVs; uav++) {{
                    const pathData = [];
                    for (let step = 0; step <= currentStep; step++) {{
                        const pos = data.positions[step][uav];
                        pathData.push([pos[1] * cellSize + cellSize/2, pos[0] * cellSize + cellSize/2]);
                    }}
                    
                    if (pathData.length > 1) {{
                        const line = d3.line()
                            .x(d => d[0])
                            .y(d => d[1]);
                        
                        svg.append('path')
                            .attr('class', 'path')
                            .attr('d', line(pathData))
                            .attr('stroke', colors[uav])
                            .attr('stroke-width', 2)
                            .attr('fill', 'none')
                            .attr('opacity', 0.6);
                    }}
                }}
                
                // Draw UAVs
                const positions = data.positions[currentStep];
                const batteries = data.batteries[currentStep];
                const spotted = data.spotted[currentStep];
                
                for (let uav = 0; uav < numUAVs; uav++) {{
                    const pos = positions[uav];
                    const battery = batteries[uav];
                    const isSpotted = spotted[uav];
                    
                    const x = pos[1] * cellSize + cellSize/2;
                    const y = pos[0] * cellSize + cellSize/2;
                    
                    // UAV circle
                    svg.append('circle')
                        .attr('class', 'uav')
                        .attr('cx', x)
                        .attr('cy', y)
                        .attr('r', battery > 0 ? 8 : 4)
                        .attr('fill', colors[uav])
                        .attr('stroke', 'black')
                        .attr('stroke-width', 2)
                        .attr('opacity', battery > 0 ? 1 : 0.5);
                    
                    // UAV label
                    svg.append('text')
                        .attr('class', 'uav')
                        .attr('x', x)
                        .attr('y', y - 15)
                        .attr('text-anchor', 'middle')
                        .attr('font-size', '10px')
                        .attr('fill', 'black')
                        .text(`UAV${{uav}} ${{battery.toFixed(0)}}%`);
                    
                    // Spotted indicator
                    if (isSpotted) {{
                        svg.append('circle')
                            .attr('class', 'uav')
                            .attr('cx', x + 10)
                            .attr('cy', y - 10)
                            .attr('r', 3)
                            .attr('fill', 'yellow')
                            .attr('stroke', 'orange')
                            .attr('stroke-width', 1);
                    }}
                }}
                
                // Update step info
                document.getElementById('step-info').innerHTML = `Step: ${{currentStep + 1}}/${{data.steps.length}}`;
                
                // Update UAV info
                let uavInfoHtml = '';
                for (let uav = 0; uav < numUAVs; uav++) {{
                    const battery = batteries[uav];
                    const isSpotted = spotted[uav];
                    const pos = positions[uav];
                    
                    uavInfoHtml += `
                        <div class="uav-info" style="border-left: 4px solid ${{colors[uav]}}">
                            <strong>UAV ${{uav}}</strong><br>
                            Position: (${{pos[0]}}, ${{pos[1]}})<br>
                            Battery: ${{battery.toFixed(1)}}%<br>
                            Status: ${{isSpotted ? 'Target Spotted' : 'Searching'}}
                        </div>
                    `;
                }}
                document.getElementById('uav-info').innerHTML = uavInfoHtml;
                
                // Update metrics chart
                updateMetricsChart();
            }}
            
            function updateMetricsChart() {{
                const svg = d3.select('#metrics');
                svg.selectAll('*').remove();
                
                const margin = {{top: 20, right: 30, bottom: 40, left: 50}};
                const width = 400 - margin.left - margin.right;
                const height = 300 - margin.top - margin.bottom;
                
                const g = svg.append('g')
                    .attr('transform', `translate(${{margin.left}},${{margin.top}})`);
                
                // Scales
                const xScale = d3.scaleLinear()
                    .domain([0, Math.max(currentStep, 1)])
                    .range([0, width]);
                
                const yScale = d3.scaleLinear()
                    .domain([0, 1])
                    .range([height, 0]);
                
                // Axes
                g.append('g')
                    .attr('transform', `translate(0,${{height}})`)
                    .call(d3.axisBottom(xScale));
                
                g.append('g')
                    .call(d3.axisLeft(yScale));
                
                // Coverage line
                const coverageData = data.visited.slice(0, currentStep + 1).map((visited, i) => {{
                    const coverage = visited.flat().reduce((sum, cell) => sum + cell, 0) / (gridSize * gridSize);
                    return [i, coverage];
                }});
                
                if (coverageData.length > 1) {{
                    const line = d3.line()
                        .x(d => xScale(d[0]))
                        .y(d => yScale(d[1]));
                    
                    g.append('path')
                        .datum(coverageData)
                        .attr('d', line)
                        .attr('stroke', 'green')
                        .attr('stroke-width', 2)
                        .attr('fill', 'none');
                }}
                
                // Battery lines
                for (let uav = 0; uav < numUAVs; uav++) {{
                    const batteryData = data.batteries.slice(0, currentStep + 1).map((batteries, i) => {{
                        return [i, batteries[uav] / 100];
                    }});
                    
                    if (batteryData.length > 1) {{
                        const line = d3.line()
                            .x(d => xScale(d[0]))
                            .y(d => yScale(d[1]));
                        
                        g.append('path')
                            .datum(batteryData)
                            .attr('d', line)
                            .attr('stroke', colors[uav])
                            .attr('stroke-width', 1)
                            .attr('fill', 'none')
                            .attr('opacity', 0.7);
                    }}
                }}
                
                // Labels
                g.append('text')
                    .attr('transform', 'rotate(-90)')
                    .attr('y', 0 - margin.left)
                    .attr('x', 0 - (height / 2))
                    .attr('dy', '1em')
                    .style('text-anchor', 'middle')
                    .text('Coverage / Battery Level');
                
                g.append('text')
                    .attr('transform', `translate(${{width / 2}}, ${{height + margin.bottom}})`)
                    .style('text-anchor', 'middle')
                    .text('Steps');
            }}
            
            function startAnimation() {{
                if (animationId) return;
                
                animationId = setInterval(() => {{
                    updateVisualization();
                    currentStep++;
                    if (currentStep >= data.steps.length) {{
                        pauseAnimation();
                    }}
                }}, animationSpeed);
            }}
            
            function pauseAnimation() {{
                if (animationId) {{
                    clearInterval(animationId);
                    animationId = null;
                }}
            }}
            
            function resetAnimation() {{
                pauseAnimation();
                currentStep = 0;
                updateVisualization();
            }}
            
            function updateSpeed() {{
                const speed = document.getElementById('speed').value;
                animationSpeed = parseInt(speed);
                if (animationId) {{
                    pauseAnimation();
                    startAnimation();
                }}
            }}
            
            // Initialize
            setupGrid();
            updateVisualization();
        </script>
    </body>
    </html>
    """
    
    # Save HTML file
    # with open('visualize/real_time_visualization.html', 'w') as f:
    #     f.write(html_content)
    
    # print("HTML visualization saved to visualize/real_time_visualization.html")
    # return html_content
    import os
    os.makedirs("visualize", exist_ok=True)
    output_path = os.path.join("visualize", "real_time_visualization.html")
    with open(output_path, 'w') as f:
        f.write(html_content)

    print(f"✅ HTML visualization saved to {output_path}")
    return html_content

def run_live_visualization():
    """Run live matplotlib visualization"""
    print("Starting real-time visualization...")
    
    # Create environment
    env = RealTimeMultiUAVEnv(grid_size=8, num_uavs=3)
    
    # Try to load trained model
    model = None
    try:
        model = PPO.load("../models/ppo_MultiUAVEnv")
        print("Using trained model")
    except:
        print("No trained model found, using random actions")
    
    # Create visualizer
    visualizer = RealTimeVisualizer(env, model)
    
    # Start animation
    ani = visualizer.animate(interval=150)
    
    return ani

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-UAV Visualization')
    parser.add_argument('--mode', choices=['live', 'html'], default='live',
                        help='Visualization mode')
    parser.add_argument('--grid-size', type=int, default=8,
                        help='Grid size for environment')
    parser.add_argument('--num-uavs', type=int, default=3,
                        help='Number of UAVs')
    
    args = parser.parse_args()
    
    if args.mode == 'live':
        run_live_visualization()
    else:
        env = RealTimeMultiUAVEnv(grid_size=args.grid_size, num_uavs=args.num_uavs)
        try:
            model = PPO.load("../models/ppo_MultiUAVEnv")
        except:
            model = None
        create_html_visualization(env, model)