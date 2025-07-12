#!/usr/bin/env python3
"""
Multi-UAV Real-Time Path Planning System - Complete Demo
========================================================

This script demonstrates the complete multi-UAV real-time path planning system:
1. Baseline data collection (before training)
2. Model training with PPO
3. Trained data collection (after training)
4. Performance comparison and analysis
5. Real-time visualization generation

Usage:
    python run_complete_system.py
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add project paths
sys.path.append('./env')
sys.path.append('./train')
sys.path.append('./test')
sys.path.append('./visualize')
sys.path.append('./utils')

# Core imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import json
from datetime import datetime

# Project imports
from env.uav_env import MultiUAVEnv
from utils.logger import UAVLogger, CSVDataLogger
from utils.metrics import UAVMetrics
from visualize.visualize import RealTimeVisualizer
from visualize.animate_3d import UAV3DVisualizer

# ML imports
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

class RealTimeCallback(BaseCallback):
    """Callback for real-time training monitoring"""
    def __init__(self, verbose=0):
        super(RealTimeCallback, self).__init__(verbose)
        self.last_update = time.time()
        self.start_time = None
    
    def _on_training_start(self) -> None:
        self.start_time = time.time()
        print("🚀 PPO Training Started...")
    
    def _on_step(self) -> bool:
        # Update every 2000 steps
        if self.n_calls % 2000 == 0:
            current_time = time.time()
            if current_time - self.last_update > 15:  # Update every 15 seconds
                elapsed = current_time - self.start_time
                progress = (self.n_calls / 50000) * 100
                print(f"   Progress: {progress:.1f}% ({self.n_calls}/50000 steps) - Time: {elapsed:.1f}s")
                self.last_update = current_time
        return True

def collect_baseline_data(env, num_episodes=3):
    """Collect baseline data before training"""
    print("\n📊 Collecting Baseline Data (Before Training)...")
    print("=" * 60)
    
    baseline_data = []
    baseline_metrics = []
    
    for episode in range(num_episodes):
        print(f"\n🎯 Baseline Episode {episode + 1}/{num_episodes}")
        obs = env.reset()
        done = False
        episode_data = []
        step = 0
        
        while not done and step < 80:  # Limit steps for demo
            # Random actions (untrained behavior)
            actions = env.action_space.sample()
            obs, reward, done, info = env.step(actions)
            
            # Collect data points
            current_data = env.get_collected_data()
            if len(current_data) > len(episode_data):
                episode_data = current_data.copy()
            
            step += 1
            
            # Show progress
            if step % 20 == 0:
                current_metrics = env.get_metrics()
                print(f"   Step {step}: Coverage {current_metrics['efficiency_score']:.1f}%, Energy {current_metrics['energy_consumed']:.1f}")
        
        # Episode summary
        final_metrics = env.get_metrics()
        baseline_data.extend(episode_data)
        baseline_metrics.append(final_metrics)
        
        print(f"   ✅ Episode {episode + 1} Results:")
        print(f"      📈 Coverage: {final_metrics['efficiency_score']:.1f}%")
        print(f"      📊 Data Points: {len(episode_data)}")
        print(f"      ⚡ Energy: {final_metrics['energy_consumed']:.1f}")
        print(f"      💥 Collisions: {final_metrics['collision_count']}")
    
    print(f"\n📈 Baseline Summary:")
    print(f"   Total Data Points: {len(baseline_data)}")
    print(f"   Avg Coverage: {np.mean([m['efficiency_score'] for m in baseline_metrics]):.1f}%")
    print(f"   Avg Energy: {np.mean([m['energy_consumed'] for m in baseline_metrics]):.1f}")
    
    return baseline_data, baseline_metrics

def train_model(env):
    """Train the PPO model"""
    print("\n🎯 Training PPO Model...")
    print("=" * 60)
    
    # Initialize PPO model
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=0,
        learning_rate=0.0003,
        n_steps=1024,
        batch_size=32,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )
    
    # Train with callback
    callback = RealTimeCallback()
    start_time = time.time()
    
    model.learn(
        total_timesteps=50000,
        callback=callback,
        progress_bar=False
    )
    
    training_time = time.time() - start_time
    
    # Save model
    model.save("./models/ppo_MultiUAVEnv")
    
    print(f"\n✅ Training Complete!")
    print(f"   Training Time: {training_time:.2f} seconds")
    print(f"   Model Saved: ./models/ppo_MultiUAVEnv")
    print(f"   Avg Time per 1000 steps: {(training_time / 50) * 1000:.2f}ms")
    
    return model, training_time

def collect_trained_data(env, model, num_episodes=3):
    """Collect data after training"""
    print("\n🎓 Collecting Trained Data (After Training)...")
    print("=" * 60)
    
    trained_data = []
    trained_metrics = []
    
    for episode in range(num_episodes):
        print(f"\n🎯 Trained Episode {episode + 1}/{num_episodes}")
        obs = env.reset()
        done = False
        episode_data = []
        step = 0
        
        while not done and step < 80:  # Limit steps for demo
            # Use trained model
            actions, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(actions)
            
            # Collect data points
            current_data = env.get_collected_data()
            if len(current_data) > len(episode_data):
                episode_data = current_data.copy()
            
            step += 1
            
            # Show progress
            if step % 20 == 0:
                current_metrics = env.get_metrics()
                print(f"   Step {step}: Coverage {current_metrics['efficiency_score']:.1f}%, Energy {current_metrics['energy_consumed']:.1f}")
        
        # Episode summary
        final_metrics = env.get_metrics()
        trained_data.extend(episode_data)
        trained_metrics.append(final_metrics)
        
        print(f"   ✅ Episode {episode + 1} Results:")
        print(f"      📈 Coverage: {final_metrics['efficiency_score']:.1f}%")
        print(f"      📊 Data Points: {len(episode_data)}")
        print(f"      ⚡ Energy: {final_metrics['energy_consumed']:.1f}")
        print(f"      💥 Collisions: {final_metrics['collision_count']}")
    
    print(f"\n📈 Trained Summary:")
    print(f"   Total Data Points: {len(trained_data)}")
    print(f"   Avg Coverage: {np.mean([m['efficiency_score'] for m in trained_metrics]):.1f}%")
    print(f"   Avg Energy: {np.mean([m['energy_consumed'] for m in trained_metrics]):.1f}")
    
    return trained_data, trained_metrics

def analyze_performance(baseline_metrics, trained_metrics, baseline_data, trained_data):
    """Analyze and compare performance"""
    print("\n📊 Performance Analysis")
    print("=" * 60)
    
    # Calculate averages
    baseline_avg_coverage = np.mean([m['efficiency_score'] for m in baseline_metrics])
    trained_avg_coverage = np.mean([m['efficiency_score'] for m in trained_metrics])
    
    baseline_avg_energy = np.mean([m['energy_consumed'] for m in baseline_metrics])
    trained_avg_energy = np.mean([m['energy_consumed'] for m in trained_metrics])
    
    baseline_avg_collisions = np.mean([m['collision_count'] for m in baseline_metrics])
    trained_avg_collisions = np.mean([m['collision_count'] for m in trained_metrics])
    
    # Calculate improvements
    coverage_improvement = ((trained_avg_coverage - baseline_avg_coverage) / baseline_avg_coverage) * 100
    energy_improvement = ((baseline_avg_energy - trained_avg_energy) / baseline_avg_energy) * 100
    collision_improvement = ((baseline_avg_collisions - trained_avg_collisions) / max(baseline_avg_collisions, 0.01)) * 100
    data_improvement = len(trained_data) - len(baseline_data)
    
    print(f"📈 COVERAGE EFFICIENCY:")
    print(f"   Baseline:    {baseline_avg_coverage:.1f}%")
    print(f"   Trained:     {trained_avg_coverage:.1f}%")
    print(f"   Improvement: {coverage_improvement:+.1f}%")
    
    print(f"\n⚡ ENERGY EFFICIENCY:")
    print(f"   Baseline:    {baseline_avg_energy:.1f}")
    print(f"   Trained:     {trained_avg_energy:.1f}")
    print(f"   Improvement: {energy_improvement:+.1f}%")
    
    print(f"\n💥 COLLISION AVOIDANCE:")
    print(f"   Baseline:    {baseline_avg_collisions:.1f}")
    print(f"   Trained:     {trained_avg_collisions:.1f}")
    print(f"   Improvement: {collision_improvement:+.1f}%")
    
    print(f"\n📊 DATA COLLECTION:")
    print(f"   Baseline:    {len(baseline_data)} points")
    print(f"   Trained:     {len(trained_data)} points")
    print(f"   Improvement: {data_improvement:+d} points")
    
    # Overall score
    overall_improvement = (coverage_improvement + energy_improvement + collision_improvement) / 3
    print(f"\n🏆 OVERALL PERFORMANCE IMPROVEMENT: {overall_improvement:+.1f}%")
    
    return {
        'coverage_improvement': coverage_improvement,
        'energy_improvement': energy_improvement,
        'collision_improvement': collision_improvement,
        'data_improvement': data_improvement,
        'overall_improvement': overall_improvement
    }

def create_visualizations(env, model):
    """Create visualizations"""
    print("\n🎬 Creating Visualizations...")
    print("=" * 60)
    
    # Generate simulation data
    print("🚁 Generating simulation data...")
    simulation_data = []
    
    obs = env.reset()
    done = False
    step = 0
    
    while not done and step < 40:  # Limit for demo
        actions, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(actions)
        
        render_data = env.render(mode='dict')
        
        step_data = {
            'episode': 0,
            'step': step,
            'uav_positions': render_data['uav_positions'].tolist(),
            'visited_cells': render_data['visited'].tolist(),
            'battery_levels': render_data['battery_levels'].tolist(),
            'metrics': render_data['metrics']
        }
        
        simulation_data.append(step_data)
        step += 1
    
    # Create HTML visualization
    print("🌐 Creating HTML visualization...")
    visualizer = RealTimeVisualizer(grid_size=10, num_uavs=3)
    html_file = visualizer.create_html_visualization(simulation_data)
    
    # Create 3D visualization
    print("🎭 Creating 3D visualization...")
    try:
        visualizer_3d = UAV3DVisualizer(grid_size=10, num_uavs=3)
        paths, altitudes, visited = visualizer_3d.create_3d_paths(env, model, num_episodes=1)
        visualizer_3d.create_static_3d_plot(paths, altitudes)
        visualizer_3d.create_altitude_profile(altitudes)
        print("   ✅ 3D plots created successfully")
    except Exception as e:
        print(f"   ⚠️ 3D visualization failed: {str(e)}")
    
    print(f"\n✅ Visualizations Created:")
    print(f"   🌐 HTML: {html_file}")
    print(f"   📊 3D Plots: ./data/uav_3d_paths.png")
    print(f"   📈 Altitude: ./data/uav_altitude_profile.png")
    
    return html_file

def save_data_files(baseline_data, trained_data, performance_results):
    """Save all data files"""
    print("\n💾 Saving Data Files...")
    print("=" * 60)
    
    # Save CSV files
    csv_logger = CSVDataLogger()
    csv_logger.save_baseline_data(baseline_data)
    csv_logger.save_trained_data(trained_data)
    
    # Save performance comparison
    comparison_df = pd.DataFrame({
        'Metric': ['Coverage (%)', 'Energy Efficiency (%)', 'Collision Reduction (%)', 'Data Points'],
        'Improvement': [
            performance_results['coverage_improvement'],
            performance_results['energy_improvement'],
            performance_results['collision_improvement'],
            performance_results['data_improvement']
        ]
    })
    comparison_df.to_csv('./data/performance_comparison.csv', index=False)
    
    # Save summary report
    summary = {
        'timestamp': datetime.now().isoformat(),
        'system_status': 'OPERATIONAL',
        'performance_results': performance_results,
        'files_generated': [
            './data/baseline_data.csv',
            './data/trained_data.csv',
            './data/performance_comparison.csv',
            './data/real_time_visualization.html',
            './models/ppo_MultiUAVEnv.zip'
        ]
    }
    
    with open('./data/system_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("✅ Data files saved successfully:")
    print("   📊 ./data/baseline_data.csv")
    print("   📊 ./data/trained_data.csv")
    print("   📊 ./data/performance_comparison.csv")
    print("   📋 ./data/system_summary.json")

def main():
    """Main execution function"""
    print("🚁 Multi-UAV Real-Time Path Planning System")
    print("=" * 60)
    print("🎯 Starting Complete System Demo...")
    
    start_time = time.time()
    
    # Configuration
    CONFIG = {
        'grid_size': 10,
        'num_uavs': 3,
        'max_steps': 200,
        'training_timesteps': 50000,
    }
    
    print(f"\n⚙️ Configuration:")
    print(f"   Grid Size: {CONFIG['grid_size']}x{CONFIG['grid_size']}")
    print(f"   UAVs: {CONFIG['num_uavs']}")
    print(f"   Max Steps: {CONFIG['max_steps']}")
    print(f"   Training Steps: {CONFIG['training_timesteps']}")
    
    # Initialize environment
    env = MultiUAVEnv(
        grid_size=CONFIG['grid_size'],
        num_uavs=CONFIG['num_uavs'],
        max_steps=CONFIG['max_steps']
    )
    
    try:
        # 1. Collect baseline data
        baseline_data, baseline_metrics = collect_baseline_data(env)
        
        # 2. Train model
        model, training_time = train_model(env)
        
        # 3. Collect trained data
        trained_data, trained_metrics = collect_trained_data(env, model)
        
        # 4. Analyze performance
        performance_results = analyze_performance(
            baseline_metrics, trained_metrics, 
            baseline_data, trained_data
        )
        
        # 5. Create visualizations
        html_file = create_visualizations(env, model)
        
        # 6. Save data files
        save_data_files(baseline_data, trained_data, performance_results)
        
        # Final summary
        total_time = time.time() - start_time
        
        print(f"\n🎉 SYSTEM DEMO COMPLETE!")
        print("=" * 60)
        print(f"✅ Status: OPERATIONAL")
        print(f"⏱️ Total Time: {total_time:.2f} seconds")
        print(f"🏆 Overall Improvement: {performance_results['overall_improvement']:+.1f}%")
        print(f"🌐 Visualization: {html_file}")
        print(f"📊 Data Files: ./data/")
        print(f"🤖 Model: ./models/ppo_MultiUAVEnv.zip")
        
        print(f"\n🎯 Next Steps:")
        print(f"   1. Open {html_file} in your browser")
        print(f"   2. Review CSV files in ./data/ directory")
        print(f"   3. Analyze performance metrics")
        print(f"   4. Run real-time simulations")
        
        print(f"\n🚁 Multi-UAV System Ready for Operation!")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()