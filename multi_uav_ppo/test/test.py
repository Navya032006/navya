import sys
sys.path.append('../')

from stable_baselines3 import PPO
from env.uav_env import MultiUAVEnv
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime

class RealTimeSimulator:
    def __init__(self, model_path="../models/ppo_MultiUAVEnv"):
        self.env = MultiUAVEnv(grid_size=10, num_uavs=3, max_steps=200)
        self.model = PPO.load(model_path)
        self.simulation_data = []
        self.real_time_metrics = []
        
    def run_real_time_simulation(self, num_episodes=5, save_data=True):
        """Run real-time simulation and collect data"""
        print("Starting real-time UAV simulation...")
        
        for episode in range(num_episodes):
            print(f"\n--- Episode {episode + 1} ---")
            obs = self.env.reset()
            done = False
            episode_data = []
            step_count = 0
            
            while not done:
                # Real-time decision making
                actions, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(actions)
                
                # Collect real-time data
                current_data = self.env.get_collected_data()
                render_data = self.env.render(mode='dict')
                
                # Store step data for visualization
                step_data = {
                    'episode': episode,
                    'step': step_count,
                    'uav_positions': render_data['uav_positions'].tolist(),
                    'visited_cells': render_data['visited'].tolist(),
                    'battery_levels': render_data['battery_levels'].tolist(),
                    'metrics': render_data['metrics'],
                    'timestamp': datetime.now().isoformat()
                }
                
                self.simulation_data.append(step_data)
                episode_data = current_data.copy()
                
                # Real-time display
                print(f"Step {step_count}: Coverage {render_data['metrics']['efficiency_score']:.1f}%")
                
                step_count += 1
                time.sleep(0.1)  # Real-time simulation delay
            
            # Episode metrics
            final_metrics = self.env.get_metrics()
            self.real_time_metrics.append(final_metrics)
            
            print(f"Episode {episode + 1} completed:")
            print(f"  Final Coverage: {final_metrics['efficiency_score']:.1f}%")
            print(f"  Data Points: {len(episode_data)}")
            print(f"  Mission Time: {final_metrics['mission_time']} steps")
            print(f"  Energy Used: {final_metrics['energy_consumed']:.1f}")
        
        if save_data:
            self.save_simulation_data()
        
        return self.simulation_data, self.real_time_metrics
    
    def save_simulation_data(self):
        """Save simulation data to files"""
        # Save step-by-step data
        df_steps = pd.DataFrame(self.simulation_data)
        df_steps.to_csv('../data/real_time_simulation.csv', index=False)
        
        # Save metrics
        df_metrics = pd.DataFrame(self.real_time_metrics)
        df_metrics.to_csv('../data/real_time_metrics.csv', index=False)
        
        print("Real-time simulation data saved successfully")
    
    def generate_performance_report(self):
        """Generate performance report"""
        if self.real_time_metrics:
            report = {
                'total_episodes': len(self.real_time_metrics),
                'avg_coverage': np.mean([m['efficiency_score'] for m in self.real_time_metrics]),
                'avg_mission_time': np.mean([m['mission_time'] for m in self.real_time_metrics]),
                'avg_energy_consumption': np.mean([m['energy_consumed'] for m in self.real_time_metrics]),
                'collision_rate': np.mean([m['collision_count'] for m in self.real_time_metrics]),
                'success_rate': len([m for m in self.real_time_metrics if m['efficiency_score'] > 80]) / len(self.real_time_metrics) * 100
            }
            
            # Save report
            with open('../data/performance_report.json', 'w') as f:
                json.dump(report, f, indent=2)
            
            print("\n=== PERFORMANCE REPORT ===")
            print(f"Average Coverage: {report['avg_coverage']:.1f}%")
            print(f"Average Mission Time: {report['avg_mission_time']:.1f} steps")
            print(f"Average Energy Consumption: {report['avg_energy_consumption']:.1f}")
            print(f"Success Rate (>80% coverage): {report['success_rate']:.1f}%")
            
            return report
        return None

def evaluate_model_performance(model_path="../models/ppo_MultiUAVEnv", num_episodes=10):
    """Evaluate trained model performance"""
    print("Evaluating trained model...")
    
    env = MultiUAVEnv(grid_size=10, num_uavs=3, max_steps=200)
    model = PPO.load(model_path)
    
    total_rewards = []
    coverage_scores = []
    mission_times = []
    energy_consumptions = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            actions, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(actions)
            episode_reward += reward
        
        metrics = env.get_metrics()
        total_rewards.append(episode_reward)
        coverage_scores.append(metrics['efficiency_score'])
        mission_times.append(metrics['mission_time'])
        energy_consumptions.append(metrics['energy_consumed'])
    
    # Calculate statistics
    evaluation_results = {
        'avg_reward': np.mean(total_rewards),
        'avg_coverage': np.mean(coverage_scores),
        'avg_mission_time': np.mean(mission_times),
        'avg_energy': np.mean(energy_consumptions),
        'std_coverage': np.std(coverage_scores),
        'success_rate': len([c for c in coverage_scores if c > 80]) / len(coverage_scores) * 100
    }
    
    print(f"\n=== MODEL EVALUATION ===")
    print(f"Episodes: {num_episodes}")
    print(f"Average Reward: {evaluation_results['avg_reward']:.2f}")
    print(f"Average Coverage: {evaluation_results['avg_coverage']:.1f}% (±{evaluation_results['std_coverage']:.1f})")
    print(f"Average Mission Time: {evaluation_results['avg_mission_time']:.1f} steps")
    print(f"Success Rate: {evaluation_results['success_rate']:.1f}%")
    
    # Save evaluation results
    with open('../data/model_evaluation.json', 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    return evaluation_results

def main():
    print("=== UAV Multi-Agent Testing ===")
    
    # 1. Evaluate model performance
    try:
        eval_results = evaluate_model_performance(num_episodes=5)
    except Exception as e:
        print(f"Model evaluation failed: {e}")
        print("Please run training first!")
        return
    
    # 2. Run real-time simulation
    simulator = RealTimeSimulator()
    simulation_data, metrics = simulator.run_real_time_simulation(num_episodes=3)
    
    # 3. Generate performance report
    report = simulator.generate_performance_report()
    
    # 4. Create summary comparison
    if report:
        print("\n=== FINAL SUMMARY ===")
        print(f"Model Performance: {eval_results['avg_coverage']:.1f}% coverage")
        print(f"Real-time Performance: {report['avg_coverage']:.1f}% coverage")
        print(f"Total Steps Simulated: {len(simulation_data)}")
        print(f"Data Files Generated: 4")
        print("\nFiles saved to ../data/:")
        print("- real_time_simulation.csv")
        print("- real_time_metrics.csv")
        print("- performance_report.json")
        print("- model_evaluation.json")

if __name__ == "__main__":
    main()