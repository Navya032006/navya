import sys
sys.path.append('../')

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from env.uav_env import MultiUAVEnv
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime

class MetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(MetricsCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.coverage_scores = []
        
    def _on_step(self) -> bool:
        # Collect metrics during training
        if 'metrics' in self.locals['infos'][0]:
            metrics = self.locals['infos'][0]['metrics']
            self.coverage_scores.append(metrics['efficiency_score'])
        return True

def collect_baseline_data(env, num_episodes=5):
    """Collect baseline data before training"""
    print("Collecting baseline data (before training)...")
    baseline_data = []
    baseline_metrics = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_data = []
        
        while not done:
            # Random actions (untrained behavior)
            actions = [env.action_space.sample() for _ in range(env.num_uavs)]
            obs, reward, done, info = env.step(actions)
            
            # Collect data points
            current_data = env.get_collected_data()
            if len(current_data) > len(episode_data):
                episode_data = current_data.copy()
        
        baseline_data.extend(episode_data)
        baseline_metrics.append(env.get_metrics())
    
    return baseline_data, baseline_metrics

def collect_trained_data(env, model, num_episodes=5):
    """Collect data after training"""
    print("Collecting trained data (after training)...")
    trained_data = []
    trained_metrics = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_data = []
        
        while not done:
            # Use trained model
            actions, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(actions)
            
            # Collect data points
            current_data = env.get_collected_data()
            if len(current_data) > len(episode_data):
                episode_data = current_data.copy()
        
        trained_data.extend(episode_data)
        trained_metrics.append(env.get_metrics())
    
    return trained_data, trained_metrics

def save_data_to_csv(data, filename):
    """Save collected data to CSV"""
    if data:
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
    else:
        print(f"No data to save to {filename}")

def save_metrics_comparison(baseline_metrics, trained_metrics, filename):
    """Save metrics comparison"""
    comparison = {
        'metric': [],
        'baseline_avg': [],
        'trained_avg': [],
        'improvement': []
    }
    
    if baseline_metrics and trained_metrics:
        for key in baseline_metrics[0].keys():
            baseline_avg = np.mean([m[key] for m in baseline_metrics])
            trained_avg = np.mean([m[key] for m in trained_metrics])
            improvement = ((trained_avg - baseline_avg) / baseline_avg * 100) if baseline_avg > 0 else 0
            
            comparison['metric'].append(key)
            comparison['baseline_avg'].append(baseline_avg)
            comparison['trained_avg'].append(trained_avg)
            comparison['improvement'].append(improvement)
    
    df = pd.DataFrame(comparison)
    df.to_csv(filename, index=False)
    print(f"Metrics comparison saved to {filename}")

def main():
    # Initialize environment
    env = MultiUAVEnv(grid_size=10, num_uavs=3, max_steps=200)
    
    # Collect baseline data (before training)
    baseline_data, baseline_metrics = collect_baseline_data(env, num_episodes=3)
    
    # Save baseline data
    save_data_to_csv(baseline_data, '../data/baseline_data.csv')
    
    # Initialize PPO model
    print("Initializing PPO model...")
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )
    
    # Create callback
    callback = MetricsCallback()
    
    # Train the model
    print("Starting training...")
    start_time = time.time()
    
    model.learn(
        total_timesteps=50000,
        callback=callback,
        progress_bar=True
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Save trained model
    model.save("../models/ppo_MultiUAVEnv")
    print("Model saved successfully")
    
    # Collect trained data (after training)
    trained_data, trained_metrics = collect_trained_data(env, model, num_episodes=3)
    
    # Save trained data
    save_data_to_csv(trained_data, '../data/trained_data.csv')
    
    # Save metrics comparison
    save_metrics_comparison(baseline_metrics, trained_metrics, '../data/metrics_comparison.csv')
    
    # Print summary
    print("\n=== TRAINING SUMMARY ===")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Baseline Data Points: {len(baseline_data)}")
    print(f"Trained Data Points: {len(trained_data)}")
    
    if baseline_metrics and trained_metrics:
        print("\n=== PERFORMANCE IMPROVEMENT ===")
        for key in baseline_metrics[0].keys():
            baseline_avg = np.mean([m[key] for m in baseline_metrics])
            trained_avg = np.mean([m[key] for m in trained_metrics])
            improvement = ((trained_avg - baseline_avg) / baseline_avg * 100) if baseline_avg > 0 else 0
            print(f"{key}: {baseline_avg:.2f} -> {trained_avg:.2f} ({improvement:+.1f}%)")

if __name__ == "__main__":
    main()