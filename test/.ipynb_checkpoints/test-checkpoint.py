import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from stable_baselines3 import PPO
from env.uav_env import RealTimeMultiUAVEnv
from utils.metrics import MetricsTracker
import numpy as np
import time

def test_model(model_path="../models/ppo_MultiUAVEnv", episodes=5):
    """Test the trained model"""
    print("Loading trained model...")
    
    try:
        model = PPO.load(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Create test environment
    env = RealTimeMultiUAVEnv(grid_size=10, num_uavs=3)
    metrics = MetricsTracker()
    
    print(f"Running {episodes} test episodes...")
    
    for episode in range(episodes):
        obs = env.reset()
        done = False
        step_count = 0
        total_reward = 0
        
        print(f"\n--- Episode {episode + 1} ---")
        
        while not done and step_count < 500:
            # Get action from trained model
            action, _ = model.predict(obs, deterministic=True)
            
            # Execute action
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            # Render every 10 steps
            if step_count % 10 == 0:
                env.render()
                time.sleep(0.1)  # Small delay for visualization
        
        # Log episode metrics
        coverage = info.get('coverage_ratio', 0)
        active_uavs = info.get('active_uavs', 0)
        total_spots = info.get('total_spots', 0)
        
        metrics.log_episode(
            episode=episode + 1,
            coverage=coverage,
            steps=step_count,
            active_uavs=active_uavs,
            total_spots=total_spots,
            total_reward=total_reward
        )
        
        print(f"Episode {episode + 1} Results:")
        print(f"  Coverage: {coverage:.2f}")
        print(f"  Steps: {step_count}")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Active UAVs: {active_uavs}")
        print(f"  Total Spots: {total_spots}")
        
        # Generate CSV for this episode
        csv_data = env.generate_csv_data(f"../data/test_episode_{episode + 1}.csv")
        print(f"  CSV data saved with {len(csv_data)} entries")
    
    # Save overall test metrics
    metrics.save_metrics("../data/test_metrics.csv")
    
    print(f"\nTest completed! Results saved to test_metrics.csv")
    return metrics

def test_real_time_performance():
    """Test real-time performance of the model"""
    print("Testing real-time performance...")
    
    env = RealTimeMultiUAVEnv(grid_size=8, num_uavs=2)
    
    try:
        model = PPO.load("../models/ppo_MultiUAVEnv")
    except:
        print("No trained model found. Please train first.")
        return
    
    obs = env.reset()
    done = False
    step_count = 0
    
    start_time = time.time()
    
    while not done and step_count < 100:
        step_start = time.time()
        
        # Get action (this should be fast)
        action, _ = model.predict(obs, deterministic=True)
        
        # Execute action
        obs, reward, done, info = env.step(action)
        
        step_time = time.time() - step_start
        
        if step_count % 10 == 0:
            print(f"Step {step_count}: {step_time:.4f}s - Coverage: {info.get('coverage_ratio', 0):.2f}")
        
        step_count += 1
        
        # Simulate real-time constraint (10 Hz)
        time.sleep(max(0, 0.1 - step_time))
    
    total_time = time.time() - start_time
    print(f"Real-time test completed in {total_time:.2f}s")
    print(f"Average step time: {total_time/step_count:.4f}s")

if __name__ == "__main__":
    # Run standard test
    test_metrics = test_model()
    
    # Run real-time performance test
    test_real_time_performance()