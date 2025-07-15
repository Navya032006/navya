# import sys
# import os
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.callbacks import BaseCallback
# from env.uav_env import RealTimeMultiUAVEnv
# from utils.logger import TrainingLogger
# from utils.metrics import MetricsTracker
# import numpy as np

# class TrainingCallback(BaseCallback):
#     def __init__(self, verbose=0):
#         super(TrainingCallback, self).__init__(verbose)
#         self.episode_count = 0
#         self.metrics = MetricsTracker()
        
#     def _on_step(self) -> bool:
#         if self.locals.get('dones', [False])[0]:
#             self.episode_count += 1
            
#             # Log metrics every 10 episodes
#             if self.episode_count % 10 == 0:
#                 env = self.training_env.get_attr('unwrapped')[0]
#                 coverage = np.sum(env.visited) / (env.grid_size ** 2)
                
#                 self.metrics.log_episode(
#                     episode=self.episode_count,
#                     coverage=coverage,
#                     steps=env.step_count,
#                     active_uavs=sum(1 for b in env.uav_batteries if b > 0),
#                     total_spots=sum(env.uav_spotted)
#                 )
                
#                 if self.verbose:
#                     print(f"Episode {self.episode_count}: Coverage={coverage:.2f}, Steps={env.step_count}")
        
#         return True

# def train_model(total_timesteps=50000, model_name="ppo_MultiUAVEnv"):
#     """Train the multi-UAV model"""
#     print("Initializing training environment...")
    
#     # Create environment
#     env = RealTimeMultiUAVEnv(grid_size=10, num_uavs=3)
    
#     # Generate pre-training CSV
#     print("Generating pre-training data...")
#     env.reset()
#     pre_training_data = env.generate_csv_data("../data/pre_training_obs.csv")
#     print(f"Pre-training data saved with {len(pre_training_data)} UAVs")
    
#     # Initialize model
#     print("Creating PPO model...")
#     model = PPO(
#         "MlpPolicy",
#         env,
#         verbose=1,
#         learning_rate=0.0003,
#         n_steps=2048,
#         batch_size=64,
#         n_epochs=10,
#         gamma=0.99,
#         gae_lambda=0.95,
#         clip_range=0.2,
#         tensorboard_log="../logs/"
#     )
    
#     # Create callback
#     callback = TrainingCallback(verbose=1)
    
#     # Train model
#     print(f"Starting training for {total_timesteps} timesteps...")
#     model.learn(
#         total_timesteps=total_timesteps,
#         callback=callback,
#         tb_log_name="MultiUAV_PPO"
#     )
    
#     # Save model
#     model_path = f"../models/{model_name}"
#     model.save(model_path)
#     print(f"Model saved to {model_path}")
    
#     # Generate post-training CSV with trained model
#     print("Generating post-training data...")
#     obs = env.reset()
#     done = False
#     step_count = 0
    
#     while not done and step_count < 200:  # Limited run for data generation
#         action, _ = model.predict(obs, deterministic=True)
#         obs, reward, done, info = env.step(action)
#         step_count += 1
    
#     post_training_data = env.generate_csv_data("../data/post_training_obs.csv")
#     print(f"Post-training data saved with {len(post_training_data)} UAVs")
    
#     # Save metrics
#     callback.metrics.save_metrics("../data/training_metrics.csv")
    
#     print("Training completed successfully!")
#     return model, callback.metrics

# if __name__ == "__main__":
#     model, metrics = train_model()
#     print("Training session finished.")

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from env.uav_env import RealTimeMultiUAVEnv
from utils.metrics import MetricsTracker
import numpy as np
import pandas as pd

# ===== Custom Callback for Metrics Logging =====
class TrainingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
        self.episode_count = 0
        self.metrics = MetricsTracker()

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        if dones is not None and any(dones):
            self.episode_count += 1
            env = self.training_env.envs[0]  # first (and only) env
            coverage = np.sum(env.visited) / (env.grid_size ** 2)
            if self.episode_count % 10 == 0:
                self.metrics.log_episode(
                    episode=self.episode_count,
                    coverage=coverage,
                    steps=env.step_count,
                    active_uavs=sum(1 for b in env.uav_batteries if b > 0),
                    total_spots=sum(env.uav_spotted)
                )
                if self.verbose:
                    print(f"Episode {self.episode_count}: Coverage={coverage:.2f}, Steps={env.step_count}")
        return True

# ===== Main Training Function =====
def train_model(total_timesteps=50000, model_name="ppo_MultiUAVEnv"):
    print("\n🚀 Initializing UAV training environment...")
    os.makedirs("../data", exist_ok=True)
    os.makedirs("../models", exist_ok=True)
    os.makedirs("../logs", exist_ok=True)

    # Use vectorized env
    vec_env = make_vec_env(lambda: RealTimeMultiUAVEnv(grid_size=10, num_uavs=3), n_envs=1)
    raw_env = vec_env.envs[0]

    # Pre-training CSV
    print("📦 Generating pre-training data...")
    raw_env.reset()
    pre_training_data = raw_env.generate_csv_data("../data/pre_training_obs.csv")
    print(pre_training_data.head())

    print("🧠 Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log="../logs/"
    )

    callback = TrainingCallback(verbose=1)

    print(f"🎯 Starting training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=callback, tb_log_name="MultiUAV_PPO")

    model_path = f"../models/{model_name}"
    model.save(model_path)
    print(f"✅ Model saved to {model_path}.zip")

    # ===== Post-training CSV Generation =====
    print("📈 Generating post-training data...")
    obs = vec_env.reset()
    done = False
    steps = 0

    while not done and steps < 200:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        steps += 1

    post_training_data = raw_env.generate_csv_data("../data/post_training_obs.csv")
    print(post_training_data.head())

    # ===== Save Metrics =====
    print("📊 Saving training metrics...")
    callback.metrics.save_metrics("../data/training_metrics.csv")

    print("🏁 Training completed successfully!")
    return model, callback.metrics

# Run
if __name__ == "__main__":
    model, metrics = train_model()
    print("🎉 All tasks complete!")
