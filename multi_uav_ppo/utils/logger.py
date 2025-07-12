import json
import csv
import os
import time
from datetime import datetime
import numpy as np
import pandas as pd

class UAVLogger:
    def __init__(self, log_dir="../data/logs"):
        self.log_dir = log_dir
        self.ensure_log_dir()
        
        # Initialize log files
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.training_log = []
        self.episode_log = []
        self.metrics_log = []
        
        # Real-time data tracking
        self.current_episode = 0
        self.start_time = time.time()
        
    def ensure_log_dir(self):
        """Ensure log directory exists"""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
    
    def log_training_start(self, config):
        """Log training configuration and start"""
        log_entry = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'event': 'training_start',
            'config': config
        }
        
        self.training_log.append(log_entry)
        print(f"Training started - Session ID: {self.session_id}")
    
    def log_episode_start(self, episode_num, total_episodes):
        """Log episode start"""
        self.current_episode = episode_num
        
        log_entry = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'event': 'episode_start',
            'episode': episode_num,
            'total_episodes': total_episodes
        }
        
        self.episode_log.append(log_entry)
    
    def log_episode_end(self, episode_num, metrics, reward, steps):
        """Log episode completion with metrics"""
        log_entry = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'event': 'episode_end',
            'episode': episode_num,
            'total_reward': reward,
            'steps': steps,
            'metrics': metrics,
            'duration': time.time() - self.start_time
        }
        
        self.episode_log.append(log_entry)
        self.metrics_log.append({
            'episode': episode_num,
            'reward': reward,
            'steps': steps,
            **metrics
        })
    
    def log_step(self, episode, step, action, reward, metrics):
        """Log individual step data"""
        log_entry = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'event': 'step',
            'episode': episode,
            'step': step,
            'action': action.tolist() if hasattr(action, 'tolist') else action,
            'reward': reward,
            'metrics': metrics
        }
        
        # Only log every 10th step to avoid too much data
        if step % 10 == 0:
            self.training_log.append(log_entry)
    
    def log_training_end(self, total_episodes, final_metrics):
        """Log training completion"""
        training_duration = time.time() - self.start_time
        
        log_entry = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'event': 'training_end',
            'total_episodes': total_episodes,
            'duration': training_duration,
            'final_metrics': final_metrics
        }
        
        self.training_log.append(log_entry)
        print(f"Training completed - Duration: {training_duration:.2f}s")
    
    def log_data_collection(self, data_points, collection_type="real_time"):
        """Log data collection events"""
        log_entry = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'event': 'data_collection',
            'collection_type': collection_type,
            'data_points': len(data_points),
            'sample_data': data_points[:3] if len(data_points) > 3 else data_points
        }
        
        self.training_log.append(log_entry)
    
    def save_logs(self):
        """Save all logs to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save training log
        training_file = os.path.join(self.log_dir, f"training_log_{timestamp}.json")
        with open(training_file, 'w') as f:
            json.dump(self.training_log, f, indent=2)
        
        # Save episode log
        episode_file = os.path.join(self.log_dir, f"episode_log_{timestamp}.json")
        with open(episode_file, 'w') as f:
            json.dump(self.episode_log, f, indent=2)
        
        # Save metrics as CSV
        if self.metrics_log:
            metrics_file = os.path.join(self.log_dir, f"metrics_{timestamp}.csv")
            df = pd.DataFrame(self.metrics_log)
            df.to_csv(metrics_file, index=False)
        
        print(f"Logs saved to {self.log_dir}")
        return training_file, episode_file
    
    def get_training_summary(self):
        """Get training summary statistics"""
        if not self.metrics_log:
            return None
        
        df = pd.DataFrame(self.metrics_log)
        
        summary = {
            'total_episodes': len(self.metrics_log),
            'avg_reward': df['reward'].mean(),
            'max_reward': df['reward'].max(),
            'min_reward': df['reward'].min(),
            'avg_steps': df['steps'].mean(),
            'avg_coverage': df['efficiency_score'].mean() if 'efficiency_score' in df.columns else 0,
            'avg_energy': df['energy_consumed'].mean() if 'energy_consumed' in df.columns else 0,
            'collision_rate': df['collision_count'].mean() if 'collision_count' in df.columns else 0
        }
        
        return summary
    
    def create_real_time_report(self, current_metrics):
        """Create real-time progress report"""
        report = {
            'session_id': self.session_id,
            'current_episode': self.current_episode,
            'elapsed_time': time.time() - self.start_time,
            'current_metrics': current_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        return report

class CSVDataLogger:
    def __init__(self, output_dir="../data"):
        self.output_dir = output_dir
        self.ensure_output_dir()
        
    def ensure_output_dir(self):
        """Ensure output directory exists"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def save_baseline_data(self, data_points):
        """Save baseline data before training"""
        filename = os.path.join(self.output_dir, "baseline_data.csv")
        
        if data_points:
            df = pd.DataFrame(data_points)
            df.to_csv(filename, index=False)
            print(f"Baseline data saved: {len(data_points)} points")
        else:
            # Create empty file with headers
            headers = ['ID', 'source_time', 'lat', 'lon', 'source_spotted', 
                      'track', 'takeoff_landing_time', 'battery', 'AoI', 'FLA']
            df = pd.DataFrame(columns=headers)
            df.to_csv(filename, index=False)
            print("Empty baseline data file created")
    
    def save_trained_data(self, data_points):
        """Save data after training"""
        filename = os.path.join(self.output_dir, "trained_data.csv")
        
        if data_points:
            df = pd.DataFrame(data_points)
            df.to_csv(filename, index=False)
            print(f"Trained data saved: {len(data_points)} points")
        else:
            # Create empty file with headers
            headers = ['ID', 'source_time', 'lat', 'lon', 'source_spotted', 
                      'track', 'takeoff_landing_time', 'battery', 'AoI', 'FLA']
            df = pd.DataFrame(columns=headers)
            df.to_csv(filename, index=False)
            print("Empty trained data file created")
    
    def compare_datasets(self, baseline_data, trained_data):
        """Compare baseline and trained datasets"""
        comparison = {
            'baseline_points': len(baseline_data),
            'trained_points': len(trained_data),
            'improvement': len(trained_data) - len(baseline_data),
            'improvement_percentage': ((len(trained_data) - len(baseline_data)) / len(baseline_data) * 100) if baseline_data else 0
        }
        
        if baseline_data and trained_data:
            # Calculate average metrics
            baseline_df = pd.DataFrame(baseline_data)
            trained_df = pd.DataFrame(trained_data)
            
            if 'battery' in baseline_df.columns and 'battery' in trained_df.columns:
                comparison['baseline_avg_battery'] = baseline_df['battery'].mean()
                comparison['trained_avg_battery'] = trained_df['battery'].mean()
                comparison['battery_improvement'] = trained_df['battery'].mean() - baseline_df['battery'].mean()
            
            if 'AoI' in baseline_df.columns and 'AoI' in trained_df.columns:
                comparison['baseline_avg_aoi'] = baseline_df['AoI'].mean()
                comparison['trained_avg_aoi'] = trained_df['AoI'].mean()
                comparison['aoi_improvement'] = trained_df['AoI'].mean() - baseline_df['AoI'].mean()
        
        # Save comparison
        comparison_file = os.path.join(self.output_dir, "dataset_comparison.json")
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        return comparison
    
    def generate_external_obs_csv(self, num_entries=10):
        """Generate external observations CSV file"""
        filename = os.path.join(self.output_dir, "external_obs.csv")
        
        # Sample data based on your format
        data = []
        for i in range(num_entries):
            entry = {
                'ID': f'UAV{i % 3}_{i}',
                'source_time': i + 1,
                'lat': 52.815992 + np.random.normal(0, 0.001),
                'lon': -4.131736 + np.random.normal(0, 0.001),
                'source_spotted': np.random.choice([0, 1]),
                'track': np.random.randint(-90, 91),
                'takeoff_landing_time': np.random.randint(10, 30),
                'battery': np.random.randint(85, 100),
                'AoI': np.random.randint(0, 6),
                'FLA': np.random.randint(1, 6)
            }
            data.append(entry)
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"External observations CSV generated: {filename}")
        return filename

def main():
    # Demo usage
    logger = UAVLogger()
    csv_logger = CSVDataLogger()
    
    # Simulate training logging
    config = {
        'algorithm': 'PPO',
        'grid_size': 10,
        'num_uavs': 3,
        'learning_rate': 0.0003,
        'total_timesteps': 50000
    }
    
    logger.log_training_start(config)
    
    # Simulate some episodes
    for episode in range(5):
        logger.log_episode_start(episode, 5)
        
        # Simulate episode metrics
        metrics = {
            'efficiency_score': np.random.uniform(60, 95),
            'energy_consumed': np.random.uniform(30, 80),
            'collision_count': np.random.randint(0, 3),
            'mission_time': np.random.randint(50, 100)
        }
        
        reward = np.random.uniform(100, 500)
        steps = np.random.randint(50, 100)
        
        logger.log_episode_end(episode, metrics, reward, steps)
    
    final_metrics = {
        'final_avg_coverage': 87.5,
        'final_avg_energy': 45.2,
        'final_collision_rate': 0.8
    }
    
    logger.log_training_end(5, final_metrics)
    
    # Save logs
    logger.save_logs()
    
    # Generate external observations
    csv_logger.generate_external_obs_csv()
    
    # Get training summary
    summary = logger.get_training_summary()
    print("\nTraining Summary:")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()