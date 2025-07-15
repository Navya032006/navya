import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os

class MetricsTracker:
    def __init__(self):
        self.episodes = []
        self.current_episode = {
            'episode': 0,
            'steps': [],
            'rewards': [],
            'coverage': [],
            'battery_levels': [],
            'spots_found': [],
            'active_uavs': []
        }
        
        self.summary_metrics = {
            'total_episodes': 0,
            'avg_coverage': 0,
            'avg_steps': 0,
            'avg_reward': 0,
            'success_rate': 0,
            'efficiency_score': 0
        }
    
    def start_episode(self, episode_num):
        """Start tracking a new episode"""
        if self.current_episode['episode'] > 0:
            self.episodes.append(self.current_episode.copy())
        
        self.current_episode = {
            'episode': episode_num,
            'steps': [],
            'rewards': [],
            'coverage': [],
            'battery_levels': [],
            'spots_found': [],
            'active_uavs': [],
            'start_time': datetime.now().isoformat()
        }
    
    def log_step(self, step, reward, coverage, battery_levels, spots_found, active_uavs):
        """Log metrics for a single step"""
        self.current_episode['steps'].append(step)
        self.current_episode['rewards'].append(reward)
        self.current_episode['coverage'].append(coverage)
        self.current_episode['battery_levels'].append(battery_levels.copy())
        self.current_episode['spots_found'].append(spots_found)
        self.current_episode['active_uavs'].append(active_uavs)
    
    def log_episode(self, episode, coverage, steps, active_uavs, total_spots, total_reward=0):
        """Log complete episode metrics"""
        episode_data = {
            'episode': episode,
            'coverage': coverage,
            'steps': steps,
            'active_uavs': active_uavs,
            'total_spots': total_spots,
            'total_reward': total_reward,
            'efficiency': coverage / max(steps, 1),  # Coverage per step
            'success': coverage >= 0.95,
            'timestamp': datetime.now().isoformat()
        }
        
        self.episodes.append(episode_data)
        self.update_summary_metrics()
    
    def update_summary_metrics(self):
        """Update summary metrics based on all episodes"""
        if not self.episodes:
            return
        
        df = pd.DataFrame(self.episodes)
        
        self.summary_metrics = {
            'total_episodes': len(self.episodes),
            'avg_coverage': df['coverage'].mean(),
            'avg_steps': df['steps'].mean(),
            'avg_reward': df.get('total_reward', pd.Series([0])).mean(),
            'success_rate': df.get('success', pd.Series([False])).mean(),
            'efficiency_score': df.get('efficiency', pd.Series([0])).mean(),
            'max_coverage': df['coverage'].max(),
            'min_steps': df['steps'].min(),
            'avg_spots': df['total_spots'].mean()
        }
    
    def get_performance_trends(self):
        """Get performance trends over episodes"""
        if not self.episodes:
            return None
        
        df = pd.DataFrame(self.episodes)
        
        # Calculate moving averages
        window = min(10, len(df))
        trends = {
            'episodes': df['episode'].tolist(),
            'coverage': df['coverage'].tolist(),
            'coverage_ma': df['coverage'].rolling(window=window).mean().tolist(),
            'steps': df['steps'].tolist(),
            'steps_ma': df['steps'].rolling(window=window).mean().tolist(),
            'efficiency': df.get('efficiency', pd.Series([0] * len(df))).tolist(),
            'success_rate': df.get('success', pd.Series([False] * len(df))).rolling(window=window).mean().tolist()
        }
        
        return trends
    
    def compare_performance(self, other_tracker):
        """Compare performance with another metrics tracker"""
        if not self.episodes or not other_tracker.episodes:
            return None
        
        self_df = pd.DataFrame(self.episodes)
        other_df = pd.DataFrame(other_tracker.episodes)
        
        comparison = {
            'coverage_improvement': self_df['coverage'].mean() - other_df['coverage'].mean(),
            'steps_improvement': other_df['steps'].mean() - self_df['steps'].mean(),  # Lower is better
            'efficiency_improvement': self_df.get('efficiency', pd.Series([0])).mean() - other_df.get('efficiency', pd.Series([0])).mean(),
            'success_rate_improvement': self_df.get('success', pd.Series([False])).mean() - other_df.get('success', pd.Series([False])).mean()
        }
        
        return comparison
    
    def generate_report(self):
        """Generate a comprehensive performance report"""
        report = {
            'summary': self.summary_metrics,
            'trends': self.get_performance_trends(),
            'episode_details': self.episodes[-10:] if len(self.episodes) > 10 else self.episodes,  # Last 10 episodes
            'generated_at': datetime.now().isoformat()
        }
        
        return report
    
    def save_metrics(self, filename):
        """Save metrics to CSV file"""
        if not self.episodes:
            print("No episodes to save")
            return
        
        df = pd.DataFrame(self.episodes)
        df.to_csv(filename, index=False)
        print(f"Metrics saved to {filename}")
        
        # Also save summary as JSON
        summary_file = filename.replace('.csv', '_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(self.summary_metrics, f, indent=2)
        print(f"Summary saved to {summary_file}")
    
    def plot_metrics(self, save_path=None):
        """Plot training metrics"""
        if not self.episodes:
            print("No episodes to plot")
            return
        
        df = pd.DataFrame(self.episodes)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Multi-UAV Training Metrics', fontsize=16)
        
        # Coverage over episodes
        axes[0, 0].plot(df['episode'], df['coverage'])
        axes[0, 0].set_title('Coverage Over Episodes')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Coverage Ratio')
        axes[0, 0].grid(True)
        
        # Steps over episodes
        axes[0, 1].plot(df['episode'], df['steps'])
        axes[0, 1].set_title('Steps Per Episode')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].grid(True)
        
        # Efficiency over episodes
        if 'efficiency' in df.columns:
            axes[1, 0].plot(df['episode'], df['efficiency'])
            axes[1, 0].set_title('Efficiency (Coverage/Steps)')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Efficiency')
            axes[1, 0].grid(True)
        
        # Success rate (rolling average)
        if 'success' in df.columns:
            success_rate = df['success'].rolling(window=10).mean()
            axes[1, 1].plot(df['episode'], success_rate)
            axes[1, 1].set_title('Success Rate (10-episode rolling average)')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Success Rate')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def export_for_analysis(self, directory="../data/analysis"):
        """Export all metrics for detailed analysis"""
        os.makedirs(directory, exist_ok=True)
        
        # Export episodes data
        if self.episodes:
            df = pd.DataFrame(self.episodes)
            df.to_csv(os.path.join(directory, "episodes.csv"), index=False)
            
            # Export detailed step data if available
            detailed_data = []
            for episode in self.episodes:
                if 'steps' in episode and isinstance(episode['steps'], list):
                    for i, step in enumerate(episode['steps']):
                        detailed_data.append({
                            'episode': episode['episode'],
                            'step': i,
                            'coverage': episode['coverage'][i] if i < len(episode.get('coverage', [])) else 0,
                            'reward': episode['rewards'][i] if i < len(episode.get('rewards', [])) else 0
                        })
            
            if detailed_data:
                pd.DataFrame(detailed_data).to_csv(os.path.join(directory, "detailed_steps.csv"), index=False)
        
        # Export summary
        with open(os.path.join(directory, "summary.json"), 'w') as f:
            json.dump(self.summary_metrics, f, indent=2)
        
        # Export full report
        report = self.generate_report()
        with open(os.path.join(directory, "full_report.json"), 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Analysis data exported to {directory}")

class RealTimeMetrics:
    def __init__(self):
        self.current_metrics = {
            'timestamp': 0,
            'step': 0,
            'coverage': 0,
            'active_uavs': 0,
            'battery_levels': [],
            'positions': [],
            'spots_found': 0
        }
        
        self.history = []
    
    def update(self, step, coverage, active_uavs, battery_levels, positions, spots_found):
        """Update current metrics"""
        self.current_metrics = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'coverage': coverage,
            'active_uavs': active_uavs,
            'battery_levels': battery_levels.copy(),
            'positions': [pos.tolist() for pos in positions],
            'spots_found': spots_found
        }
        
        self.history.append(self.current_metrics.copy())
    
    def get_current_status(self):
        """Get current status for display"""
        return {
            'Step': self.current_metrics['step'],
            'Coverage': f"{self.current_metrics['coverage']:.1%}",
            'Active UAVs': self.current_metrics['active_uavs'],
            'Targets Found': self.current_metrics['spots_found'],
            'Avg Battery': f"{np.mean(self.current_metrics['battery_levels']):.1f}%"
        }
    
    def export_real_time_data(self, filename):
        """Export real-time data to CSV"""
        if not self.history:
            return
        
        # Flatten the data for CSV export
        flattened_data = []
        for entry in self.history:
            base_data = {
                'timestamp': entry['timestamp'],
                'step': entry['step'],
                'coverage': entry['coverage'],
                'active_uavs': entry['active_uavs'],
                'spots_found': entry['spots_found']
            }
            
            # Add UAV-specific data
            for i, (battery, pos) in enumerate(zip(entry['battery_levels'], entry['positions'])):
                base_data[f'uav_{i}_battery'] = battery
                base_data[f'uav_{i}_x'] = pos[0]
                base_data[f'uav_{i}_y'] = pos[1]
            
            flattened_data.append(base_data)
        
        df = pd.DataFrame(flattened_data)
        df.to_csv(filename, index=False)
        print(f"Real-time data exported to {filename}")