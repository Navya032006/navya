import numpy as np
import pandas as pd
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

class UAVMetrics:
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.episode_metrics = []
        self.step_metrics = []
        self.real_time_metrics = []
        self.current_episode = 0
        self.start_time = time.time()
        
    def calculate_coverage_efficiency(self, visited_cells: np.ndarray, total_cells: int) -> float:
        """Calculate coverage efficiency percentage"""
        covered_cells = np.sum(visited_cells)
        return (covered_cells / total_cells) * 100
    
    def calculate_energy_efficiency(self, energy_consumed: float, coverage: float) -> float:
        """Calculate energy efficiency (coverage per unit energy)"""
        return coverage / energy_consumed if energy_consumed > 0 else 0
    
    def calculate_path_efficiency(self, path_length: int, coverage: float) -> float:
        """Calculate path efficiency (coverage per unit path length)"""
        return coverage / path_length if path_length > 0 else 0
    
    def calculate_mission_success_rate(self, target_coverage: float = 80.0) -> float:
        """Calculate mission success rate based on coverage threshold"""
        if not self.episode_metrics:
            return 0.0
        
        successful_episodes = sum(1 for m in self.episode_metrics 
                                if m.get('coverage', 0) >= target_coverage)
        return (successful_episodes / len(self.episode_metrics)) * 100
    
    def calculate_collision_rate(self) -> float:
        """Calculate average collision rate per episode"""
        if not self.episode_metrics:
            return 0.0
        
        total_collisions = sum(m.get('collision_count', 0) for m in self.episode_metrics)
        return total_collisions / len(self.episode_metrics)
    
    def calculate_battery_utilization(self, initial_battery: float, final_battery: float) -> float:
        """Calculate battery utilization percentage"""
        return ((initial_battery - final_battery) / initial_battery) * 100
    
    def calculate_coordination_score(self, uav_positions: List[List[Tuple]], visited_cells: np.ndarray) -> float:
        """Calculate coordination score based on area coverage distribution"""
        if not uav_positions:
            return 0.0
        
        # Calculate coverage overlap
        total_positions = sum(len(positions) for positions in uav_positions)
        unique_positions = len(set(tuple(pos) for positions in uav_positions for pos in positions))
        
        # Higher score for less overlap
        overlap_score = unique_positions / total_positions if total_positions > 0 else 0
        
        # Calculate area distribution
        grid_size = int(np.sqrt(visited_cells.size))
        quadrants = [
            visited_cells[:grid_size//2, :grid_size//2],
            visited_cells[:grid_size//2, grid_size//2:],
            visited_cells[grid_size//2:, :grid_size//2],
            visited_cells[grid_size//2:, grid_size//2:]
        ]
        
        quadrant_coverage = [np.sum(q) for q in quadrants]
        total_coverage = sum(quadrant_coverage)
        
        if total_coverage > 0:
            # Calculate distribution evenness
            distribution_score = 1.0 - (np.std(quadrant_coverage) / np.mean(quadrant_coverage))
        else:
            distribution_score = 0.0
        
        return (overlap_score + distribution_score) / 2 * 100
    
    def add_episode_metrics(self, episode: int, metrics: Dict):
        """Add metrics for a completed episode"""
        enhanced_metrics = {
            'episode': episode,
            'timestamp': datetime.now().isoformat(),
            'elapsed_time': time.time() - self.start_time,
            **metrics
        }
        
        # Calculate derived metrics
        if 'coverage' in metrics and 'energy_consumed' in metrics:
            enhanced_metrics['energy_efficiency'] = self.calculate_energy_efficiency(
                metrics['energy_consumed'], metrics['coverage']
            )
        
        if 'coverage' in metrics and 'path_length' in metrics:
            enhanced_metrics['path_efficiency'] = self.calculate_path_efficiency(
                metrics['path_length'], metrics['coverage']
            )
        
        self.episode_metrics.append(enhanced_metrics)
    
    def add_step_metrics(self, episode: int, step: int, metrics: Dict):
        """Add metrics for a single step"""
        step_data = {
            'episode': episode,
            'step': step,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        
        self.step_metrics.append(step_data)
    
    def add_real_time_metrics(self, metrics: Dict):
        """Add real-time metrics"""
        real_time_data = {
            'timestamp': datetime.now().isoformat(),
            'elapsed_time': time.time() - self.start_time,
            **metrics
        }
        
        self.real_time_metrics.append(real_time_data)
    
    def get_episode_statistics(self) -> Dict:
        """Get comprehensive episode statistics"""
        if not self.episode_metrics:
            return {}
        
        df = pd.DataFrame(self.episode_metrics)
        
        stats = {
            'total_episodes': len(self.episode_metrics),
            'avg_coverage': df['coverage'].mean() if 'coverage' in df.columns else 0,
            'max_coverage': df['coverage'].max() if 'coverage' in df.columns else 0,
            'min_coverage': df['coverage'].min() if 'coverage' in df.columns else 0,
            'std_coverage': df['coverage'].std() if 'coverage' in df.columns else 0,
            'avg_energy': df['energy_consumed'].mean() if 'energy_consumed' in df.columns else 0,
            'avg_mission_time': df['mission_time'].mean() if 'mission_time' in df.columns else 0,
            'avg_collisions': df['collision_count'].mean() if 'collision_count' in df.columns else 0,
            'success_rate': self.calculate_mission_success_rate(),
            'collision_rate': self.calculate_collision_rate()
        }
        
        # Add efficiency metrics if available
        if 'energy_efficiency' in df.columns:
            stats['avg_energy_efficiency'] = df['energy_efficiency'].mean()
        
        if 'path_efficiency' in df.columns:
            stats['avg_path_efficiency'] = df['path_efficiency'].mean()
        
        return stats
    
    def get_performance_trends(self) -> Dict:
        """Get performance trends over episodes"""
        if not self.episode_metrics:
            return {}
        
        df = pd.DataFrame(self.episode_metrics)
        
        trends = {}
        
        # Calculate trends for key metrics
        metrics_to_analyze = ['coverage', 'energy_consumed', 'mission_time', 'collision_count']
        
        for metric in metrics_to_analyze:
            if metric in df.columns:
                values = df[metric].values
                if len(values) > 1:
                    # Calculate trend (slope)
                    x = np.arange(len(values))
                    trend = np.polyfit(x, values, 1)[0]
                    trends[f'{metric}_trend'] = trend
                    
                    # Calculate improvement percentage
                    if len(values) >= 2:
                        initial = np.mean(values[:len(values)//3]) if len(values) >= 3 else values[0]
                        final = np.mean(values[-len(values)//3:]) if len(values) >= 3 else values[-1]
                        
                        if initial != 0:
                            improvement = ((final - initial) / initial) * 100
                            trends[f'{metric}_improvement'] = improvement
        
        return trends
    
    def compare_baseline_vs_trained(self, baseline_metrics: List[Dict], trained_metrics: List[Dict]) -> Dict:
        """Compare baseline vs trained performance"""
        comparison = {
            'baseline_episodes': len(baseline_metrics),
            'trained_episodes': len(trained_metrics),
            'comparison_date': datetime.now().isoformat()
        }
        
        if baseline_metrics and trained_metrics:
            baseline_df = pd.DataFrame(baseline_metrics)
            trained_df = pd.DataFrame(trained_metrics)
            
            metrics_to_compare = ['coverage', 'energy_consumed', 'mission_time', 'collision_count']
            
            for metric in metrics_to_compare:
                if metric in baseline_df.columns and metric in trained_df.columns:
                    baseline_avg = baseline_df[metric].mean()
                    trained_avg = trained_df[metric].mean()
                    
                    comparison[f'baseline_{metric}'] = baseline_avg
                    comparison[f'trained_{metric}'] = trained_avg
                    comparison[f'{metric}_improvement'] = trained_avg - baseline_avg
                    
                    if baseline_avg != 0:
                        comparison[f'{metric}_improvement_pct'] = ((trained_avg - baseline_avg) / baseline_avg) * 100
        
        return comparison
    
    def generate_performance_report(self, save_to_file: bool = True) -> Dict:
        """Generate comprehensive performance report"""
        report = {
            'report_date': datetime.now().isoformat(),
            'total_runtime': time.time() - self.start_time,
            'episode_statistics': self.get_episode_statistics(),
            'performance_trends': self.get_performance_trends(),
            'data_quality': {
                'total_episodes': len(self.episode_metrics),
                'total_steps': len(self.step_metrics),
                'real_time_samples': len(self.real_time_metrics)
            }
        }
        
        if save_to_file:
            filename = f"../data/performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Performance report saved to {filename}")
        
        return report
    
    def create_visualization_plots(self, save_plots: bool = True) -> Dict:
        """Create visualization plots for metrics"""
        if not self.episode_metrics:
            return {}
        
        df = pd.DataFrame(self.episode_metrics)
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('UAV Performance Metrics', fontsize=16)
        
        plots_created = {}
        
        # Plot 1: Coverage over episodes
        if 'coverage' in df.columns:
            axes[0, 0].plot(df['episode'], df['coverage'], marker='o', linewidth=2)
            axes[0, 0].set_title('Coverage Efficiency Over Episodes')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Coverage (%)')
            axes[0, 0].grid(True, alpha=0.3)
            plots_created['coverage_plot'] = True
        
        # Plot 2: Energy consumption over episodes
        if 'energy_consumed' in df.columns:
            axes[0, 1].plot(df['episode'], df['energy_consumed'], marker='s', color='orange', linewidth=2)
            axes[0, 1].set_title('Energy Consumption Over Episodes')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Energy Consumed')
            axes[0, 1].grid(True, alpha=0.3)
            plots_created['energy_plot'] = True
        
        # Plot 3: Mission time over episodes
        if 'mission_time' in df.columns:
            axes[1, 0].plot(df['episode'], df['mission_time'], marker='^', color='green', linewidth=2)
            axes[1, 0].set_title('Mission Time Over Episodes')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Mission Time (steps)')
            axes[1, 0].grid(True, alpha=0.3)
            plots_created['mission_time_plot'] = True
        
        # Plot 4: Collision count over episodes
        if 'collision_count' in df.columns:
            axes[1, 1].plot(df['episode'], df['collision_count'], marker='D', color='red', linewidth=2)
            axes[1, 1].set_title('Collision Count Over Episodes')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Collision Count')
            axes[1, 1].grid(True, alpha=0.3)
            plots_created['collision_plot'] = True
        
        plt.tight_layout()
        
        if save_plots:
            filename = f"../data/performance_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Performance plots saved to {filename}")
            plots_created['saved_file'] = filename
        
        plt.close()
        
        return plots_created
    
    def export_metrics_to_csv(self, filename: Optional[str] = None) -> str:
        """Export all metrics to CSV files"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"../data/metrics_export_{timestamp}.csv"
        
        if self.episode_metrics:
            df = pd.DataFrame(self.episode_metrics)
            df.to_csv(filename, index=False)
            print(f"Episode metrics exported to {filename}")
        
        # Export step metrics if available
        if self.step_metrics:
            step_filename = filename.replace('.csv', '_steps.csv')
            step_df = pd.DataFrame(self.step_metrics)
            step_df.to_csv(step_filename, index=False)
            print(f"Step metrics exported to {step_filename}")
        
        return filename

def calculate_multi_uav_coordination_metrics(uav_positions: List[List[Tuple]], 
                                           visited_cells: np.ndarray) -> Dict:
    """Calculate coordination metrics for multiple UAVs"""
    if not uav_positions:
        return {}
    
    num_uavs = len(uav_positions)
    
    # Calculate coverage overlap
    total_positions = sum(len(positions) for positions in uav_positions)
    unique_positions = len(set(tuple(pos) for positions in uav_positions for pos in positions))
    
    overlap_ratio = 1.0 - (unique_positions / total_positions) if total_positions > 0 else 0
    
    # Calculate inter-UAV distances
    distances = []
    for i in range(num_uavs):
        for j in range(i + 1, num_uavs):
            if uav_positions[i] and uav_positions[j]:
                pos_i = uav_positions[i][-1]  # Last position
                pos_j = uav_positions[j][-1]
                dist = np.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)
                distances.append(dist)
    
    avg_distance = np.mean(distances) if distances else 0
    
    # Calculate coverage distribution
    grid_size = int(np.sqrt(visited_cells.size))
    coverage_per_uav = []
    
    for positions in uav_positions:
        coverage = 0
        for pos in positions:
            if 0 <= pos[0] < grid_size and 0 <= pos[1] < grid_size:
                coverage += 1
        coverage_per_uav.append(coverage)
    
    coverage_balance = 1.0 - (np.std(coverage_per_uav) / np.mean(coverage_per_uav)) if coverage_per_uav else 0
    
    return {
        'overlap_ratio': overlap_ratio,
        'avg_inter_uav_distance': avg_distance,
        'coverage_balance': coverage_balance,
        'coordination_score': (1 - overlap_ratio) * coverage_balance * 100
    }

def main():
    # Demo usage
    metrics = UAVMetrics()
    
    # Simulate episode data
    for episode in range(10):
        episode_metrics = {
            'coverage': np.random.uniform(60, 95),
            'energy_consumed': np.random.uniform(30, 80),
            'mission_time': np.random.randint(50, 100),
            'collision_count': np.random.randint(0, 3),
            'path_length': np.random.randint(100, 200)
        }
        
        metrics.add_episode_metrics(episode, episode_metrics)
    
    # Generate report
    report = metrics.generate_performance_report()
    print("Performance Report:")
    print(json.dumps(report, indent=2))
    
    # Create plots
    plots = metrics.create_visualization_plots()
    print(f"Plots created: {plots}")
    
    # Export to CSV
    csv_file = metrics.export_metrics_to_csv()
    print(f"Metrics exported to: {csv_file}")

if __name__ == "__main__":
    main()