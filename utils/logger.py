import logging
import os
import time
from datetime import datetime
import json

class TrainingLogger:
    def __init__(self, log_dir="../logs", log_level=logging.INFO):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create timestamp for this session
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup logging
        self.setup_logging(log_level)
        
        # Session data
        self.session_data = {
            'session_id': self.session_id,
            'start_time': time.time(),
            'episodes': [],
            'training_params': {}
        }
    
    def setup_logging(self, log_level):
        """Setup logging configuration"""
        log_file = os.path.join(self.log_dir, f"training_{self.session_id}.log")
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def log_training_start(self, params):
        """Log training start with parameters"""
        self.session_data['training_params'] = params
        self.logger.info(f"Training session {self.session_id} started")
        self.logger.info(f"Parameters: {params}")
    
    def log_episode(self, episode, metrics):
        """Log episode results"""
        episode_data = {
            'episode': episode,
            'timestamp': time.time(),
            'metrics': metrics
        }
        
        self.session_data['episodes'].append(episode_data)
        
        if episode % 10 == 0:  # Log every 10 episodes
            self.logger.info(f"Episode {episode}: {metrics}")
    
    def log_training_end(self, final_metrics):
        """Log training completion"""
        self.session_data['end_time'] = time.time()
        self.session_data['duration'] = self.session_data['end_time'] - self.session_data['start_time']
        self.session_data['final_metrics'] = final_metrics
        
        self.logger.info(f"Training completed in {self.session_data['duration']:.2f} seconds")
        self.logger.info(f"Final metrics: {final_metrics}")
        
        # Save session data
        self.save_session_data()
    
    def save_session_data(self):
        """Save session data to JSON file"""
        json_file = os.path.join(self.log_dir, f"session_{self.session_id}.json")
        
        with open(json_file, 'w') as f:
            json.dump(self.session_data, f, indent=2)
        
        self.logger.info(f"Session data saved to {json_file}")
    
    def log_error(self, error_msg, exception=None):
        """Log error messages"""
        self.logger.error(f"Error: {error_msg}")
        if exception:
            self.logger.error(f"Exception: {str(exception)}")
    
    def log_warning(self, warning_msg):
        """Log warning messages"""
        self.logger.warning(warning_msg)
    
    def log_info(self, info_msg):
        """Log info messages"""
        self.logger.info(info_msg)

class RealTimeLogger:
    def __init__(self, log_file="../logs/realtime.log"):
        self.log_file = log_file
        self.start_time = time.time()
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Initialize log file
        with open(log_file, 'w') as f:
            f.write(f"Real-time UAV mission log started at {datetime.now()}\n")
            f.write("=" * 50 + "\n")
    
    def log_step(self, step, uav_states, metrics):
        """Log a single step of the mission"""
        timestamp = time.time() - self.start_time
        
        log_entry = {
            'timestamp': timestamp,
            'step': step,
            'uav_states': uav_states,
            'metrics': metrics
        }
        
        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp:.2f}s] Step {step}: {json.dumps(log_entry)}\n")
    
    def log_mission_complete(self, final_metrics):
        """Log mission completion"""
        duration = time.time() - self.start_time
        
        with open(self.log_file, 'a') as f:
            f.write("\n" + "=" * 50 + "\n")
            f.write(f"Mission completed in {duration:.2f} seconds\n")
            f.write(f"Final metrics: {json.dumps(final_metrics)}\n")
    
    def get_log_data(self):
        """Read and return log data"""
        try:
            with open(self.log_file, 'r') as f:
                return f.read()
        except FileNotFoundError:
            return "No log data available"