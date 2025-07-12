# Multi-UAV Real-Time Path Planning System

A comprehensive real-time path planning system for multi-UAV navigation using PPO reinforcement learning with efficient data collection and metrics tracking.

## 🚁 Features

- **Real-time path planning** with intelligent decision making
- **Dynamic collision avoidance** between multiple UAVs
- **Energy-efficient navigation** with battery management
- **HTML-based visualization** for interactive monitoring
- **CSV data generation** with before/after training comparisons
- **3D visualization** (optional) for advanced analysis
- **Comprehensive metrics** tracking and performance analysis

## 📁 Project Structure

```
multi_uav_ppo/
├── main.ipynb              # Main Jupyter notebook
├── env/
│   └── uav_env.py          # Enhanced UAV environment
├── train/
│   └── train.py            # Training script with metrics
├── test/
│   └── test.py             # Testing and evaluation
├── visualize/
│   ├── visualize.py        # Real-time visualization
│   └── animate_3d.py       # 3D visualization (optional)
├── utils/
│   ├── logger.py           # Logging utilities
│   └── metrics.py          # Metrics tracking
├── models/
│   └── ppo_MultiUAVEnv.zip # Trained model
├── data/
│   ├── external_obs.csv    # External observations
│   ├── baseline_data.csv   # Pre-training data
│   ├── trained_data.csv    # Post-training data
│   └── real_time_visualization.html
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Install requirements
pip install stable-baselines3 matplotlib pandas numpy gym seaborn

# Navigate to project directory
cd multi_uav_ppo
```

### 2. Training the Model
```bash
# Run training script
python train/train.py
```

### 3. Testing the System
```bash
# Run testing and evaluation
python test/test.py
```

### 4. Real-time Visualization
```bash
# Generate visualization
python visualize/visualize.py
```

### 5. Jupyter Notebook (Recommended)
```bash
# Start Jupyter notebook
jupyter notebook main.ipynb
```

## 📊 Key Components

### Environment (`env/uav_env.py`)
- **Grid-based navigation** with 10x10 default grid
- **Multi-UAV coordination** with collision detection
- **Energy consumption modeling** with battery levels
- **Data collection simulation** with realistic metrics
- **Real-time rendering** capabilities

### Training (`train/train.py`)
- **PPO algorithm** implementation
- **Baseline data collection** before training
- **Trained data collection** after training
- **Performance metrics** comparison
- **CSV data generation** for analysis

### Visualization (`visualize/visualize.py`)
- **HTML-based real-time visualization**
- **Interactive controls** (start, pause, speed control)
- **Real-time metrics display**
- **Battery level monitoring**
- **Coverage progress tracking**

### Metrics (`utils/metrics.py`)
- **Coverage efficiency** calculation
- **Energy efficiency** metrics
- **Collision rate** analysis
- **Mission success rate** tracking
- **Performance trend** analysis

## 🎯 Usage Examples

### Basic Training
```python
from env.uav_env import MultiUAVEnv
from stable_baselines3 import PPO

# Initialize environment
env = MultiUAVEnv(grid_size=10, num_uavs=3)

# Train model
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)
model.save("./models/ppo_MultiUAVEnv")
```

### Real-time Simulation
```python
# Load trained model
model = PPO.load("./models/ppo_MultiUAVEnv")

# Run simulation
obs = env.reset()
done = False
while not done:
    actions, _ = model.predict(obs)
    obs, reward, done, info = env.step(actions)
    env.render()
```

### Visualization
```python
from visualize.visualize import RealTimeVisualizer

# Create visualizer
visualizer = RealTimeVisualizer(grid_size=10, num_uavs=3)

# Generate HTML visualization
html_file = visualizer.create_html_visualization(simulation_data)
```

## 📈 Performance Metrics

The system tracks comprehensive performance metrics:

- **Coverage Efficiency**: Percentage of grid covered
- **Energy Consumption**: Total energy used by all UAVs
- **Mission Time**: Steps required to complete mission
- **Collision Count**: Number of UAV collisions
- **Data Points Collected**: Simulated sensor data
- **Battery Utilization**: Energy efficiency metrics

## 🎮 Interactive Features

### HTML Visualization Controls
- **Start/Pause**: Control animation playback
- **Speed Control**: Adjust animation speed
- **Reset**: Restart simulation
- **Real-time Metrics**: Live performance monitoring
- **Battery Levels**: Individual UAV battery status

### CSV Data Analysis
- **Baseline Data**: Performance before training
- **Trained Data**: Performance after training
- **Comparison Metrics**: Side-by-side analysis
- **Improvement Tracking**: Quantified enhancements

## 🔧 Configuration

### Environment Settings
```python
CONFIG = {
    'grid_size': 10,           # Grid dimensions
    'num_uavs': 3,             # Number of UAVs
    'max_steps': 200,          # Maximum episode steps
    'training_timesteps': 50000, # Training duration
    'learning_rate': 0.0003,   # PPO learning rate
    'real_time_delay': 0.1,    # Visualization delay
}
```

### Training Parameters
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Policy**: MlpPolicy (Multi-layer Perceptron)
- **Batch Size**: 32
- **Learning Rate**: 0.0003
- **Discount Factor**: 0.99

## 📊 Data Format

### External Observations CSV
```csv
ID,source_time,lat,lon,source_spotted,track,takeoff_landing_time,battery,AoI,FLA
AL2,1,52.815992,-4.131736,1,90,12,90,0,2
OI3,2,52.815992,-4.131736,0,-35,26,100,0,3
```

### Performance Metrics CSV
```csv
episode,coverage,energy_consumed,mission_time,collision_count,path_length
0,75.2,45.3,120,1,180
1,82.1,38.7,95,0,165
```

## 🎯 Real-time Capabilities

### Live Monitoring
- **UAV Position Tracking**: Real-time position updates
- **Coverage Visualization**: Live grid coverage display
- **Battery Monitoring**: Real-time battery level tracking
- **Collision Detection**: Immediate collision alerts
- **Performance Metrics**: Live efficiency calculations

### Dynamic Path Planning
- **Obstacle Avoidance**: Real-time obstacle detection
- **Route Optimization**: Dynamic path adjustments
- **Energy Management**: Battery-aware navigation
- **Coordination**: Multi-UAV cooperation

## 🚁 Mission Scenarios

### Area Coverage
- **Complete grid coverage** with minimal overlap
- **Energy-efficient paths** to maximize battery life
- **Coordinated navigation** to avoid collisions
- **Data collection** at specified locations

### Real-time Surveillance
- **Dynamic target tracking** with adaptive paths
- **Multi-UAV coordination** for comprehensive coverage
- **Real-time decision making** based on environment
- **Continuous monitoring** with live updates

## 📋 Output Files

### Training Results
- `baseline_data.csv` - Pre-training performance
- `trained_data.csv` - Post-training performance
- `performance_comparison.csv` - Improvement metrics
- `ppo_MultiUAVEnv.zip` - Trained model

### Visualization
- `real_time_visualization.html` - Interactive visualization
- `uav_3d_paths.png` - 3D path visualization
- `uav_altitude_profile.png` - Altitude analysis
- `performance_plots.png` - Metrics graphs

## 🔍 System Status

✅ **Environment**: Multi-UAV navigation environment  
✅ **Training**: PPO reinforcement learning  
✅ **Visualization**: Real-time HTML display  
✅ **Data Collection**: CSV generation and analysis  
✅ **Metrics**: Comprehensive performance tracking  
✅ **3D Visualization**: Optional advanced display  

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Based on the DAMIAN (Delay-Aware Multi-Aerial Navigation) framework
- Implements PPO from Stable Baselines3
- Uses OpenAI Gym for environment management
- Inspired by real-world UAV coordination challenges

---

**Status**: ✅ **OPERATIONAL** - The multi-UAV real-time path planning system is fully functional with trained intelligence, real-time visualization, and comprehensive analytics.