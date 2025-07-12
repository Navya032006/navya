#!/usr/bin/env python3
"""
Simple script to run 3D UAV visualization from any directory
"""

import os
import sys
import subprocess

def main():
    # Get the current directory
    current_dir = os.getcwd()
    
    # Check if we're in the workspace directory
    if "multi_uav_ppo" in os.listdir('.'):
        # We're in the workspace root
        script_path = os.path.join('multi_uav_ppo', 'visualize', 'animate_3d.py')
        working_dir = os.path.join('.', 'multi_uav_ppo', 'visualize')
    elif os.path.basename(current_dir) == 'multi_uav_ppo':
        # We're in the multi_uav_ppo directory
        script_path = os.path.join('visualize', 'animate_3d.py')
        working_dir = os.path.join('.', 'visualize')
    else:
        print("❌ Error: Please run this script from either:")
        print("   - /workspace (root directory)")
        print("   - /workspace/multi_uav_ppo")
        return
    
    if not os.path.exists(script_path):
        print(f"❌ Error: Could not find {script_path}")
        return
    
    print("🚁 Starting 3D UAV Visualization...")
    print(f"📁 Working directory: {os.path.abspath(working_dir)}")
    print(f"🐍 Running script: {script_path}")
    print("-" * 50)
    
    try:
        # Change to the correct directory and run the script
        os.chdir(working_dir)
        result = subprocess.run([sys.executable, 'animate_3d.py'], 
                              capture_output=False, text=True)
        
        if result.returncode == 0:
            print("-" * 50)
            print("✅ 3D Visualization completed successfully!")
            print("📁 Check the data/ directory for generated files:")
            print("   - uav_3d_animation.gif")
            print("   - uav_3d_paths.png") 
            print("   - uav_altitude_profile.png")
        else:
            print("❌ Error occurred during visualization")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Return to the original directory
        os.chdir(current_dir)

if __name__ == "__main__":
    main()