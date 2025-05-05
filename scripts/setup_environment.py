import subprocess
import sys
import os
from pathlib import Path

def setup_environment():
    """Set up the development environment with required dependencies"""
    print("Setting up vAIn P2P AGI development environment...")
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    requirements_file = project_root / "requirements.txt"
    
    if not requirements_file.exists():
        print(f"Error: requirements.txt not found at {requirements_file}")
        return False
    
    # Install dependencies
    print("Installing dependencies from requirements.txt...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])
        print("Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False
    
    # Create directories if they don't exist
    directories = [
        "data",
        "logs",
        "models",
        "metrics",
        "metrics/learning"
    ]
    
    for directory in directories:
        dir_path = project_root / directory
        if not dir_path.exists():
            print(f"Creating directory: {directory}")
            dir_path.mkdir(parents=True)
    
    print("Environment setup complete!")
    return True

if __name__ == "__main__":
    success = setup_environment()
    sys.exit(0 if success else 1)