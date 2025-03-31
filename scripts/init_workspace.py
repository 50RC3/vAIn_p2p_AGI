#!/usr/bin/env python3
"""
Initialize the vAIn P2P AGI workspace.
This script creates directories, checks dependencies, and prepares the environment.
"""

import os
import sys
import shutil
import logging
from pathlib import Path

# Add project root to path for importing
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("init_workspace")

# Import project modules
try:
    from core.constants import (
        BASE_DIR, TEMP_DIR, DATA_DIR, CONFIG_DIR, LOGS_DIR, 
        MODELS_DIR, CACHE_DIR, SESSION_SAVE_PATH
    )
    from utils.dependency_checker import check_dependencies, install_dependencies
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.info("Make sure you're running from the project root directory")
    sys.exit(1)

def create_directories():
    """Create all necessary directories for the project"""
    directories = [
        TEMP_DIR, DATA_DIR, CONFIG_DIR, LOGS_DIR, MODELS_DIR, CACHE_DIR, SESSION_SAVE_PATH
    ]
    
    for directory in directories:
        try:
            directory.mkdir(exist_ok=True, parents=True)
            logger.info(f"Created directory: {directory}")
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")
    
    # Create subdirectories for organization
    subdirs = {
        DATA_DIR: ["training", "evaluation", "checkpoints"],
        MODELS_DIR: ["saved", "pretrained", "exports"],
        LOGS_DIR: ["network", "training", "system"],
        CONFIG_DIR: ["profiles", "templates"]
    }
    
    for parent, children in subdirs.items():
        for child in children:
            try:
                (parent / child).mkdir(exist_ok=True)
            except Exception as e:
                logger.error(f"Failed to create subdirectory {parent/child}: {e}")

def check_env():
    """Check for environment requirements"""
    logger.info("Checking environment...")
    
    # Python version check
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        logger.warning(f"Python 3.8+ is recommended, you have {sys.version}")
    else:
        logger.info(f"Python version: {sys.version}")
    
    # Check for required dependencies
    success, missing = check_dependencies(required_only=True)
    
    if success:
        logger.info("✓ All required dependencies are installed.")
    else:
        logger.warning(f"Missing core dependencies: {', '.join(missing)}")
        
        # Ask if user wants to install
        if input("Install missing dependencies now? (y/n): ").lower().startswith('y'):
            if install_dependencies(missing):
                logger.info("✓ Successfully installed all dependencies.")
            else:
                logger.error("Failed to install some dependencies.")
                sys.exit(1)
        else:
            logger.warning("Please install the missing dependencies before proceeding.")
            sys.exit(1)

def create_default_configs():
    """Create default configuration files if they don't exist"""
    default_configs = {
        CONFIG_DIR / "logging.json": {
            "level": "INFO",
            "format": "%(asctime)s | %(name)s | %(levelname)s | [%(filename)s:%(lineno)d] | %(message)s",
            "file_logging": True,
            "max_bytes": 10485760,
            "backup_count": 5,
        },
        CONFIG_DIR / "network.json": {
            "node_env": "development",
            "port": 8468,
            "max_connections": 10,
            "timeout": 30
        }
    }
    
    import json
    
    for config_path, default_content in default_configs.items():
        if not config_path.exists():
            try:
                with open(config_path, 'w') as f:
                    json.dump(default_content, f, indent=2)
                logger.info(f"Created default config: {config_path}")
            except Exception as e:
                logger.error(f"Failed to create config {config_path}: {e}")

def main():
    """Main initialization function"""
    logger.info("Initializing vAIn P2P AGI workspace...")
    
    # Create directories
    create_directories()
    
    # Check environment
    check_env()
    
    # Create default configs
    create_default_configs()
    
    logger.info("Workspace initialization complete!")
    logger.info(f"Project root: {BASE_DIR}")
    
    # Create README if it doesn't exist
    readme_path = BASE_DIR / "README.md"
    if not readme_path.exists():
        try:
            with open(readme_path, 'w') as f:
                f.write("# vAIn P2P AGI\n\n")
                f.write("A peer-to-peer artificial general intelligence system.\n\n")
                f.write("## Getting Started\n\n")
                f.write("1. Run `python scripts/init_workspace.py` to initialize the workspace\n")
                f.write("2. Run `python start.py` to launch the system\n")
        except Exception as e:
            logger.error(f"Failed to create README: {e}")
    
    logger.info("To start the system, run: python start.py")

if __name__ == "__main__":
    main()
