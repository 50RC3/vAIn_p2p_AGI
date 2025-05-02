#!/usr/bin/env python3
"""
Unified setup script for vAIn_p2p_AGI
- Creates virtual environment
- Installs dependencies
- Validates installation
- Creates default configurations
"""
import os
import sys
import subprocess
import argparse
import platform
import logging
from pathlib import Path

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("setup")

# Define constants
PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / "venv"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
CONFIG_DIR = PROJECT_ROOT / "config"
LOGS_DIR = PROJECT_ROOT / "logs"

def create_virtual_environment() -> bool:
    """Create a virtual environment for the project."""
    if VENV_DIR.exists():
        logger.info("Virtual environment already exists at %s", VENV_DIR)
        return True
    
    logger.info("Creating virtual environment at %s...", VENV_DIR)
    try:
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)])
        logger.info("Virtual environment created successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error("Failed to create virtual environment: %s", e)
        return False

def get_venv_python() -> str:
    """Get the path to the Python executable in the virtual environment."""
    if platform.system() == "Windows":
        return str(VENV_DIR / "Scripts" / "python.exe")
    return str(VENV_DIR / "bin" / "python")

def install_dependencies(update: bool = False) -> bool:
    """Install required Python packages from requirements.txt."""
    logger.info("Installing required packages...")
    
    venv_python = get_venv_python()
    pip_cmd = [venv_python, "-m", "pip"]
    
    if update:
        # Update pip itself first
        try:
            subprocess.check_call(pip_cmd + ["install", "--upgrade", "pip"])
            logger.info("Upgraded pip to the latest version")
        except subprocess.CalledProcessError as e:
            logger.error("Failed to upgrade pip: %s", e)
    
    # Install from requirements.txt
    try:
        cmd = pip_cmd + ["install", "-r", str(REQUIREMENTS_FILE)]
        if update:
            cmd.append("--upgrade")
        subprocess.check_call(cmd)
        logger.info("Successfully installed Python dependencies")
        return True
    except subprocess.CalledProcessError as e:
        logger.error("Failed to install dependencies: %s", e)
        return False

def setup_nodejs_dependencies(update: bool = False) -> bool:
    """Install Node.js dependencies if package.json exists."""
    package_json = PROJECT_ROOT / "package.json"
    backend_dir = PROJECT_ROOT / "backend"
    
    if not package_json.exists():
        logger.info("No package.json found, skipping Node.js setup")
        return True
    
    try:
        # Install root project dependencies
        logger.info("Installing Node.js dependencies...")
        cmd = ["npm", "install"]
        if update:
            cmd.append("--update")
        subprocess.check_call(cmd, cwd=str(PROJECT_ROOT))
        
        # Install backend dependencies if they exist
        if (backend_dir / "package.json").exists():
            logger.info("Installing backend Node.js dependencies...")
            subprocess.check_call(cmd, cwd=str(backend_dir))
            
        logger.info("Successfully installed Node.js dependencies")
        return True
    except subprocess.CalledProcessError as e:
        logger.error("Failed to install Node.js dependencies: %s", e)
        return False
    except FileNotFoundError:
        logger.error("npm not found. Please install Node.js and npm first.")
        return False

def create_default_directories() -> bool:
    """Create default directories needed by the application."""
    directories = [
        CONFIG_DIR,
        CONFIG_DIR / "backups",
        LOGS_DIR,
        LOGS_DIR / "metrics",
    ]
    
    for directory in directories:
        directory.mkdir(exist_ok=True, parents=True)
    
    logger.info("Created default directories")
    return True

def create_default_configs() -> bool:
    """Create default configuration files."""
    venv_python = get_venv_python()
    
    try:
        # Run the config manager to create default configurations
        cmd = [
            venv_python, "-c", 
            "from tools.config_manager import ConfigManager; ConfigManager().create_default_configs()"
        ]
        
        subprocess.check_call(cmd)
        logger.info("Created default configuration files")
        return True
    except subprocess.CalledProcessError as e:
        logger.error("Failed to create default configurations: %s", e)
        logger.info("You may need to run the application once first")
        return False
    except Exception as e:
        logger.warning("Could not create default configurations: %s", e)
        return False

def verify_installation() -> bool:
    """Verify the installation by running dependency checks."""
    venv_python = get_venv_python()
    
    try:
        cmd = [
            venv_python, "-c", 
            "from utils.verify_dependencies import verify_dependencies; "
            "print('Installation verification successful') "
            "if verify_dependencies(show_system=False) else print('Installation verification failed')"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if "Installation verification successful" in result.stdout:
            logger.info("Installation verified successfully")
            return True
        else:
            logger.warning("Installation verification found issues:")
            logger.warning(result.stdout.strip())
            return False
    except subprocess.CalledProcessError as e:
        logger.error("Failed to verify installation: %s", e)
        return False

def activate_instructions() -> None:
    """Print instructions for activating the virtual environment."""
    if platform.system() == "Windows":
        activate_script = VENV_DIR / "Scripts" / "activate"
        logger.info("\nTo activate the virtual environment, run:\n> %s", activate_script)
    else:
        activate_script = VENV_DIR / "bin" / "activate"
        logger.info("\nTo activate the virtual environment, run:\n$ source %s", activate_script)

def main() -> int:
    """Main entry point for setup script."""
    parser = argparse.ArgumentParser(description="Setup vAIn_p2p_AGI project")
    parser.add_argument("--update", action="store_true", help="Update existing dependencies")
    parser.add_argument("--no-venv", action="store_true", help="Skip virtual environment creation")
    parser.add_argument("--no-nodejs", action="store_true", help="Skip Node.js setup")
    parser.add_argument("--verify", action="store_true", help="Only verify the installation")
    
    args = parser.parse_args()
    
    if args.verify:
        return verify_installation()
    
    # Create directories first
    create_default_directories()
    
    # Setup process
    success = True
    
    if not args.no_venv:
        success = create_virtual_environment() and success
    
    if success:
        success = install_dependencies(update=args.update) and success
    
    if not args.no_nodejs:
        success = setup_nodejs_dependencies(update=args.update) and success
    
    # Create default configurations
    success = create_default_configs() and success
    
    # Verify installation
    if success:
        success = verify_installation() and success
    
    # Print activation instructions
    if success and not args.no_venv:
        activate_instructions()
    
    logger.info("Setup completed successfully" if success else "Setup completed with errors")
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())

from setuptools import setup, find_packages

setup(
    name="vain_p2p_agi",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # List your requirements here
        "torch>=1.10.0",
        "numpy>=1.19.0,<2.0.0",
        "nltk>=3.8.1",
    ],
)
