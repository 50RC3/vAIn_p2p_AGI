import os
import sys
import subprocess
import platform
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REQUIRED_PYTHON = (3, 8, 0)

def check_python_version():
    """Verify that the correct version of Python is being used."""
    if sys.version_info < REQUIRED_PYTHON:
        logger.error(f"This project requires Python {'.'.join(map(str, REQUIRED_PYTHON))} or higher")
        sys.exit(1)
    logger.info(f"Python version check passed: {sys.version.split()[0]}")

def create_venv():
    """Create a virtual environment."""
    venv_path = Path("venv")
    if venv_path.exists():
        logger.warning("Virtual environment already exists")
        if input("Recreate virtual environment? (y/N): ").lower() != 'y':
            return False
        import shutil
        shutil.rmtree(venv_path)
    
    logger.info("Creating virtual environment...")
    subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
    return True

def install_dependencies():
    """Install project dependencies."""
    pip_cmd = "venv/Scripts/pip" if platform.system() == "Windows" else "venv/bin/pip"
    
    logger.info("Installing build dependencies...")
    build_deps = [
        "setuptools>=65.0.0",
        "wheel>=0.37.0",
        "cython>=0.29.0",
        "ninja>=1.10.0"
    ]
    
    try:
        # Upgrade pip first
        subprocess.run([pip_cmd, "install", "--upgrade", "pip"], check=True)
        
        # Install build dependencies
        for dep in build_deps:
            subprocess.run([pip_cmd, "install", dep], check=True)
        
        # Read and install requirements
        if os.path.exists("requirements.txt"):
            logger.info("Installing project dependencies from requirements.txt...")
            subprocess.run([pip_cmd, "install", "-r", "requirements.txt"], check=True)
        else:
            logger.warning("requirements.txt not found. Skipping project dependencies.")
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        raise

def print_activation_instructions():
    """Print instructions for activating the virtual environment."""
    if platform.system() == "Windows":
        activate_cmd = ".\\venv\\Scripts\\activate"
    else:
        activate_cmd = "source venv/bin/activate"
        
    print("\nVirtual environment setup complete!")
    print("To activate the virtual environment, run:")
    print(f"  {activate_cmd}")

def main():
    """Main setup function."""
    try:
        check_python_version()
        
        print("\nvAIn Virtual Environment Setup")
        print("=" * 30)
        
        if create_venv():
            install_dependencies()
        print_activation_instructions()
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\nSetup cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
