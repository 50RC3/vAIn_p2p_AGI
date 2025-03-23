import os
import sys
import subprocess
import platform
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REQUIRED_PYTHON = ">=3.8.0"

def check_python_version():
    """Verify that the correct version of Python is being used."""
    import pkg_resources
    try:
        pkg_resources.require(f"python{REQUIRED_PYTHON}")
    except pkg_resources.VersionConflict:
        logger.error(f"This project requires Python {REQUIRED_PYTHON}")
        sys.exit(1)

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
    
    logger.info("Installing dependencies...")
    subprocess.run([pip_cmd, "install", "--upgrade", "pip"], check=True)
    subprocess.run([pip_cmd, "install", "-r", "requirements.txt"], check=True)

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
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
