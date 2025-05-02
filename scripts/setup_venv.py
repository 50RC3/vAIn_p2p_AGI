import os
import sys
import subprocess
import platform

############  ✨ Windsurf Command 🌟  #############
def create_venv() -> None:
    """Create and setup Python virtual environment.

    This function will create a virtual environment, upgrade pip and install
    the Python packages specified in the requirements.txt file.
    """

    """Create and setup Python virtual environment"""
    try:
        # Create venv
        subprocess.check_call([sys.executable, '-m', 'venv', 'venv'])

        # Get venv Python path
        if platform.system() == "Windows":
            python_path = os.path.join("venv", "Scripts", "python")
        else:
            python_path = os.path.join("venv", "bin", "python")

        # Upgrade pip
        subprocess.check_call([python_path, '-m', 'pip', 'install', '--upgrade', 'pip'])

        # Install requirements
        subprocess.check_call([python_path, '-m', 'pip', 'install', '-r', 'requirements.txt'])

        print("Virtual environment setup complete!")

    except subprocess.CalledProcessError as e:
        print(f"Error setting up virtual environment: {str(e)}")
        sys.exit(1)
########  c377719d-3126-4cbb-8200-696b58f963ed  ########

if __name__ == "__main__":
    create_venv()
