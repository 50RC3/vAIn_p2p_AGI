import subprocess
import sys

def install_requirements():
    """Install required Python packages"""
    print("Installing required packages...")
    requirements = [
        'torch',
        'tqdm',
        'numpy', 
        'pandas',
        'fastapi',
        'pydantic',
        'python-dotenv',
        'psutil',
        'aiofiles',
        'aiologger'
    ]
    
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError:
            print(f"Error installing {package}")
            return False
    return True

if __name__ == '__main__':
    if install_requirements():
        print("\nAll dependencies installed successfully!")
    else:
        print("\nError installing one or more dependencies")
