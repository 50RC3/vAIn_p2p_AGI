import subprocess
import sys
import os

def install_debug_tools():
    """Install Python debugging tools"""
    print("Installing debugging tools...")
    debug_packages = [
        'debugpy',
        'ptvsd',  # Additional debugging support
    ]
    
    success = True
    for package in debug_packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError:
            print(f"Error installing {package}")
            success = False
    
    return success

if __name__ == '__main__':
    # Check if we're in a virtual environment
    in_venv = sys.prefix != sys.base_prefix
    if not in_venv:
        print("Warning: You are not running in a virtual environment.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Installation aborted.")
            sys.exit(1)
    
    if install_debug_tools():
        print("\nDebugging tools installed successfully!")
        print("You can now use: python -m debugpy --listen 5678 --wait-for-client your_script.py")
    else:
        print("\nError installing one or more debugging tools")
