#!/usr/bin/env python3
"""
Terminal command shortcuts and aliases for the vAIn system.
This script provides useful shortcuts for common development tasks.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Repository root detection
REPO_ROOT = Path(__file__).parent.parent.absolute()

# Define common commands with shortcuts
COMMAND_SHORTCUTS = {
    # Launcher shortcuts
    "l0": "python main.py", 
    "l1": "python main.py --debug",
    "l2": "python main.py --verbose",
    "l9": "python main.py --interactive",
    
    # Development tools
    "d0": "pytest",
    "d1": "pytest -xvs",
    "d2": "flake8",
    "d3": "mypy .",
    
    # Network commands
    "n0": "python -m network.tools.node_status",
    "n1": "python -m network.tools.peer_discovery",
    
    # Model commands
    "m0": "python -m ai_core.tools.model_status",
    "m1": "python -m training.train --local",
    
    # Configuration commands
    "c0": "python -m tools.config_manager list",
    "c1": "python -m tools.config_manager show",
    "c2": "python -m tools.config_manager update",
    "c3": "python -m tools.config_manager update-all",
    "c4": "python -m tools.config_manager validate",
    
    # Utility commands
    "u0": "python -m tools.system_check",
    "u1": "python -m tools.benchmark",
    
    # Documentation generation 
    "docs": "python -m tools.generate_docs",
    
    # Help command
    "help": "list-commands"
}

def list_commands():
    """Display all available command shortcuts."""
    print("\nAvailable vAIn command shortcuts:")
    print("-" * 50)
    print(f"{'Shortcut':<10} {'Command':<40} {'Description'}")
    print("-" * 50)
    
    # Launcher commands
    print("\n🚀 Launcher Commands:")
    print(f"{'l0':<10} {'python main.py':<40} {'Standard launch'}")
    print(f"{'l1':<10} {'python main.py --debug':<40} {'Debug mode'}")
    print(f"{'l2':<10} {'python main.py --verbose':<40} {'Verbose mode'}")
    print(f"{'l9':<10} {'python main.py --interactive':<40} {'Interactive mode'}")
    
    # Development tools
    print("\n🛠️  Development Tools:")
    print(f"{'d0':<10} {'pytest':<40} {'Run all tests'}")
    print(f"{'d1':<10} {'pytest -xvs':<40} {'Detailed test output'}")
    print(f"{'d2':<10} {'flake8':<40} {'Code style check'}")
    print(f"{'d3':<10} {'mypy .':<40} {'Type checking'}")
    
    # Network commands
    print("\n🌐 Network Commands:")
    print(f"{'n0':<10} {'network.tools.node_status':<40} {'Show node status'}")
    print(f"{'n1':<10} {'network.tools.peer_discovery':<40} {'Discover peers'}")
    
    # Model commands
    print("\n🧠 Model Commands:")
    print(f"{'m0':<10} {'ai_core.tools.model_status':<40} {'Show model status'}")
    print(f"{'m1':<10} {'training.train --local':<40} {'Train model locally'}")
    
    # Configuration commands
    print("\n⚙️ Configuration Commands:")
    print(f"{'c0':<10} {'tools.config_manager list':<40} {'List all configs'}")
    print(f"{'c1':<10} {'tools.config_manager show':<40} {'Show config details'}")
    print(f"{'c2':<10} {'tools.config_manager update':<40} {'Update a config'}")
    print(f"{'c3':<10} {'tools.config_manager update-all':<40} {'Update all configs'}")
    print(f"{'c4':<10} {'tools.config_manager validate':<40} {'Validate configs'}")
    
    # Utility commands
    print("\n🔧 Utility Commands:")
    print(f"{'u0':<10} {'tools.system_check':<40} {'System check'}")
    print(f"{'u1':<10} {'tools.benchmark':<40} {'System benchmark'}")
    
    # Documentation
    print("\n📚 Documentation:")
    print(f"{'docs':<10} {'tools.generate_docs':<40} {'Generate documentation'}")
    print()

def main():
    parser = argparse.ArgumentParser(description="vAIn terminal command shortcuts")
    parser.add_argument("command", nargs='?', default="help", 
                        help="Command shortcut to run")
    parser.add_argument("args", nargs=argparse.REMAINDER,
                        help="Additional arguments to pass to the command")
    
    args = parser.parse_args()
    
    if args.command == "help" or args.command == "--help":
        list_commands()
        return 0
        
    if args.command in COMMAND_SHORTCUTS:
        cmd = COMMAND_SHORTCUTS[args.command]
        if cmd == "list-commands":
            list_commands()
            return 0
            
        # Build and execute the command
        full_cmd = f"{cmd} {' '.join(args.args)}"
        print(f"Executing: {full_cmd}")
        try:
            result = subprocess.run(full_cmd, shell=True, check=True, cwd=REPO_ROOT)
            return result.returncode
        except subprocess.CalledProcessError as e:
            print(f"Command failed with exit code {e.returncode}")
            return e.returncode
        except Exception as e:
            print(f"Error executing command: {e}")
            return 1
    else:
        print(f"Unknown command shortcut: '{args.command}'")
        list_commands()
        return 1

if __name__ == "__main__":
    sys.exit(main())
