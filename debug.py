#!/usr/bin/env python3
"""
Debug entry point for the vAIn_p2p_AGI project
"""
import sys
import os
from pathlib import Path
from utils.debug_launcher import launch_with_debugpy

def main():
    # Get the path to main.py relative to this script
    main_script = str(Path(__file__).parent / "main.py")
    
    # Pass any command line arguments to the main script
    script_args = sys.argv[1:] if len(sys.argv) > 1 else None
    
    # Launch with debugpy, using auto_resolve to automatically handle port conflicts
    return launch_with_debugpy(
        main_script, 
        script_args=script_args,
        auto_resolve=True  # Always automatically resolve port conflicts
    )

if __name__ == "__main__":
    sys.exit(main())
