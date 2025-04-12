#!/usr/bin/env python3
"""
Debug entry point for the vAIn_p2p_AGI project
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from pathlib import Path
from utils.debug_launcher import launch_with_debugpy

def main():
    # Get the path to main.py relative to this script
    main_script = str(Path(__file__).parent / "main.py")
    
    # Check if we should configure with default configs
    if "--init-config" in sys.argv or any(arg.startswith("--config=") for arg in sys.argv):
        try:
            # Add proper path for importing
            sys.path.insert(0, str(Path(__file__).parent))
            
            # Create default config files if needed
            from tools.config_manager import ConfigManager
            ConfigManager().create_default_configs(overwrite="--force-config" in sys.argv)
            print("Default configurations created")
            
            # Remove flags that were handled
            if "--init-config" in sys.argv:
                sys.argv.remove("--init-config")
            if "--force-config" in sys.argv:
                sys.argv.remove("--force-config")
        except Exception as e:
            print(f"Failed to initialize configurations: {e}")
    
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
