#!/usr/bin/env python
"""
Debug Launcher - Utility to launch debugging sessions with port conflict handling
"""
import os
import sys
import argparse
import logging
from pathlib import Path

# Add the project root to the path to allow importing from the project
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.debug_port_manager import DebugPortManager

logger = logging.getLogger(__name__)

def setup_logging(verbose: bool = False):
    """Configure logging for the script."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Debug Launcher with port conflict handling')
    parser.add_argument('script', help='Path to the script to debug')
    parser.add_argument('--port', type=int, default=DebugPortManager.DEFAULT_DEBUG_PORT,
                      help=f'Debug port (default: {DebugPortManager.DEFAULT_DEBUG_PORT})')
    parser.add_argument('--auto-resolve', action='store_true', 
                      help='Automatically resolve port conflicts')
    parser.add_argument('--kill-existing', action='store_true',
                      help='Kill processes using the specified port')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    
    return parser.parse_args()

def main():
    """Main entry point for the debug launcher."""
    args = parse_args()
    setup_logging(args.verbose)
    
    # Normalize script path
    script_path = os.path.abspath(args.script)
    if not os.path.exists(script_path):
        logger.error(f"Script not found: {script_path}")
        return 1
    
    port = args.port
    
    # Check if the port is already in use
    if DebugPortManager.is_port_in_use(port):
        proc = DebugPortManager.find_process_using_port(port)
        proc_info = f"{proc.name()} (PID: {proc.pid})" if proc else "an unknown process"
        
        logger.warning(f"Port {port} is already in use by {proc_info}")
        
        if args.kill_existing and proc:
            # Try to kill the process
            if DebugPortManager.kill_process_using_port(port, force=True):
                logger.info(f"Successfully killed process using port {port}")
            else:
                logger.error(f"Failed to kill process using port {port}")
                
                if args.auto_resolve:
                    # Find an alternative port
                    port = DebugPortManager.find_available_port()
                    logger.info(f"Using alternative port: {port}")
                else:
                    logger.error(f"Debug port {args.port} is in use and --auto-resolve is not enabled")
                    return 1
        elif args.auto_resolve:
            # Find an alternative port
            port = DebugPortManager.find_available_port()
            logger.info(f"Using alternative port: {port}")
        else:
            logger.error("Port conflict detected. Use --auto-resolve to find another port "
                       "or --kill-existing to terminate the process using the port.")
            return 1
    
    # Construct and execute the debug command
    cmd = DebugPortManager.get_debug_command(script_path, port)
    logger.info(f"Starting debug session with command: {cmd}")
    
    return os.system(cmd)

if __name__ == "__main__":
    sys.exit(main())
