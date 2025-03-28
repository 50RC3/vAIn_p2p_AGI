#!/usr/bin/env python3
"""
Debug Launcher - Handles port conflicts for debugging sessions
"""
import os
import sys
import subprocess
import logging
from typing import Optional, List, Union
import argparse
from pathlib import Path

# Import the debug port manager
from utils.debug_port_manager import DebugPortManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("DebugLauncher")

def launch_with_debugpy(
    target_script: str, 
    port: int = None, 
    wait_for_client: bool = True, 
    script_args: List[str] = None,
    auto_resolve: bool = False
) -> int:
    """
    Launch a Python script with debugpy attached, handling port conflicts automatically.
    
    Args:
        target_script: Path to the Python script to debug
        port: Debug port to use (if None, will use default or find available port)
        wait_for_client: Whether to wait for client to attach before running
        script_args: Additional arguments to pass to the target script
        auto_resolve: If True, automatically find an available port without asking
        
    Returns:
        Exit code from the process
    """
    if port is None:
        port = DebugPortManager.DEFAULT_DEBUG_PORT
    
    # Check if port is in use
    if DebugPortManager.is_port_in_use(port):
        logger.warning(f"Debug port {port} is already in use")
        
        # Find the process using this port
        process = DebugPortManager.find_process_using_port(port)
        
        if auto_resolve:
            # Automatically find an available port without asking
            new_port = DebugPortManager.find_available_port()
            logger.info(f"Automatically switching to available port: {new_port}")
            port = new_port
        elif process:
            logger.info(f"Port {port} is being used by process {process.pid} ({process.name()})")
            
            # Ask user what to do
            print(f"\nPort {port} is already in use by {process.name()} (PID: {process.pid})")
            choice = input("Options: [K]ill process, [F]ind another port, [A]bort? ").strip().upper()
            
            if choice.startswith('K'):
                # Try to kill the process
                if DebugPortManager.kill_process_using_port(port):
                    logger.info(f"Successfully killed process using port {port}")
                else:
                    logger.error(f"Failed to kill process using port {port}")
                    # Find another port
                    port = DebugPortManager.find_available_port()
                    logger.info(f"Found available port: {port}")
            elif choice.startswith('F'):
                # Find another port
                port = DebugPortManager.find_available_port()
                logger.info(f"Found available port: {port}")
            else:
                logger.info("Debugging session aborted by user")
                return 1
        else:
            # Process not found but port is in use, find another port
            port = DebugPortManager.find_available_port()
            logger.info(f"Found available port: {port}")
    
    # Build command with proper arguments
    cmd = [sys.executable, "-m", "debugpy", "--listen", str(port)]
    
    if wait_for_client:
        cmd.append("--wait-for-client")
    
    cmd.append(target_script)
    
    if script_args:
        cmd.extend(script_args)
    
    # Show the command that will be executed
    logger.info(f"Launching: {' '.join(cmd)}")
    
    # Execute the command
    try:
        print(f"\nStarting debug session on port {port}...")
        if wait_for_client:
            print("Waiting for client to attach...")
        
        process = subprocess.run(cmd)
        return process.returncode
    except KeyboardInterrupt:
        print("\nDebug session terminated by user")
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        logger.error(f"Error launching debug session: {e}")
        return 1

def main():
    """Command-line entry point"""
    parser = argparse.ArgumentParser(description="Launch Python scripts with debugpy, handling port conflicts")
    parser.add_argument("script", help="Path to the Python script to debug")
    parser.add_argument("--port", "-p", type=int, default=DebugPortManager.DEFAULT_DEBUG_PORT,
                        help=f"Debug port (default: {DebugPortManager.DEFAULT_DEBUG_PORT})")
    parser.add_argument("--no-wait", action="store_true", help="Don't wait for client to attach")
    parser.add_argument("--auto-resolve", "-a", action="store_true", 
                        help="Automatically find an available port if the specified port is in use")
    parser.add_argument("args", nargs="*", help="Arguments to pass to the script")
    
    args = parser.parse_args()
    
    script_path = Path(args.script)
    if not script_path.exists():
        logger.error(f"Script not found: {script_path}")
        return 1
    
    return launch_with_debugpy(
        str(script_path.resolve()), 
        port=args.port, 
        wait_for_client=not args.no_wait,
        script_args=args.args,
        auto_resolve=args.auto_resolve
    )

if __name__ == "__main__":
    sys.exit(main())
