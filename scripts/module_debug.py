#!/usr/bin/env python3
"""
Docker build debugging module for vAIn_p2p_AGI.
Helps diagnose and fix common Docker build issues.
"""

import os
import sys
import subprocess
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("docker.debug")

class DockerDebugError(Exception):
    """Exception raised for errors in the Docker debugging module."""
    pass

def check_docker_installed() -> bool:
    """Check if Docker is installed and accessible."""
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            logger.info(f"Docker installed: {result.stdout.strip()}")
            return True
        else:
            logger.error(f"Docker check failed: {result.stderr.strip()}")
            return False
    except FileNotFoundError:
        logger.error("Docker not found in PATH")
        return False

def validate_docker_build_command(args: List[str]) -> Tuple[bool, str]:
    """
    Validate a Docker build command arguments.
    
    Args:
        args: List of command arguments
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not args:
        return False, "No arguments provided for docker build command"
    
    # Check for required -t/--tag argument with value
    has_tag = False
    i = 0
    while i < len(args):
        if args[i] in ['-t', '--tag']:
            if i + 1 >= len(args) or args[i+1].startswith('-'):
                return False, f"Flag needs an argument: '{args[i][1:]}' in {args[i]}"
            has_tag = True
            i += 2
        else:
            i += 1
    
    if not has_tag:
        return False, "Missing required tag (-t or --tag) in docker build command"
    
    # Check if a build context is provided
    if not any(not arg.startswith('-') for arg in args):
        return False, "Missing build context (PATH | URL | -)"
    
    return True, ""

def fix_docker_build_command(args: List[str]) -> List[str]:
    """
    Attempt to fix a Docker build command.
    
    Args:
        args: Original command arguments
        
    Returns:
        Fixed command arguments
    """
    fixed_args = args.copy()
    
    # Fix missing tag value
    i = 0
    while i < len(fixed_args):
        if fixed_args[i] in ['-t', '--tag'] and (i + 1 >= len(fixed_args) or fixed_args[i+1].startswith('-')):
            # Insert default image name
            project_dir = Path.cwd().name
            fixed_args.insert(i+1, f"{project_dir.lower()}:latest")
            logger.info(f"Added default tag: {project_dir.lower()}:latest")
            break
        i += 1
    
    # If no -t/--tag at all, add it with default value
    if not any(arg in ['-t', '--tag'] for arg in fixed_args):
        project_dir = Path.cwd().name
        fixed_args.extend(['-t', f"{project_dir.lower()}:latest"])
        logger.info(f"Added missing tag option: -t {project_dir.lower()}:latest")
    
    # Add build context if missing
    if not any(not arg.startswith('-') for arg in fixed_args):
        fixed_args.append('.')
        logger.info("Added current directory as build context")
    
    return fixed_args

def build_with_diagnostics(args: List[str]) -> int:
    """
    Run docker build with diagnostics.
    
    Args:
        args: Build command arguments
        
    Returns:
        Return code from docker build
    """
    # Check if Docker is installed
    if not check_docker_installed():
        logger.error("Docker is not installed or not in PATH")
        print("\nTo install Docker, visit: https://docs.docker.com/get-docker/")
        return 1
    
    # Validate build arguments
    is_valid, error_message = validate_docker_build_command(args)
    if not is_valid:
        logger.error(f"Invalid docker build command: {error_message}")
        fixed_args = fix_docker_build_command(args)
        
        print("\nOriginal command was invalid:")
        print(f"  docker build {' '.join(args)}")
        print("\nSuggested fix:")
        print(f"  docker build {' '.join(fixed_args)}")
        print("\nWould you like to run the fixed command? (y/n)")
        
        choice = input().strip().lower()
        if choice != 'y':
            logger.info("User chose not to run the fixed command")
            return 1
        
        args = fixed_args
    
    # Run the build with full output
    logger.info(f"Running: docker build {' '.join(args)}")
    
    try:
        # Use subprocess to run the command with real-time output
        process = subprocess.Popen(
            ["docker", "build"] + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Print output in real-time
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
            sys.stdout.flush()
        
        process.wait()
        return process.returncode
    except Exception as e:
        logger.error(f"Error executing docker build: {str(e)}")
        return 1

def show_docker_help() -> None:
    """Display Docker build help information."""
    try:
        subprocess.run(["docker", "buildx", "build", "--help"], check=False)
    except Exception as e:
        logger.error(f"Error showing Docker help: {str(e)}")

def main() -> int:
    """Main entry point for the Docker debug module."""
    parser = argparse.ArgumentParser(description="Debug Docker build issues for vAIn_p2p_AGI")
    
    parser.add_argument("--check", action="store_true", help="Check Docker installation")
    parser.add_argument("--fix", action="store_true", help="Fix and run Docker build command")
    parser.add_argument("--help-docker", action="store_true", help="Show Docker build help")
    parser.add_argument("build_args", nargs="*", help="Docker build arguments")
    
    args = parser.parse_args()
    
    try:
        if args.check:
            check_docker_installed()
            return 0
        
        if args.help_docker:
            show_docker_help()
            return 0
        
        if args.fix or args.build_args:
            return build_with_diagnostics(args.build_args)
        
        # If no arguments provided, show help
        parser.print_help()
        return 0
        
    except DockerDebugError as e:
        logger.error(str(e))
        return 1
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Unexpected error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
