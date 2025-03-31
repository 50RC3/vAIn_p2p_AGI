"""
System utilities for vAIn P2P AGI system.
Provides functionality for system management, process control, and diagnostics.
"""

import os
import sys
import logging
import platform
import psutil
import asyncio
import subprocess
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


def get_system_info() -> Dict[str, str]:
    """
    Get basic system information
    
    Returns:
        Dict containing system information
    """
    info = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
    }
    
    # Add CPU info
    try:
        info["cpu_cores"] = str(psutil.cpu_count(logical=False))
        info["cpu_threads"] = str(psutil.cpu_count(logical=True))
    except Exception:
        info["cpu_cores"] = "Unknown"
        info["cpu_threads"] = "Unknown"
        
    # Add memory info
    try:
        vm = psutil.virtual_memory()
        info["total_memory"] = f"{vm.total / (1024**3):.2f} GB"
    except Exception:
        info["total_memory"] = "Unknown"
        
    return info


async def get_resource_usage() -> Dict[str, Union[float, str]]:
    """
    Get current resource usage of the system
    
    Returns:
        Dict containing resource usage information
    """
    usage = {}
    
    try:
        # Memory usage
        memory = psutil.virtual_memory()
        usage["memory_percent"] = memory.percent
        usage["memory_used"] = f"{memory.used / (1024**3):.2f} GB"
        usage["memory_available"] = f"{memory.available / (1024**3):.2f} GB"
        
        # CPU usage
        usage["cpu_percent"] = psutil.cpu_percent(interval=0.5)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        usage["disk_percent"] = disk.percent
        usage["disk_free"] = f"{disk.free / (1024**3):.2f} GB"
        
        # Process information
        process = psutil.Process(os.getpid())
        usage["process_memory"] = f"{process.memory_info().rss / (1024**2):.2f} MB"
        usage["process_cpu"] = process.cpu_percent(interval=0.5)
        usage["process_threads"] = process.num_threads()
        
        # Check if GPU is available
        if "torch" in sys.modules:
            import torch
            if torch.cuda.is_available():
                usage["gpu_available"] = True
                usage["gpu_count"] = torch.cuda.device_count()
                usage["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "N/A"
                
                # Get GPU memory usage for the first GPU
                if torch.cuda.device_count() > 0:
                    usage["gpu_memory_allocated"] = f"{torch.cuda.memory_allocated(0) / (1024**2):.2f} MB"
                    usage["gpu_memory_reserved"] = f"{torch.cuda.memory_reserved(0) / (1024**2):.2f} MB"
            else:
                usage["gpu_available"] = False
    except Exception as e:
        logger.error(f"Error getting resource usage: {e}")
        usage["error"] = str(e)
        
    return usage


async def check_port_available(port: int) -> bool:
    """
    Check if a port is available
    
    Args:
        port: Port number to check
        
    Returns:
        True if port is available, False otherwise
    """
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) != 0


async def verify_system_compatibility() -> Tuple[bool, List[str]]:
    """
    Verify that the system meets minimum requirements
    
    Returns:
        Tuple of (is_compatible, warnings)
    """
    warnings = []
    is_compatible = True
    
    # Check Python version
    if sys.version_info < (3, 8):
        warnings.append(f"Python 3.8+ is required, but you have {platform.python_version()}")
        is_compatible = False
    
    # Check memory
    try:
        vm = psutil.virtual_memory()
        if vm.total < 4 * 1024 * 1024 * 1024:  # 4 GB
            warnings.append(f"At least 4GB of RAM recommended, but you have {vm.total / (1024**3):.1f} GB")
            if vm.total < 2 * 1024 * 1024 * 1024:  # 2 GB
                is_compatible = False
    except Exception:
        warnings.append("Could not verify system memory")
    
    # Check if CUDA is available when torch is installed
    if "torch" in sys.modules:
        import torch
        if not torch.cuda.is_available():
            warnings.append("CUDA is not available - AI operations will be slower")
    
    # Check disk space
    try:
        disk = psutil.disk_usage('/')
        if disk.free < 5 * 1024 * 1024 * 1024:  # 5 GB
            warnings.append(f"At least 5GB of free disk space recommended, but you have {disk.free / (1024**3):.1f} GB")
            if disk.free < 1 * 1024 * 1024 * 1024:  # 1 GB
                is_compatible = False
    except Exception:
        warnings.append("Could not verify free disk space")
    
    return is_compatible, warnings


async def setup_environment() -> None:
    """Setup environment for proper system operation"""
    # Ensure critical paths exist
    from core.constants import (
        TEMP_DIR, DATA_DIR, CONFIG_DIR, LOGS_DIR, MODELS_DIR, CACHE_DIR
    )
    
    for directory in [TEMP_DIR, DATA_DIR, CONFIG_DIR, LOGS_DIR, MODELS_DIR, CACHE_DIR]:
        directory.mkdir(exist_ok=True, parents=True)
    
    # Setup logging directories with date-based organization
    log_date_dir = LOGS_DIR / datetime.now().strftime("%Y-%m-%d")
    log_date_dir.mkdir(exist_ok=True)
    
    # Set environment variables if needed
    os.environ["VAIN_LOG_DIR"] = str(log_date_dir)


async def register_shutdown_handler(handler) -> None:
    """Register a shutdown handler for graceful application exit"""
    import atexit
    import signal
    
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, initiating shutdown...")
        if asyncio.iscoroutinefunction(handler):
            # Create a new event loop for the shutdown handler
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(handler())
            finally:
                loop.close()
        else:
            handler()
        # Exit with a special code for signals
        sys.exit(128 + sig)
    
    # Register for SIGINT and SIGTERM
    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(sig, signal_handler)
    
    # Register normal exit handler
    atexit.register(lambda: handler() if not asyncio.iscoroutinefunction(handler) 
                   else print("Cannot run async shutdown handler from atexit"))
