"""
Recovery script for restarting the system when critical failures are detected.
"""
import os
import sys
import time
import logging
import asyncio
import subprocess
from pathlib import Path

async def recover(alert, logger):
    """
    Attempt to restart the system when a critical failure is detected.
    
    Args:
        alert: The LogAlert object that triggered the recovery
        logger: Logger to use for logging recovery actions
        
    Returns:
        bool: True if recovery was successful, False otherwise
    """
    logger.info(f"Running system restart recovery for alert: {alert.message}")
    
    # 1. Attempt to save any unsaved data
    try:
        await save_critical_data(logger)
    except Exception as e:
        logger.error(f"Error saving critical data: {e}")
    
    # 2. Stop critical services
    try:
        await stop_services(logger)
    except Exception as e:
        logger.error(f"Error stopping services: {e}")
    
    # 3. Restart the main process
    try:
        restart_main_process(logger)
        return True
    except Exception as e:
        logger.error(f"Error restarting main process: {e}")
        return False

async def save_critical_data(logger):
    """Save any critical data before restart"""
    logger.info("Saving critical data...")
    
    try:
        # Try to save model state if available
        try:
            from ai_core.system_coordinator import SystemCoordinator
            coordinator = SystemCoordinator.get_instance()
            if coordinator:
                await coordinator.save_state(emergency=True)
                logger.info("Saved system state")
        except ImportError:
            pass
    except Exception as e:
        logger.error(f"Error saving system state: {e}")

async def stop_services(logger):
    """Stop services gracefully"""
    logger.info("Stopping services...")
    
    try:
        # Try to stop services through system coordinator
        try:
            from ai_core.system_coordinator import SystemCoordinator
            coordinator = SystemCoordinator.get_instance()
            if coordinator:
                await coordinator.shutdown()
                logger.info("Stopped services through system coordinator")
                return
        except ImportError:
            pass
            
        # Fallback: use direct process management
        if sys.platform == 'win32':
            os.system('taskkill /f /im python.exe')
        else:
            # More careful approach for Unix
            my_pid = os.getpid()
            os.system(f'pkill -TERM -P {my_pid} python')
            
    except Exception as e:
        logger.error(f"Error stopping services: {e}")

def restart_main_process(logger):
    """Restart the main process"""
    logger.info("Restarting main process...")
    
    # Find the main script path (start.py, main.py, or production.py)
    possible_scripts = ["start.py", "main.py", "production.py", "run_app.py"]
    main_script = None
    
    for script in possible_scripts:
        if os.path.exists(script):
            main_script = script
            break
    
    if main_script:
        # Start new process
        cmd = [sys.executable, main_script] + sys.argv[1:]
        logger.info(f"Restarting with command: {' '.join(cmd)}")
        
        # Use different methods based on platform
        if sys.platform == 'win32':
            subprocess.Popen(
                cmd, 
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
            )
        else:
            subprocess.Popen(
                cmd, 
                start_new_session=True
            )
        
        # Exit after a short delay to allow new process to start
        time.sleep(1)
        return True
    else:
        logger.error("Could not find main script to restart")
        return False