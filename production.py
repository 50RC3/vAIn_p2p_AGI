#!/usr/bin/env python3
"""
Production entry point for vAIn P2P AGI system.
This script initializes all components and ensures proper coordination.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import asyncio
import logging
import signal
import argparse
import traceback
from typing import Dict, Any, Optional
import json
import time

# Configure logging first
from utils.logger_init import init_logger
logger = init_logger(name="vAIn", log_dir="logs", default_level=logging.INFO)

# Import system components
try:
    from ai_core.system_coordinator import SystemCoordinator, SystemCoordinatorConfig
except ImportError as e:
    logger.critical(f"Failed to import core components: {e}")
    sys.exit(1)

# Global variables
coordinator = None
running = True

def signal_handler(sig, frame):
    """Handle interruption signals"""
    global running
    if running:
        logger.info(f"Signal {sig} received, initiating graceful shutdown...")
        running = False
    else:
        logger.warning("Forced termination")
        sys.exit(1)

async def shutdown_system():
    """Shutdown the system gracefully"""
    global coordinator
    if coordinator:
        logger.info("Shutting down system")
        try:
            # Set a timeout for shutdown to avoid hanging
            shutdown_task = asyncio.create_task(coordinator.shutdown())
            try:
                success = await asyncio.wait_for(shutdown_task, timeout=30.0)
                if success:
                    logger.info("System shutdown successfully")
                else:
                    logger.warning("System shutdown completed with issues")
            except asyncio.TimeoutError:
                logger.error("System shutdown timed out after 30 seconds")
        except Exception as e:
            logger.error(f"Error during system shutdown: {e}")
            logger.debug(traceback.format_exc())
    else:
        logger.warning("System coordinator not initialized, nothing to shut down")

async def system_monitor():
    """Monitor system health periodically"""
    global coordinator, running
    check_interval = 60  # seconds
    
    try:
        while running:
            if coordinator and coordinator.is_initialized:
                try:
                    # Get system status
                    status = await coordinator.get_system_status()
                    
                    # Log basic health information
                    logger.info(f"System health: All components running={all(status['components'].values())}")
                    
                except Exception as e:
                    logger.error(f"Error checking system health: {e}")
            
            # Wait for next check, while being responsive to shutdown signals
            for _ in range(check_interval):
                if not running:
                    break
                await asyncio.sleep(1)
    except asyncio.CancelledError:
        logger.info("System monitor task cancelled")
    except Exception as e:
        logger.error(f"Error in system monitor: {e}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="vAIn P2P AGI Production System")
    
    parser.add_argument("--config", type=str, default="config/production.json",
                      help="Path to configuration file")
    parser.add_argument("--log-level", type=str, default="INFO",
                      choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                      help="Logging level")
    parser.add_argument("--no-interactive", action="store_true",
                      help="Disable interactive mode")
    parser.add_argument("--metrics-path", type=str, default="./logs/metrics",
                      help="Path to store metrics data")
    
    return parser.parse_args()

async def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file"""
    try:
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                return json.load(f)
        else:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {}
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        return {}

async def main():
    """Main entry point for the system"""
    global coordinator, running
    
    # Parse command line arguments
    args = parse_args()
    
    try:
        # Set up signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Load configuration
        config_data = await load_config(args.config)
        
        # Create system coordinator config
        coordinator_config = SystemCoordinatorConfig(
            interactive=not args.no_interactive,
            log_level=args.log_level,
            metrics_storage_path=args.metrics_path,
            # Override with config file values if present
            **config_data.get("coordinator", {})
        )
        
        # Create and initialize system coordinator
        logger.info("Initializing system coordinator")
        coordinator = SystemCoordinator(coordinator_config)
        
        # Configure system status callback for monitoring
        coordinator.register_callback('error', lambda data: logger.error(f"System error: {data}"))
        coordinator.register_callback('shutdown_complete', lambda data: setattr(sys.modules[__name__], 'running', False))
        
        # Initialize the system with timeout
        try:
            init_task = asyncio.create_task(coordinator.initialize())
            init_success = await asyncio.wait_for(init_task, timeout=60.0)
        except asyncio.TimeoutError:
            logger.error("System initialization timed out after 60 seconds")
            return 1
        
        if not init_success:
            logger.error(f"System initialization failed: {coordinator.initialization_error}")
            return 1
            
        logger.info("System initialization complete")
        
        # Start system monitor
        monitor_task = asyncio.create_task(system_monitor())
        
        # Continue running until interrupted
        while running:
            await asyncio.sleep(1)
            
        # System shutdown was requested
        logger.info("Shutting down...")
        
        # Cancel monitor task
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
            
        # Perform graceful shutdown
        await shutdown_system()
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        await shutdown_system()
        return 1
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}")
        logger.critical(traceback.format_exc())
        await shutdown_system()
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received during startup")
        sys.exit(1)
