#!/usr/bin/env python3
"""
vAIn P2P AGI System Startup Script

This script initializes all system components in the correct order,
ensuring proper dependency resolution and configuration.
"""
import os
import sys
import asyncio
import logging
import argparse
import time
from pathlib import Path

# Configure basic logging before imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("startup")

def setup_paths():
    """Add project root to Python path to ensure imports work properly"""
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        
setup_paths()

# Import components only after path setup
from .utils.dependency_checker import check_dependencies, check_module_dependencies, install_dependencies
from .core.constants import BASE_DIR
from .core.interactive_utils import InteractionLevel


async def initialize_system(args):
    """Initialize the core system components"""
    try:
        # Set up logging first
        setup_logging(args.log_level)
        
        # Initialize resource manager
        resource_manager = await core.resource_manager.ResourceManager.create()
        
        # Initialize configuration
        config = load_configuration(args.config)
        
        return resource_manager, config
    except Exception as e:
        logger.critical(f"Failed to initialize system: {e}", exc_info=True)
        return None, None

async def start_services(args, config, resource_manager):
    """Start all required services based on configuration"""
    try:
        services = []
        
        # Start P2P networking if enabled
        if args.enable_p2p:
            p2p_service = await initialize_p2p_network(config, args.port)
            services.append(p2p_service)
            
        # Start API server if enabled
        if args.enable_api:
            api_service = await initialize_api_server(config, args.api_port)
            services.append(api_service)
            
        return services
    except Exception as e:
        logger.critical(f"Failed to start services: {e}", exc_info=True)
        return []

async def main():
    """Main entry point for the application"""
    parser = argparse.ArgumentParser(description='Start the P2P AGI system')
    parser.add_argument('--config', type=str, default='config.json', help='Path to configuration file')
    parser.add_argument('--log-level', type=str, default='info', choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help='Set the logging level')
    parser.add_argument('--enable-p2p', action='store_true', help='Enable P2P networking')
    parser.add_argument('--port', type=int, default=8000, help='P2P network port')
    parser.add_argument('--enable-api', action='store_true', help='Enable API server')
    parser.add_argument('--api-port', type=int, default=5000, help='API server port')
    parser.add_argument('--ui-mode', type=str, default='web', choices=['web', 'gui', 'cli', 'none'],
                        help='User interface mode')
    
    args = parser.parse_args()
    
    # Initialize system
    resource_manager, config = await initialize_system(args)
    if not resource_manager or not config:
        sys.exit(1)
    
    # Start services
    services = await start_services(args, config, resource_manager)
    
    # Start UI based on mode
    if args.ui_mode == 'web':
        ui = await initialize_web_ui(config)
    elif args.ui_mode == 'gui':
        ui = await initialize_gui(config)
    elif args.ui_mode == 'cli':
        ui = await initialize_cli(config)
    else:
        ui = None
        
    try:
        # Run the main event loop
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutdown signal received")
    finally:
        # Clean shutdown
        await shutdown_services(services, resource_manager, ui)
        logger.info("System shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())
