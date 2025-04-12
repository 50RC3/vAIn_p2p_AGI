#!/usr/bin/env python3
"""
vAIn P2P AGI System Startup Script

This script initializes all system components in the correct order,
ensuring proper dependency resolution and configuration.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import asyncio
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

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
from utils.dependency_checker import check_dependencies, check_module_dependencies, install_dependencies
from core.constants import FEATURES, BASE_DIR
from core.interactive_utils import InteractionLevel


async def check_system_dependencies(args):
    """Check and install system dependencies if needed"""
    logger.info("Checking system dependencies...")
    
    success, missing = check_dependencies(required_only=args.minimal)
    
    if success:
        logger.info("✓ All required dependencies are installed.")
        return True
        
    logger.warning(f"Missing dependencies: {', '.join(missing)}")
    
    if args.auto_install or (not args.non_interactive and 
                            input("Install missing dependencies? (y/n): ").lower().startswith('y')):
        logger.info("Installing missing dependencies...")
        if install_dependencies(missing):
            logger.info("✓ Successfully installed all dependencies.")
            return True
        else:
            logger.error("Failed to install some dependencies.")
            if args.strict:
                return False
    elif args.strict:
        logger.error("Missing dependencies and --strict flag is set. Exiting.")
        return False
        
    return not args.strict


async def check_module_readiness(module_name: str, args) -> bool:
    """Check if a specific module is ready with all dependencies"""
    logger.info(f"Checking {module_name} module dependencies...")
    
    success, missing = check_module_dependencies(module_name)
    if success:
        logger.info(f"✓ Module {module_name} dependencies satisfied")
        return True
        
    logger.warning(f"Module {module_name} has missing dependencies: {', '.join(missing)}")
    
    if args.auto_install or (not args.non_interactive and 
                           input(f"Install {module_name} dependencies? (y/n): ").lower().startswith('y')):
        if install_dependencies(missing):
            logger.info(f"✓ Successfully installed {module_name} dependencies")
            return True
    
    if args.strict:
        logger.error(f"Cannot continue without {module_name} dependencies")
        return False
        
    return not args.strict


async def initialize_core_systems(args):
    """Initialize core system components"""
    logger.info("Initializing core systems...")
    
    # First check if the AI core module is ready
    if not await check_module_readiness("ai_core", args):
        return False
    
    try:
        # Import here to allow dependency installation first
        from ai_core.resource_management import ResourceManager
        from ai_core.module_registry import ModuleRegistry
        
        # Initialize resource manager first
        logger.info("Initializing resource manager...")
        resource_manager = ResourceManager()
        await resource_manager.initialize()
        logger.info("✓ Resource manager initialized")
        
        # Initialize module registry
        logger.info("Initializing module registry...")
        registry = ModuleRegistry.get_instance()
        success = await registry.initialize(resource_manager=resource_manager)
        if not success:
            logger.error("Failed to initialize module registry")
            return False
        logger.info("✓ Module registry initialized")
        
        # Load configurations
        from config import get_config
        config = get_config(interactive=not args.non_interactive)
        
        # Check if we need to update config interactively at startup
        if args.update_config and not args.non_interactive:
            logger.info("Updating configuration interactively")
            config.update_interactive()
        
        # Validate configuration
        if not args.skip_config_validation:
            valid = config.validate()
            if not valid:
                logger.error("Configuration validation failed")
                if args.strict:
                    return False
        
        logger.info("✓ Configuration loaded and validated")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize core systems: {e}", exc_info=True)
        return False


async def init_memory_system(args):
    """Initialize memory subsystem"""
    logger.info("Initializing memory subsystem...")
    
    if not await check_module_readiness("ai_core", args):
        return False
    
    try:
        # Import memory system
        from memory import MemoryManager
        
        # Initialize simple placeholder memory manager
        memory_manager = MemoryManager()
        logger.info("✓ Memory system initialized")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize memory system: {e}")
        if args.strict:
            return False
        return True  # Continue anyway unless strict mode


async def init_model_system(args):
    """Initialize model subsystem"""
    logger.info("Initializing model subsystem...")
    
    if not await check_module_readiness("ai_core", args):
        return False
    
    try:
        # Import model system
        from models import HybridMemorySystem
        
        # We don't initialize here as it depends on memory system
        # Just validate imports
        logger.info("✓ Model system imports validated")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize model system: {e}")
        if args.strict:
            return False
        return True  # Continue anyway unless strict mode


async def start_network_services(args):
    """Start networking services"""
    if args.no_network:
        logger.info("Network services disabled (--no-network)")
        return True
    
    if not await check_module_readiness("network", args):
        if args.strict:
            return False
        logger.warning("Network services may not function properly due to missing dependencies")
        
    try:
        logger.info("Starting network services...")
        # Import here to ensure dependencies were checked first
        from network.dht import DHT
        
        # Load network configuration
        try:
            from config import get_config
            config = get_config(interactive=False)
            network_config = {
                'bootstrap_nodes': args.bootstrap_nodes or ['localhost:8468'],
                'port': config.network.port or args.port,
                'max_retries': 3,
                'retry_delay': 1.0,
            }
            logger.info("Using network configuration from config system")
        except Exception:
            # Fallback to args
            network_config = {
                'bootstrap_nodes': args.bootstrap_nodes or ['localhost:8468'],
                'port': args.port,
                'max_retries': 3,
                'retry_delay': 1.0,
            }
            logger.warning("Using fallback network configuration")
        
        # Initialize DHT but don't start it yet
        dht = DHT(node_id=f"node_{int(time.time())}", config=network_config, interactive=not args.non_interactive)
        logger.info("✓ Network services initialized (not started)")
        
        # Store DHT instance for later use
        return dht
        
    except Exception as e:
        logger.error(f"Failed to initialize network services: {e}")
        if args.strict:
            return False
        return None  # Return None to indicate failure but not strict failure


async def start_ui(args):
    """Start the appropriate user interface"""
    if args.no_ui:
        logger.info("UI disabled (--no-ui)")
        return None
    
    try:
        logger.info("Starting user interface...")
        
        # Try to load interaction level from config
        interaction_level = InteractionLevel.NORMAL
        try:
            from config import get_config
            config = get_config(interactive=not args.non_interactive)
            if hasattr(config, 'system') and hasattr(config.system, 'get_interaction_level'):
                interaction_level = config.system.get_interaction_level()
                logger.info(f"Using interaction level from config: {interaction_level.name}")
        except Exception:
            # Determine interaction level from args as fallback
            if args.minimal:
                interaction_level = InteractionLevel.MINIMAL
            elif args.verbose:
                interaction_level = InteractionLevel.VERBOSE
            
        if args.console:
            try:
                from core.interactive_console import InteractiveConsole
                
                console = InteractiveConsole(interaction_level=interaction_level)
                logger.info("✓ Interactive console ready")
                return console
            except Exception as e:
                logger.error(f"Failed to start interactive console: {e}")
                return None
        else:
            try:
                # Check if ai_core.ui is available
                try:
                    from ai_core.ui import CommandLineInterface
                    cli = CommandLineInterface()
                    logger.info("✓ Command line interface ready")
                    return cli
                except ImportError:
                    # Fall back to interactive console
                    from core.interactive_console import InteractiveConsole
                    console = InteractiveConsole()
                    logger.info("✓ Falling back to interactive console")
                    return console
            except Exception as e:
                logger.error(f"Failed to start user interface: {e}")
                return None
    except Exception as e:
        logger.error(f"Error setting up UI: {e}")
        return None


async def startup_sequence(args):
    """Execute full system startup sequence"""
    start_time = time.time()
    logger.info("Starting vAIn P2P AGI System...")
    
    # Create necessary directories
    os.makedirs(BASE_DIR / "logs", exist_ok=True)
    os.makedirs(BASE_DIR / "config", exist_ok=True)
    
    # Check if config directory is empty and create default configs if needed
    config_dir = BASE_DIR / "config"
    if not list(config_dir.glob("*.json")) or args.init_config:
        logger.info("Creating default configurations...")
        try:
            from tools.config_manager import ConfigManager
            ConfigManager().create_default_configs()
        except ImportError:
            logger.warning("ConfigManager not found, skipping default config creation")
        except Exception as e:
            logger.error(f"Error creating default configurations: {e}")
    
    # Check dependencies first
    if not await check_system_dependencies(args):
        return False
    
    # Initialize core systems
    if not await initialize_core_systems(args):
        return False
    
    # Initialize memory subsystem
    if not await init_memory_system(args):
        if args.strict:
            return False
    
    # Initialize model subsystem
    if not await init_model_system(args):
        if args.strict:
            return False
    
    # Start network if enabled
    network = await start_network_services(args)
    if network is False:  # Strict failure
        return False
    
    # Start UI
    ui = await start_ui(args)
    
    # Print startup complete message
    elapsed = time.time() - start_time
    logger.info(f"Startup completed in {elapsed:.2f} seconds")
    
    # Launch interactive console if available and requested
    if ui and isinstance(ui, object) and hasattr(ui, 'start'):
        try:
            await ui.start()
        except Exception as e:
            logger.error(f"Error in UI: {e}")
    
    return True


def main():
    """Main entry point for system startup"""
    parser = argparse.ArgumentParser(description="vAIn P2P AGI System")
    
    # Dependency management
    parser.add_argument("--auto-install", action="store_true", 
                      help="Automatically install missing dependencies")
    parser.add_argument("--strict", action="store_true",
                      help="Exit if dependencies are missing")
    parser.add_argument("--minimal", action="store_true",
                      help="Only check for core dependencies")
    
    # Network options
    parser.add_argument("--no-network", action="store_true",
                      help="Don't start network services")
    parser.add_argument("--port", type=int, default=8468,
                      help="Network port to use")
    parser.add_argument("--bootstrap-nodes", type=str, nargs='+',
                      help="Bootstrap nodes to connect to")
    
    # UI options
    parser.add_argument("--no-ui", action="store_true",
                      help="Don't start user interface")
    parser.add_argument("--console", action="store_true",
                      help="Use interactive console instead of CLI")
    parser.add_argument("--verbose", action="store_true",
                      help="Enable verbose output")
    parser.add_argument("--non-interactive", action="store_true",
                      help="Run in non-interactive mode")
                      
    # Configuration options
    parser.add_argument("--update-config", action="store_true",
                      help="Update configuration interactively at startup")
    parser.add_argument("--skip-config-validation", action="store_true",
                      help="Skip configuration validation")
    parser.add_argument("--init-config", action="store_true",
                      help="Initialize default configurations")
    
    args = parser.parse_args()
    
    try:
        if sys.platform == 'win32':
            # Use ProactorEventLoop on Windows for full async support
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
        return asyncio.run(startup_sequence(args))
    except KeyboardInterrupt:
        logger.info("Startup interrupted by user")
        return False
    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
