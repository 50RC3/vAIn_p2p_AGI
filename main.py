"""Main entry point for vAIn P2P AGI system"""
import asyncio
import logging
import os
import sys
import signal
import traceback
from pathlib import Path
from functools import partial

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("vAIn")

# Add file logging
os.makedirs("logs", exist_ok=True)
file_handler = logging.FileHandler("logs/main.log")
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Signal handling to gracefully exit on ctrl+c
def handle_signals():
    """Set up signal handlers for graceful termination."""
    loop = asyncio.get_running_loop()
    
    def signal_handler():
        logger.info("Shutdown signal received, initiating graceful exit...")
        # Stop all running tasks
        for task in asyncio.all_tasks(loop):
            if task is not asyncio.current_task():
                task.cancel()
    
    # Add signal handlers for graceful termination
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

async def initialize_module(name, init_func):
    """Initialize a module with proper error handling and reporting"""
    try:
        logger.info(f"Initializing {name}...")
        
        # Check if init_func is callable
        if not callable(init_func):
            logger.error(f"Initialization function for {name} is not callable")
            return None
            
        # Check if it's async or regular function
        if asyncio.iscoroutinefunction(init_func):
            result = await init_func()
        else:
            result = init_func()
            
        if result is None:
            logger.warning(f"{name} initialization returned None")
        elif result is False:
            logger.error(f"{name} initialization failed (returned False)")
            return None
            
        logger.info(f"Successfully initialized {name}")
        return result
    except Exception as e:
        logger.error(f"Error initializing {name}: {str(e)}")
        logger.debug(f"Initialization error details for {name}: {traceback.format_exc()}")
        return None

async def shutdown_module(name, module):
    """Shutdown a module with proper error handling"""
    try:
        logger.info(f"Shutting down {name}...")
        
        if hasattr(module, 'shutdown'):
            if asyncio.iscoroutinefunction(module.shutdown):
                await module.shutdown()
            else:
                module.shutdown()
            logger.info(f"Successfully shut down {name}")
        elif hasattr(module, 'stop'):
            if asyncio.iscoroutinefunction(module.stop):
                await module.stop()
            else:
                module.stop()
            logger.info(f"Successfully stopped {name}")
        elif hasattr(module, 'close'):
            if asyncio.iscoroutinefunction(module.close):
                await module.close()
            else:
                module.close()
            logger.info(f"Successfully closed {name}")
        else:
            logger.debug(f"No shutdown method found for {name}, skipping")
            
    except Exception as e:
        logger.error(f"Error shutting down {name}: {str(e)}")
        if '--debug' in sys.argv or '--interactive' in sys.argv:
            logger.debug(traceback.format_exc())

async def ensure_directories_exist():
    """Ensure necessary directories exist."""
    essential_dirs = [
        "logs", 
        "config", 
        "data", 
        "models",
        "checkpoints"
    ]
    
    for directory in essential_dirs:
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory}")

async def check_system_requirements():
    """Check if system meets all requirements for running the application"""
    try:
        # Check Python version
        if sys.version_info < (3, 8):
            logger.error(f"Python 3.8 or higher is required. Current version: {sys.version}")
            return False
        
        # Check for critical directories
        required_dirs = ['logs', 'config', 'data']
        missing_dirs = []
        for dir_name in required_dirs:
            if not os.path.exists(dir_name):
                missing_dirs.append(dir_name)
                try:
                    os.makedirs(dir_name)
                    logger.info(f"Created missing directory: {dir_name}")
                except Exception as e:
                    logger.error(f"Failed to create directory {dir_name}: {e}")
                    return False
        
        # Check if we can write to the log directory
        try:
            test_file_path = os.path.join('logs', 'test_write.tmp')
            with open(test_file_path, 'w') as f:
                f.write('test')
            os.unlink(test_file_path)
        except Exception as e:
            logger.error(f"Log directory is not writable: {e}")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error checking system requirements: {e}")
        return False

async def main():
    """Initialize and run the vAIn P2P AGI system"""
    # Modules to be cleaned up in finally block
    modules = {}
    
    try:
        # Ensure necessary directories exist
        await ensure_directories_exist()
        
        logger.info("Initializing vAIn P2P AGI system...")
        
        # Set up proper signal handling for graceful termination
        handle_signals()
        
        # Determine if we're in debug mode
        debug_mode = '--debug' in sys.argv or '--interactive' in sys.argv
        if debug_mode:
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug mode enabled")
            # Configure more verbose logging
            for handler in logger.handlers:
                handler.setLevel(logging.DEBUG)
        
        # Check for system requirements before proceeding
        if not await check_system_requirements():
            logger.error("System requirements check failed")
            return 1

        # Initialize configuration
        try:
            from config.network_config import NetworkConfig
            network_config = NetworkConfig.from_env()
        except ImportError as e:
            logger.error(f"Could not import NetworkConfig: {e}")
            logger.error("Make sure the config module is properly installed.")
            return 1
        except Exception as e:
            logger.error(f"Failed to load network configuration: {str(e)}")
            if debug_mode:
                logger.debug(traceback.format_exc())
            return 1
        
        # Initialize module registry first
        try:
            from ai_core.module_registry import ModuleRegistry
            registry = ModuleRegistry.get_instance()
            initialized = await initialize_module("module_registry", registry.initialize)
            if initialized:
                modules["module_registry"] = registry
                logger.debug("Module registry initialized successfully")
            else:
                logger.warning("Module registry initialization returned False")
        except ImportError as e:
            logger.warning(f"Module registry not available: {e}")
            logger.warning("Continuing without component registration")
        except Exception as e:
            logger.error(f"Failed to initialize module registry: {str(e)}")
            if debug_mode:
                logger.debug(traceback.format_exc())
        
        # Initialize memory manager with registration check
        try:
            from memory.memory_manager import MemoryManager
            from core.constants import MAX_CACHE_SIZE
            
            # Check if memory manager has already been registered
            memory_manager = None
            if 'module_registry' in modules:
                registry = modules['module_registry']
                if registry.is_module_registered("memory_manager"):
                    logger.info("Using existing registered memory_manager")
                    memory_manager = registry.get_module("memory_manager")
            
            if memory_manager is None:
                memory_manager = await initialize_module(
                    "memory_manager",
                    lambda: MemoryManager(max_cache_size=MAX_CACHE_SIZE)
                )
                
                # Register with module registry if available
                if 'module_registry' in modules and memory_manager is not None:
                    await registry.register_module(
                        "memory_manager", 
                        MemoryManager, 
                        dependencies=[], 
                        replace=True
                    )
                
                modules["memory_manager"] = memory_manager
        except ImportError as e:
            logger.error(f"Could not import memory management modules: {e}")
            memory_manager = None
        except Exception as e:
            logger.error(f"Failed to initialize memory manager: {str(e)}")
            if debug_mode:
                logger.debug(traceback.format_exc())
            memory_manager = None
        
        # Initialize cognitive system with proper parameters
        try:
            from ai_core import initialize_cognitive_system
            
            # Define cognitive parameters - but don't pass as 'config' since the function doesn't accept it
            cognitive_params = {
                'memory_vector_dim': 40,
                'hidden_size': 64, 
                'memory_size': 128,
                'nhead': 4,
                'num_layers': 2
            }
            
            # Pass configuration to cognitive system with proper error handling
            # NOTE: Not using 'config' parameter as it's not supported
            cognitive_init = partial(initialize_cognitive_system,
                memory_manager=memory_manager,
                resource_metrics=cognitive_params  # Pass as resource_metrics instead of config
            )
            
            result = await initialize_module("cognitive_system", cognitive_init)
            if result:
                unified_system, cognitive_system = result
                modules["cognitive_system"] = cognitive_system
            else:
                unified_system, cognitive_system = None, None
        except ImportError as e:
            logger.error(f"Could not import cognitive system modules: {e}")
            unified_system, cognitive_system = None, None
        except Exception as e:
            logger.error(f"Failed to initialize cognitive system: {str(e)}")
            if debug_mode:
                logger.debug(traceback.format_exc())
            unified_system, cognitive_system = None, None
        
        # Initialize P2P network with proper error handling
        try:
            from network.p2p_network import P2PNetwork
            node_id = os.environ.get('NODE_ID', None)
            
            # Ensure network config has all required components
            network_dict = network_config.to_dict()
            
            # Add encryption key if missing
            if 'security' in network_dict and 'encryption' in network_dict['security'] and network_dict['security']['encryption']:
                if 'encryption_key' not in network_dict['security']:
                    import secrets
                    network_dict['security']['encryption_key'] = secrets.token_hex(16)
                    logger.info("Generated encryption key for secure communication")
            
            # Network initialization with proper error handling
            p2p_init = partial(P2PNetwork,
                node_id=node_id or f"vAIn-{os.getpid()}",
                network_config=network_dict,
                interactive=True
            )
            
            p2p_network = await initialize_module("P2P network", p2p_init)
            if p2p_network:
                modules["p2p_network"] = p2p_network
        except ImportError as e:
            logger.error(f"Could not import network modules: {e}")
            p2p_network = None
        except Exception as e:
            logger.error(f"Failed to initialize P2P network: {str(e)}")
            if debug_mode:
                logger.debug(traceback.format_exc())
            p2p_network = None
        
        # Initialize UI with proper error handling
        try:
            from ui.interface_manager import UserInterfaceManager
            from ui.terminal_ui import TerminalUI
            
            ui_manager = UserInterfaceManager()
            terminal_ui = TerminalUI(interactive=True)
            ui_manager.register_interface(terminal_ui)
            
            # Connect components if they exist
            if cognitive_system:
                ui_manager.connect_system(cognitive_system)
            else:
                # Connect UI to a null system or other fallback
                from models.null_system import NullSystem
                ui_manager.connect_system(NullSystem())
            
            # Start UI
            ui_manager.start()
            modules["ui_manager"] = ui_manager
            
            logger.info("User interface initialized")
        except ImportError as e:
            logger.error(f"Could not import UI modules: {e}")
            ui_manager = None
        except Exception as e:
            logger.error(f"Failed to initialize UI: {str(e)}")
            if debug_mode:
                logger.debug(traceback.format_exc())
            ui_manager = None
        
        # Final status message
        has_critical_components = cognitive_system is not None and p2p_network is not None and ui_manager is not None
        if has_critical_components:
            logger.info("vAIn P2P AGI system initialized and ready")
        else:
            logger.warning("vAIn P2P AGI system initialized with limited functionality")
        
        if cognitive_system:
            try:
                active_learning = getattr(cognitive_system, '_active_learning', False)
                logger.info(f"Cognitive evolution active: {active_learning}")
            except Exception:
                pass
        
        # Run the UI event loop instead of just sleeping
        if ui_manager:
            await run_ui_loop(ui_manager)
        else:
            # Keep the main application running until interrupted
            logger.info("Running in headless mode (no UI). Press Ctrl+C to exit.")
            while True:
                await asyncio.sleep(1)
                
    except asyncio.CancelledError:
        logger.info("Main task cancelled, shutting down gracefully...")
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    except Exception as e:
        logger.error(f"System initialization failed: {str(e)}")
        if '--debug' in sys.argv or '--interactive' in sys.argv:
            logger.debug(traceback.format_exc())
    finally:
        # Cleanup resources in reverse order of initialization
        logger.info("Cleaning up resources...")
        
        # Shutdown modules in reverse dependency order
        for module_name, module in reversed(list(modules.items())):
            await shutdown_module(module_name, module)
            
        logger.info("System shutdown complete")

async def run_ui_loop(ui_manager):
    """Run the UI event loop as an awaitable task."""
    try:
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        
        def ui_thread_func():
            try:
                ui_manager.run_event_loop()
            except Exception as e:
                loop.call_soon_threadsafe(lambda: future.set_exception(e))
            else:
                loop.call_soon_threadsafe(future.set_result, None)
        
        import threading
        ui_thread = threading.Thread(target=ui_thread_func, daemon=True)
        ui_thread.start()
        
        await future
    except asyncio.CancelledError:
        logger.info("UI loop cancelled, shutting down UI...")
        ui_manager.shutdown()
        raise

if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\nExiting due to keyboard interrupt")
        sys.exit(0)
    except Exception as e:
        print(f"\nFatal error: {e}")
        if '--debug' in sys.argv:
            traceback.print_exc()
        sys.exit(1)

