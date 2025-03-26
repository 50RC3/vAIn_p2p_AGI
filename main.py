import os
from config import Config
from models.simple_nn import SimpleNN
from training.federated_training import FederatedTraining
from data.data_loader import DataLoader
from network.p2p_network import P2PNetwork
import logging
from utils.logger_init import init_logger
import signal
import sys
import asyncio
from core.interactive_utils import InteractiveSession, InteractiveConfig, InteractionLevel
from core.constants import INTERACTION_TIMEOUTS
import psutil

MAX_RETRY_ATTEMPTS = 3
STARTUP_TIMEOUT = 60

async def cleanup_resources(session, network):
    """Cleanup resources gracefully"""
    try:
        if network:
            network.stop()
        if session:
            await session.__aexit__(None, None, None)
    except Exception as e:
        logging.error(f"Error during cleanup: {e}")

def setup_signal_handlers(network: P2PNetwork, session: InteractiveSession):
    def signal_handler(sig, frame):
        logging.info("Initiating graceful shutdown...")
        try:
            # Run cleanup in event loop
            loop = asyncio.get_event_loop()
            loop.run_until_complete(cleanup_resources(session, network))
        except Exception as e:
            logging.error(f"Error during shutdown: {e}")
        finally:
            sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

async def verify_system_resources():
    """Verify sufficient system resources before startup"""
    try:
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Define minimum requirements
        MIN_FREE_MEMORY = 2 * 1024 * 1024 * 1024  # 2GB
        MIN_FREE_DISK = 5 * 1024 * 1024 * 1024    # 5GB
        
        warnings = []
        if cpu_percent > 80:
            warnings.append(f"High CPU usage: {cpu_percent}%")
        if memory.available < MIN_FREE_MEMORY:
            warnings.append(f"Low memory: {memory.available / 1024**3:.1f}GB available")
        if disk.free < MIN_FREE_DISK:
            warnings.append(f"Low disk space: {disk.free / 1024**3:.1f}GB available")
            
        if warnings:
            logger.warning("Resource warnings:\n" + "\n".join(warnings))
            
        # Optimize Python memory allocator
        if memory.available < MIN_FREE_MEMORY * 2:
            import gc
            gc.collect()
            
        return len(warnings) == 0
            
    except Exception as e:
        logger.error(f"Resource verification failed: {str(e)}")
        return False

async def run_node(config: Config, logger: logging.Logger):
    """Run node with enhanced interactive session management"""
    session = None
    network = None
    
    try:
        # Verify resources first
        if not await verify_system_resources():
            logger.error("Failed system resource verification")
            return

        # Initialize interactive session with recovery
        session = InteractiveSession(
            level=InteractionLevel.NORMAL,
            config=InteractiveConfig(
                timeout=INTERACTION_TIMEOUTS["default"],
                persistent_state=True,
                safe_mode=True,
                recovery_enabled=True,
                max_cleanup_wait=30,
                max_retries=MAX_RETRY_ATTEMPTS,
                heartbeat_interval=30,
                memory_threshold=0.9
            )
        )

        async with session:
            # Monitor system resources
            resource_monitor = asyncio.create_task(session.monitor_resources())
            
            try:
                network = P2PNetwork(config.node_id, config.network)
                setup_signal_handlers(network, session)
                
                logger.info(f"Starting vAIn node {config.node_id}...")
                
                # Enhanced interactive startup with timeout
                if session.level != InteractionLevel.NONE:
                    async with session.timeout(STARTUP_TIMEOUT):
                        proceed = await session.confirm_with_timeout(
                            "\nStart node with current configuration?",
                            timeout=INTERACTION_TIMEOUTS["confirmation"]
                        )
                        if not proceed:
                            logger.info("Node startup cancelled by user")
                            return

                # Register cleanup handlers
                session.register_cleanup(network.cleanup)
                
                # Start network with monitoring and auto-recovery
                await network.start_interactive()

            except asyncio.TimeoutError:
                logger.error("Node startup timed out")
                raise
            finally:
                resource_monitor.cancel()

    except Exception as e:
        logger.error(f"Error running vAIn node: {e}")
        raise
    finally:
        await cleanup_resources(session, network)

def main():
    # Initialize logger first
    logger = init_logger(
        name="vAIn",
        config_path="config/logging.json",
        log_dir="logs"
    )
    
    try:
        # Initialize config with environment variables
        config = Config()
        
        # Run node with asyncio
        asyncio.run(run_node(config, logger))
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
