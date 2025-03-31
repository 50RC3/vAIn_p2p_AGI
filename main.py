"""Main entry point for vAIn P2P AGI system"""
import asyncio
import logging
import os
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("vAIn")

async def main():
    """Initialize and run the vAIn P2P AGI system"""
    try:
        logger.info("Initializing vAIn P2P AGI system...")
        
        # Initialize configuration
        from config.network_config import NetworkConfig
        network_config = NetworkConfig.from_env()
        
        # Initialize memory manager
        from memory.memory_manager import MemoryManager
        from core.constants import MAX_CACHE_SIZE
        memory_manager = MemoryManager(max_cache_size=MAX_CACHE_SIZE)
        
        # Initialize cognitive system
        from ai_core import initialize_cognitive_system
        unified_system, cognitive_system = await initialize_cognitive_system(
            memory_manager=memory_manager
        )
        
        # Initialize P2P network
        from network.p2p_network import P2PNetwork
        node_id = os.environ.get('NODE_ID', None)
        p2p_network = P2PNetwork(
            node_id=node_id or f"vAIn-{os.getpid()}",
            network_config=network_config.__dict__,
            interactive=True
        )
        
        # Final status message
        logger.info("vAIn P2P AGI system initialized and ready")
        logger.info(f"Cognitive evolution active: {cognitive_system._active_learning}")
        
        # Keep the system running
        while True:
            await asyncio.sleep(60)
            
    except KeyboardInterrupt:
        logger.info("Shutdown requested, terminating...")
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())

