from typing import Dict, Optional, List
import asyncio
import logging
from core.interactive_utils import InteractiveSession, InteractionLevel, InteractiveConfig
from core.constants import INTERACTION_TIMEOUTS
from memory.memory_manager import MemoryManager
from network.consensus import ConsensusManager
from training.model_interface import ModelInterface

logger = logging.getLogger(__name__)

class ComponentCoordinator:
    """Coordinates interaction between major system components"""
    
    def __init__(self, memory_manager: MemoryManager, consensus_manager: ConsensusManager):
        self.memory_manager = memory_manager
        self.consensus_manager = consensus_manager
        self.interfaces: Dict[str, ModelInterface] = {}
        self.interactive = True
        self._cleanup_lock = asyncio.Lock()
        self.session = None

    async def __aenter__(self):
        if self.interactive:
            self.session = InteractiveSession(
                level=InteractionLevel.NORMAL,
                config=InteractiveConfig(
                    timeout=INTERACTION_TIMEOUTS["default"],
                    persistent_state=True,
                    safe_mode=True
                )
            )
            await self.session.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.session:
            await self.session.__aexit__(exc_type, exc, tb)
        await self._cleanup()

    async def register_interface(self, interface_id: str, interface: ModelInterface) -> bool:
        """Register model interface with coordination layer"""
        try:
            # Ensure memory system registration
            await self.memory_manager.register_memory_system(
                f"interface_{interface_id}",
                interface.model
            )

            # Validate with consensus layer if needed
            if hasattr(interface.model, 'get_voting_power'):
                voting_power = await interface.model.get_voting_power()
                self.consensus_manager.update_voting_power(interface_id, voting_power)

            self.interfaces[interface_id] = interface
            return True

        except Exception as e:
            logger.error(f"Failed to register interface {interface_id}: {e}")
            return False

    async def coordinate_state_updates(self) -> None:
        """Coordinate state updates between components"""
        try:
            # Memory state coordination
            memory_state = await self.memory_manager.coordinate_systems_state()
            
            # Update interfaces
            for interface in self.interfaces.values():
                await interface.share_state('memory', memory_state)

            # Update consensus layer if needed
            await self.consensus_manager.update_system_state({
                'memory_state': memory_state,
                'interfaces': list(self.interfaces.keys())
            })

        except Exception as e:
            logger.error(f"State coordination failed: {e}")
            if self.session:
                await self.session.log_error(f"State coordination error: {e}")

    async def _cleanup(self) -> None:
        """Clean up component resources"""
        async with self._cleanup_lock:
            try:
                # Clean up memory systems
                await self.memory_manager._cleanup_storage()
                
                # Clean up interfaces
                for interface in self.interfaces.values():
                    await interface.cleanup()
                
                self.interfaces.clear()
                
            except Exception as e:
                logger.error(f"Cleanup failed: {e}")
