from typing import Dict, Optional
from models import ModelOutput, ModelState, get_resource_metrics
from memory.memory_manager import MemoryManager
from training.federated_client import FederatedClient
from training.model_interface import ModelInterface

class ModelCoordinator:
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self.model_interfaces: Dict[str, ModelInterface] = {}
        self.active_clients: Dict[str, FederatedClient] = {}
        self._shared_resources = {}

    async def register_model(self, model_id: str, model: torch.nn.Module) -> bool:
        try:
            interface = ModelInterface(model, interactive=True)
            self.model_interfaces[model_id] = interface
            
            # Initialize shared memory
            await self.memory_manager.register_memory_system(
                f"model_{model_id}", 
                model.memory_system if hasattr(model, 'memory_system') else None
            )
            return True
        except Exception as e:
            logger.error(f"Failed to register model {model_id}: {e}")
            return False

    async def coordinate_training(self, model_id: str, 
                                client: FederatedClient) -> Optional[ModelOutput]:
        try:
            interface = self.model_interfaces.get(model_id)
            if not interface:
                raise ValueError(f"Model {model_id} not registered")

            # Monitor resources
            metrics = get_resource_metrics()
            if metrics.memory_usage > 90:
                await self.memory_manager._force_cleanup()

            # Coordinate training
            state = ModelState(
                model_hash=model_id,
                version=self._shared_resources.get(model_id, {}).get('version', 0) + 1,
                timestamp=time.time(),
                metrics=metrics
            )
            
            output = await client.train()
            if output:
                await self.memory_manager.cache_tensor_interactive(
                    f"{model_id}_state_{state.version}",
                    output['state_dict']
                )
                self._shared_resources[model_id] = state.__dict__
            return output

        except Exception as e:
            logger.error(f"Training coordination failed: {e}")
            return None
