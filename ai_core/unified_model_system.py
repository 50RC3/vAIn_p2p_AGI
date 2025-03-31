import torch
import logging
from typing import Dict, Optional, List, Any, Tuple
from dataclasses import dataclass
from models import ModelOutput, ModelState, get_resource_metrics, ModelRole
from memory.memory_manager import MemoryManager
from training.model_interface import ModelInterface
from training.federated_client import FederatedClient

logger = logging.getLogger(__name__)

@dataclass
class CognitiveState:
    """Represents the current cognitive state of the system"""
    current_focus: torch.Tensor
    memory_state: Dict[str, Any]
    attention_patterns: Optional[torch.Tensor] = None

class UnifiedModelSystem:
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self.models: Dict[str, torch.nn.Module] = {}
        self.interfaces: Dict[str, ModelInterface] = {}
        self.shared_memory = {}
        self._resource_allocations = {}
        self._active_tasks = set()
        self.model_roles: Dict[str, ModelRole] = {}
        
    async def register_model(self, model_id: str, model: torch.nn.Module, 
                           role: ModelRole = ModelRole.GENERAL) -> bool:
        """
        Register a model with the unified system
        
        Args:
            model_id: Unique identifier for the model
            model: The PyTorch model to register
            role: The role this model plays in the cognitive system
            
        Returns:
            bool: True if registration was successful
        """
        try:
            if model_id in self.models:
                logger.warning(f"Model {model_id} already registered, replacing")
                
            # Store model
            self.models[model_id] = model
            self.model_roles[model_id] = role
            
            # Create interface
            from training.model_interface import ModelInterface
            interface = ModelInterface(model, self.memory_manager)
            self.interfaces[model_id] = interface
            
            logger.info(f"Registered model {model_id} with role {role.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register model {model_id}: {e}")
            return False
            
    async def coordinate_inference(self, input_data: torch.Tensor) -> ModelOutput:
        """Coordinate model inference with resource management"""
        return await self.cognitive_step(input_data)
        
    async def cognitive_step(self, input_data: torch.Tensor) -> ModelOutput:
        """Execute one step of cognitive processing"""
        try:
            # Track cognitive state
            current_state = CognitiveState(
                current_focus=input_data,
                memory_state={},
                attention_patterns=None
            )

            # Check if we have models registered
            if not self.models:
                logger.warning("No models registered for cognitive processing")
                return ModelOutput(
                    output=input_data,
                    attention=None,
                    memory_state={},
                    metadata={'status': 'no_models_registered'}
                )

            # Process through model pipeline with cognitive tracking
            # Order: MEMORY → PROCESSING → META
            processing_order = {
                ModelRole.MEMORY: 1,
                ModelRole.PROCESSING: 2, 
                ModelRole.META: 3,
                ModelRole.GENERAL: 4
            }
            
            # Sort models by role
            sorted_models = sorted(
                self.models.items(),
                key=lambda x: processing_order.get(self.model_roles.get(x[0], ModelRole.GENERAL), 99)
            )
            
            for model_id, model in sorted_models:
                interface = self.interfaces.get(model_id)
                if not interface:
                    logger.warning(f"No interface for model {model_id}, skipping")
                    continue
                
                # Forward pass with resource management
                output = await interface.forward_interactive(current_state.current_focus)
                
                # Update cognitive state
                if isinstance(output, tuple):
                    current_state.current_focus = output[0]  
                    # If attention weights are available
                    if len(output) > 1 and output[1] is not None:
                        if current_state.attention_patterns is None:
                            current_state.attention_patterns = output[1]
                        else:
                            # Combine attention patterns
                            current_state.attention_patterns = torch.cat(
                                [current_state.attention_patterns, output[1]], 
                                dim=0
                            )
                else:
                    current_state.current_focus = output
                
                # Collect memory state
                if hasattr(model, 'get_memory_state'):
                    mem_state = await model.get_memory_state()
                    if mem_state:
                        current_state.memory_state[model_id] = mem_state

            return ModelOutput(
                output=current_state.current_focus,
                attention=current_state.attention_patterns,
                memory_state=current_state.memory_state,
                metadata={'cognitive_state': current_state}
            )

        except Exception as e:
            logger.error(f"Cognitive step failed: {e}")
            raise

    async def _share_memory_state(self, model_id: str, memory_state: Dict):
        """Share memory state between models"""
        try:
            for key, tensor in memory_state.items():
                memory_key = f"{model_id}_{key}"
                self.shared_memory[memory_key] = tensor
                
                # Notify other models about memory update
                for other_id, model in self.models.items():
                    if other_id != model_id and hasattr(model, 'update_external_memory'):
                        await model.update_external_memory(memory_key, tensor)
                
        except Exception as e:
            logger.error(f"Memory sharing failed: {e}")

    async def optimize_resource_usage(self):
        """Balance resources between models"""
        try:
            metrics = get_resource_metrics()
            total_memory = sum(alloc.get('memory', 0) 
                             for alloc in self._resource_allocations.values())

            # Adjust allocations if needed
            if metrics.memory_usage > 85:
                # High memory usage - rebalance resources
                await self._rebalance_resources()
                
                # Clear CUDA cache if available
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                    
                # Notify models about resource constraint
                for model_id, interface in self.interfaces.items():
                    if hasattr(interface, 'notify_resource_constraint'):
                        await interface.notify_resource_constraint('memory', metrics.memory_usage)

        except Exception as e:
            logger.error(f"Resource optimization failed: {e}")

    async def _rebalance_resources(self):
        """Rebalance resources between models"""
        try:
            # Free up resources from inactive models
            for model_id in self.models:
                allocation = self._resource_allocations.get(model_id, {})
                
                # Check if model has been inactive
                if not any(task_id.startswith(f"inference_{model_id}") 
                         for task_id in self._active_tasks):
                    # Mark as inactive
                    allocation['active'] = False
                    
                    # Optimize memory usage for this model
                    interface = self.interfaces.get(model_id)
                    if interface and hasattr(interface, 'optimize_memory'):
                        await interface.optimize_memory()

        except Exception as e:
            logger.error(f"Resource rebalancing failed: {e}")

    def _register_role_handlers(self, model_id: str, role: str):
        """Register role-specific functionality"""
        if role == 'memory':
            self._register_memory_handlers(model_id)
        elif role == 'processing':
            self._register_processing_handlers(model_id)
        elif role == 'meta':
            self._register_meta_handlers(model_id)
    
    def _register_memory_handlers(self, model_id: str):
        """Register memory-specific handlers"""
        pass
        
    def _register_processing_handlers(self, model_id: str):
        """Register processing-specific handlers"""
        pass
        
    def _register_meta_handlers(self, model_id: str):
        """Register meta-learning handlers"""
        pass
