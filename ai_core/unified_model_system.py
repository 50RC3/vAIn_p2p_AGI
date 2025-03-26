import torch
from typing import Dict, Optional, List
from models import ModelOutput, ModelState, get_resource_metrics
from memory.memory_manager import MemoryManager
from training.model_interface import ModelInterface
from training.federated_client import FederatedClient
import logging

logger = logging.getLogger(__name__)

class UnifiedModelSystem:
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self.models: Dict[str, torch.nn.Module] = {}
        self.interfaces: Dict[str, ModelInterface] = {}
        self.shared_memory = {}
        self._resource_allocations = {}
        self._active_tasks = set()

    async def register_model(self, model_id: str, model: torch.nn.Module, role: str) -> bool:
        try:
            # Register model with interface
            interface = ModelInterface(model, interactive=True)
            self.models[model_id] = model
            self.interfaces[model_id] = interface

            # Set up memory sharing
            if hasattr(model, 'memory_system'):
                await self.memory_manager.register_memory_system(
                    f"model_{model_id}", 
                    model.memory_system
                )

            # Register role-specific handlers
            self._register_role_handlers(model_id, role)

            return True
        except Exception as e:
            logger.error(f"Failed to register model {model_id}: {e}")
            return False

    async def coordinate_inference(self, input_data: torch.Tensor) -> ModelOutput:
        """Coordinate inference across multiple models"""
        memory_state = {}
        current_output = input_data

        try:
            # Process through model pipeline
            for model_id, model in self.models.items():
                interface = self.interfaces[model_id]
                
                # Share memory state
                if memory_state:
                    await self._share_memory_state(model_id, memory_state)
                
                # Forward pass with memory coordination
                output = await interface.forward(current_output)
                current_output = output.output
                
                # Update shared memory state
                if output.memory_state:
                    memory_state.update(output.memory_state)

            return ModelOutput(
                output=current_output,
                memory_state=memory_state
            )

        except Exception as e:
            logger.error(f"Inference coordination failed: {e}")
            raise

    async def cognitive_step(self, input_data: torch.Tensor) -> ModelOutput:
        """Execute one step of cognitive processing"""
        try:
            # Track cognitive state
            current_state = CognitiveState(
                current_focus=input_data,
                memory_state={},
                attention_patterns=None
            )

            # Process through model pipeline with cognitive tracking
            for model_id, model in self.models.items():
                interface = self.interfaces[model_id]
                
                # Update cognitive state
                if hasattr(model, 'get_cognitive_state'):
                    current_state.metacognitive_state = await model.get_cognitive_state()
                
                # Process with cognitive awareness
                output = await interface.forward(current_state.current_focus)
                current_state.current_focus = output.output
                
                if output.attention is not None:
                    current_state.attention_patterns = output.attention
                if output.memory_state:
                    current_state.memory_state.update(output.memory_state)

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
                await self.memory_manager.share_tensor(
                    source_id='shared_pool',
                    target_id=f"model_{model_id}",
                    tensor_key=key
                )
        except Exception as e:
            logger.error(f"Memory sharing failed: {e}")

    async def optimize_resource_usage(self):
        """Balance resources between models"""
        try:
            metrics = get_resource_metrics()
            total_memory = sum(alloc.get('memory', 0) 
                             for alloc in self._resource_allocations.values())

            # Adjust allocations if needed
            if metrics.memory_usage > 85:  # High memory usage
                await self._rebalance_resources()

        except Exception as e:
            logger.error(f"Resource optimization failed: {e}")

    async def _rebalance_resources(self):
        """Rebalance resources between models"""
        try:
            # Free up resources from inactive models
            for model_id in self.models:
                if model_id not in self._active_tasks:
                    await self.memory_manager._cleanup_storage()
                    if hasattr(self.models[model_id], 'release_resources'):
                        await self.models[model_id].release_resources()

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
