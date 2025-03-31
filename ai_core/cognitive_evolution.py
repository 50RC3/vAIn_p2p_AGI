import torch
import time
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from models import ModelOutput, ModelRole, get_resource_metrics
from memory.memory_manager import MemoryManager
from .unified_model_system import UnifiedModelSystem
from models.hybrid_memory_system import HybridMemorySystem
import logging

logger = logging.getLogger(__name__)

@dataclass
class CognitiveState:
    """Represents the current cognitive state of the system"""
    current_focus: torch.Tensor
    memory_state: Dict[str, Any]
    attention_patterns: Optional[torch.Tensor] = None


class CognitiveEvolution:
    def __init__(self, unified_system: UnifiedModelSystem, memory_manager: MemoryManager):
        self.unified_system = unified_system
        self.memory_manager = memory_manager
        self.cognitive_states = {}
        self.evolution_history = []
        self._active_learning = False
        self._initialized = False
        
    async def initialize_cognitive_network(self):
        """Initialize the cognitive processing network"""
        try:
            if self._initialized:
                logger.info("Cognitive network already initialized")
                return True
                
            logger.info("Initializing cognitive processing network")
            
            # Register models with unified system
            # First, register memory system
            memory_model = HybridMemorySystem(
                input_size=512,
                hidden_size=256,
                memory_size=128,
                word_size=64,
                num_heads=4
            )
            await self.unified_system.register_model(
                "memory_encoder", 
                memory_model,
                ModelRole.MEMORY
            )
            
            # Then register cognitive processor (transformer)
            # Import locally to avoid circular imports
            from models.transformers import vAInTransformer
            processor = vAInTransformer(
                dim=512,
                depth=6,
                heads=8,
                dim_head=64
            )
            await self.unified_system.register_model(
                "cognitive_processor",
                processor, 
                ModelRole.PROCESSING
            )
            
            # Finally register meta-learning component
            # Import locally to avoid circular imports
            from training.meta_reptile import MetaReptile
            meta_learner = MetaReptile(
                model=processor,
                inner_lr=0.03,
                outer_lr=0.1
            )
            await self.unified_system.register_model(
                "meta_learner",
                meta_learner,
                ModelRole.META
            )
            
            # Initialize cognitive states tracking
            self.cognitive_states = {}
            self.evolution_history = []
            
            self._active_learning = True
            self._initialized = True
            
            logger.info("Cognitive processing network initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize cognitive network: {e}")
            self._active_learning = False
            self._initialized = False
            return False
            
    async def cognitive_cycle(self, input_data: torch.Tensor) -> ModelOutput:
        """Execute one cognitive processing cycle"""
        try:
            # Ensure initialization
            if not self._initialized:
                await self.initialize_cognitive_network()
                
            # Process through cognitive pipeline
            memory_output = await self.unified_system.cognitive_step(input_data)
            
            # Update cognitive state
            self.cognitive_states[len(self.evolution_history)] = {
                'input': input_data,
                'output': memory_output,
                'timestamp': time.time()
            }
            
            # Trigger evolution if needed
            if self._should_evolve():
                await self._evolve_cognitive_system()
                
            return memory_output
            
        except Exception as e:
            logger.error(f"Cognitive cycle failed: {e}")
            raise

    async def _evolve_cognitive_system(self):
        """Evolve the cognitive system based on accumulated experience"""
        try:
            # Analyze performance
            metrics = self._analyze_cognitive_performance()
            
            # Update models based on performance
            for model_id, model in self.unified_system.models.items():
                if metrics[model_id]['improvement_needed']:
                    await self._adapt_model(model_id, metrics[model_id])
                    
            # Record evolution step
            self.evolution_history.append({
                'timestamp': time.time(),
                'metrics': metrics,
                'cognitive_state': self.cognitive_states.copy()
            })
            
            # Cleanup old states
            self._cleanup_old_states()
            
        except Exception as e:
            logger.error(f"Evolution step failed: {e}")
            raise

    def _should_evolve(self) -> bool:
        """Determine if system should evolve based on performance"""
        if len(self.cognitive_states) < 100:
            return False
            
        recent_performance = self._analyze_recent_performance()
        return recent_performance['stagnation'] or recent_performance['errors'] > 0.1

    async def _adapt_model(self, model_id: str, metrics: Dict):
        """Adapt a specific model based on performance metrics"""
        try:
            model = self.unified_system.models[model_id]
            interface = self.unified_system.interfaces[model_id]
            
            # Get relevant experience for adaptation
            support_set = self._get_adaptation_support_set(model_id)
            
            # Adapt model
            if hasattr(interface, 'adapt'):
                success = await interface.adapt(support_set)
                if success:
                    logger.info(f"Successfully adapted model {model_id}")
                    
        except Exception as e:
            logger.error(f"Model adaptation failed: {e}")

    def _cleanup_old_states(self):
        """Remove old cognitive states to prevent memory bloat"""
        max_states = 1000
        if len(self.cognitive_states) > max_states:
            oldest_states = sorted(self.cognitive_states.keys())[:len(self.cognitive_states) - max_states]
            for key in oldest_states:
                del self.cognitive_states[key]
