import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field
import os
import json
import torch

# Import learning modules
from ai_core.learning.unsupervised import UnsupervisedLearningModule
from ai_core.learning.self_supervised import SelfSupervisedLearning
from .rl_trainer import RLTrainer, RLConfig
from ai_core.evolution.cognitive_evolution import CognitiveEvolution, EvolutionConfig

logger = logging.getLogger(__name__)

@dataclass
class LearningStats:
    """Statistics for different learning modules"""
    self_supervised: Dict[str, Any] = field(default_factory=dict)
    unsupervised: Dict[str, Any] = field(default_factory=dict)
    reinforcement: Dict[str, Any] = field(default_factory=dict)
    last_update: float = field(default_factory=time.time)
    total_examples: int = 0

@dataclass
class LearningCoordinatorConfig:
    """Configuration for learning coordinator"""
    enable_self_supervised: bool = True
    enable_unsupervised: bool = True
    enable_reinforcement: bool = True
    enable_cognitive_evolution: bool = True
    model_path: str = "./models"
    stats_save_interval: int = 100
    min_sample_length: int = 5
    max_buffer_size: int = 1000
    self_supervised_model: str = "bert-base-uncased"
    unsupervised_clusters: int = 10
    batch_size: int = 8
    save_interval: int = 500

class LearningCoordinator:
    """Coordinates different learning approaches to ensure they work together harmoniously"""
    
    def __init__(self, chatbot_interface: "ChatbotInterface"):
        self.interface = chatbot_interface
        self.processing_lock = asyncio.Lock()
        self.last_update = 0
        self.metrics_history = []
        
        # Initialize cognitive evolution system
        self.config = getattr(self.interface, 'learning_config', LearningCoordinatorConfig())
        evolution_config = EvolutionConfig(
            save_path=os.path.join(self.config.model_path, "evolution")
        )
        self.cognitive_evolution = CognitiveEvolution(evolution_config) if self.config.enable_cognitive_evolution else None

    async def coordinate_training_cycle(self) -> Dict[str, Any]:
        """Run a coordinated training cycle across all learning systems"""
        async with self.processing_lock:
            metrics = {}
            
            try:
                # 1. First run unsupervised learning to identify patterns/clusters
                if self.interface.unsupervised_module:
                    await self._update_clusters()
                    metrics["unsupervised"] = {
                        "clusters": self.interface.unsupervised_module.n_clusters,
                        "samples_processed": len(self.interface.unsupervised_module.buffer)
                    }
                
                # 2. Then run self-supervised to improve representations
                if self.interface.self_supervised_module and len(self.interface.history) > 5:
                    ss_metrics = await self._update_self_supervised()
                    metrics["self_supervised"] = ss_metrics
                
                # 3. Finally run RL to optimize decision making with improved embeddings
                if self.interface.rl_trainer:
                    rl_metrics = await self._update_reinforcement()
                    metrics["reinforcement"] = rl_metrics
                
                # 4. Apply cross-learning to share insights between systems
                await self.interface._apply_cross_learning()
                
                # 5. Update cognitive evolution metrics if enabled
                if self.cognitive_evolution:
                    evolution_metrics = await self.cognitive_evolution.update_metrics(metrics)
                    metrics["evolution"] = evolution_metrics.to_dict()
                
                # Track metrics history
                self.metrics_history.append(metrics)
                if len(self.metrics_history) > 100:
                    self.metrics_history = self.metrics_history[-100:]
                    
                return metrics
                
            except Exception as e:
                logger.error(f"Coordinated training cycle failed: {e}")
                return {"error": str(e)}

    async def _update_clusters(self) -> None:
        """Update unsupervised clustering with recent data"""
        if not self.interface.unsupervised_module or len(self.interface.history) < 3:
            return
            
        messages = [msg for msg, _ in self.interface.history[-50:]]
        for msg in messages:
            self.interface.unsupervised_module.add_to_buffer(msg)
            
        await self.interface.unsupervised_module.update_clusters()

    async def _update_self_supervised(self) -> Dict[str, float]:
        """Update self-supervised learning with recent examples"""
        metrics = {}
        
        if not self.interface.self_supervised_module:
            return metrics
            
        # Get recent examples
        examples = []
        for msg, resp in self.interface.history[-20:]:
            examples.extend([msg, resp])
            
        # Train on examples
        total_loss = 0
        for example in examples:
            if len(example.strip()) > 10:  # Skip very short examples
                loss = await self.interface._process_self_supervised(example)
                if loss is not None:
                    total_loss += loss
                    
        metrics["avg_loss"] = total_loss / max(1, len(examples))
        return metrics

    async def _update_reinforcement(self) -> Dict[str, float]:
        """Update reinforcement learning with recent feedback"""
        if not self.interface.rl_trainer:
            return {}
            
        # Get RL stats before update
        stats = self.interface.rl_trainer.get_training_stats()
        
        # Perform update if we have enough data
        await self.interface.rl_trainer.update()
        
        return stats
    
    def get_evolution_metrics(self) -> Dict[str, Any]:
        """Get current cognitive evolution metrics"""
        if self.cognitive_evolution:
            return self.cognitive_evolution.get_current_metrics().to_dict()
        return {}
    
    def get_evolution_history(self) -> List[Dict[str, Any]]:
        """Get cognitive evolution history"""
        if self.cognitive_evolution:
            return self.cognitive_evolution.get_evolution_history()
        return []
