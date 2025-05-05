"""
Learning Coordinator Module

Coordinates different learning approaches to ensure they work together harmoniously
and maintains cross-learning between different AI components in the P2P network.
"""

import asyncio
import logging
import time
import torch
import numpy as np
from typing import Dict, Any
from dataclasses import dataclass, field
import os
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class LearningStats:
    """Statistics for different learning modules"""
    self_supervised: Dict[str, Any] = field(default_factory=dict)
    unsupervised: Dict[str, Any] = field(default_factory=dict)
    reinforcement: Dict[str, Any] = field(default_factory=dict)
    last_update: float = field(default_factory=time.time)
    total_examples: int = 0
    cross_learning_transfers: int = 0
    training_cycles: int = 0

@dataclass
class LearningCoordinatorConfig:
    """Configuration for learning coordinator"""
    enable_self_supervised: bool = True
    enable_unsupervised: bool = True
    enable_reinforcement: bool = True
    enable_cross_learning: bool = True
    model_path: str = "./models"
    stats_save_interval: int = 100
    min_sample_length: int = 5
    max_buffer_size: int = 1000
    self_supervised_model: str = "distilbert-base-uncased"
    unsupervised_clusters: int = 10
    batch_size: int = 8
    save_interval: int = 500
    training_interval: int = 60  # seconds
    cross_learning_ratio: float = 0.2
    knowledge_sharing_threshold: float = 0.7
    experimental_features: bool = False
    resource_awareness: bool = True
    p2p_sharing: bool = True
    p2p_sharing_threshold: float = 0.8
    p2p_receive_threshold: float = 0.7


class LearningCoordinator:
    """Coordinates different learning approaches to ensure they work together harmoniously"""
    
    def __init__(self, chatbot_interface):
        """Initialize the learning coordinator
        
        Args:
            chatbot_interface: The chatbot interface to coordinate learning for
        """
        self.interface = chatbot_interface
        self.lock = asyncio.Lock()
        self.last_update = time.time()
        self.metrics_history = []
        self.task = None
        self.active = True
        self.stats = LearningStats()
        
        # Load config from interface if available
        if hasattr(self.interface, 'learning_config'):
            self.config = LearningCoordinatorConfig()
            # Copy attributes from interface config to coordinator config
            for key, value in vars(self.interface.learning_config).items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        else:
            self.config = LearningCoordinatorConfig()
        
        # Initialize metrics storage
        self.metrics_dir = Path("./metrics/learning")
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Track cluster information for cross-learning
        self.cluster_affinities = {}
        self.high_quality_clusters = set()
        
        logger.info("Learning coordinator initialized")
        
    async def start_coordinator(self):
        """Start the coordinator task"""
        if self.task is not None and not self.task.done():
            return
            
        self.active = True
        self.task = asyncio.create_task(self._coordinator_loop())
        logger.info("Learning coordinator started")
        
    async def _coordinator_loop(self):
        """Background task for coordinating learning systems"""
        while self.active:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                now = time.time()
                if now - self.last_update >= self.config.training_interval:
                    metrics = await self.coordinate_training_cycle()
                    self.last_update = now
                    
                    # Save metrics periodically
                    if self.stats.training_cycles % self.config.stats_save_interval == 0:
                        self._save_metrics()
                        
            except asyncio.CancelledError:
                logger.info("Learning coordinator task cancelled")
                break
            except (asyncio.TimeoutError, ConnectionError, RuntimeError) as e:
                logger.error(f"Error in coordinator loop: {e}", exc_info=True)
                await asyncio.sleep(60)  # Back off on error
            except Exception as e:  # Fallback for unexpected errors
                logger.critical(f"Unexpected error in coordinator loop: {e}", exc_info=True)
                await asyncio.sleep(120)  # Longer back off for unexpected errors
                
    def _save_metrics(self):
        """Save learning metrics to disk"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics_file = self.metrics_dir / f"learning_metrics_{timestamp}.json"
            
            # Convert any non-serializable items to strings
            def prepare_for_json(obj):
                if isinstance(obj, (np.ndarray, torch.Tensor)):
                    return obj.tolist() if hasattr(obj, 'tolist') else str(obj)
                elif isinstance(obj, (datetime, timedelta)):
                    return str(obj)
                return obj
            
            # Create a serializable copy of the metrics
            metrics_data = {
                "timestamp": time.time(),
                "training_cycles": self.stats.training_cycles,
                "total_examples": self.stats.total_examples,
                "self_supervised": {k: prepare_for_json(v) for k, v in self.stats.self_supervised.items()},
                "unsupervised": {k: prepare_for_json(v) for k, v in self.stats.unsupervised.items()},
                "reinforcement": {k: prepare_for_json(v) for k, v in self.stats.reinforcement.items()},
                "cross_learning_transfers": self.stats.cross_learning_transfers
            }
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
                
            logger.debug(f"Saved learning metrics to {metrics_file}")
                
        except Exception as e:
            logger.error(f"Error saving learning metrics: {e}")
    
    async def coordinate_training_cycle(self) -> Dict[str, Any]:
        """Run a coordinated training cycle across all learning systems
        
        Returns:
            Dict with metrics from the training cycle
        """
        async with self.lock:
            metrics = {}
            
            try:
                # First check if learning is enabled
                if not self.interface.learning_enabled:
                    return {"status": "learning_disabled"}
                    
                # Check system resources if resource awareness is enabled
                if self.config.resource_awareness:
                    import psutil
                    cpu_usage = psutil.cpu_percent()
                    memory_usage = psutil.virtual_memory().percent
                    
                    if cpu_usage > 85 or memory_usage > 90:
                        logger.warning(f"Skipping training due to high resource usage: CPU {cpu_usage}%, Memory {memory_usage}%")
                        return {"status": "skipped", "reason": "high_resource_usage"}
                
                # 1. First run unsupervised learning to identify patterns/clusters
                if self.interface.unsupervised_module and self.config.enable_unsupervised:
                    cluster_success = await self._update_clusters()
                    metrics["unsupervised"] = {
                        "clusters_updated": cluster_success,
                        "clusters": self.interface.unsupervised_module.n_clusters if hasattr(self.interface.unsupervised_module, 'n_clusters') else 0,
                        "buffer_size": len(self.interface.unsupervised_module.buffer) if hasattr(self.interface.unsupervised_module, 'buffer') else 0
                    }
                    self.stats.unsupervised = metrics["unsupervised"]
                
                # 2. Then run self-supervised to improve representations
                if self.interface.self_supervised_module and self.config.enable_self_supervised:
                    ss_metrics = await self._update_self_supervised()
                    metrics["self_supervised"] = ss_metrics
                    self.stats.self_supervised = ss_metrics
                
                # 3. Finally run RL to optimize decision making with improved embeddings
                if self.interface.rl_trainer and self.config.enable_reinforcement:
                    await self._train_rl_from_history()
                    
                    # Get RL training stats
                    if hasattr(self.interface.rl_trainer, 'get_training_stats'):
                        rl_stats = self.interface.rl_trainer.get_training_stats()
                        metrics["reinforcement"] = {k: float(v) if isinstance(v, (int, float)) else v 
                                                 for k, v in rl_stats.items()}
                        self.stats.reinforcement = metrics["reinforcement"]
                
                # 4. Apply cross-learning to share insights between systems
                if self.config.enable_cross_learning:
                    cross_metrics = await self._apply_cross_learning()
                    metrics["cross_learning"] = cross_metrics
                
                # 5. Handle P2P sharing of high-quality learning samples
                if self.config.p2p_sharing and hasattr(self.interface, 'p2p_network') and self.interface.p2p_network:
                    await self._handle_p2p_sharing()
                    
                # Update statistics
                self.stats.training_cycles += 1
                
                # Track metrics history
                self.metrics_history.append(metrics)
                if len(self.metrics_history) > 100:
                    self.metrics_history = self.metrics_history[-100:]
                
                logger.info(f"Completed learning cycle {self.stats.training_cycles}")
                return metrics
                
            except Exception as e:
                logger.error(f"Coordinated training cycle failed: {e}", exc_info=True)
                return {"error": str(e), "status": "failed"}
                        
    async def _apply_cross_learning(self) -> Dict[str, Any]:
        """Apply cross-learning between different learning systems
        
        Returns:
            Dict[str, Any]: Metrics from cross-learning
        """
        try:
            cross_metrics = {
                "transfers": 0,
                "insights_shared": 0
            }
            
            # Skip if required modules aren't available
            if not (self.interface.unsupervised_module and 
                    self.interface.self_supervised_module and 
                    self.interface.rl_trainer):
                return {"status": "skipped", "reason": "missing_modules"}
            
            # 1. Use unsupervised clusters to guide self-supervised learning
            if hasattr(self.interface.unsupervised_module, 'get_cluster_centers'):
                # Get representative samples from clusters
                cluster_centers = self.interface.unsupervised_module.get_cluster_centers()
                if cluster_centers is not None and len(cluster_centers) > 0:
                    # Use cluster insights to improve representations
                    if hasattr(self.interface.self_supervised_module, 'incorporate_clusters'):
                        await self.interface.self_supervised_module.incorporate_clusters(cluster_centers)
                        cross_metrics["transfers"] += 1
            
            # 2. Use self-supervised representations to improve RL state representation
            if hasattr(self.interface.self_supervised_module, 'get_model_improvements'):
                improvements = self.interface.self_supervised_module.get_model_improvements()
                if improvements and hasattr(self.interface.rl_trainer, 'update_state_representation'):
                    await self.interface.rl_trainer.update_state_representation(improvements)
                    cross_metrics["insights_shared"] += 1
            
            # Track cross-learning transfers
            self.stats.cross_learning_transfers += cross_metrics["transfers"]
            
            return cross_metrics
            
        except Exception as e:
            logger.error("Error in cross-learning: %s", e, exc_info=True)
            return {"error": str(e), "status": "failed"}

    async def _handle_p2p_sharing(self) -> Dict[str, Any]:
        """Handle P2P sharing of high-quality learning samples
        
        Returns:
            Dict[str, Any]: Metrics from P2P sharing
        """
        try:
            # Implementation depends on your P2P network setup
            # This is a placeholder for the actual implementation
            return {"status": "not_implemented"}
        except Exception as e:
            logger.error("Error in P2P sharing: %s", e, exc_info=True)
            return {"error": str(e), "status": "failed"}
            
    async def _update_clusters(self) -> bool:
        """Update unsupervised clustering with recent data
        
        Returns:
            bool: Whether clusters were updated successfully
        """
        if not self.interface.unsupervised_module or not hasattr(self.interface, 'history') or len(self.interface.history) < 3:
            return False
            
        try:
            # Get recent text samples for clustering
            messages = [msg for msg, _ in self.interface.history[-30:]]
            for msg in messages:
                # Add messages to buffer for clustering
                if len(msg.split()) >= self.config.min_sample_length:
                    try:
                        # Get embedding first if possible
                        embedding = await self.interface.get_embedding(msg)
                        if embedding is not None:
                            self.interface.unsupervised_module.add_to_buffer(embedding.cpu().numpy())
                    except Exception as e:
                        logger.error("Error adding message to unsupervised buffer: %s", e)
            
            # After adding to buffer, update clusters if possible
            if hasattr(self.interface.unsupervised_module, 'update_clusters'):
                await self.interface.unsupervised_module.update_clusters()
            return True
        except Exception as e:
            logger.error("Error updating clusters: %s", e)
            return False
                
    async def _update_self_supervised(self) -> Dict[str, Any]:
        """Update self-supervised learning with recent conversation data
        
        Returns:
            Dict[str, Any]: Metrics from self-supervised learning
        """
        if not self.interface.self_supervised_module or not hasattr(self.interface, 'history'):
            return {"status": "skipped", "reason": "no_module_or_history"}
            
        try:
            # Get recent conversation samples for self-supervised learning
            samples = []
            if len(self.interface.history) >= 2:
                # Create training pairs from conversation history
                for i in range(1, len(self.interface.history)):
                    prev_msg, _ = self.interface.history[i-1]
                    curr_msg, _ = self.interface.history[i]
                    
                    if len(prev_msg.split()) >= self.config.min_sample_length and \
                       len(curr_msg.split()) >= self.config.min_sample_length:
                        samples.append((prev_msg, curr_msg))
                        
            # Train on collected samples
            if samples:
                metrics = await self.interface.self_supervised_module.train(samples, 
                                                                         batch_size=self.config.batch_size)
                # Update total examples count
                self.stats.total_examples += len(samples)
                return metrics
            
            return {"status": "no_samples"}
            
        except Exception as e:
            logger.error("Error in self-supervised learning: %s", e, exc_info=True)
            return {"error": str(e), "status": "failed"}
                
    async def _train_rl_from_history(self) -> Dict[str, Any]:
        """Update reinforcement learning module with recent conversation history
        
        Returns:
            Dict[str, Any]: Metrics from reinforcement learning
        """
        if not self.interface.rl_trainer or not hasattr(self.interface, 'history'):
            return {"status": "skipped", "reason": "no_module_or_history"}
            
        try:
            # Get recent conversation exchanges for RL training
            if len(self.interface.history) >= 2:
                # Extract state-action-reward sequences from history
                # This is a simplified version - actual implementation would depend on your RL setup
                training_data = []
                for i in range(1, min(30, len(self.interface.history))):
                    user_msg, _ = self.interface.history[i-1]
                    bot_msg, feedback = self.interface.history[i]
                    
                    # Get reward from feedback if available
                    reward = 0.0
                    if isinstance(feedback, dict) and 'rating' in feedback:
                        reward = float(feedback['rating']) / 5.0  # Normalize to [0,1]
                    
                    if len(user_msg) > 0 and len(bot_msg) > 0:
                        training_data.append({
                            'state': user_msg,
                            'action': bot_msg,
                            'reward': reward
                        })
                
                # Train RL system on collected data
                if training_data and hasattr(self.interface.rl_trainer, 'train_from_samples'):
                    metrics = await self.interface.rl_trainer.train_from_samples(training_data)
                    return metrics
            
            return {"status": "no_samples"}
            
        except Exception as e:
            logger.error("Error in reinforcement learning: %s", e, exc_info=True)
            return {"error": str(e), "status": "failed"}
