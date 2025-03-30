"""
Learning integration module for the chatbot system.
Provides integration between the chatbot interface and advanced learning capabilities.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
from datetime import datetime
import json
import os
from pathlib import Path

from .interface import ChatbotInterface
from .rl_trainer import RLTrainer
from ..learning.advanced_learning import AdvancedLearning, LearningConfig
from network.p2p_network import P2PNetwork

logger = logging.getLogger(__name__)

class LearningIntegration:
    """
    Integrates advanced learning capabilities with the chatbot interface.
    Manages the learning lifecycle and provides simplified access to learning features.
    """
    
    def __init__(
        self, 
        chatbot: ChatbotInterface,
        config_path: Optional[str] = None
    ):
        """
        Initialize learning integration.
        
        Args:
            chatbot: The chatbot interface to integrate with
            config_path: Optional path to a JSON configuration file
        """
        self.chatbot = chatbot
        self.learning_system = None
        self.rl_trainer = None
        self.p2p_network = None
        self.config = self._load_config(config_path)
        self.ready = False
        self.training_tasks = []
        
        # Event callbacks
        self.callbacks = {
            'training_started': [],
            'training_completed': [],
            'training_error': [],
            'new_cluster': [],
            'sample_added': []
        }
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from JSON file or use defaults"""
        config = {
            "unsupervised_batch_size": 32,
            "supervised_batch_size": 16,
            "rl_batch_size": 8,
            "learning_rate": 1e-5,
            "unsupervised_ratio": 0.4,
            "contrastive_weight": 0.3,
            "training_interval": 600,
            "min_samples_for_training": 50,
            "max_history_length": 10000,
            "enable_peer_learning": True,
            "contrastive_temperature": 0.07,
            "similarity_threshold": 0.85,
            "memory_cleanup_interval": 3600,
            "max_peer_samples": 5000,
            "save_path": "models/advanced_learning",
            "save_interval": 10,
            "interactive_mode": False,
            "interactive_timeout": 30
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    config.update(user_config)
                    logger.info(f"Loaded learning config from {config_path}")
            except Exception as e:
                logger.error(f"Error loading config from {config_path}: {e}")
        
        return config
    
    async def initialize(
        self, 
        rl_trainer: Optional[RLTrainer] = None, 
        p2p_network: Optional[P2PNetwork] = None
    ) -> bool:
        """
        Initialize the learning system.
        
        Args:
            rl_trainer: Optional RL trainer (will use chatbot's if not provided)
            p2p_network: Optional P2P network for distributed learning
            
        Returns:
            bool: True if initialization was successful
        """
        try:
            # Get RL trainer from chatbot if not provided
            self.rl_trainer = rl_trainer or getattr(self.chatbot, 'rl_trainer', None)
            if not self.rl_trainer:
                logger.error("No RL trainer available, learning system cannot be initialized")
                return False
            
            self.p2p_network = p2p_network
            
            # Convert dict config to LearningConfig
            learning_config = LearningConfig()
            for key, value in self.config.items():
                if hasattr(learning_config, key):
                    setattr(learning_config, key, value)
            
            # Initialize learning system
            self.learning_system = AdvancedLearning(
                chatbot=self.chatbot,
                rl_trainer=self.rl_trainer,
                config=learning_config,
                p2p_network=self.p2p_network
            )
            
            # Try to load previous state
            try:
                await self.learning_system.load_state()
            except Exception as e:
                logger.warning(f"Could not load previous learning state: {e}")
            
            self.ready = True
            logger.info("Learning integration initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize learning integration: {e}")
            return False
    
    async def process_conversation(
        self, 
        message: str, 
        response: str, 
        feedback: Optional[float] = None
    ) -> None:
        """
        Process a conversation for learning.
        
        Args:
            message: The user message
            response: The chatbot response
            feedback: Optional feedback score (0.0 - 1.0)
        """
        if not self.ready or not self.learning_system:
            logger.warning("Learning system not ready, skipping conversation processing")
            return
        
        try:
            await self.learning_system.add_conversation(message, response, feedback)
            await self._notify_callbacks('sample_added', {
                'type': 'conversation',
                'message': message[:100] + '...' if len(message) > 100 else message,
                'has_feedback': feedback is not None
            })
        except Exception as e:
            logger.error(f"Error processing conversation: {e}")
    
    async def process_text_sample(self, text: str) -> None:
        """
        Process a raw text sample for unsupervised learning.
        
        Args:
            text: The text sample
        """
        if not self.ready or not self.learning_system:
            logger.warning("Learning system not ready, skipping text sample processing")
            return
        
        try:
            await self.learning_system.add_unsupervised_sample(text)
            await self._notify_callbacks('sample_added', {
                'type': 'unsupervised',
                'text_length': len(text)
            })
        except Exception as e:
            logger.error(f"Error processing text sample: {e}")
    
    async def process_peer_sample(self, sample: Dict[str, Any]) -> None:
        """
        Process a sample received from a peer node.
        
        Args:
            sample: The sample data dict
        """
        if not self.ready or not self.learning_system:
            logger.warning("Learning system not ready, skipping peer sample processing")
            return
        
        try:
            await self.learning_system.receive_peer_sample(sample)
        except Exception as e:
            logger.error(f"Error processing peer sample: {e}")
    
    async def force_training_cycle(self) -> Dict[str, Any]:
        """
        Force an immediate training cycle.
        
        Returns:
            Dict containing training metrics or error information
        """
        if not self.ready or not self.learning_system:
            return {"error": "Learning system not ready"}
        
        try:
            await self._notify_callbacks('training_started', {})
            results = await self.learning_system.force_training_cycle()
            
            if 'error' not in results:
                await self._notify_callbacks('training_completed', results)
            else:
                await self._notify_callbacks('training_error', {'message': results['error']})
                
            return results
        except Exception as e:
            error_info = {"error": str(e)}
            await self._notify_callbacks('training_error', error_info)
            return error_info
    
    async def get_learning_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics from the learning system.
        
        Returns:
            Dict of metrics
        """
        if not self.ready or not self.learning_system:
            return {"error": "Learning system not ready"}
        
        try:
            return await self.learning_system.get_metrics()
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return {"error": str(e)}
    
    async def get_cluster_information(self) -> Dict[str, Any]:
        """
        Get information about conversation clusters.
        
        Returns:
            Dict mapping cluster IDs to cluster information
        """
        if not self.ready or not self.learning_system:
            return {"error": "Learning system not ready"}
        
        try:
            return await self.learning_system.get_cluster_info()
        except Exception as e:
            logger.error(f"Error getting cluster info: {e}")
            return {"error": str(e)}
    
    def register_callback(self, event: str, callback: Callable[[Dict[str, Any]], None]) -> bool:
        """
        Register a callback for learning system events.
        
        Args:
            event: Event name ('training_started', 'training_completed', etc.)
            callback: Async callback function
            
        Returns:
            bool: True if registration was successful
        """
        if event in self.callbacks:
            self.callbacks[event].append(callback)
            return True
        return False
    
    async def _notify_callbacks(self, event: str, data: Dict[str, Any]) -> None:
        """Notify registered callbacks about an event"""
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    logger.error(f"Error in {event} callback: {e}")
    
    async def cleanup(self) -> None:
        """Clean up resources and save state"""
        if self.learning_system:
            try:
                # Save the current state
                await self.learning_system._save_state()
                
                # Cancel any running training tasks
                for task in self.training_tasks:
                    if not task.done():
                        task.cancel()
                
                logger.info("Learning integration cleaned up successfully")
            except Exception as e:
                logger.error(f"Error during learning cleanup: {e}")
