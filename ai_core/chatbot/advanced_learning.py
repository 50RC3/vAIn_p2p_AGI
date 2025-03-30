"""
Integration module that provides a simplified interface to the AdvancedLearning system.
This re-exports the core functionality from ai_core.learning.advanced_learning.
"""

import logging
from typing import Dict, Any, Optional

# Re-export from the learning module
from ai_core.learning.advanced_learning import (
    AdvancedLearning, 
    LearningConfig
)

from ai_core.chatbot.rl_trainer import RLTrainer
# Import from the correct location
from ai_core.chatbot.interface import ChatbotInterface
from network.p2p_network import P2PNetwork

logger = logging.getLogger(__name__)

async def initialize_learning_system(
    chatbot: ChatbotInterface,
    rl_trainer: RLTrainer,
    config: Optional[Dict[str, Any]] = None,
    p2p_network: Optional[P2PNetwork] = None
) -> AdvancedLearning:
    """
    Initialize the advanced learning system with the given components.
    
    Args:
        chatbot: The chatbot interface
        rl_trainer: The reinforcement learning trainer
        config: Optional dictionary of configuration parameters
        p2p_network: Optional P2P network for distributed learning
        
    Returns:
        Configured AdvancedLearning instance
    """
    # Convert dict config to LearningConfig if provided
    learning_config = None
    if config:
        learning_config = LearningConfig()
        for key, value in config.items():
            if hasattr(learning_config, key):
                setattr(learning_config, key, value)
    
    # Create and initialize the learning system
    learning_system = AdvancedLearning(
        chatbot=chatbot,
        rl_trainer=rl_trainer,
        config=learning_config,
        p2p_network=p2p_network
    )
    
    # Load previous state if available
    try:
        await learning_system.load_state()
    except (FileNotFoundError, PermissionError, IOError) as e:
        logger.warning("Could not load previous learning state: %s", e)
    
    logger.info("Advanced learning system initialized and ready")
    return learning_system

async def add_learning_sample(
    learning_system: AdvancedLearning,
    message: str,
    response: Optional[str] = None,
    feedback: Optional[float] = None
) -> None:
    """
    Add a learning sample to the system.
    
    Args:
        learning_system: The advanced learning system
        message: The message text
        response: Optional response text (if None, treated as unsupervised sample)
        feedback: Optional feedback score (0.0 - 1.0)
    """
    if response is None:
        await learning_system.add_unsupervised_sample(message)
    else:
        await learning_system.add_conversation(message, response, feedback)
