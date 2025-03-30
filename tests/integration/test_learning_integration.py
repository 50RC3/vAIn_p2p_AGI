import asyncio
import logging
import torch
from ai_core.chatbot.interface import ChatbotInterface, LearningConfig
from models.simple_nn import SimpleNN
from ai_core.model_storage import ModelStorage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_integrated_learning():
    """Test that all learning components work together"""
    
    # Initialize components
    model = SimpleNN()
    storage = ModelStorage()
    learning_config = LearningConfig(
        enable_unsupervised=True,
        enable_self_supervised=True,
        enable_reinforcement=True,
        batch_size=4,  # Small batch for testing
        training_interval=5  # Quick training for test
    )
    
    # Create interface
    interface = ChatbotInterface(
        model=model,
        storage=storage,
        learning_config=learning_config
    )
    
    # Test conversation flow
    test_messages = [
        "Hello, how are you?",
        "Tell me about reinforcement learning",
        "What is the current time?",
        "How do neural networks work?",
        "Thank you for the information"
    ]
    
    # Process test messages
    for msg in test_messages:
        logger.info(f"Processing: {msg}")
        response = await interface.process_message(msg)
        logger.info(f"Response: {response.text}")
        
        # Add feedback
        feedback = 0.7  # Positive feedback
        await interface.store_feedback(response.id, feedback)
    
    # Wait for background training
    logger.info("Waiting for background training...")
    await asyncio.sleep(10)
    
    # Verify integration
    if interface.learning_enabled:
        logger.info("Learning is enabled")
        
        if interface.unsupervised_module:
            logger.info(f"Unsupervised module has processed {len(interface.unsupervised_module.buffer)} examples")
            
        if interface.self_supervised_module:
            logger.info("Self-supervised module initialized")
            
        if interface.rl_trainer:
            stats = interface.rl_trainer.get_training_stats() 
            logger.info(f"RL stats: {stats}")
            
    await interface.shutdown()
    logger.info("Integration test completed")

if __name__ == "__main__":
    asyncio.run(test_integrated_learning())