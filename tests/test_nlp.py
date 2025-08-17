"""
Test script for NLP capabilities of the chatbot.
This tests both spaCy and NLTK integrations.
"""
import asyncio
import logging
import sys
import os
import pytest

# Skip the entire test module if cognitive_evolution is not available
pytestmark = pytest.importorskip("ai_core.chatbot", reason="Chatbot module not implemented yet")

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.simple_nn import SimpleNN
from ai_core.chatbot.interface import ChatbotInterface
from ai_core.model_storage import ModelStorage

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_nlp_integration():
    """Test the NLP capabilities of the chatbot."""
    logger.info("Starting NLP integration test")
    
    # Initialize model and storage
    model = SimpleNN(input_size=512, output_size=512, hidden_size=256)
    storage = ModelStorage()
    
    # Initialize chatbot interface
    chatbot = ChatbotInterface(model, storage, interactive=True)
    
    # Initialize NLP components
    if hasattr(chatbot, '_initialize_nlp'):
        await chatbot._initialize_nlp()
    if hasattr(chatbot, '_initialize_nltk'):
        await chatbot._initialize_nltk()
    
    # Start a session
    await chatbot.start_session()
    
    # Test questions with different NLP features
    test_messages = [
        "Hello there, how are you doing today?",  # Greeting intent
        "What can you tell me about natural language processing?",  # Question type: what
        "Who created you and what is your purpose?",  # Question type: who
        "I'm really happy with how well this system works!",  # Positive sentiment
        "This doesn't seem to be working properly.",  # Negative sentiment
        "Can you analyze the sentiment of this text?",  # Technical question
        "Thank you for your help!",  # Thanks intent
        "Goodbye for now, I'll talk to you later.",  # Farewell intent
        "Python programming is really interesting and versatile.",  # Topic extraction
    ]
    
    logger.info("Testing %d messages with NLP processing", len(test_messages))
    
    # Process each test message
    for i, message in enumerate(test_messages):
        logger.info("\n%d) TESTING: %s", i+1, message)
        
        response = await chatbot.process_message(message)
        
        # Log response
        logger.info("RESPONSE: %s", response.text)
        
        # Check if context contains NLP analysis
        message_key = chatbot._generate_cache_key(message, None)
        
        # Display NLP analysis results if available
        if message_key in chatbot.response_cache:
            logger.info("Confidence: %.2f, Latency: %.4f", 
                     response.confidence, response.latency)
            
            # Extract context info
            message_hash = response.text.__hash__()
            if message_hash in chatbot.context_cache:
                context = chatbot.context_cache[message_hash]
                logger.info("Context: %s", context)
        
        logger.info("-" * 40)
    
    # Clear session
    await chatbot.clear_session()
    logger.info("NLP integration test completed")

def test_nlp_processing():
    from ai_core.chatbot.interface import ChatbotInterface
    # Actual test code here...

if __name__ == "__main__":
    asyncio.run(test_nlp_integration())
