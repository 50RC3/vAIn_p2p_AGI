"""
Test script for greeting recognition in the chatbot.
Tests both simple and complex greeting patterns.
"""
import asyncio
import logging
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.simple_nn import SimpleNN
from ai_core.chatbot.interface import ChatbotInterface
from ai_core.model_storage import ModelStorage
from ai_core.nlp.nltk_utils import determine_intent

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_greeting_detection():
    """Test the chatbot's ability to detect and respond to greetings."""
    # Initialize NLTK intent detection
    try:
        from ai_core.nlp.nltk_utils import download_nltk_resources, initialize_nltk_components
        await download_nltk_resources()
        initialize_nltk_components()
    except ImportError:
        logger.error("NLTK not available")
        return
        
    # Test greeting patterns
    greeting_patterns = [
        "hi",
        "hello",
        "hey",
        "hello there",
        "good morning",
        "hi there, how are you?",
        "hey, what's up?",
        "greetings",
        "howdy",
        "good evening",
        "yo",
        "hiya",
    ]
    
    success_count = 0
    
    logger.info("Testing greeting intent detection...")
    for pattern in greeting_patterns:
        intent_info = determine_intent(pattern)
        intent = intent_info.get("intent", "unknown")
        confidence = intent_info.get("confidence", 0)
        
        if intent == "greeting":
            success_count += 1
            logger.info(f"✓ '{pattern}' correctly detected as greeting (confidence: {confidence:.2f})")
        else:
            logger.error(f"✗ '{pattern}' not detected as greeting. Got: {intent}")
    
    success_rate = success_count / len(greeting_patterns) * 100
    logger.info(f"Greeting detection success rate: {success_rate:.1f}% ({success_count}/{len(greeting_patterns)})")

async def test_chatbot_greeting_responses():
    """Test the chatbot's responses to greetings."""
    logger.info("Testing chatbot greeting responses")
    
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
    
    # Test greeting messages
    greetings = [
        "hi",
        "hello",
        "hey there",
        "good morning",
    ]
    
    greeting_responses = 0
    
    for greeting in greetings:
        logger.info(f"TESTING: '{greeting}'")
        
        response = await chatbot.process_message(greeting)
        
        # Check if this is a greeting response
        is_greeting = any(phrase in response.text.lower() for phrase in 
                         ["hello", "hi", "greetings", "welcome", "assist"])
        
        if is_greeting:
            greeting_responses += 1
            logger.info(f"✓ Greeting response: {response.text}")
        else:
            logger.error(f"✗ Not a greeting response: {response.text}")
        
        logger.info("-" * 40)
    
    success_rate = greeting_responses / len(greetings) * 100
    logger.info(f"Greeting response success rate: {success_rate:.1f}% ({greeting_responses}/{len(greetings)})")
    
    # Clear session
    await chatbot.clear_session()

if __name__ == "__main__":
    asyncio.run(test_greeting_detection())
    asyncio.run(test_chatbot_greeting_responses())
