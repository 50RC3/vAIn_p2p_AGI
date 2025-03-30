"""
Debug script to directly test intent detection and greeting responses.
"""
import asyncio
import logging
import sys
import os
import time

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Add path for project imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def test_intent_detection():
    """Direct test of intent detection and response generation."""
    try:
        # Import required components
        from ai_core.nlp.nltk_utils import download_nltk_resources, initialize_nltk_components, determine_intent
        from models.simple_nn import SimpleNN
        
        # Initialize NLTK
        logger.info("Downloading NLTK resources...")
        await download_nltk_resources()
        initialize_nltk_components()
        
        # Create model
        model = SimpleNN(input_size=512, hidden_size=256, output_size=512)
        
        # Test greetings
        test_cases = [
            "hi",
            "hello",
            "hey there",
            "good morning",
            "how are you?",
            "what's up",
        ]
        
        logger.info("=== TESTING INTENT DETECTION ===")
        for text in test_cases:
            # First, test raw intent detection
            result = determine_intent(text)
            intent = result.get("intent")
            confidence = result.get("confidence", 0)
            logger.info(f"Text: '{text}' -> Intent: {intent} (Confidence: {confidence:.2f})")
            
            # Then test model response
            start_time = time.time()
            response = model._generate_nlp_response(text)
            elapsed = time.time() - start_time
            logger.info(f"Response [{elapsed:.3f}s]: '{response}'")
            
            # Check if it's a greeting response
            is_greeting_response = any(phrase in response.lower() for phrase in 
                                      ["hello", "hi", "greetings", "how can i", "how may i"])
            if is_greeting_response:
                logger.info("✓ CORRECT: Greeting response generated")
            else:
                logger.error("✗ ERROR: Not a greeting response!")
            
            logger.info("-" * 50)
    
    except Exception as e:
        logger.exception(f"Error in intent detection test: {e}")

if __name__ == "__main__":
    asyncio.run(test_intent_detection())
