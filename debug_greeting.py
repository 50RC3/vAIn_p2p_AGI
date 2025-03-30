"""
Debug script to test greeting detection in isolation.
"""
import asyncio
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Add trace logging for NLTK utils
nltk_logger = logging.getLogger("ai_core.nlp.nltk_utils")
nltk_logger.setLevel(logging.DEBUG)

async def test_greeting_detection():
    """Test greeting detection directly with the NLTK utils."""
    try:
        # Import needed functions
        from ai_core.nlp.nltk_utils import download_nltk_resources, initialize_nltk_components, determine_intent

        # Initialize NLTK
        logger.info("Downloading NLTK resources...")
        await download_nltk_resources()
        initialize_nltk_components()
        
        # Simple test cases
        test_cases = [
            "hi",
            "hello",
            "hey there",
            "good morning",
            "how are you?",
            "what's up",
            "goodbye",  # Not a greeting
            "thanks",   # Not a greeting
            "help me",  # Not a greeting
        ]
        
        logger.info("Testing greeting detection...")
        for text in test_cases:
            # Get the intent
            result = determine_intent(text)
            intent = result.get("intent")
            confidence = result.get("confidence", 0)
            
            logger.info(f"Text: '{text}' -> Intent: {intent} (Confidence: {confidence:.2f})")
            
            # Check if greeting was detected correctly
            if text in ["hi", "hello", "hey there", "good morning", "how are you?", "what's up"]:
                if intent == "greeting":
                    logger.info(f"✅ CORRECT: '{text}' detected as greeting")
                else:
                    logger.error(f"❌ ERROR: '{text}' not detected as greeting!")
            else:
                if intent != "greeting":
                    logger.info(f"✅ CORRECT: '{text}' not detected as greeting")
                else:
                    logger.error(f"❌ ERROR: '{text}' incorrectly detected as greeting!")
    
    except Exception as e:
        logger.exception(f"Error in greeting detection test: {e}")

if __name__ == "__main__":
    asyncio.run(test_greeting_detection())
