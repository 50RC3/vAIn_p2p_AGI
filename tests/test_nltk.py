"""
Test script for NLTK integration in the chatbot.
"""
import asyncio
import logging
import sys
import os
from typing import Dict, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_nltk_functionality():
    """Test NLTK functionality"""
    try:
        # Import NLTK utils
        from ai_core.nlp.nltk_utils import (
            download_nltk_resources,
            initialize_nltk_components,
            analyze_text_nltk, 
            get_nltk_sentiment,
            determine_intent,
            extract_keywords_nltk,
            get_text_statistics
        )
        
        # Download resources
        logger.info("Downloading NLTK resources...")
        success = await download_nltk_resources()
        if not success:
            logger.error("Failed to download NLTK resources")
            return
            
        # Initialize components
        initialize_nltk_components()
        
        # Test text samples
        test_texts = [
            "Hello, how are you doing today?",
            "I'm really happy with the results of this project.",
            "This system is not working as expected, I'm disappointed.",
            "What is the capital of France?",
            "Python is a powerful programming language for AI development.",
            "Thank you for all your help with this issue!",
            "Goodbye, I'll talk to you later."
        ]
        
        for i, text in enumerate(test_texts):
            logger.info(f"\nTest {i+1}: '{text}'")
            
            # Get intent
            intent = determine_intent(text)
            logger.info(f"Intent: {intent.get('intent', 'unknown')} (confidence: {intent.get('confidence', 0)})")
            
            # Get sentiment
            sentiment = get_nltk_sentiment(text)
            logger.info(f"Sentiment: {sentiment}")
            
            # Get keywords
            keywords = extract_keywords_nltk(text)
            logger.info(f"Keywords: {keywords}")
            
            # Get text statistics
            stats = get_text_statistics(text)
            logger.info(f"Statistics: {stats}")
            
            # Full analysis
            logger.info("Running full analysis...")
            analysis = analyze_text_nltk(text)
            logger.info(f"Entities found: {analysis.get('entities', {})}")
            logger.info(f"Lemmatized tokens: {analysis.get('lemmatized_tokens', [][:5])}...")
            
            logger.info("-" * 50)
            
    except ImportError as e:
        logger.error(f"NLTK not available: {e}")
    except Exception as e:
        logger.error(f"Error in NLTK testing: {e}")

if __name__ == "__main__":
    asyncio.run(test_nltk_functionality())
