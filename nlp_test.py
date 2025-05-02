"""
Simple test script for NLTK functionality.
Run this from the project root directory.
"""
import sys
import os

# Add the project directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Import from the project's modules
try:
    from ai_core.nlp.nltk_utils import analyze_text_nltk, initialize_nltk_components
    print("Successfully imported ai_core.nlp.nltk_utils module")

    # Initialize NLTK components
    initialize_nltk_components()

    # Test text analysis
    test_text = "GPT models are transforming AI capabilities worldwide."
    result = analyze_text_nltk(test_text)

    print("\nNLTK Analysis Result:")
    print(f"Tokens: {result.get('tokens', [])[:5]}...")
    print(f"Sentences: {len(result.get('sentences', []))}")
    print(f"Named Entities: {result.get('named_entities', [])}")
    print(f"Sentiment: {result.get('sentiment', {})}")

except ImportError as e:
    print(f"Import error: {e}")

    # Try to install required packages
    print("\nAttempting to install required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])

    print("\nDownloading NLTK resources...")
    import nltk  # type: ignore
    nltk.download('punkt')
