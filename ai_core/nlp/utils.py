import spacy
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
import asyncio
from functools import lru_cache

logger = logging.getLogger(__name__)

# Global variable to store the spaCy model
_nlp = None

async def load_spacy_model(model_name: str = "en_core_web_sm") -> None:
    """
    Load spaCy model asynchronously.
    
    Args:
        model_name: The name of the spaCy model to load
        
    Note:
        Call this function at application startup to load the model.
        This function supports: "en_core_web_sm", "en_core_web_md", "en_core_web_lg"
    """
    global _nlp
    
    # Load spaCy model in a separate thread to avoid blocking
    def _load_model():
        try:
            import spacy
            
            # Try to load the model
            try:
                return spacy.load(model_name)
            except OSError:
                # Model not found, try to download it
                logger.info(f"Downloading spaCy model: {model_name}")
                spacy.cli.download(model_name)
                return spacy.load(model_name)
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {str(e)}")
            return None
    
    # Load model in executor to avoid blocking
    _nlp = await asyncio.get_event_loop().run_in_executor(None, _load_model)
    logger.info(f"SpaCy model '{model_name}' loaded: {_nlp is not None}")

def get_nlp():
    """Get the loaded spaCy model"""
    global _nlp
    if _nlp is None:
        # Fallback to synchronous loading if not loaded yet
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm")
            logger.warning("Loading spaCy model synchronously, consider using load_spacy_model at startup")
        except OSError:
            logger.error("SpaCy model not found. Please run load_spacy_model first")
            raise RuntimeError("SpaCy model not loaded")
    return _nlp

@lru_cache(maxsize=1000)
def analyze_text(text: str) -> Dict[str, Any]:
    """
    Analyze text using spaCy model and return structured information.
    
    Args:
        text: The text to analyze
        
    Returns:
        Dictionary containing entities, sentiment, POS tags, etc.
    """
    try:
        nlp = get_nlp()
        doc = nlp(text)
        
        # Extract entities
        entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
        
        # Extract noun chunks (noun phrases)
        noun_chunks = [chunk.text for chunk in doc.noun_chunks]
        
        # Extract POS tags
        pos_tags = [(token.text, token.pos_) for token in doc]
        
        # Extract dependency relations
        dependencies = [(token.text, token.dep_) for token in doc]
        
        # Extract key tokens (excluding stopwords and punctuation)
        key_tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
        
        # Return structured analysis
        return {
            "entities": entities,
            "noun_chunks": noun_chunks,
            "pos_tags": pos_tags,
            "dependencies": dependencies,
            "key_tokens": key_tokens,
            "tokens": [token.text for token in doc],
            "sentence_count": len(list(doc.sents))
        }
    except Exception as e:
        logger.error(f"Error analyzing text: {str(e)}")
        return {"error": str(e)}

def get_sentiment(text: str) -> float:
    """
    Get sentiment score for text (-1 to 1 where -1 is negative, 1 is positive)
    
    Simple implementation that can be enhanced with a proper sentiment model
    """
    try:
        nlp = get_nlp()
        doc = nlp(text)
        
        # This is a simplistic implementation
        # For better sentiment analysis, consider using a dedicated model/library
        
        # Count positive and negative words using basic lexicon approach
        positive_words = {"good", "great", "excellent", "amazing", "wonderful", 
                         "happy", "best", "love", "like", "nice", "thanks",
                         "thank", "helpful", "positive", "beautiful"}
        
        negative_words = {"bad", "terrible", "awful", "horrible", "worst",
                         "hate", "dislike", "stupid", "ugly", "negative",
                         "wrong", "error", "fail", "failed", "not"}
        
        tokens = [token.text.lower() for token in doc]
        
        positive_count = sum(1 for token in tokens if token in positive_words)
        negative_count = sum(1 for token in tokens if token in negative_words)
        
        # Calculate simple sentiment score
        if positive_count + negative_count == 0:
            return 0  # Neutral
            
        sentiment = (positive_count - negative_count) / (positive_count + negative_count)
        return max(-1, min(1, sentiment))  # Clamp between -1 and 1
    except Exception as e:
        logger.error(f"Error getting sentiment: {str(e)}")
        return 0  # Default to neutral

def get_main_topic(text: str) -> str:
    """Extract the main topic of the text"""
    try:
        analysis = analyze_text(text)
        
        # First look for entities
        if analysis["entities"]:
            # Prioritize organizations, people, or other named entities
            for entity in analysis["entities"]:
                if entity["label"] in ["ORG", "PERSON", "GPE", "LOC", "PRODUCT"]:
                    return entity["text"]
            
            # If no priority entities, return the first one
            return analysis["entities"][0]["text"]
        
        # If no entities, look for noun chunks
        if analysis["noun_chunks"]:
            return analysis["noun_chunks"][0]
        
        # If no noun chunks, look for key tokens
        if analysis["key_tokens"]:
            return analysis["key_tokens"][0]
        
        # Fallback to first token if nothing else works
        if analysis["tokens"]:
            return analysis["tokens"][0]
            
        return ""
    except Exception as e:
        logger.error(f"Error getting main topic: {str(e)}")
        return ""

def get_question_type(text: str) -> str:
    """
    Identify the type of question being asked.
    
    Returns:
        String with question type: 'what', 'who', 'when', 'where', 'why', 'how', 'yes_no', 'other'
    """
    try:
        nlp = get_nlp()
        doc = nlp(text)
        
        # Check for standard question words
        first_token = doc[0].text.lower() if len(doc) > 0 else ""
        
        if first_token in ["what"]:
            return "what"
        elif first_token in ["who"]:
            return "who"
        elif first_token in ["when"]:
            return "when"
        elif first_token in ["where"]:
            return "where"
        elif first_token in ["why"]:
            return "why"
        elif first_token in ["how"]:
            return "how"
        
        # Check for yes/no questions (verb at the beginning)
        if len(doc) > 1 and doc[0].pos_ == "AUX":
            return "yes_no"
            
        # Check for question marks
        if text.strip().endswith("?"):
            return "other_question"
            
        return "statement"
    except Exception as e:
        logger.error(f"Error identifying question type: {str(e)}")
        return "unknown"

def extract_key_phrases(text: str, max_phrases: int = 3) -> List[str]:
    """
    Extract key phrases from the text.
    
    Args:
        text: The text to analyze
        max_phrases: Maximum number of phrases to extract
        
    Returns:
        List of key phrases
    """
    try:
        analysis = analyze_text(text)
        
        # Combine entities and noun chunks
        key_phrases = []
        
        # Add entity texts
        for entity in analysis["entities"]:
            phrase = entity["text"]
            if phrase not in key_phrases:
                key_phrases.append(phrase)
        
        # Add noun chunks
        for chunk in analysis["noun_chunks"]:
            if chunk not in key_phrases:
                key_phrases.append(chunk)
        
        # Return up to max_phrases
        return key_phrases[:max_phrases]
    except Exception as e:
        logger.error(f"Error extracting key phrases: {str(e)}")
        return []
