import logging
import asyncio
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from functools import lru_cache
import re
import string

logger = logging.getLogger(__name__)

try:
    import spacy  # type: ignore
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)

try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords, wordnet
    from nltk.stem import WordNetLemmatizer, PorterStemmer
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from nltk.tag import pos_tag
    from nltk.chunk import ne_chunk
    from textblob import TextBlob
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK/TextBlob not available. NLTK features will be disabled.")

# Global variables for NLTK resources
_lemmatizer = None
_stemmer = None
_sentiment_analyzer = None
_downloaded_resources = set()

async def download_nltk_resources() -> bool:
    """
    Download required NLTK resources asynchronously
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not NLTK_AVAILABLE:
        return False
    
    required_resources = [
        ('punkt', 'tokenizers/punkt'),
        ('stopwords', 'corpora/stopwords'),
        ('wordnet', 'corpora/wordnet'),
        ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger'),
        ('vader_lexicon', 'sentiment/vader_lexicon'),
        ('maxent_ne_chunker', 'chunkers/maxent_ne_chunker'),
        ('words', 'corpora/words')
    ]
    
    def _download():
        success = True
        for name, path in required_resources:
            if name not in _downloaded_resources:
                try:
                    nltk.download(name)
                    _downloaded_resources.add(name)
                    logger.info(f"Downloaded NLTK resource: {name}")
                except Exception as e:
                    logger.error(f"Failed to download NLTK resource '{name}': {str(e)}")
                    success = False
        return success
    
    try:
        return await asyncio.get_event_loop().run_in_executor(None, _download)
    except Exception as e:
        logger.error(f"Error downloading NLTK resources: {str(e)}")
        return False

def initialize_nltk_components():
    """Initialize global NLTK components"""
    global _lemmatizer, _stemmer, _sentiment_analyzer
    
    if not NLTK_AVAILABLE:
        return False
    
    try:
        if _lemmatizer is None:
            _lemmatizer = WordNetLemmatizer()
        
        if _stemmer is None:
            _stemmer = PorterStemmer()
            
        if _sentiment_analyzer is None:
            _sentiment_analyzer = SentimentIntensityAnalyzer()
            
        return True
    except Exception as e:
        logger.error(f"Failed to initialize NLTK components: {str(e)}")
        return False

def get_wordnet_pos(tag: str) -> str:
    """
    Map POS tag to WordNet POS tag
    
    Args:
        tag: POS tag from nltk.pos_tag
        
    Returns:
        WordNet POS tag
    """
    tag = tag[0].upper()
    tag_dict = {
        'J': wordnet.ADJ,
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'R': wordnet.ADV
    }
    return tag_dict.get(tag, wordnet.NOUN)

@lru_cache(maxsize=500)
def preprocess_text(text: str, remove_stopwords: bool = True) -> List[str]:
    """
    Preprocess text: tokenize, lowercase, remove punctuation, stopwords
    
    Args:
        text: Input text
        remove_stopwords: Whether to remove stopwords
        
    Returns:
        List of processed tokens
    """
    if not NLTK_AVAILABLE:
        return text.lower().split()
        
    try:
        # Tokenize and lowercase
        tokens = word_tokenize(text.lower())
        
        # Remove punctuation
        tokens = [token for token in tokens if token not in string.punctuation]
        
        # Remove stopwords
        if remove_stopwords:
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token not in stop_words]
            
        return tokens
    except Exception as e:
        logger.error(f"Error preprocessing text: {str(e)}")
        return text.lower().split()

@lru_cache(maxsize=500)
def lemmatize_text(text: str) -> List[str]:
    """
    Lemmatize text (reduce words to base form)
    
    Args:
        text: Input text
        
    Returns:
        List of lemmatized tokens
    """
    if not NLTK_AVAILABLE or _lemmatizer is None:
        return text.lower().split()
    
    try:
        tokens = preprocess_text(text)
        pos_tags = pos_tag(tokens)
        return [_lemmatizer.lemmatize(word, get_wordnet_pos(tag)) 
                for word, tag in pos_tags]
    except Exception as e:
        logger.error(f"Error lemmatizing text: {str(e)}")
        return text.lower().split()

def get_nltk_sentiment(text: str) -> Dict[str, float]:
    """
    Get detailed sentiment analysis using NLTK's VADER
    
    Args:
        text: Input text
        
    Returns:
        Dictionary with sentiment scores
    """
    if not NLTK_AVAILABLE or _sentiment_analyzer is None:
        return {"compound": 0, "pos": 0, "neu": 0, "neg": 0}
    
    try:
        return _sentiment_analyzer.polarity_scores(text)
    except Exception as e:
        logger.error(f"Error getting sentiment: {str(e)}")
        return {"compound": 0, "pos": 0, "neu": 0, "neg": 0}

def extract_entities_nltk(text: str) -> Dict[str, List[str]]:
    """
    Extract named entities using NLTK
    
    Args:
        text: Input text
        
    Returns:
        Dictionary of entity types and values
    """
    if not NLTK_AVAILABLE:
        return {}
    
    try:
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        named_entities = ne_chunk(pos_tags)
        
        entities = {}
        for chunk in named_entities:
            if hasattr(chunk, 'label'):
                entity_type = chunk.label()
                entity_text = ' '.join(c[0] for c in chunk)
                
                if entity_type not in entities:
                    entities[entity_type] = []
                entities[entity_type].append(entity_text)
                
        return entities
    except Exception as e:
        logger.error(f"Error extracting entities: {str(e)}")
        return {}

def get_text_statistics(text: str) -> Dict[str, Any]:
    """
    Get comprehensive text statistics
    
    Args:
        text: Input text
        
    Returns:
        Dictionary of text statistics
    """
    if not NLTK_AVAILABLE:
        return {}
    
    try:
        # Basic stats
        sentences = sent_tokenize(text)
        tokens = word_tokenize(text)
        words = [word for word in tokens if word.isalnum()]
        
        # Lexical diversity (unique words / total words)
        lexical_diversity = len(set(words)) / len(words) if words else 0
        
        # Average word length
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        # Average sentence length
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        
        # TextBlob analysis
        blob = TextBlob(text)
        
        return {
            "sentence_count": len(sentences),
            "word_count": len(words),
            "unique_word_count": len(set(words)),
            "lexical_diversity": lexical_diversity,
            "avg_word_length": avg_word_length,
            "avg_sentence_length": avg_sentence_length,
            "polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity
        }
    except Exception as e:
        logger.error(f"Error getting text statistics: {str(e)}")
        return {}

def extract_keywords_nltk(text: str, top_n: int = 5) -> List[str]:
    """
    Extract important keywords from text
    
    Args:
        text: Input text
        top_n: Number of keywords to return
        
    Returns:
        List of keywords
    """
    if not NLTK_AVAILABLE:
        return []
    
    try:
        # Preprocess
        tokens = preprocess_text(text, remove_stopwords=True)
        
        # Get POS tags
        tagged = pos_tag(tokens)
        
        # Filter for nouns and adjectives
        keywords = [word for word, tag in tagged 
                    if tag.startswith('NN') or tag.startswith('JJ')]
        
        # Count frequencies
        freq = {}
        for word in keywords:
            freq[word] = freq.get(word, 0) + 1
            
        # Sort by frequency
        sorted_keywords = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        
        # Return top N
        return [word for word, count in sorted_keywords[:top_n]]
    except Exception as e:
        logger.error(f"Error extracting keywords: {str(e)}")
        return []

def analyze_text_nltk(text: str) -> Dict[str, Any]:
    """
    Comprehensive text analysis using NLTK and TextBlob
    
    Args:
        text: Input text
        
    Returns:
        Dictionary with comprehensive analysis
    """
    if not NLTK_AVAILABLE:
        return {"error": "NLTK not available"}
    
    try:
        initialize_nltk_components()
        
        # Get basic stats
        stats = get_text_statistics(text)
        
        # Get sentiment
        sentiment = get_nltk_sentiment(text)
        
        # Get entities
        entities = extract_entities_nltk(text)
        
        # Get keywords
        keywords = extract_keywords_nltk(text)
        
        # Lemmatize
        lemmatized = lemmatize_text(text)
        
        # Combine all analyses
        return {
            "statistics": stats,
            "sentiment": sentiment,
            "entities": entities,
            "keywords": keywords,
            "lemmatized_tokens": lemmatized,
            "language": detect_language_nltk(text)
        }
    except Exception as e:
        logger.error(f"Error analyzing text with NLTK: {str(e)}")
        return {"error": str(e)}

def detect_language_nltk(text: str) -> str:
    """
    Detect language of text using TextBlob
    
    Args:
        text: Input text
        
    Returns:
        ISO code for detected language
    """
    if not NLTK_AVAILABLE:
        return "en"
    
    try:
        blob = TextBlob(text)
        return blob.detect_language()
    except Exception as e:
        logger.error(f"Error detecting language: {str(e)}")
        return "en"

def determine_intent(text: str) -> Dict[str, Any]:
    """
    Determine user intent from text
    
    Args:
        text: Input text
        
    Returns:
        Dictionary with intent information
    """
    if not NLTK_AVAILABLE:
        return {"intent": "unknown"}
    
    try:
        # Normalize and clean the text first
        text = text.lower().strip()
        
        # Debug log to track greeting detection
        logger.info(f"NLTK intent detection for: '{text}'")
        
        # CRITICAL FIX: Immediately detect single-word greetings before any other processing
        if text in {"hi", "hello", "hey", "greetings", "howdy", "hiya", "sup", "yo"}:
            logger.info(f"Single-word greeting detected: '{text}'")
            return {"intent": "greeting", "confidence": 0.95}
            
        # Tokenize only after checking for exact matches
        try:
            tokens = preprocess_text(text.lower())
            logger.info(f"Tokens after preprocessing: {tokens}")
        except Exception as e:
            logger.error(f"Error preprocessing text: {e}")
            tokens = text.lower().split()
            
        # Expanded greeting patterns
        greetings = {
            "hello", "hi", "hey", "greetings", "sup", "what's up", "howdy", 
            "good morning", "good afternoon", "good evening", "morning", "evening",
            "afternoon", "yo", "hiya", "heya", "hi there", "hello there"
        }
        
        # Check for full phrase matches and common greeting starts
        if any(text.startswith(greeting) for greeting in greetings):
            logger.info(f"Phrase-level greeting detected: '{text}'")
            return {"intent": "greeting", "confidence": 0.9}
        
        # Word-level check for greetings
        if any(token in greetings for token in tokens):
            logger.info(f"Token-level greeting detected in: '{text}'")
            return {"intent": "greeting", "confidence": 0.9}
        
        # Special case for "how are you" type greetings
        if "how" in tokens and any(word in tokens for word in ["you", "ya", "u"]):
            logger.info(f"'How are you' type greeting detected: '{text}'")
            return {"intent": "greeting", "confidence": 0.9}
        
        # Check for farewell
        farewells = {"bye", "goodbye", "see you", "later", "farewell", "good night", "take care"}
        if any(token in farewells for token in tokens):
            return {"intent": "farewell", "confidence": 0.9}
        
        # Check for thanks
        thanks = {"thanks", "thank", "appreciate", "grateful", "thx"}
        if any(token in thanks for token in tokens):
            return {"intent": "thanks", "confidence": 0.9}
        
        # Check for help request
        help_words = {"help", "assist", "support", "guide", "how"}
        if any(token in help_words for token in tokens):
            return {"intent": "help_request", "confidence": 0.8}
        
        # Check if question
        question_words = {"what", "when", "where", "which", "who", "whom", "whose", "why", "how"}
        if any(token in question_words for token in tokens) or text.rstrip().endswith("?"):
            return {"intent": "question", "confidence": 0.8}
        
        # Default to statement
        return {"intent": "statement", "confidence": 0.6}
    except Exception as e:
        logger.error(f"Error determining intent: {str(e)}")
        return {"intent": "unknown", "error": str(e)}

def process_text(text):
    if SPACY_AVAILABLE:
        # spaCy implementation
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        # process with spaCy
    else:
        # NLTK fallback implementation
        from nltk import word_tokenize, pos_tag, ne_chunk
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        # process with NLTK
