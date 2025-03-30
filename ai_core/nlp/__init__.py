from .utils import (
    load_spacy_model,
    analyze_text,
    get_sentiment,
    get_main_topic,
    get_question_type,
    extract_key_phrases
)

# Import NLTK utilities if available
try:
    from .nltk_utils import (
        download_nltk_resources,
        initialize_nltk_components,
        analyze_text_nltk,
        get_nltk_sentiment,
        extract_keywords_nltk,
        determine_intent,
        get_text_statistics,
        extract_entities_nltk,
        lemmatize_text,
        preprocess_text
    )
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

__all__ = [
    # spaCy utilities
    'load_spacy_model',
    'analyze_text',
    'get_sentiment',
    'get_main_topic',
    'get_question_type',
    'extract_key_phrases',
    
    # NLTK utilities (if available)
    'NLTK_AVAILABLE'
]

# Add NLTK functions to __all__ if available
if NLTK_AVAILABLE:
    __all__.extend([
        'download_nltk_resources',
        'initialize_nltk_components',
        'analyze_text_nltk',
        'get_nltk_sentiment',
        'extract_keywords_nltk',
        'determine_intent',
        'get_text_statistics',
        'extract_entities_nltk',
        'lemmatize_text',
        'preprocess_text'
    ])
