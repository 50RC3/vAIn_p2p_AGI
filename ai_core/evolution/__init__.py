"""Evolution module for AI cognitive capabilities growth."""

try:
    from .cognitive_evolution import CognitiveEvolution, EvolutionConfig, EvolutionMetrics
    
    __all__ = [
        'CognitiveEvolution',
        'EvolutionConfig',
        'EvolutionMetrics'
    ]
except ImportError as e:
    import logging
    logging.warning(f"Could not import all evolution modules: {e}")
    __all__ = []
