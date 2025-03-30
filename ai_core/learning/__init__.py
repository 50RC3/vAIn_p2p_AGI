"""Learning components for AI core."""

try:
    from .unsupervised import UnsupervisedLearningModule
    from .self_supervised import SelfSupervisedLearning
    from .advanced_learning import AdvancedLearning, LearningConfig, LearningMetrics, LearningError
    
    __all__ = [
        'UnsupervisedLearningModule',
        'SelfSupervisedLearning',
        'AdvancedLearning',
        'LearningConfig',
        'LearningMetrics',
        'LearningError'
    ]
except ImportError as e:
    import logging
    logging.warning(f"Could not import all learning modules: {e}")
    __all__ = []
