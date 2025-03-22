import logging
from typing import Optional, Callable, List, Dict
from .validation_coordinator import ValidationCoordinator 
from .model_storage import ModelStorage
from .model_evaluation import ModelEvaluator

logger = logging.getLogger(__name__)

def default_consensus_strategy(validations: List[Dict], threshold: float) -> bool:
    """Default strategy requiring threshold % of positive validations"""
    if not validations:
        return False
    positive_votes = sum(1 for v in validations if v.get('approved', False))
    return (positive_votes / len(validations)) >= threshold

def create_validation_coordinator(
    storage: ModelStorage,
    evaluator: ModelEvaluator,
    validation_threshold: float = 0.85,
    validation_timeout: int = 60,
    consensus_strategy: Optional[Callable[[List[Dict], float], bool]] = None
) -> ValidationCoordinator:
    """
    Factory function to create properly configured ValidationCoordinator instances.
    
    Args:
        storage: Model storage implementation
        evaluator: Model evaluator implementation 
        validation_threshold: Threshold for validation consensus (0-1)
        validation_timeout: Timeout in seconds for validation operations
        consensus_strategy: Optional custom consensus strategy
        
    Returns:
        Configured ValidationCoordinator instance
        
    Raises:
        ValueError: If input parameters are invalid
    """
    # Input validation
    if not isinstance(storage, ModelStorage):
        raise ValueError("storage must be a ModelStorage instance")
    if not isinstance(evaluator, ModelEvaluator):
        raise ValueError("evaluator must be a ModelEvaluator instance")
    if not 0 <= validation_threshold <= 1:
        raise ValueError("validation_threshold must be between 0 and 1")
    if validation_timeout < 1:
        raise ValueError("validation_timeout must be at least 1 second")

    try:
        # Initialize coordinator with monitoring
        coordinator = ValidationCoordinator(
            storage=storage,
            evaluator=evaluator,
            validation_threshold=validation_threshold,
            validation_timeout=validation_timeout,
            consensus_strategy=consensus_strategy or default_consensus_strategy
        )
        
        logger.info(
            f"Created ValidationCoordinator with threshold={validation_threshold}, "
            f"timeout={validation_timeout}s"
        )
        
        return coordinator
        
    except Exception as e:
        logger.error(f"Failed to create ValidationCoordinator: {str(e)}")
        raise RuntimeError(f"Coordinator creation failed: {str(e)}")
