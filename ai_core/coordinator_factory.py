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
        RuntimeError: If coordinator creation fails
    """
    # Input validation with detailed error messages
    if not isinstance(storage, ModelStorage):
        raise ValueError(f"storage must be a ModelStorage instance, got {type(storage).__name__}")
    if not isinstance(evaluator, ModelEvaluator):
        raise ValueError(f"evaluator must be a ModelEvaluator instance, got {type(evaluator).__name__}")
    if not 0 <= validation_threshold <= 1:
        raise ValueError(f"validation_threshold must be between 0 and 1, got {validation_threshold}")
    if validation_timeout < 1:
        raise ValueError(f"validation_timeout must be at least 1 second, got {validation_timeout}")

    try:
        # Use the default consensus strategy if none provided
        actual_consensus_strategy = consensus_strategy or default_consensus_strategy
        
        # Initialize coordinator with monitoring
        coordinator = ValidationCoordinator(
            storage=storage,
            evaluator=evaluator,
            threshold=validation_threshold,
            timeout=validation_timeout,
            consensus_strategy=actual_consensus_strategy
        )
        
        # Validate the coordinator was created correctly
        if not hasattr(coordinator, 'evaluator') or coordinator.evaluator is None:
            raise ValueError("Coordinator was created without a valid evaluator")
            
        logger.info(
            "Created ValidationCoordinator with threshold=%s, timeout=%ss",
            validation_threshold, validation_timeout
        )
        
        return coordinator
        
    except Exception as e:
        logger.error("Failed to create ValidationCoordinator: %s", str(e), exc_info=True)
        raise RuntimeError(f"Coordinator creation failed: {str(e)}") from e
