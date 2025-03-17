from typing import Dict, List
import torch
from .model_storage import ModelStorage
from core.model_evaluation import ModelEvaluator

class ValidationCoordinator:
    def __init__(self, storage: ModelStorage, evaluator: ModelEvaluator):
        self.storage = storage
        self.evaluator = evaluator
        self.validation_threshold = 0.85
        
    async def coordinate_validation(self, model: torch.nn.Module, 
                                  validators: List[str]) -> Dict:
        """Coordinate model validation across multiple nodes."""
        # Store model for validation
        model_hash = self.storage.store_model(model, {
            "timestamp": str(torch.cuda.current_device()),
            "architecture": model.__class__.__name__
        })
        
        validation_results = []
        for validator in validators:
            result = await self._validate_on_node(model_hash, validator)
            validation_results.append(result)
            
        consensus = self._reach_consensus(validation_results)
        return {
            "model_hash": model_hash,
            "consensus": consensus,
            "results": validation_results
        }
        
    def _reach_consensus(self, results: List[Dict]) -> bool:
        """Determine if model passes validation consensus."""
        if not results:
            return False
            
        approved = sum(1 for r in results if r.get("approved", False))
        return approved / len(results) >= self.validation_threshold
