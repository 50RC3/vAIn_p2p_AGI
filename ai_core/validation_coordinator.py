from typing import Dict, List, Optional, Callable, Any
import asyncio
import logging
import time
import torch
from .model_storage import ModelStorage
from .model_evaluation import ModelEvaluator
from dataclasses import dataclass
from asyncio import Event

logger = logging.getLogger(__name__)

@dataclass
class ValidationProgress:
    total: int
    completed: int
    successful: int
    failed: int
    consensus_reached: bool
    progress_event: Event

class ValidationCoordinator:
    """
    Coordinates the validation of a machine learning model across multiple nodes,
    allowing for configurable consensus mechanisms and enhanced error handling.
    """
    def __init__(
        self,
        storage: ModelStorage,
        evaluator: ModelEvaluator,
        validation_threshold: float = 0.85,
        validation_timeout: int = 60,
        consensus_strategy: Optional[Callable[[List[Dict], float], bool]] = None,
    ):
        if not 0 < validation_threshold <= 1:
            raise ValueError("Validation threshold must be between 0 and 1")
        if validation_timeout <= 0:
            raise ValueError("Validation timeout must be a positive integer")

        self.storage = storage
        self.evaluator = evaluator
        self.validation_threshold = validation_threshold
        self.validation_timeout = validation_timeout
        self.metrics: Dict[str, Dict] = {}
        self._consensus_strategy = consensus_strategy or self._default_consensus
        self.active_validations: Dict[str, ValidationProgress] = {}
        self._progress_callbacks: List[Callable[[str, ValidationProgress], None]] = []
        self.error_manager = ErrorFeedbackManager(interactive=True)

    def register_progress_callback(self, callback: Callable[[str, ValidationProgress], None]):
        """Register a callback to receive validation progress updates"""
        self._progress_callbacks.append(callback)

    def _notify_progress(self, model_hash: str, progress: ValidationProgress):
        """Notify all registered callbacks of validation progress"""
        for callback in self._progress_callbacks:
            try:
                callback(model_hash, progress)
            except Exception as e:
                logger.error(f"Error in progress callback: {str(e)}")

    async def coordinate_validation(self, model: torch.nn.Module, 
                                  validators: List[str],
                                  interactive: bool = True) -> Dict:
        """Coordinate model validation with progress tracking"""
        if model is None or not validators:
            raise ValueError("Invalid model or empty validator list")

        try:
            model_hash = await self._store_model(model)
            progress = ValidationProgress(
                total=len(validators),
                completed=0,
                successful=0,
                failed=0,
                consensus_reached=False,
                progress_event=Event()
            )
            self.active_validations[model_hash] = progress

            tasks = []
            for validator in validators:
                task = asyncio.create_task(
                    self._validate_with_progress(model_hash, validator, progress)
                )
                tasks.append(task)

            validation_results = await asyncio.gather(*tasks, return_exceptions=True)
            valid_results = [r for r in validation_results if isinstance(r, dict)]
            
            consensus = self._consensus_strategy(valid_results, self.validation_threshold)
            progress.consensus_reached = consensus
            progress.progress_event.set()
            
            if interactive:
                self._notify_progress(model_hash, progress)

            self._update_metrics(model_hash, valid_results, consensus)
            del self.active_validations[model_hash]
            
            return {
                "model_hash": model_hash,
                "consensus": consensus,
                "results": valid_results,
                "metrics": self.metrics.get(model_hash, {}),
                "progress": {
                    "total": progress.total,
                    "completed": progress.completed,
                    "successful": progress.successful,
                    "failed": progress.failed
                }
            }
            
        except Exception as e:
            error_context = {
                'source': 'validation_coordinator',
                'model_hash': model_hash,
                'validators': validators,
                'metrics': self.metrics.get(model_hash, {})
            }
            
            error_response = await self.error_manager.process_error(e, error_context)
            
            if error_response['action'] == 'critical_intervention':
                if interactive:
                    self._notify_progress(model_hash, ValidationProgress(
                        total=len(validators),
                        completed=0,
                        successful=0,
                        failed=1,
                        consensus_reached=False,
                        progress_event=asyncio.Event()
                    ))
                    
            logger.error(f"Validation coordination failed: {str(e)}")
            if model_hash in self.active_validations:
                del self.active_validations[model_hash]
            raise

    async def _validate_with_timeout(self, model_hash: str, validator: str) -> Dict:
        start_time = time.time()
        try:
            result = await asyncio.wait_for(
                self._validate_on_node(model_hash, validator),
                timeout=self.validation_timeout,
            )
            result["latency"] = time.time() - start_time
            return result
        except asyncio.TimeoutError:
            return {"validator": validator, "error": "timeout"}
        except Exception as e:
            return {"validator": validator, "error": str(e)}

    async def _store_model(self, model: torch.nn.Module) -> str:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            return self.storage.store_model(model, {
                "timestamp": time.time(),
                "architecture": model.__class__.__name__,
                "device": str(device),
            })
        except Exception as e:
            logger.error(f"Error storing model: {str(e)}", exc_info=True)
            raise

    def _update_metrics(self, model_hash: str, results: List[Dict], consensus: bool):
        successful_validations = sum(1 for r in results if r.get("approved"))
        total_validators = len(results)
        average_latency = sum(r.get("latency", 0) for r in results) / total_validators if total_validators > 0 else 0

        self.metrics[model_hash] = {
            "total_validators": total_validators,
            "successful_validations": successful_validations,
            "consensus_reached": consensus,
            "timestamp": time.time(),
            "average_latency": average_latency,
        }

    async def _validate_with_progress(
        self, 
        model_hash: str, 
        validator: str,
        progress: ValidationProgress
    ) -> Dict:
        """Validate on a node and track progress"""
        try:
            result = await self._validate_with_timeout(model_hash, validator)
            
            progress.completed += 1
            if result.get("approved", False):
                progress.successful += 1
            else:
                progress.failed += 1
                
            self._notify_progress(model_hash, progress)
            return result
            
        except Exception as e:
            progress.completed += 1
            progress.failed += 1
            self._notify_progress(model_hash, progress)
            return {"validator": validator, "error": str(e)}

    async def _validate_on_node(self, model_hash: str, validator: str) -> Dict:
        """Validate model on a specific node."""
        model = self.storage.load_model(model_hash)
        return await self.evaluator.evaluate(model, validator)

    async def _get_stored_results(self, model_hash: str) -> List[Dict]:
        """Retrieve stored validation results for a model."""
        try:
            return self.metrics.get(model_hash, {}).get("results", [])
        except Exception as e:
            logger.error(f"Error retrieving stored results for {model_hash}: {str(e)}", exc_info=True)
            return []

    async def get_validation_status(self, model_hash: str) -> Dict:
        """Get current validation status for a model.
        
        Args:
            model_hash: Hash identifier of the model
            
        Returns:
            Dict containing validation progress and current results
        """
        validation_results = await self._get_stored_results(model_hash)
        return {
            "model_hash": model_hash,
            "total_validators": len(validation_results),
            "approved_count": sum(1 for r in validation_results if r.get("approved", False)),
            "consensus_reached": self._consensus_strategy(validation_results, self.validation_threshold),
            "results": validation_results
        }
        
    async def get_active_validations(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all active validations"""
        return {
            model_hash: {
                "total": progress.total,
                "completed": progress.completed,
                "successful": progress.successful,
                "failed": progress.failed,
                "consensus_reached": progress.consensus_reached
            }
            for model_hash, progress in self.active_validations.items()
        }

    def _default_consensus(self, results: List[Dict], threshold: float) -> bool:
        """Default consensus strategy based on a simple majority rule."""
        successful_validations = sum(1 for r in results if r.get("approved"))
        return (successful_validations / len(results)) >= threshold if results else False
