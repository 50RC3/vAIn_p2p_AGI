from dataclasses import dataclass
import logging
import time
from typing import Optional

from core.types import ModelState
from core.interactive_utils import InteractiveSession
from ai_core.model_storage import ModelStorage
from training.model_interface import ModelInterface
from utils.resource_metrics import ResourceMetrics

# Update ModelStorage class if it doesn't have the required method
if not hasattr(ModelStorage, 'store_model_state'):
    async def store_model_state(self, model_id: str, state: ModelState) -> None:
        """Store model state in storage"""
        # Implementation depends on how ModelStorage actually works
        # This is a temporary placeholder
        await self.save_model(model_id, state)
    
    ModelStorage.store_model_state = store_model_state

# Define a placeholder for FederatedClient if the module doesn't exist
class FederatedClient:
    """Placeholder for the FederatedClient interface"""
    async def cleanup(self) -> None:
        """Placeholder for cleanup method"""
        return None
        
    async def notify_failure(self, model_id: str) -> None:
        """Placeholder for notify_failure method"""

logger = logging.getLogger(__name__)

class CoordinatorError(Exception):
    """Base exception for coordinator errors"""

class ResourceExhaustedError(CoordinatorError):
    """Raised when system resources are exhausted"""

class TrainingError(CoordinatorError):
    """Raised when training fails"""
@dataclass
class ModelCoordinator:
    model_storage: ModelStorage
    model_interface: ModelInterface
    federated_client: Optional[FederatedClient] = None
    resource_metrics: Optional[ResourceMetrics] = None
    
    async def initialize(self) -> None:
        """Initialize coordinator and verify connections"""
        try:
            # Verify connection to storage
            # Check if storage is accessible by performing a simple operation
            test_state = {
                "value": 0.0, 
                "model_id": "test_connection", 
                "status": "testing"
            }
            await self.model_storage.store_model_state("test_connection", test_state)
            
            # Initialize resource metrics if not already provided
            if not self.resource_metrics:
                self.resource_metrics = ResourceMetrics.get_current()
                
            logger.info("ModelCoordinator initialized successfully")
            return True
        except ConnectionError as e:
            logger.error("Failed to initialize coordinator: %s", str(e))
            raise
        except Exception as e:
            logger.error("Unexpected error during coordinator initialization: %s", str(e))
            raise
            
    async def coordinate_training(self, model_id: str) -> Optional[ModelState]:
        """Coordinate model training with resource monitoring"""
        try:
            # Check if model exists first
            existing_state = await self.model_storage.load_model_state(model_id)
            if existing_state and existing_state.get('status') == 'training':
                raise CoordinatorError(f"Model {model_id} is already being trained")
            # Update state to indicate training has started
            await self.model_storage.store_model_state(
                model_id,
                {"value": 0.0, "model_id": model_id, "status": "training"}
            )
            
            # Monitor resources
            metrics = ResourceMetrics.get_current()
            if metrics.memory_usage > 90 or metrics.cpu_usage > 95:
                await self._handle_failure(model_id, "System resources too constrained for training")
                raise ResourceExhaustedError("System resources too constrained for training")

            # Start training session
            logger.info("Starting training for model %s", model_id)
            async with InteractiveSession(session_id=model_id) as session:
                # Get training data
                training_data = session.get_data()
                if not training_data:
                    await self._handle_failure(model_id, "No training data available")
                    raise TrainingError("No training data available")
                
                # Perform actual training
                training_metrics = await self.model_interface.train(
                    training_data=training_data,
                    model_id=model_id  # Updated parameter name to match interface
                )
                
                if not training_metrics:
                    await self._handle_failure(model_id, "Training produced no metrics")
                    raise TrainingError("Training produced no metrics")

                # Cleanup federated resources if available
                if self.federated_client:
                    await self.federated_client.cleanup()

                # Update metrics
                accuracy = training_metrics.get('accuracy', 0.0)
                timestamp = training_metrics.get('timestamp', time.time())
                
                final_state = {
                    "value": accuracy,
                    "model_id": model_id,
                    "status": "completed",
                    "timestamp": timestamp
                }
                # Store the final state
                await self.model_storage.store_model_state(model_id, final_state)
                
                logger.info("Training completed for model %s with accuracy %s", model_id, accuracy)
                return final_state
                    
        except (CoordinatorError, ResourceExhaustedError, TrainingError) as e:
            # These exceptions have already been handled by _handle_failure
            logger.error("Training coordination error: %s", str(e))
            return {
                "value": 0.0,
                "model_id": model_id,
                "status": "failed",
                "error": str(e)
            }
            
        except Exception as e:
            error_message = str(e)
            logger.error("Unexpected error during training coordination: %s", error_message)
            await self._handle_failure(model_id, error_message)
            raise CoordinatorError(f"Training failed: {error_message}") from e
            
    async def _handle_failure(self, model_id: str, error: str) -> None:
        """Handle training failures gracefully"""
        try:
            # Log the failure first
            logger.warning("Training failure for model %s: %s", model_id, error)
            
            # Store failure state in the model storage
            await self.model_storage.store_model_state(
                model_id,
                {
                    "value": 0.0,
                    "model_id": model_id,
                    "status": "failed",
                    "error": error
                }
            )
            if self.federated_client:
                await self.federated_client.notify_failure(model_id)
                
        except (ConnectionError, ValueError, RuntimeError, IOError) as e:
            logger.error("Error handling failure for model %s: %s", model_id, e)
        
    async def get_resource_status(self) -> dict:
            """Get current resource utilization status"""
            metrics = ResourceMetrics.get_current()
            return {
                'cpu_usage': metrics.cpu_usage,
                'memory_usage': metrics.memory_usage,
                'gpu_usage': metrics.gpu_usage,
                'disk_usage': metrics.disk_usage,
                'healthy': self._is_healthy(metrics)
            }

    def _is_healthy(self, metrics: ResourceMetrics) -> bool:
        """Check if system resources are healthy"""
        return (
            metrics.cpu_usage < 90 and
            metrics.memory_usage < 85 and
            metrics.disk_usage < 95 and
            (metrics.gpu_usage is None or metrics.gpu_usage < 0.9)
        )

    async def get_model_status(self, model_id: str) -> dict:
        """Get the current status of a model
        
        Args:
            model_id: ID of the model
        
        Returns:
            Dictionary with model status information
        """
        try:
            # Get the model state
            state = await self.model_storage.load_model_state(model_id)
            if not state:
                return {
                    "model_id": model_id,
                    "exists": False,
                    "status": "not_found"
                }
                
            # Get resource metrics for this model if available
            metrics = {}
            if self.resource_metrics:
                metrics = {
                    "cpu_usage": self.resource_metrics.cpu_usage,
                    "memory_usage": self.resource_metrics.memory_usage,
                    "healthy": self._is_healthy(self.resource_metrics)
                }
                
            # Combine state and metrics
            return {
                "model_id": model_id,
                "exists": True,
                "status": state.get("status", "unknown"),
                "value": state.get("value", 0.0),
                "timestamp": state.get("timestamp", 0),
                "resources": metrics
            }
            
        except (ConnectionError, PermissionError, OSError, ValueError) as e:
            logger.error("Error getting status for model %s: %s", model_id, str(e))
            return {
                "model_id": model_id,
                "exists": False,
                "status": "error",
                "error": str(e)
            }
