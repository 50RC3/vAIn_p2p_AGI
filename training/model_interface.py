"""Model interface definitions for federated learning"""

from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import logging
import asyncio
import os
import psutil
import time

logger = logging.getLogger(__name__)

class ModelInterface:
    """Interface for model operations in federated learning"""
    
    def __init__(self, model: nn.Module, model_name: str):
        """Initialize model interface
        
        Args:
            model: The PyTorch model
            model_name: Name of the model
        """
        self.model = model
        self.model_name = model_name
        self.version = 1
        self.metadata = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training = False
        self.initialized = False
        self.event_handlers = {}
        self.training_stats = {}
        self.p2p_network = None  # Will be set by the system if P2P is enabled
        self.interactive_session = None  # Will be set when running in interactive mode
        self.resource_monitor = None  # Will be set by the system for resource monitoring

    def get_parameters(self) -> Dict[str, Any]:
        """Get model parameters
        
        Returns:
            Dict containing model parameters
        """
        return {k: v.cpu().detach().numpy() for k, v in self.model.state_dict().items()}
    
    def update_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Update model parameters
        
        Args:
            parameters: New parameters to apply
            
        Returns:
            bool: Success status
        """
        try:
            state_dict = {k: torch.tensor(v) for k, v in parameters.items()}
            self.model.load_state_dict(state_dict)
            self.version += 1
            return True
        except Exception as e:
            logger.error("Failed to update parameters: %s", e)
            return False
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get model metadata
        
        Returns:
            Dict containing metadata
        """
        return {
            "name": self.model_name,
            "version": self.version,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "extra": self.metadata
        }
    
    async def initialize(self):
        """Initialize the model asynchronously"""
        if self.initialized:
            return
            
        try:
            logger.info(f"Initializing model")
            # Notify any listeners that initialization is starting
            await self._trigger_event('initialization_started', {
                'model_name': self.model_name,
                'device': str(self.device)
            })
            
            # Model initialization logic here
            self.model = self.model.to(self.device)
                
            self.initialized = True
            logger.info("Model initialized successfully")
            
            # Notify any listeners that initialization is complete
            await self._trigger_event('initialization_completed', {
                'success': True,
                'model_type': type(self.model).__name__
            })
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            self.initialized = False
            # Notify any listeners about initialization failure
            await self._trigger_event('initialization_failed', {
                'error': str(e)
            })
            
    async def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run prediction with the model"""
        if not self.initialized:
            await self.initialize()
            
        try:
            # Ensure inputs are on the correct device
            inputs = inputs.to(self.device)
            
            # Check resource constraints if monitoring enabled
            if self.resource_monitor:
                await self._check_resources()
            
            # Notify prediction start
            await self._trigger_event('prediction_started', {
                'input_shape': list(inputs.shape)
            })
            
            # Run prediction
            with torch.no_grad():
                output = self.model(inputs)
                
            # Notify prediction complete
            await self._trigger_event('prediction_completed', {
                'output_shape': list(output.shape)
            })
            
            return output
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            await self._trigger_event('prediction_failed', {
                'error': str(e)
            })
            raise RuntimeError(f"Prediction failed: {str(e)}")
            
    async def train(self, training_data, validation_data=None, epochs=10):
        """Train the model with the provided data"""
        if not self.initialized:
            await self.initialize()
            
        try:
            self.training = True
            logger.info(f"Starting training for {epochs} epochs")
            
            # Notify training started
            await self._trigger_event('training_started', {
                'epochs': epochs,
                'has_validation': validation_data is not None
            })
            
            # Simple training loop - replace with actual training logic
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            # Initialize or reset training stats
            self.training_stats = {
                'current_epoch': 0,
                'total_epochs': epochs,
                'start_time': time.time(),
                'losses': [],
                'validation_losses': []
            }
            
            for epoch in range(epochs):
                if not self.training:
                    logger.info("Training interrupted")
                    break
                
                # Update training stats
                self.training_stats['current_epoch'] = epoch + 1
                
                # Notify epoch start
                await self._trigger_event('epoch_started', {
                    'epoch': epoch + 1,
                    'total_epochs': epochs
                })
                
                # Check resources before each epoch if monitoring enabled
                if self.resource_monitor and not await self._check_resources():
                    logger.warning("Insufficient resources, pausing training")
                    # Ask for confirmation if in interactive mode
                    if self.interactive_session:
                        continue_training = await self._get_interactive_confirmation(
                            "Resource usage is high. Continue training?")
                        if not continue_training:
                            break
                
                # Training logic here
                inputs, targets = training_data
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                # Record loss
                self.training_stats['losses'].append(loss.item())
                
                # Run validation if provided
                validation_loss = None
                if validation_data:
                    val_inputs, val_targets = validation_data
                    val_inputs = val_inputs.to(self.device)
                    val_targets = val_targets.to(self.device)
                    with torch.no_grad():
                        val_outputs = self.model(val_inputs)
                        validation_loss = criterion(val_outputs, val_targets).item()
                        self.training_stats['validation_losses'].append(validation_loss)
                
                # Notify epoch progress
                await self._trigger_event('epoch_completed', {
                    'epoch': epoch + 1,
                    'total_epochs': epochs,
                    'loss': loss.item(),
                    'validation_loss': validation_loss
                })
                
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}" + 
                          (f", Validation Loss: {validation_loss:.4f}" if validation_loss else ""))
                
                # Check if we should stop training
                if await self._should_stop_training():
                    logger.info("Training stopped early")
                    break
                    
            # Calculate training duration
            training_duration = time.time() - self.training_stats['start_time']
            self.training_stats['duration'] = training_duration
            self.training = False
            logger.info(f"Training completed in {training_duration:.2f} seconds")
            
            # Notify training completion
            await self._trigger_event('training_completed', {
                'epochs_completed': epoch + 1,
                'duration': training_duration,
                'final_loss': self.training_stats['losses'][-1] if self.training_stats['losses'] else None
            })
                
            return {
                "status": "success", 
                "epochs_completed": epoch + 1,
                "duration": training_duration,
                "final_loss": self.training_stats['losses'][-1] if self.training_stats['losses'] else None
            }
        except Exception as e:
            self.training = False
            logger.error(f"Training error: {str(e)}")
            # Notify training failure
            await self._trigger_event('training_failed', {
                'error': str(e)
            })
            raise RuntimeError(f"Training failed: {str(e)}")
            
    async def _should_stop_training(self) -> bool:
        """Check if training should be stopped early"""
        # Check system resources
        if self.resource_monitor:
            # Get memory usage
            try:
                metrics = self.resource_monitor.get_metrics()
                if metrics.memory_usage > 95:
                    logger.warning(f"Memory usage too high: {metrics.memory_usage:.1f}%, stopping training")
                    return True
                
                # GPU memory check if available
                if metrics.gpu_usage is not None and metrics.gpu_usage > 90:
                    logger.warning(f"GPU memory usage too high: {metrics.gpu_usage:.1f}%, stopping training")
                    return True
            except Exception as e:
                logger.warning(f"Error checking resources: {str(e)}")
        
        # Use psutil as fallback if resource monitor not available
        if psutil.virtual_memory().percent > 95:
            logger.warning("Memory usage too high, stopping training")
            return True
            
        return False
    
    async def _check_resources(self) -> bool:
        """Check if system resources are sufficient for operation"""
        if self.resource_monitor:
            try:
                metrics = self.resource_monitor.get_metrics()
                # Return True if resources are available, False otherwise
                return metrics.memory_usage < 90 and (
                    metrics.gpu_usage is None or metrics.gpu_usage < 90)
            except Exception as e:
                logger.warning(f"Error checking resources: {str(e)}")
                
        # Fallback to psutil if no resource monitor
        return psutil.virtual_memory().percent < 90
    
    async def _get_interactive_confirmation(self, message: str) -> bool:
        """Get confirmation from user if in interactive mode"""
        if self.interactive_session:
            try:
                return await self.interactive_session.prompt_yes_no(message)
            except Exception as e:
                logger.warning(f"Error getting interactive confirmation: {str(e)}")
                return False
        return True  # Default to continuing if not interactive
        
    def stop_training(self):
        """Signal to stop the training process"""
        if self.training:
            logger.info("Requested to stop training")
            self.training = False
            
    def register_handler(self, event_type: str, handler: Callable[..., Any]) -> None:
        """Register a handler for an event"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = set()
        self.event_handlers[event_type].add(handler)
    
    def unregister_handler(self, event_type: str, handler: Callable[..., Any]) -> None:
        """Unregister a handler for an event"""
        if event_type in self.event_handlers and handler in self.event_handlers[event_type]:
            self.event_handlers[event_type].remove(handler)
    
    async def _trigger_event(self, event_type: str, data: Any) -> None:
        """Trigger an event for registered handlers"""
        if event_type not in self.event_handlers:
            return
            
        for handler in self.event_handlers[event_type]:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                logger.error(f"Error in event handler for '{event_type}': {e}")
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        if not self.training:
            return {"status": "idle"}
            
        return {
            "status": "training",
            "epochs_completed": self.training_stats.get('current_epoch', 0),
            "total_epochs": self.training_stats.get('total_epochs', 0),
            "duration": time.time() - self.training_stats.get('start_time', time.time()),
            "latest_loss": self.training_stats.get('losses', [])[-1] if self.training_stats.get('losses') else None
        }
    
    async def set_interactive_session(self, session):
        """Set the interactive session for user interactions"""
        self.interactive_session = session
        logger.info("Interactive session connected to model interface")
        
    async def set_resource_monitor(self, monitor):
        """Set the resource monitor for tracking system resources"""
        self.resource_monitor = monitor
        logger.info("Resource monitor connected to model interface")
        
    async def cleanup(self):
        """Clean up resources before shutdown"""
        logger.info("Cleaning up model interface resources")
        
        # Stop any training in progress
        if self.training:
            self.stop_training()
            # Give a short time for training to stop
            await asyncio.sleep(0.5)
        
        # Clear memory explicitly
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Clear handlers
        self.event_handlers.clear()
        
        # Clear references to other components
        self.interactive_session = None
        self.resource_monitor = None
        
        logger.info("Model interface cleanup completed")
