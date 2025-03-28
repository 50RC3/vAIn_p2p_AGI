# ai/federated_learning.py
"""Module implementing federated learning algorithms."""
import logging
from typing import Dict, List, Tuple, Any
import torch

from ai.compression import compress_gradients, decompress_gradients
from ai.exceptions import (
    TrainingError, AggregationError, CompressionError
)
from ai.metrics import ModelMetrics
from core.cross_domain_transfer import CrossDomainTransfer, DomainType
from core.symbolic_reasoning import PropositionalLogic

class FederatedLearner:
    """Handles federated learning operations for a model."""
    
    def __init__(self, model, optimizer, loss_fn, device, 
                 error_feedback=True, compression_rate=0.1,
                 secure_aggregation=False, differential_privacy=False):
        """
        Initialize a federated learner.
        
        Args:
            model: The model to be trained
            optimizer: The optimizer to use
            loss_fn: The loss function
            device: Device to run computations on
            error_feedback: Whether to use error feedback
            compression_rate: Compression rate for gradients
            secure_aggregation: Whether to use secure aggregation
            differential_privacy: Whether to use differential privacy
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.error_feedback = error_feedback
        self.error_residuals = {}
        self.compression_rate = compression_rate
        self.secure_aggregation = secure_aggregation
        self.differential_privacy = differential_privacy
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initialized FederatedLearner with error_feedback=%s",
                        error_feedback)
        
        self.metrics = ModelMetrics()
        self.domain_transfer = CrossDomainTransfer()
        self.symbolic_logic = PropositionalLogic()
        
        # Setup domain logic rules
        self.symbolic_logic.add_variable("valid_update", True)
        self.symbolic_logic.add_variable("domain_compatible", True)
        self.symbolic_logic.add_variable("compression_valid", True)
        
    def get_model_parameters(self):
        """
        Get model parameters.
        
        Returns:
            Dict: Model parameters
        """
        return {name: param.data.clone() for name, param in self.model.named_parameters()}
    
    def set_model_parameters(self, parameters):
        """
        Set model parameters.
        
        Args:
            parameters: Dict of parameter tensors
        """
        for name, param in self.model.named_parameters():
            if name in parameters:
                param.data.copy_(parameters[name])
    
    def compute_gradients(self, data_loader):
        """
        Compute gradients based on the data.
        
        Args:
            data_loader: Data loader containing training data
            
        Returns:
            Tuple[Dict, float]: Dictionary of gradients and average loss
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        total_loss = 0
        num_samples = 0
        
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            
            total_loss += loss.item() * len(inputs)
            num_samples += len(inputs)
        
        gradients = {name: param.grad.clone() for name, param in self.model.named_parameters() 
                    if param.grad is not None}
        
        avg_loss = total_loss / num_samples if num_samples > 0 else 0
        
        return gradients, avg_loss
    
    def apply_gradients(self, gradients):
        """
        Apply gradients to the model.
        
        Args:
            gradients: Dictionary of gradients
        """
        for name, param in self.model.named_parameters():
            if name in gradients and gradients[name] is not None:
                if param.grad is None:
                    param.grad = gradients[name].clone()
                else:
                    param.grad.copy_(gradients[name])
        
        self.optimizer.step()
    
    def apply_error_feedback(self, gradients):
        """
        Apply error feedback to gradients.
        
        Args:
            gradients: Dictionary of gradients
            
        Returns:
            Dict: Updated gradients with error feedback
        """
        if not self.error_feedback:
            return gradients
        
        updated_gradients = {}
        
        # Initialize error residuals if not already done
        if not self.error_residuals:
            self.error_residuals = {name: torch.zeros_like(grad) 
                                  for name, grad in gradients.items()}
        
        # Apply error feedback
        for name, grad in gradients.items():
            if name in self.error_residuals:
                updated_grad = grad + self.error_residuals[name]
                updated_gradients[name] = updated_grad
            else:
                updated_gradients[name] = grad
                self.error_residuals[name] = torch.zeros_like(grad)
        
        return updated_gradients
    
    def compress_gradients(self, gradients):
        """
        Compress gradients.
        
        Args:
            gradients: Dictionary of gradients
            
        Returns:
            Dict: Compressed gradients
        """
        try:
            compressed = compress_gradients(gradients, self.compression_rate)
            
            # Update error residuals
            if self.error_feedback:
                for name, grad in gradients.items():
                    if name in compressed:
                        self.error_residuals[name] = grad - compressed[name]
            
            return compressed
        except Exception as e:
            self.logger.error("Gradient compression failed: %s", str(e))
            raise CompressionError(f"Gradient compression failed: {str(e)}") from e
    
    def aggregate_updates(self, updates):
        """
        Aggregate updates from multiple clients.
        
        Args:
            updates: List of updates (each update is a tuple of gradients and weight)
            
        Returns:
            Dict: Aggregated update
        """
        try:
            if not updates:
                return None
            
            # Extract gradients and weights
            all_gradients = [grad for grad, _ in updates]
            weights = [weight for _, weight in updates]
            total_weight = sum(weights)
            
            # Normalize weights
            normalized_weights = ([w / total_weight for w in weights] 
                                if total_weight > 0 
                                else [1.0 / len(weights)] * len(weights))
            
            # Perform weighted average of gradients
            aggregated = {}
            for name in all_gradients[0].keys():
                tensors = [grads[name] for grads in all_gradients 
                          if name in grads and grads[name] is not None]
                if tensors:
                    weighted_sum = torch.zeros_like(tensors[0])
                    for i, tensor in enumerate(tensors):
                        weighted_sum += tensor * normalized_weights[i]
                    aggregated[name] = weighted_sum
            
            return aggregated
        except Exception as e:
            self.logger.error("Aggregation failed: %s", str(e))
            raise AggregationError(f"Aggregation failed: {str(e)}") from e
    
    def hierarchical_aggregate(self, updates_by_group):
        """
        Perform hierarchical aggregation of updates.
        
        Args:
            updates_by_group: Dictionary mapping group ID to list of updates
            
        Returns:
            Dict: Aggregated update
        """
        group_aggregates = {}
        
        # First level: aggregate within groups
        for group_id, group_updates in updates_by_group.items():
            try:
                group_agg = self.aggregate_updates(group_updates)
                if group_agg is not None:
                    group_aggregates[group_id] = (group_agg, sum(w for _, w in group_updates))
            except Exception as e:
                self.logger.warning("Aggregation failed for group %s: %s", 
                                   group_id, str(e))
        
        # Second level: aggregate across groups
        if not group_aggregates:
            return None
        
        return self.aggregate_updates(list(group_aggregates.values()))
    
    def train(self, data_loader, epochs=1, local_updates=True):
        """
        Train the model.
        
        Args:
            data_loader: Data loader for training data
            epochs: Number of epochs to train
            local_updates: Whether to apply updates locally
            
        Returns:
            Tuple[Dict, Dict]: Gradients and training metrics
        """
        try:
            metrics = {"loss": 0.0}
            
            for _ in range(epochs):
                gradients, loss = self.compute_gradients(data_loader)
                metrics["loss"] += loss / epochs
            
            # Apply error feedback if enabled
            gradients = self.apply_error_feedback(gradients)
            
            # Apply updates locally if requested
            if local_updates:
                self.apply_gradients(gradients)
            
            return gradients, metrics
        except Exception as e:
            self.logger.error("Training failed: %s", str(e))
            raise TrainingError(f"Training failed: {str(e)}") from e
    
    def validate(self, data_loader, update=None):
        """
        Validate the model.
        
        Args:
            data_loader: Data loader for validation data
            update: Optional update to apply temporarily for validation
            
        Returns:
            Dict: Validation metrics
        """
        # Save original parameters if we're applying temporary updates
        original_params = None
        if update is not None:
            original_params = self.get_model_parameters()
            self.set_model_parameters(update)
        
        # Set model to evaluation mode
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                
                total_loss += loss.item() * len(inputs)
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)
        
        # Restore original parameters if needed
        if original_params is not None:
            self.set_model_parameters(original_params)
        
        metrics = {
            "val_loss": total_loss / total if total > 0 else float('inf'),
            "val_accuracy": correct / total if total > 0 else 0.0
        }
        
        return metrics
    
    def evaluate(self, data_loader, update=None):
        """
        Evaluate the model.
        
        Args:
            data_loader: Data loader for test data
            update: Optional update to apply temporarily for evaluation
            
        Returns:
            Dict: Evaluation metrics
        """
        return self.validate(data_loader, update)
    
    async def aggregate_models(self, model_updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Hierarchical FedAvg aggregation with cross-domain support"""
        try:
            if len(model_updates) < 2:
                raise AggregationError(f"Not enough clients ({len(model_updates)}/2)")
            
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()
            
            # Memory management
            self._check_memory_usage(model_updates)
            
            # Validate updates using symbolic reasoning
            decompressed_updates = []
            for i, update in enumerate(model_updates):
                self.symbolic_logic.set_variable("valid_update", 
                    self._validate_update(update))
                self.symbolic_logic.set_variable("domain_compatible",
                    self._check_domain_compatibility(update))
                
                if not self.symbolic_logic.evaluate_expression(
                    "valid_update and domain_compatible"):
                    self.logger.warning("Update failed symbolic validation")
                    continue
                    
                # Apply cross-domain adaptation if needed
                if "domain_type" in update:
                    update = self.domain_transfer.transfer_knowledge(
                        update["domain_type"],
                        None,
                        update
                    )
                
                try:
                    update_with_feedback = self.apply_error_feedback(update)
                    decompressed = decompress_gradients(update_with_feedback)
                    decompressed_updates.append(decompressed)
                except Exception as e:
                    self.logger.warning(f"Failed to process update {i}: {str(e)}")
                    continue

            # Aggregate hierarchically
            aggregated_model = {}
            for key in decompressed_updates[0].keys():
                aggregated_model[key] = self.hierarchical_aggregate(
                    [update[key] for update in decompressed_updates]
                )
            
            end_time.record()
            torch.cuda.synchronize()
            agg_time = start_time.elapsed_time(end_time)
            
            # Update metrics
            self.metrics.update_aggregation_stats(
                num_updates=len(model_updates),
                time_taken=agg_time,
                memory_used=torch.cuda.memory_allocated()
            )
            
            return aggregated_model
            
        except Exception as e:
            self.logger.error(f"Aggregation failed: {str(e)}")
            raise AggregationError(f"Aggregation failed: {str(e)}")

    def _check_memory_usage(self, updates: List[Dict[str, torch.Tensor]]) -> None:
        """Monitor and manage memory usage"""
        estimated_memory = sum(param.element_size() * param.nelement() 
                             for update in updates 
                             for param in update.values())
        if estimated_memory > 16 * 1e9:
            torch.cuda.empty_cache()
            if torch.cuda.memory_allocated() + estimated_memory > 16 * 1e9:
                raise MemoryError("Insufficient memory for aggregation")

    async def train_round(self, local_data, epochs: int = 1) -> Dict[str, torch.Tensor]:
        """Local model training with monitoring"""
        try:
            self.model.train()
            optimizer = torch.optim.Adam(self.model.parameters())
            
            total_batches = len(local_data)
            total_loss = 0
            
            for epoch in range(epochs):
                epoch_loss = 0
                for batch_idx, batch in enumerate(local_data):
                    try:
                        optimizer.zero_grad()
                        loss = self.model.training_step(batch)
                        loss.backward()
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                        
                        if batch_idx % 10 == 0:
                            self.logger.debug(f"Epoch {epoch+1}/{epochs}, "
                                       f"Batch {batch_idx+1}/{total_batches}, "
                                       f"Loss: {loss.item():.4f}")
                            
                    except RuntimeError as e:
                        self.logger.warning(f"Error in batch {batch_idx}: {str(e)}")
                        continue
                        
                avg_epoch_loss = epoch_loss / total_batches
                total_loss += avg_epoch_loss
                self.logger.info(f"Epoch {epoch+1} completed, Average loss: {avg_epoch_loss:.4f}")
                
            self.metrics.update_training_stats(
                avg_loss=total_loss / epochs,
                num_epochs=epochs,
                num_batches=total_batches
            )
            
            return {k: v.cpu() for k, v in self.model.state_dict().items()}
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise TrainingError(f"Training failed: {str(e)}")
            
    def _validate_update(self, update: Dict) -> bool:
        """Validate model update using symbolic rules"""
        # Add validation logic here
        return True
        
    def _check_domain_compatibility(self, update: Dict) -> bool:
        """Check domain compatibility for transfer"""
        # Add compatibility check here
        return True
