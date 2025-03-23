import torch
import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
from .compression import compress_gradients, decompress_gradients
from .exceptions import AggregationError, TrainingError
from .metrics import ModelMetrics
from core.cross_domain_transfer import CrossDomainTransfer, DomainType
from core.symbolic_reasoning import PropositionalLogic

logger = logging.getLogger(__name__)

class FederatedLearner:
    def __init__(self, model: torch.nn.Module, config: Dict):
        """Initialize federated learner with configuration"""
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.model.to(self.device)
        
        # Configuration
        self.min_clients = config.get('min_clients', 2)
        self.compression_rate = config.get('compression_rate', 0.01)
        self.max_memory_usage = config.get('max_memory_gb', 16) * 1e9
        self.timeout = config.get('aggregation_timeout', 300)
        
        self.error_feedback = {}
        self.metrics = ModelMetrics()
        self._validate_config(config)
        
        # Initialize cross-domain transfer
        self.domain_transfer = CrossDomainTransfer()
        self.symbolic_logic = PropositionalLogic()
        
        # Setup domain logic rules
        self.symbolic_logic.add_variable("valid_update", True)
        self.symbolic_logic.add_variable("domain_compatible", True)
        self.symbolic_logic.add_variable("compression_valid", True)

    def _validate_config(self, config: Dict) -> None:
        """Validate configuration parameters"""
        if self.min_clients < 2:
            raise ValueError("min_clients must be at least 2")
        if not 0 < self.compression_rate <= 1:
            raise ValueError("compression_rate must be between 0 and 1")
            
    async def aggregate_models(self, model_updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Hierarchical FedAvg aggregation with cross-domain support"""
        try:
            if len(model_updates) < self.min_clients:
                raise AggregationError(f"Not enough clients ({len(model_updates)}/{self.min_clients})")
            
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
                    logger.warning("Update failed symbolic validation")
                    continue
                    
                # Apply cross-domain adaptation if needed
                if "domain_type" in update:
                    update = self.domain_transfer.transfer_knowledge(
                        update["domain_type"],
                        self.config.target_domain,
                        update
                    )
                
                try:
                    update_with_feedback = self._apply_error_feedback(update)
                    decompressed = decompress_gradients(update_with_feedback)
                    decompressed_updates.append(decompressed)
                except Exception as e:
                    logger.warning(f"Failed to process update {i}: {str(e)}")
                    continue

            # Aggregate hierarchically
            aggregated_model = {}
            for key in decompressed_updates[0].keys():
                aggregated_model[key] = self._hierarchical_aggregate(
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
            logger.error(f"Aggregation failed: {str(e)}")
            raise AggregationError(f"Aggregation failed: {str(e)}")

    def _check_memory_usage(self, updates: List[Dict[str, torch.Tensor]]) -> None:
        """Monitor and manage memory usage"""
        estimated_memory = sum(param.element_size() * param.nelement() 
                             for update in updates 
                             for param in update.values())
        if estimated_memory > self.max_memory_usage:
            torch.cuda.empty_cache()
            if torch.cuda.memory_allocated() + estimated_memory > self.max_memory_usage:
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
                            logger.debug(f"Epoch {epoch+1}/{epochs}, "
                                       f"Batch {batch_idx+1}/{total_batches}, "
                                       f"Loss: {loss.item():.4f}")
                            
                    except RuntimeError as e:
                        logger.warning(f"Error in batch {batch_idx}: {str(e)}")
                        continue
                        
                avg_epoch_loss = epoch_loss / total_batches
                total_loss += avg_epoch_loss
                logger.info(f"Epoch {epoch+1} completed, Average loss: {avg_epoch_loss:.4f}")
                
            self.metrics.update_training_stats(
                avg_loss=total_loss / epochs,
                num_epochs=epochs,
                num_batches=total_batches
            )
            
            return {k: v.cpu() for k, v in self.model.state_dict().items()}
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise TrainingError(f"Training failed: {str(e)}")
            
    def _validate_update(self, update: Dict) -> bool:
        """Validate model update using symbolic rules"""
        # Add validation logic here
        return True
        
    def _check_domain_compatibility(self, update: Dict) -> bool:
        """Check domain compatibility for transfer"""
        # Add compatibility check here
        return True
