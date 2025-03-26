import torch
from torch.optim import Adam
import torch.nn.functional as F
import logging
from typing import Optional, Dict, Tuple
import psutil
import time
from dataclasses import dataclass
from .distillation import DistillationTrainer

logger = logging.getLogger(__name__)

@dataclass
class TrainingMetrics:
    loss: float = 0.0
    accuracy: float = 0.0
    duration: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0

class FederatedClientError(Exception):
    pass

class ResourceExhaustedError(FederatedClientError):
    pass

class FederatedClient:
    def __init__(self, model, data_loader, config):
        try:
            if not torch.cuda.is_available() and config.device == 'cuda':
                raise FederatedClientError("CUDA requested but not available")
                
            self.model = model.to(config.device)
            self.data_loader = data_loader
            self.config = config
            self.optimizer = Adam(model.parameters(), lr=config.learning_rate)
            self.criterion = F.cross_entropy
            self.metrics = TrainingMetrics()
            self._validate_config()
            self._check_resources()
            self.model_interface = ModelInterface(model, interactive=config.interactive)
            self.data_shard = None
            self.shard_id = None
            self.initial_model_state = None
            logger.info(f"Initialized FederatedClient on device: {config.device}")
        except Exception as e:
            logger.error(f"Failed to initialize FederatedClient: {str(e)}")
            raise FederatedClientError(f"Initialization failed: {str(e)}")

    def _validate_config(self):
        """Validate configuration parameters"""
        required = {'learning_rate', 'num_epochs', 'batch_size', 'device'}
        if not all(hasattr(self.config, attr) for attr in required):
            raise FederatedClientError(f"Missing required config attributes: {required}")
        
        if self.config.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.config.num_epochs <= 0:
            raise ValueError("Number of epochs must be positive")

    def _check_resources(self):
        """Verify sufficient system resources"""
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            raise ResourceExhaustedError("Insufficient memory available")
            
        if psutil.cpu_percent(interval=1) > 90:
            raise ResourceExhaustedError("CPU usage too high")

    def _update_metrics(self, batch_loss: float):
        """Update training metrics"""
        self.metrics.memory_usage = psutil.virtual_memory().percent
        self.metrics.cpu_usage = psutil.cpu_percent()
        self.metrics.loss = batch_loss

    async def train(self) -> Optional[Dict]:
        """Enhanced training with data parallelism"""
        try:
            if not self.data_shard:
                raise ValueError("Data shard not assigned")

            # Store initial model state for computing updates
            self.initial_model_state = {
                k: v.clone() for k, v in self.model.state_dict().items()
            }

            total_batches = len(self.data_shard) * self.config.epochs
            epoch_loss = 0
            
            for epoch in range(self.config.epochs):
                for batch_idx, (data, target) in enumerate(self.data_shard):
                    try:
                        data, target = data.to(self.config.device), target.to(self.config.device)
                        self.optimizer.zero_grad()
                        output = await self.model_interface.forward(data) 
                        loss = self._train_batch(output.output, target)
                        
                        # Update progress tracking
                        progress = ((epoch * len(self.data_shard) + batch_idx + 1) / total_batches) * 100
                        self._update_metrics(loss)
                        logger.info(f"Shard {self.shard_id} - Progress: {progress:.1f}% Loss: {loss:.4f}")
                        
                        epoch_loss += loss
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            logger.error("GPU OOM in training batch")
                            if hasattr(torch.cuda, 'empty_cache'):
                                torch.cuda.empty_cache()
                        raise

            # Compute model updates
            updates = {}
            for key, final_param in self.model.state_dict().items():
                updates[key] = final_param - self.initial_model_state[key]

            # Compress updates before sending
            compressed_updates = self.compression.compress_updates(updates)
            
            return {
                'shard_id': self.shard_id,
                'updates': compressed_updates,
                'metrics': self.metrics.to_dict()
            }

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

    def _train_batch(self, data: torch.Tensor, target: torch.Tensor) -> float:
        """Train a single batch"""
        self.optimizer.zero_grad()
        output = self.model(data)
        if isinstance(output, tuple):
            output = output[0]
        loss = self.criterion(output, target)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        return loss.item()

    async def coordinate_with_memory(self, memory_manager: MemoryManager) -> bool:
        """Coordinate with memory manager for efficient resource usage"""
        try:
            model_id = id(self.model)
            await memory_manager.coordinate_memory_systems(str(model_id))
            
            # Share model tensors
            for name, param in self.model.named_parameters():
                await memory_manager.share_tensor(
                    str(model_id),
                    'shared_pool',
                    f'{name}_grad'
                )
            return True
        except Exception as e:
            logger.error(f"Memory coordination failed: {e}")
            return False
