import torch
from torch.optim import Adam
import torch.nn.functional as F
import logging
from typing import Optional, Dict, Tuple
import psutil
import time
from dataclasses import dataclass

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

    def train(self) -> Optional[Dict]:
        """Train the model and return updated parameters"""
        try:
            self.model.train()
            start_time = time.time()
            total_loss = 0
            total_batches = len(self.data_loader) * self.config.num_epochs

            for epoch in range(self.config.num_epochs):
                self._check_resources()  # Monitor resources each epoch
                epoch_loss = 0
                
                for batch_idx, (data, target) in enumerate(self.data_loader):
                    try:
                        data, target = data.to(self.config.device), target.to(self.config.device)
                        loss = self._train_batch(data, target)
                        epoch_loss += loss
                        self._update_metrics(loss)
                        
                        # Log progress
                        if batch_idx % max(1, len(self.data_loader) // 10) == 0:
                            progress = ((epoch * len(self.data_loader) + batch_idx + 1) / total_batches) * 100
                            logger.info(f"Training progress: {progress:.1f}% - Loss: {loss:.4f}")
                            
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            if hasattr(torch.cuda, 'empty_cache'):
                                torch.cuda.empty_cache()
                            logger.error("GPU OOM, attempting recovery")
                            continue
                        raise e
                    except Exception as e:
                        logger.error(f"Batch training failed: {str(e)}")
                        continue

                avg_epoch_loss = epoch_loss / len(self.data_loader)
                logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}, Loss: {avg_epoch_loss:.4f}")
                total_loss += avg_epoch_loss

            self.metrics.duration = time.time() - start_time
            return {
                'state_dict': self.model.state_dict(),
                'metrics': self.metrics.__dict__
            }
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return None

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
