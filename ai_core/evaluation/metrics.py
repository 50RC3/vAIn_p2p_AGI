import torch
import numpy as np
import logging
import psutil
from typing import Dict, Tuple, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, model: torch.nn.Module, device: str = 'cuda', 
                 batch_size_limit: Optional[int] = None):
        self.model = model
        self.device = device
        self.batch_size_limit = batch_size_limit
        self._validate_setup()
        
    def _validate_setup(self):
        """Validate model and device setup"""
        if not isinstance(self.model, torch.nn.Module):
            raise ValueError("Model must be a torch.nn.Module instance")
        if not torch.cuda.is_available() and self.device == 'cuda':
            logger.warning("CUDA not available, falling back to CPU")
            self.device = 'cpu'
        self.model.to(self.device)
        
    def evaluate_performance(self, data_loader: DataLoader, 
                           interactive: bool = True) -> Dict[str, float]:
        """Run model evaluation with progress tracking and resource monitoring"""
        try:
            self._check_dataloader(data_loader)
            metrics = {
                'resource_usage': [],
                'batch_metrics': []
            }
            
            correct = total = 0
            self.model.eval()
            
            # Progress bar for interactive mode
            iterator = tqdm(data_loader) if interactive else data_loader
            
            with torch.no_grad():
                for batch_idx, (inputs, labels) in enumerate(iterator):
                    # Monitor resources every 10 batches
                    if batch_idx % 10 == 0:
                        metrics['resource_usage'].append(self._get_resource_usage())
                    
                    try:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        
                        if self.batch_size_limit and inputs.size(0) > self.batch_size_limit:
                            raise ValueError(f"Batch size {inputs.size(0)} exceeds limit {self.batch_size_limit}")
                        
                        outputs = self.model(inputs)
                        _, predicted = outputs.max(1)
                        total += labels.size(0)
                        correct += predicted.eq(labels).sum().item()
                        
                        # Track batch-level metrics
                        metrics['batch_metrics'].append({
                            'batch_accuracy': predicted.eq(labels).float().mean().item(),
                            'batch_size': inputs.size(0)
                        })
                        
                        if interactive:
                            iterator.set_description(f"Acc: {(correct/total):.4f}")
                            
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            logger.error("GPU OOM - Try reducing batch size")
                            torch.cuda.empty_cache()
                        raise
                        
            metrics.update({
                'accuracy': correct / total,
                'loss': self._compute_detailed_metrics(data_loader)[0],
                'per_class_accuracy': self._compute_detailed_metrics(data_loader)[1],
                'model_size_mb': self._get_model_size(),
                'param_count': self._count_parameters(),
                'avg_resource_usage': {
                    k: np.mean([d[k] for d in metrics['resource_usage']])
                    for k in metrics['resource_usage'][0].keys()
                }
            })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise
            
    def _check_dataloader(self, data_loader: DataLoader):
        """Validate dataloader configuration"""
        if not isinstance(data_loader, DataLoader):
            raise ValueError("data_loader must be a torch.utils.data.DataLoader instance")
        if not data_loader.dataset:
            raise ValueError("DataLoader has no dataset")
            
    def _get_resource_usage(self) -> Dict[str, float]:
        """Monitor system resource usage"""
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'gpu_memory_used': torch.cuda.memory_allocated(self.device) / 1024**2 
                             if torch.cuda.is_available() else 0
        }
        
    def _compute_accuracy(self, data_loader: DataLoader) -> float:
        correct = total = 0
        self.model.eval()
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        return correct / total
