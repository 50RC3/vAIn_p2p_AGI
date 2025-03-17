import torch
import numpy as np
from typing import Dict, Tuple
from torch.utils.data import DataLoader

class ModelEvaluator:
    def __init__(self, model: torch.nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        
    def evaluate_performance(self, data_loader: DataLoader) -> Dict[str, float]:
        accuracy = self._compute_accuracy(data_loader)
        loss, per_class_acc = self._compute_detailed_metrics(data_loader)
        
        return {
            'accuracy': accuracy,
            'loss': loss,
            'per_class_accuracy': per_class_acc,
            'model_size_mb': self._get_model_size(),
            'param_count': self._count_parameters()
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
