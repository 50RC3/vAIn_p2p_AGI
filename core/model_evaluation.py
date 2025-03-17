import torch
from typing import Dict, Any, Optional
from dataclasses import dataclass
import time
import psutil

@dataclass
class EvaluationMetrics:
    accuracy: float
    loss: float
    f1_score: float
    latency: float
    memory_usage: float

class ModelEvaluator:
    def __init__(self, model: torch.nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def _compute_accuracy(self, data_loader) -> float:
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total

    def _compute_loss(self, data_loader) -> float:
        total_loss = 0
        num_batches = 0
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                num_batches += 1
        return total_loss / num_batches

    def _compute_f1_score(self, data_loader) -> float:
        # ... rest of implementation
        pass

    def _measure_inference_latency(self, data_loader) -> float:
        start_time = time.time()
        with torch.no_grad():
            for inputs, _ in data_loader:
                inputs = inputs.to(self.device)
                _ = self.model(inputs)
        end_time = time.time()
        return (end_time - start_time) * 1000  # Convert to ms

    def _measure_memory_usage(self) -> float:
        return psutil.Process().memory_info().rss / 1024 / 1024  # Convert to MB

    def evaluate(self, data_loader) -> EvaluationMetrics:
        self.model.eval()
        metrics = {
            'accuracy': self._compute_accuracy(data_loader),
            'loss': self._compute_loss(data_loader),
            'f1_score': self._compute_f1_score(data_loader),
            'latency': self._measure_inference_latency(data_loader),
            'memory_usage': self._measure_memory_usage()
        }
        return EvaluationMetrics(**metrics)
