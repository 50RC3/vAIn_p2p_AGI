import torch
import numpy as np
from typing import Dict, Any
from utils.metrics import compute_accuracy
import logging

logger = logging.getLogger(__name__)

class BenchmarkError(Exception):
    pass

class ModelBenchmark:
    def __init__(self, model, test_loader, device='cuda'):
        try:
            self.model = model
            self.test_loader = test_loader
            self.device = device
            logger.info(f"Initialized ModelBenchmark with device: {device}")
        except Exception as e:
            logger.error(f"Failed to initialize ModelBenchmark: {str(e)}")
            raise BenchmarkError(f"Initialization failed: {str(e)}")

    def evaluate(self) -> Dict[str, Any]:
        try:
            result = {
                'accuracy': compute_accuracy(self.model, self.test_loader, self.device),
                'device': self.device,
                'total_params': sum(p.numel() for p in self.model.parameters())
            }
            logger.info(f"Evaluation complete - Accuracy: {result['accuracy']:.4f}")
            return result
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise BenchmarkError(f"Evaluation failed: {str(e)}")
