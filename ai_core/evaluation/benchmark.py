import torch
import numpy as np
from typing import Dict, Any, Optional
from utils.metrics import compute_accuracy
import logging
import time
import json
from pathlib import Path
import psutil
import GPUtil

logger = logging.getLogger(__name__)

class BenchmarkError(Exception):
    pass

class ModelBenchmark:
    def __init__(self, model, test_loader, device='cuda', 
                 save_path: Optional[str] = None,
                 early_stop_threshold: float = 0.95):
        try:
            self.model = model
            self.test_loader = test_loader
            self.device = device
            self.save_path = Path(save_path) if save_path else None
            self.early_stop_threshold = early_stop_threshold
            self.monitoring_stats = {}
            logger.info(f"Initialized ModelBenchmark with device: {device}")
        except Exception as e:
            logger.error(f"Failed to initialize ModelBenchmark: {str(e)}")
            raise BenchmarkError(f"Initialization failed: {str(e)}")

    def get_resource_usage(self) -> Dict[str, float]:
        """Monitor system resource usage"""
        stats = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / (1024**3)
        }
        
        if torch.cuda.is_available():
            gpu = GPUtil.getGPUs()[0]
            stats.update({
                'gpu_utilization': gpu.load * 100,
                'gpu_memory_used_gb': gpu.memoryUsed / 1024
            })
        return stats

    def evaluate(self, interactive: bool = True) -> Dict[str, Any]:
        """Run model evaluation with progress tracking and resource monitoring"""
        try:
            start_time = time.time()
            total_batches = len(self.test_loader)
            correct = 0
            total = 0
            
            self.model.eval()
            resource_samples = []

            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(self.test_loader):
                    # Resource monitoring
                    if batch_idx % 10 == 0:  # Sample every 10 batches
                        resource_samples.append(self.get_resource_usage())
                    
                    # Progress tracking
                    if interactive and batch_idx % 5 == 0:
                        progress = (batch_idx / total_batches) * 100
                        logger.info(f"Progress: {progress:.1f}% - "
                                  f"Batch {batch_idx}/{total_batches}")

                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
                    
                    # Early stopping check
                    current_accuracy = correct / total
                    if current_accuracy >= self.early_stop_threshold:
                        logger.info("Early stopping threshold reached")
                        break

            # Compute final results
            accuracy = correct / total
            elapsed_time = time.time() - start_time
            
            result = {
                'accuracy': accuracy,
                'device': self.device,
                'total_params': sum(p.numel() for p in self.model.parameters()),
                'elapsed_time': elapsed_time,
                'batches_processed': batch_idx + 1,
                'early_stopped': accuracy >= self.early_stop_threshold,
                'resource_usage': {
                    'mean': {k: np.mean([s[k] for s in resource_samples]) 
                            for k in resource_samples[0].keys()},
                    'max': {k: np.max([s[k] for s in resource_samples]) 
                           for k in resource_samples[0].keys()}
                }
            }

            # Save results if path provided
            if self.save_path:
                self.save_results(result)

            logger.info(f"Evaluation complete:\n"
                       f"- Accuracy: {accuracy:.4f}\n"
                       f"- Time: {elapsed_time:.2f}s\n"
                       f"- Batches: {batch_idx + 1}/{total_batches}")
            return result

        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise BenchmarkError(f"Evaluation failed: {str(e)}")

    def save_results(self, result: Dict[str, Any]) -> None:
        """Save benchmark results to file"""
        try:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filepath = self.save_path / f"benchmark_{timestamp}.json"
            
            with open(filepath, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Saved benchmark results to {filepath}")
            
        except Exception as e:
            logger.warning(f"Failed to save benchmark results: {str(e)}")
