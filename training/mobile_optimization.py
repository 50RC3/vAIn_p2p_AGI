import torch
import numpy as np
from typing import Dict, Tuple, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class MobileOptimizer:
    def __init__(self, compression_rate: float = 0.1):
        self.compression_rate = compression_rate
        self.prev_state = {}
        self.change_threshold = 0.01  # Only send changes above 1%
        self.update_history = defaultdict(list)
        
    def compress_for_mobile(self, updates: Dict[str, torch.Tensor]) -> Dict:
        """Compress model updates specifically for mobile devices"""
        compressed = {}
        
        for key, tensor in updates.items():
            # Calculate delta from previous state
            prev = self.prev_state.get(key, torch.zeros_like(tensor))
            delta = tensor - prev
            
            # Only keep significant changes
            mask = torch.abs(delta) > self.change_threshold
            if mask.any():
                # Sparse representation
                values = delta[mask]
                indices = mask.nonzero()
                
                # Store compressed format
                compressed[key] = {
                    'values': values.cpu().numpy().tolist(),
                    'indices': indices.cpu().numpy().tolist(),
                    'shape': list(tensor.shape)
                }
                
                self.prev_state[key] = tensor
        
        return compressed

    def aggregate_metrics(self, metrics: Dict) -> Dict:
        """Aggregate metrics to minimize monitoring data"""
        return {
            'avg_cpu': np.mean(metrics.get('cpu_history', [0])),
            'peak_memory': max(metrics.get('memory_history', [0])),
            'total_bandwidth': sum(metrics.get('bandwidth_history', [0])),
            'error_count': len(metrics.get('errors', []))
        }

    def decompress_mobile_update(self, compressed: Dict) -> Dict[str, torch.Tensor]:
        """Decompress updates from mobile devices"""
        decompressed = {}
        
        for key, data in compressed.items():
            tensor = torch.zeros(*data['shape'])
            indices = torch.tensor(data['indices'])
            values = torch.tensor(data['values'])
            
            if len(indices) > 0:
                tensor[indices[:, 0], indices[:, 1]] = values
            decompressed[key] = tensor
            
        return decompressed
