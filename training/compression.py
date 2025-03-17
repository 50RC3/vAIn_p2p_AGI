import torch
from typing import Dict, Tuple
from torch import nn
import numpy as np

class AdaptiveCompression:
    def __init__(self, base_compression_rate=0.1, min_rate=0.01, max_rate=0.3):
        self.base_rate = base_compression_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.network_quality = 1.0  # 1.0 = good, 0.0 = poor
        self.decay_factor = 0.95
        self.reward_history = []
        self.learning_rate = 0.01
        self.exploration_rate = 0.1

    def compress_model_updates(self, model_update: Dict[str, torch.Tensor]) -> Tuple[Dict, float]:
        compression_rate = self._calculate_compression_rate()
        compressed = {}
        
        for name, tensor in model_update.items():
            # Select top k% of gradients based on magnitude
            k = max(1, int(tensor.numel() * compression_rate))
            values, indices = torch.topk(tensor.abs().flatten(), k)
            threshold = values[-1]
            
            # Create sparse tensor with only significant gradients
            mask = tensor.abs() >= threshold
            compressed[name] = {
                'values': tensor[mask],
                'indices': mask.nonzero(),
                'shape': tensor.shape
            }
            
        return compressed, compression_rate

    def _calculate_compression_rate(self) -> float:
        """Calculate compression rate using exponential decay and RL"""
        base_rate = self._get_rl_rate()
        network_factor = np.exp(-self.decay_factor * (1 - self.network_quality))
        rate = base_rate * network_factor
        return max(self.min_rate, min(self.max_rate, rate))

    def _get_rl_rate(self) -> float:
        """Use reinforcement learning to optimize base rate"""
        if np.random.random() < self.exploration_rate:
            return np.random.uniform(self.min_rate, self.max_rate)
        
        if not self.reward_history:
            return self.base_rate
            
        # Update base rate based on rewards
        reward_avg = np.mean(self.reward_history[-10:])
        if reward_avg > 0:
            self.base_rate += self.learning_rate
        else:
            self.base_rate -= self.learning_rate
            
        return self.base_rate

    def update_reward(self, accuracy_delta: float, bandwidth_usage: float):
        """Update RL rewards based on performance metrics"""
        reward = accuracy_delta - 0.5 * (bandwidth_usage / self.network_quality)
        self.reward_history.append(reward)
        if len(self.reward_history) > 100:
            self.reward_history.pop(0)

    def decompress_model_updates(self, compressed: Dict) -> Dict[str, torch.Tensor]:
        decompressed = {}
        
        for name, data in compressed.items():
            tensor = torch.zeros(data['shape'])
            tensor[tuple(data['indices'].t())] = data['values']
            decompressed[name] = tensor
            
        return decompressed
