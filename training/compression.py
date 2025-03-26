import torch
from typing import Dict, Tuple, Optional, Any, List
from torch import nn
import numpy as np
import logging
import warnings
from dataclasses import dataclass
import json
import os
import asyncio
from .feature_optimization import FeatureOptimizer

logger = logging.getLogger(__name__)

@dataclass
class CompressionStats:
    """Track compression performance metrics"""
    original_size: int
    compressed_size: int
    compression_ratio: float
    bandwidth_saved: float
    accuracy_impact: float

class CompressionError(Exception):
    """Custom exception for compression-related errors"""
    pass

class AdaptiveCompression:
    def __init__(self, 
                 base_compression_rate: float = 0.1,
                 min_rate: float = 0.01, 
                 max_rate: float = 0.3,
                 eps: float = 1e-8,
                 stats_history_size: int = 1000):
        self._validate_init_params(base_compression_rate, min_rate, max_rate)
        self.base_rate = base_compression_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.network_quality = 1.0  # 1.0 = good, 0.0 = poor
        self.decay_factor = 0.95
        self.reward_history = []
        self.learning_rate = 0.01
        self.exploration_rate = 0.1
        self.eps = eps
        self.stats_history: List[CompressionStats] = []
        self.stats_history_size = stats_history_size
        self._initialize_monitoring()
        
        # Add domain-aware compression tracking
        self.domain_compression_rates = {}
        self.cross_domain_stats = {}

        self.batch_config = {
            'target_latency': 0.05,  # 50ms target
            'min_size': 1024,        # Minimum batch 1KB
            'max_size': 1024*1024,   # Maximum batch 1MB
            'parallel_limit': 4       # Max parallel compression tasks
        }
        self.compression_queue = asyncio.Queue()
        self.feature_optimizer = FeatureOptimizer()

    def _validate_init_params(self, base_rate: float, min_rate: float, max_rate: float) -> None:
        """Validate initialization parameters"""
        if not (0 < min_rate <= base_rate <= max_rate < 1):
            raise ValueError(f"Invalid rates: min={min_rate}, base={base_rate}, max={max_rate}")

    def _initialize_monitoring(self) -> None:
        """Initialize performance monitoring"""
        self.total_compressed = 0
        self.total_original = 0
        self.compression_time = 0.0
        self._load_state()

    def compress_model_updates(self, model_update: Dict[str, torch.Tensor]) -> Tuple[Dict, float]:
        """Compress model updates with advanced optimization"""
        try:
            self._validate_model_updates(model_update)
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            
            # Optimize features and compress data
            optimized_update, selected_features = self.feature_optimizer.select_features(model_update)
            
            # Apply quantization to compressed updates
            compressed = {}
            original_size = 0
            compressed_size = 0
            
            for name, tensor in optimized_update.items():
                original_size += tensor.numel() * tensor.element_size()
                
                if name in selected_features:
                    # Quantize to 8-bit precision
                    scale = tensor.abs().max() / 127
                    quantized = torch.quantize_per_tensor(
                        tensor.float(), scale=scale, zero_point=0, dtype=torch.qint8
                    )
                    
                    compressed[name] = {
                        'values': quantized.dequantize(),
                        'scale': scale,
                        'shape': tensor.shape,
                    }
                    compressed_size += compressed[name]['values'].numel()

            end_time.record()
            torch.cuda.synchronize()
            self.compression_time += start_time.elapsed_time(end_time)
            
            # Update statistics
            stats = CompressionStats(
                original_size=original_size,
                compressed_size=compressed_size,
                compression_ratio=compressed_size/original_size,
                bandwidth_saved=(original_size-compressed_size)/original_size,
                accuracy_impact=0.0  # Will be updated later with update_reward
            )
            self._update_stats(stats)
            self._update_domain_stats(domain_type, stats)
            self._save_state()
            
            return compressed, compression_rate
            
        except Exception as e:
            logger.error(f"Compression failed: {str(e)}")
            raise CompressionError(f"Compression failed: {str(e)}")

    def _validate_model_updates(self, updates: Dict[str, torch.Tensor]) -> None:
        """Validate model updates before compression"""
        if not updates:
            raise CompressionError("Empty model updates")
        for name, tensor in updates.items():
            if torch.isnan(tensor).any():
                raise CompressionError(f"NaN values in {name}")
            if torch.isinf(tensor).any():
                raise CompressionError(f"Inf values in {name}")

    def decompress_model_updates(self, compressed: Dict) -> Dict[str, torch.Tensor]:
        """Decompress updates with validation"""
        try:
            decompressed = {}
            for name, data in compressed.items():
                tensor = torch.tensor(data['values'])
                mask = torch.tensor(data['mask'], dtype=torch.bool)
                full_tensor = torch.zeros_like(data['shape'])
                full_tensor[mask] = tensor
                decompressed[name] = full_tensor
            return decompressed
        except Exception as e:
            logger.error(f"Decompression failed: {str(e)}")
            raise CompressionError(f"Decompression failed: {str(e)}")

    def _update_stats(self, stats: CompressionStats) -> None:
        """Update compression statistics"""
        self.stats_history.append(stats)
        if len(self.stats_history) > self.stats_history_size:
            self.stats_history.pop(0)
        self.total_compressed += stats.compressed_size
        self.total_original += stats.original_size

    def _save_state(self) -> None:
        """Persist compression state"""
        state = {
            'base_rate': self.base_rate,
            'network_quality': self.network_quality,
            'reward_history': self.reward_history[-100:],
            'total_compressed': self.total_compressed,
            'total_original': self.total_original
        }
        try:
            with open('compression_state.json', 'w') as f:
                json.dump(state, f)
        except Exception as e:
            logger.warning(f"Failed to save compression state: {e}")

    def _load_state(self) -> None:
        """Load persisted compression state"""
        try:
            if os.path.exists('compression_state.json'):
                with open('compression_state.json', 'r') as f:
                    state = json.load(f)
                self.base_rate = state['base_rate']
                self.network_quality = state['network_quality']
                self.reward_history = state['reward_history']
                self.total_compressed = state['total_compressed']
                self.total_original = state['total_original']
        except Exception as e:
            logger.warning(f"Failed to load compression state: {e}")

    def _calculate_compression_rate(self) -> float:
        """Calculate adaptive compression rate"""
        if not self.reward_history:
            return self.base_rate
        recent_rewards = self.reward_history[-10:]
        avg_reward = sum(recent_rewards) / len(recent_rewards)
        return max(self.min_rate, 
                  min(self.max_rate,
                      self.base_rate * (1 - avg_reward)))

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

    def _get_domain_compression_rate(self, domain_type: str) -> float:
        """Get optimal compression rate for domain"""
        if domain_type not in self.domain_compression_rates:
            self.domain_compression_rates[domain_type] = self.base_rate
        return self.domain_compression_rates[domain_type]
        
    def _update_domain_stats(self, domain_type: str, stats: Dict):
        """Update compression statistics per domain"""
        if domain_type not in self.cross_domain_stats:
            self.cross_domain_stats[domain_type] = []
        self.cross_domain_stats[domain_type].append(stats)
        
        # Prune old stats
        if len(self.cross_domain_stats[domain_type]) > self.stats_history_size:
            self.cross_domain_stats[domain_type].pop(0)

    async def compress_batch_parallel(self, updates: List[Dict[str, torch.Tensor]]) -> List[Tuple[Dict, float]]:
        """Process compression in optimized parallel batches"""
        tasks = []
        results = []
        
        for i in range(0, len(updates), self.batch_config['parallel_limit']):
            batch = updates[i:i + self.batch_config['parallel_limit']]
            batch_tasks = [
                asyncio.create_task(self.compress_model_updates(update))
                for update in batch
            ]
            tasks.extend(batch_tasks)
            
            if len(tasks) >= self.batch_config['parallel_limit']:
                done, tasks = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )
                results.extend([t.result() for t in done])
        
        if tasks:
            done, _ = await asyncio.wait(tasks)
            results.extend([t.result() for t in done])
            
        return results

    def compress_updates(self, updates: Dict[str, torch.Tensor]) -> Dict:
        """Compress model updates with sparse encoding"""
        try:
            compressed = {}
            for key, tensor in updates.items():
                # Use top-k sparsification
                k = max(1, int(tensor.numel() * self.base_rate))
                values, indices = torch.topk(tensor.abs().flatten(), k)
                threshold = values[-1].item()
                
                # Create sparse tensor
                mask = tensor.abs() >= threshold
                compressed[key] = {
                    'values': tensor[mask].cpu(),
                    'indices': mask.nonzero().cpu(),
                    'shape': tensor.shape
                }

            return compressed

        except Exception as e:
            logger.error(f"Update compression failed: {str(e)}")
            raise

    def decompress_updates(self, compressed: Dict) -> Dict[str, torch.Tensor]:
        """Decompress model updates"""
        try:
            decompressed = {}
            for key, data in compressed.items():
                tensor = torch.zeros(data['shape'])
                tensor[data['indices'][:, 0], data['indices'][:, 1]] = data['values']
                decompressed[key] = tensor
            return decompressed

        except Exception as e:
            logger.error(f"Update decompression failed: {str(e)}")
            raise
