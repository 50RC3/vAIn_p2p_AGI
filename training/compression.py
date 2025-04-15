"""
Module for compression techniques used in federated learning.
"""
import asyncio
import logging
import time
from typing import Dict, Any, Tuple, List, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

class CompressionError(Exception):
    """Exception raised for errors in the compression module."""
    pass

class CompressionStats:
    """Statistics for compression operations"""
    def __init__(self):
        self.compression_ratios = []
        self.bandwidth_saved = []
        self.quality_scores = []
        self.compression_times = []
        self.decompression_times = []
        
    def update(self, ratio: float = 0.0, bandwidth_saved: float = 0.0, quality: float = 1.0,
               comp_time: float = 0.0, decomp_time: float = 0.0):
        """Update compression statistics"""
        self.compression_ratios.append(ratio)
        self.bandwidth_saved.append(bandwidth_saved)
        self.quality_scores.append(quality)
        self.compression_times.append(comp_time)
        self.decompression_times.append(decomp_time)
    
    def get_average_stats(self) -> Dict[str, float]:
        """Get average statistics"""
        return {
            "avg_compression_ratio": np.mean(self.compression_ratios) if self.compression_ratios else 0.0,
            "avg_bandwidth_saved": np.mean(self.bandwidth_saved) if self.bandwidth_saved else 0.0,
            "avg_quality_score": np.mean(self.quality_scores) if self.quality_scores else 0.0,
            "avg_compression_time": np.mean(self.compression_times) if self.compression_times else 0.0,
            "avg_decompression_time": np.mean(self.decompression_times) if self.decompression_times else 0.0,
        }

class AdaptiveCompression:
    """
    Adaptive compression for model updates based on network conditions.
    Dynamically adjusts compression ratio based on quality requirements.
    """
    
    def __init__(self, base_rate: float = 0.1, min_rate: float = 0.01, 
                 max_rate: float = 0.5, quality_threshold: float = 0.9):
        """
        Initialize the adaptive compression.
        
        Args:
            base_rate: Starting compression rate (fraction of data to keep)
            min_rate: Minimum compression rate
            max_rate: Maximum compression rate
            quality_threshold: Minimum quality score to maintain (0-1)
        """
        self.base_rate = base_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.quality_threshold = quality_threshold
        self.current_rate = base_rate
        self.stats = CompressionStats()
        self.last_quality = 1.0
        self.learning_rate = 0.1  # Learning rate for adaptation
        self.history = []  # Compression history
        
    async def compress_model_updates(self, updates: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """
        Compress model updates adaptively.
        
        Args:
            updates: Dictionary of model updates to compress
            
        Returns:
            Tuple of (compressed_updates, compression_ratio)
        """
        start_time = time.time()
        
        try:
            # Adjust compression rate based on previous quality
            self._adjust_compression_rate()
            
            # Do actual compression with the current rate
            compressed = self._compress_gradients(updates, self.current_rate)
            
            # Calculate compression ratio (compressed size / original size)
            # This is approximate since we don't have the exact byte sizes
            original_size = sum(tensor.numel() * tensor.element_size() 
                              for tensor in updates.values() if isinstance(tensor, torch.Tensor))
            
            # Estimate compressed size based on kept values
            compressed_size = 0
            for name, item in compressed.items():
                if isinstance(item, dict) and 'indices' in item:
                    compressed_size += len(item['indices']) * 8  # Assuming 8 bytes per index/value pair
            
            compression_ratio = compressed_size / max(1, original_size)
            bandwidth_saved = 1.0 - compression_ratio
            
            # Update stats
            comp_time = time.time() - start_time
            self.stats.update(
                ratio=compression_ratio,
                bandwidth_saved=bandwidth_saved,
                comp_time=comp_time
            )
            
            self.history.append({
                'timestamp': time.time(),
                'rate': self.current_rate,
                'ratio': compression_ratio,
                'bandwidth_saved': bandwidth_saved
            })
            
            if len(self.history) > 100:
                self.history = self.history[-100:]  # Keep last 100 entries
                
            return compressed, compression_ratio
            
        except Exception as e:
            logger.error(f"Compression failed: {str(e)}")
            raise CompressionError(f"Failed to compress model updates: {str(e)}") from e
    
        """
        Decompress model updates received from peers.
        This asynchronous method takes compressed model updates and restores them to their
        original format. It measures decompression time and updates internal statistics.
        
        Args:
            compressed (Dict[str, Any]): Dictionary containing compressed model updates.
        
        Returns:
            Dict[str, Any]: Decompressed model updates in their original format.
            
        Raises:
            CompressionError: If the decompression process fails for any reason.
            
        Notes:
            - Updates internal statistics with decompression time
            - Uses the internal _decompress_gradients implementation for the actual decompression
        """
        start_time = time.time()
        
        try:
            # Decompress the updates
            decompressed = self._decompress_gradients(compressed)
            
            # Update stats
            decomp_time = time.time() - start_time
            self.stats.update(
                decomp_time=decomp_time
            )
            
            return decompressed
            
        except Exception as e:
            logger.error(f"Decompression failed: {str(e)}")
            raise CompressionError(f"Failed to decompress model updates: {str(e)}") from e
    
    def _adjust_compression_rate(self) -> None:
        """Adjust compression rate based on quality feedback"""
        # If quality is below threshold, reduce compression (increase rate)
        if self.last_quality < self.quality_threshold:
            new_rate = min(self.max_rate, self.current_rate + self.learning_rate)
            logger.debug(f"Quality below threshold ({self.last_quality:.2f} < {self.quality_threshold:.2f}). "
                        f"Adjusting compression rate: {self.current_rate:.2f} -> {new_rate:.2f}")
            self.current_rate = new_rate
        else:
            # Otherwise, try to increase compression (decrease rate)
            new_rate = max(self.min_rate, self.current_rate - 0.5 * self.learning_rate)
            self.current_rate = new_rate
    
    def update_quality_feedback(self, quality_score: float) -> None:
        """
        Update compression quality feedback.
        
        Args:
            quality_score: Quality score of the last compression (0-1)
        """
        self.last_quality = max(0.0, min(1.0, quality_score))
        self.stats.quality_scores.append(self.last_quality)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get compression statistics"""
        return {
            "current_rate": self.current_rate,
            "min_rate": self.min_rate,
            "max_rate": self.max_rate,
            "quality_threshold": self.quality_threshold,
            "last_quality": self.last_quality,
            **self.stats.get_average_stats(),
            "history": self.history[-10:]  # Last 10 entries
        }
        
    async def cleanup(self) -> None:
        """Clean up resources used by the compressor"""
        self.history.clear()
        self.stats = CompressionStats()
        
    def _compress_gradients(self, gradients: Dict[str, torch.Tensor], compression_ratio: float = 0.1) -> Dict[str, Any]:
        """
        Compress neural network gradients to reduce communication overhead.
        
        Args:
            gradients: Dictionary mapping parameter names to gradient tensors
            compression_ratio: The ratio of values to keep (between 0 and 1)
            
        Returns:
            Dictionary containing compressed gradients
        """
        compressed = {}
        
        for name, grad in gradients.items():
            if grad is None or not torch.is_tensor(grad):
                compressed[name] = grad
                continue
                
            # Convert to numpy for processing
            grad_np = grad.detach().cpu().numpy()
            
            if compression_ratio < 1.0:
                # Top-k sparsification
                size = grad_np.size
                k = max(1, int(size * compression_ratio))
                
                # Flatten the tensor
                flattened = grad_np.flatten()
                
                # Find indices of top k values by magnitude
                indices = np.argsort(np.abs(flattened))[-k:]
                values = flattened[indices]
                
                # Store as sparse representation
                compressed[name] = {
                    'shape': grad_np.shape,
                    'indices': indices.tolist(),
                    'values': values.tolist(),
                }
            else:
                # Just store the full tensor
                compressed[name] = {
                    'shape': grad_np.shape,
                    'data': grad_np.tolist(),
                }
        
        return compressed

    def _decompress_gradients(self, compressed_grads: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Decompress gradients that were compressed with _compress_gradients.
        
        Args:
            compressed_grads: Dictionary of compressed gradients
            
        Returns:
            Dictionary of decompressed gradients
        """
        result = {}
        
        for name, item in compressed_grads.items():
            if isinstance(item, dict) and 'shape' in item:
                shape = tuple(item['shape'])
                
                if 'indices' in item:  # Sparse representation
                    # Initialize with zeros
                    grad_np = np.zeros(np.prod(shape))
                    
                    # Put values back in their original positions
                    indices = item['indices']
                    values = item['values']
                    for idx, val in zip(indices, values):
                        grad_np[idx] = val
                    
                    # Reshape back to original dimensions
                    grad_np = grad_np.reshape(shape)
                else:  # Full representation
                    grad_np = np.array(item['data']).reshape(shape)
                
                # Convert back to PyTorch tensor
                result[name] = torch.tensor(grad_np)
            else:
                result[name] = item  # Keep as is (None or non-tensor)
        
        return result
