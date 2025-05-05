import torch
import numpy as np
import time

class CompressionError(Exception):
    """Exception raised for errors in the compression module."""

class CompressionStats:
    def __init__(self):
        self.compression_times = []
        self.decompression_times = []
        self.compression_ratios = []
        self.quality_feedbacks = []

    def get_average_stats(self):
        return {
            "avg_compression_time": sum(self.compression_times) / max(1, len(self.compression_times)),
            "avg_decompression_time": sum(self.decompression_times) / max(1, len(self.decompression_times)),
            "avg_compression_ratio": sum(self.compression_ratios) / max(1, len(self.compression_ratios)),
            "avg_quality_feedback": sum(self.quality_feedbacks) / max(1, len(self.quality_feedbacks)) if self.quality_feedbacks else 0
        }

class AdaptiveCompression:
    def __init__(self, base_rate=0.1, min_rate=0.05, max_rate=0.9):
        self.base_rate = base_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.current_rate = base_rate
        self.stats = CompressionStats()
        self.quality_feedback = 1.0  # Default is best quality
    
    async def compress_model_updates(self, model_updates):
        start_time = time.time()
        compressed_updates = self._compress_gradients(model_updates)
        compression_time = time.time() - start_time
        
        # Calculate compression ratio (size before / size after)
        original_size = sum(tensor.numel() * 4 for tensor in model_updates.values() if torch.is_tensor(tensor))  # Assuming float32
        compressed_size = self._estimate_compressed_size(compressed_updates)
        ratio = original_size / max(1, compressed_size)  # Avoid division by zero
        
        # Update stats
        self.stats.compression_times.append(compression_time)
        self.stats.compression_ratios.append(ratio)
        
        # Adjust compression rate based on feedback
        self._adjust_compression_rate()
        
        return compressed_updates, ratio
    
    async def decompress_model_updates(self, compressed_updates):
        start_time = time.time()
        try:
            decompressed = {}
            
            # Empty input check
            if not compressed_updates:
                return {}
            
            # Process each parameter
            for param_name, param_data in compressed_updates.items():
                if param_data is None:
                    decompressed[param_name] = None
                    continue
                
                if isinstance(param_data, dict):
                    if 'indices' in param_data and 'values' in param_data:
                        # Sparse format
                        shape = tuple(param_data['shape'])
                        indices = param_data['indices']
                        values = param_data['values']
                        
                        tensor = torch.zeros(np.prod(shape))
                        for idx, val in zip(indices, values):
                            tensor[idx] = val
                        decompressed[param_name] = tensor.reshape(shape)
                    elif 'data' in param_data:
                        # Full tensor format
                        tensor_data = torch.tensor(param_data['data'])
                        decompressed[param_name] = tensor_data
                    else:
                        raise CompressionError(f"Invalid compressed format for {param_name}")
                else:
                    # Non-tensor data, keep as is
                    decompressed[param_name] = param_data
            
            decompression_time = time.time() - start_time
            self.stats.decompression_times.append(decompression_time)
            
            return decompressed
        except Exception as e:
            raise CompressionError(f"Decompression failed: {str(e)}") from e
    
    def _compress_gradients(self, model_updates):
        compressed = {}
        for name, tensor in model_updates.items():
            if not torch.is_tensor(tensor):
                # Non-tensor data, keep as is
                compressed[name] = tensor
                continue
                
            # Compress tensors
            tensor_np = tensor.detach().cpu().numpy()
            flattened = tensor_np.flatten()
            
            # Keep top k% values
            k = int(max(1, self.current_rate * len(flattened)))
            indices = np.argsort(np.abs(flattened))[-k:]
            values = flattened[indices]
            
            compressed[name] = {
                'shape': tensor_np.shape,
                'indices': indices.tolist(),
                'values': values.tolist()
            }
        
        return compressed
    
    def _decompress_gradients(self, compressed_updates):
        # This is implemented in decompress_model_updates
        return compressed_updates
    
    def _estimate_compressed_size(self, compressed_updates):
        total_size = 0
        for param_data in compressed_updates.values():
            if isinstance(param_data, dict):
                if 'indices' in param_data and 'values' in param_data:
                    # Sparse format: indices + values + shape
                    total_size += len(param_data['indices']) * 4  # indices (int32)
                    total_size += len(param_data['values']) * 4   # values (float32)
                    total_size += len(param_data['shape']) * 4    # shape dimensions
                elif 'data' in param_data:
                    # Full tensor format
                    if isinstance(param_data['data'], list):
                        total_size += len(np.array(param_data['data']).flatten()) * 4
            # Add estimated overhead for strings and other data
            total_size += 100  # approximate overhead per parameter
        
        return total_size
    
    def update_quality_feedback(self, quality_score):
        """Update the quality feedback (0 to 1, where 1 is best)"""
        self.quality_feedback = max(0, min(1, quality_score))
        self.stats.quality_feedbacks.append(quality_score)
    
    def _adjust_compression_rate(self):
        # Adjust compression rate based on quality feedback
        # Lower quality feedback results in higher compression rate (less compression)
        adjustment = 0.05 * (2 - self.quality_feedback)  # Range from -0.05 to +0.05
        self.current_rate = max(self.min_rate, min(self.max_rate, self.current_rate + adjustment))
    
    def get_stats(self):
        avg_stats = self.stats.get_average_stats()
        return {
            "current_rate": self.current_rate,
            "avg_compression_ratio": avg_stats["avg_compression_ratio"],
            "avg_compression_time": avg_stats["avg_compression_time"],
            "avg_decompression_time": avg_stats["avg_decompression_time"],
            "quality_feedback": self.quality_feedback,
            "history": {
                "compression_ratios": self.stats.compression_ratios,
                "quality_feedbacks": self.stats.quality_feedbacks
            }
        }