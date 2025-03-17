import torch
import gc
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class MemoryStatus:
    total: int
    used: int
    free: int
    cached_tensors: int

class MemoryManager:
    def __init__(self, max_cache_size: int = 1024):
        self.max_cache_size = max_cache_size
        self.tensor_cache = {}
        
    def cache_tensor(self, key: str, tensor: torch.Tensor) -> None:
        if self._get_cache_size() + tensor.element_size() * tensor.nelement() > self.max_cache_size:
            self._evict_cache()
        self.tensor_cache[key] = tensor
        
    def get_memory_status(self) -> MemoryStatus:
        torch.cuda.empty_cache()
        return MemoryStatus(
            total=torch.cuda.get_device_properties(0).total_memory,
            used=torch.cuda.memory_allocated(),
            free=torch.cuda.memory_reserved(),
            cached_tensors=len(self.tensor_cache)
        )
