from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional
import time
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

class CacheLevel(Enum):
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"

@dataclass
class CachePolicy:
    max_size: int
    ttl: int  # Time to live in seconds
    level: CacheLevel

@dataclass
class CacheEntry:
    data: Any
    timestamp: float
    metadata: Dict
    level: CacheLevel

class CacheManager:
    def __init__(self, policies: Dict[CacheLevel, CachePolicy]):
        self.policies = policies
        self.caches: Dict[CacheLevel, Dict] = {
            level: {} for level in CacheLevel
        }
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0
        }

    def get(self, key: str, level: CacheLevel = CacheLevel.MEMORY) -> Optional[Any]:
        """Get item from cache with specified level"""
        if entry := self.caches[level].get(key):
            if time.time() - entry.timestamp <= self.policies[level].ttl:
                self.stats["hits"] += 1
                return entry.data
            self._evict(key, level)
        self.stats["misses"] += 1
        return None

    def put(self, key: str, value: Any, metadata: Dict = None, 
            level: CacheLevel = CacheLevel.MEMORY) -> bool:
        """Put item in cache with specified level"""
        if not metadata:
            metadata = {}
            
        self._ensure_capacity(level)
        
        entry = CacheEntry(
            data=value,
            timestamp=time.time(),
            metadata=metadata,
            level=level
        )
        
        self.caches[level][key] = entry
        return True

    def _ensure_capacity(self, level: CacheLevel) -> None:
        """Ensure cache has capacity for new items"""
        cache = self.caches[level]
        policy = self.policies[level]
        
        while len(cache) >= policy.max_size:
            oldest_key = min(cache.keys(), key=lambda k: cache[k].timestamp)
            self._evict(oldest_key, level)

    def _evict(self, key: str, level: CacheLevel) -> None:
        """Evict item from cache"""
        if key in self.caches[level]:
            del self.caches[level][key]
            self.stats["evictions"] += 1

    @lru_cache(maxsize=100)
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return self.stats.copy()

    def clear(self, level: Optional[CacheLevel] = None) -> None:
        """Clear cache at specified level or all levels"""
        if level:
            self.caches[level].clear()
        else:
            for cache in self.caches.values():
                cache.clear()
        self.stats = {k: 0 for k in self.stats}
