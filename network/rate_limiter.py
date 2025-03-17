from datetime import datetime, timedelta
from collections import defaultdict
import asyncio
from typing import Dict
import time

class RateLimiter:
    def __init__(self, max_requests: int, time_window: int):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = defaultdict(list)
        
    async def allow_request(self, client_id: str) -> bool:
        now = datetime.now()
        window_start = now - timedelta(seconds=self.time_window)
        
        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > window_start
        ]
        
        # Check rate limit
        if len(self.requests[client_id]) >= self.max_requests:
            return False
            
        self.requests[client_id].append(now)
        return True

class AdaptiveRateLimiter:
    def __init__(self, initial_rate: float = 1000.0, window_size: int = 60):
        self.base_rate = initial_rate  # bytes per second
        self.window_size = window_size  # seconds
        self.usage = defaultdict(list)
        self.last_adjustment = time.time()
        
    def can_send(self, node_id: str, size: int) -> bool:
        now = time.time()
        self._cleanup_old_data(node_id, now)
        
        current_usage = sum(usage[1] for usage in self.usage[node_id])
        allowed_usage = self.base_rate * self.window_size
        
        if current_usage + size <= allowed_usage:
            self.usage[node_id].append((now, size))
            return True
        return False
        
    def _cleanup_old_data(self, node_id: str, now: float):
        cutoff = now - self.window_size
        self.usage[node_id] = [
            (t, s) for t, s in self.usage[node_id] 
            if t > cutoff
        ]

    def adjust_rate(self, congestion_level: float):
        """Adjust rate based on network congestion (0-1 scale)"""
        self.base_rate *= (1 - congestion_level)
        self.last_adjustment = time.time()
