import asyncio
from collections import defaultdict
from enum import IntEnum
from typing import Dict, Optional
import time
import logging

logger = logging.getLogger(__name__)

class TrafficPriority(IntEnum):
    CRITICAL = 0  # System coordination, emergencies
    HIGH = 1      # Model updates, important metrics
    MEDIUM = 2    # Regular peer messages
    LOW = 3       # Discovery, non-essential updates
    
class TrafficShaper:
    def __init__(self, max_bandwidth: float = 1000.0):
        self.max_bandwidth = max_bandwidth  # bytes per second
        self.priority_allocations = {
            TrafficPriority.CRITICAL: 0.4,  # 40% for critical
            TrafficPriority.HIGH: 0.3,      # 30% for high
            TrafficPriority.MEDIUM: 0.2,    # 20% for medium
            TrafficPriority.LOW: 0.1        # 10% for low
        }
        self.usage = defaultdict(lambda: defaultdict(list))
        self.window_size = 1.0  # 1 second window
        self._lock = asyncio.Lock()

    async def can_send(self, size: int, priority: TrafficPriority, node_id: str) -> bool:
        async with self._lock:
            now = time.time()
            await self._cleanup_old_data(priority, node_id, now)
            
            # Calculate available bandwidth for priority level
            max_rate = self.max_bandwidth * self.priority_allocations[priority]
            current_usage = sum(s for _, s in self.usage[priority][node_id])
            
            # Allow burst for CRITICAL priority
            if priority == TrafficPriority.CRITICAL and current_usage < max_rate * 2:
                self.usage[priority][node_id].append((now, size))
                return True
                
            if current_usage + size <= max_rate:
                self.usage[priority][node_id].append((now, size))
                return True
            return False

    async def _cleanup_old_data(self, priority: TrafficPriority, node_id: str, now: float):
        cutoff = now - self.window_size
        self.usage[priority][node_id] = [
            (t, s) for t, s in self.usage[priority][node_id] 
            if t > cutoff
        ]

    def adjust_allocation(self, priority: TrafficPriority, allocation: float):
        """Adjust bandwidth allocation for a priority level"""
        if not 0 < allocation < 1:
            raise ValueError("Allocation must be between 0 and 1")
        
        total = sum(self.priority_allocations[p] for p in TrafficPriority 
                   if p != priority) + allocation
        if total > 1:
            raise ValueError("Total allocations cannot exceed 1")
        
        self.priority_allocations[priority] = allocation
