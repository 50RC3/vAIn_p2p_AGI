from datetime import datetime, timedelta
from collections import defaultdict
import asyncio
from typing import Dict, Optional
import time
import logging
from core.constants import INTERACTION_TIMEOUTS, InteractionLevel
from core.interactive_utils import InteractiveSession, InteractiveConfig
from .traffic_shaper import TrafficShaper, TrafficPriority

logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self, max_requests: int, time_window: int, interactive: bool = True):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = defaultdict(list)
        self.interactive = interactive
        self.session = None
        self._cleanup_event = asyncio.Event()
        self._interrupt_requested = False
        self._lock = asyncio.Lock()
        
    async def allow_request(self, client_id: str) -> bool:
        async with self._lock:
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

    async def allow_request_interactive(self, client_id: str) -> bool:
        """Interactive request handling with monitoring"""
        try:
            if self.interactive:
                self.session = InteractiveSession(
                    level=InteractionLevel.NORMAL,
                    config=InteractiveConfig(
                        timeout=INTERACTION_TIMEOUTS["default"],
                        persistent_state=True,
                        safe_mode=True
                    )
                )

            async with self.session:
                allowed = await self.allow_request(client_id)
                if not allowed and self.interactive:
                    logger.warning(f"Rate limit exceeded for client {client_id}")
                return allowed

        except Exception as e:
            logger.error(f"Rate limit check failed: {str(e)}")
            return False
        finally:
            if self.session:
                await self.session.__aexit__(None, None, None)

    async def cleanup(self):
        """Cleanup resources"""
        try:
            self._cleanup_event.set()
            self.requests.clear()
            logger.info("Rate limiter cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")

    def request_shutdown(self):
        """Request graceful shutdown"""
        self._interrupt_requested = True

class AdaptiveRateLimiter:
    def __init__(self, initial_rate: float = 1000.0, window_size: int = 60, 
                 interactive: bool = True):
        self.base_rate = initial_rate  # bytes per second
        self.window_size = window_size  # seconds
        self.usage = defaultdict(list)
        self.last_adjustment = time.time()
        self.interactive = interactive
        self.session = None
        self._cleanup_event = asyncio.Event()
        self._interrupt_requested = False
        self.adjustment_history = []
        self._lock = asyncio.Lock()
        self.traffic_shaper = TrafficShaper(initial_rate)

    async def can_send(self, node_id: str, size: int, priority: TrafficPriority = TrafficPriority.MEDIUM) -> bool:
        async with self._lock:
            try:
                # First check traffic shaping rules
                if not await self.traffic_shaper.can_send(size, priority, node_id):
                    return False
                    
                # Then apply general rate limiting
                now = time.time()
                await self._cleanup_old_data(node_id, now)
                
                current_usage = sum(usage[1] for usage in self.usage[node_id])
                allowed_usage = self.base_rate * self.window_size
                
                if current_usage + size <= allowed_usage:
                    self.usage[node_id].append((now, size))
                    return True
                return False
                
            except Exception as e:
                logger.error(f"Error in can_send for node {node_id}: {str(e)}")
                return False
        
    async def _cleanup_old_data(self, node_id: str, now: float):
        async with self._lock:
            try:
                cutoff = now - self.window_size
                self.usage[node_id] = [
                    (t, s) for t, s in self.usage[node_id] 
                    if t > cutoff
                ]
            except Exception as e:
                logger.error(f"Error cleaning old data for node {node_id}: {str(e)}")
                raise

    async def adjust_rate(self, congestion_level: float):
        """Adjust rate based on network congestion (0-1 scale)"""
        async with self._lock:
            self.base_rate *= (1 - congestion_level)
            self.last_adjustment = time.time()

    async def can_send_interactive(self, node_id: str, size: int) -> bool:
        """Interactive send check with monitoring"""
        try:
            if self.interactive:
                self.session = InteractiveSession(
                    level=InteractionLevel.NORMAL,
                    config=InteractiveConfig(
                        timeout=INTERACTION_TIMEOUTS["default"],
                        persistent_state=True,
                        safe_mode=True
                    )
                )

            async with self.session:
                if size > self.base_rate * 2:  # Large message warning
                    if self.interactive:
                        logger.warning(f"Large message detected: {size} bytes")

                allowed = await self.can_send(node_id, size)
                if not allowed and self.interactive:
                    logger.warning(f"Rate limit exceeded for node {node_id}")
                return allowed

        except Exception as e:
            logger.error(f"Send check failed: {str(e)}")
            return False
        finally:
            if self.session:
                await self.session.__aexit__(None, None, None)

    async def adjust_rate_interactive(self, congestion_level: float):
        """Interactive rate adjustment with validation"""
        try:
            if not 0 <= congestion_level <= 1:
                raise ValueError("Congestion level must be between 0 and 1")

            old_rate = self.base_rate
            await self.adjust_rate(congestion_level)
            
            self.adjustment_history.append({
                'timestamp': time.time(),
                'old_rate': old_rate,
                'new_rate': self.base_rate,
                'congestion': congestion_level
            })

            if self.interactive:
                logger.info(f"Rate adjusted: {old_rate:.2f} -> {self.base_rate:.2f} B/s")

        except Exception as e:
            logger.error(f"Rate adjustment failed: {str(e)}")
            raise

    async def cleanup(self):
        """Cleanup resources"""
        try:
            self._cleanup_event.set()
            self.usage.clear()
            self.adjustment_history.clear()
            logger.info("Adaptive rate limiter cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")

    def request_shutdown(self):
        """Request graceful shutdown"""
        self._interrupt_requested = True
        logger.info("Shutdown requested for rate limiter")
