import asyncio
from typing import Dict, Optional, Tuple, DefaultDict
from collections import defaultdict
import time
import logging
from dataclasses import dataclass, field
import psutil
import errno
import socket
from core.interactive_utils import InteractiveSession, InteractiveConfig, InteractionLevel
from core.constants import INTERACTION_TIMEOUTS

logger = logging.getLogger(__name__)

@dataclass 
class ConnectionMetrics:
    created_at: float
    last_used: float
    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    error_count: int = 0
    reconnect_attempts: int = 0
    error_types: DefaultDict[str, int] = field(default_factory=lambda: defaultdict(int))
    timeouts: DefaultDict[str, float] = field(default_factory=lambda: defaultdict(float))

class Connection:
    def __init__(self, peer_id: str, custom_timeouts: Optional[Dict[str, float]] = None):
        self.peer_id = peer_id
        self.writer: Optional[asyncio.StreamWriter] = None
        self.reader: Optional[asyncio.StreamReader] = None
        self.last_used = time.time()
        self.metrics = ConnectionMetrics(
            created_at=time.time(),
            last_used=time.time()
        )
        self.is_healthy = True
        self.timeouts = {**INTERACTION_TIMEOUTS, **(custom_timeouts or {})}

    def update_usage(self):
        self.last_used = time.time()
        self.metrics.last_used = time.time()

class ConnectionPool:
    def __init__(self, max_connections: int = 100, ttl: int = 300, 
                 interactive: bool = True, default_timeouts: Optional[Dict[str, float]] = None):
        self.connections: Dict[str, Connection] = {}
        self.max_connections = max_connections
        self.ttl = ttl
        self.interactive = interactive
        self.session: Optional[InteractiveSession] = None
        self._cleanup_event = asyncio.Event()
        self._interrupt_requested = False
        self._monitor_task = None
        self.retry_config = {
            'max_retries': 3,
            'base_delay': 1.0,
            'max_delay': 30.0,
            'backoff_factor': 2.0
        }
        self.cleanup_thresholds = {
            'memory_percent': 85.0,  # Trigger aggressive cleanup at 85% memory
            'unhealthy_ratio': 0.3,  # Cleanup when >30% connections unhealthy
            'min_idle_time': 60,     # Minimum idle time for cleanup
            'max_idle_time': 600,    # Maximum idle time before forced cleanup
            'grace_period': 30,      # Grace period before closing inactive connections
            'max_grace_periods': 3    # Maximum number of grace periods per connection
        }
        self.grace_periods: Dict[str, int] = {}  # Track grace periods used per connection
        self.monitor_interval = 60  # Initial monitor interval
        self._timeout_history = []
        self.default_timeouts = {**INTERACTION_TIMEOUTS, **(default_timeouts or {})}

    async def __aenter__(self):
        if self.interactive:
            self.session = InteractiveSession(
                level=InteractionLevel.NORMAL,
                config=InteractiveConfig(
                    timeout=INTERACTION_TIMEOUTS["batch"],
                    persistent_state=True,
                    safe_mode=True
                )
            )
            await self.session.__aenter__()
            self._monitor_task = asyncio.create_task(self._monitor_connections())
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._interrupt_requested = True
        if self._monitor_task:
            self._cleanup_event.set()
            await self._monitor_task
        await self._cleanup_all_connections()
        if self.session:
            await self.session.__aexit__(exc_type, exc, tb)

    async def get_connection(self, peer_id: str, custom_timeouts: Optional[Dict[str, float]] = None) -> Optional[Connection]:
        try:
            if peer_id in self.connections:
                conn = self.connections[peer_id]
                if not await self._validate_connection(conn):
                    await self._close_connection(peer_id)
                else:
                    conn.update_usage()
                    return conn
            return await self._create_connection(peer_id, custom_timeouts)
        except Exception as e:
            logger.error(f"Error getting connection to {peer_id}: {str(e)}")
            return None

    async def _create_connection(self, peer_id: str, custom_timeouts: Optional[Dict[str, float]] = None) -> Optional[Connection]:
        if len(self.connections) >= self.max_connections:
            await self._adaptive_cleanup()

        retries = 0
        last_error = None
        timeouts = {**self.default_timeouts, **(custom_timeouts or {})}
        
        while retries < self.retry_config['max_retries']:
            try:
                delay = min(
                    self.retry_config['base_delay'] * (self.retry_config['backoff_factor'] ** retries),
                    self.retry_config['max_delay']
                )
                if retries > 0:
                    await asyncio.sleep(delay)
                    logger.info(f"Retry {retries} connecting to {peer_id} after {delay:.1f}s delay")

                timeout = self._calculate_dynamic_timeout()
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(peer_id, 8000),
                    timeout=timeout
                )
                conn = Connection(peer_id, custom_timeouts=timeouts)
                conn.reader = reader
                conn.writer = writer
                self.connections[peer_id] = conn
                logger.info(f"Created new connection to {peer_id}")
                self._update_timeout_stats(timeout, success=True)
                return conn

            except asyncio.TimeoutError:
                last_error = "Connection timeout"
                self._update_timeout_stats(timeout, success=False)
                conn.metrics.error_types["timeout"] += 1
            except socket.gaierror as e:
                last_error = f"DNS resolution error: {e}"
                conn.metrics.error_types["dns"] += 1
            except ConnectionRefusedError:
                last_error = "Connection refused"
                conn.metrics.error_types["refused"] += 1
            except OSError as e:
                if e.errno == errno.ENETUNREACH:
                    last_error = "Network unreachable"
                    conn.metrics.error_types["unreachable"] += 1
                else:
                    last_error = f"Network error: {e}"
                    conn.metrics.error_types["other"] += 1
            except Exception as e:
                last_error = str(e)
                conn.metrics.error_types["unknown"] += 1
            retries += 1
            conn.metrics.reconnect_attempts += 1

        logger.error(f"Failed to connect to {peer_id} after {retries} retries. Last error: {last_error}")
        return None

    async def _validate_connection(self, conn: Connection) -> bool:
        if not conn.writer or conn.writer.is_closing():
            return False
            
        idle_time = time.time() - conn.last_used
        if idle_time > self.ttl:
            # Check if connection is in grace period
            grace_used = self.grace_periods.get(conn.peer_id, 0)
            if grace_used < self.cleanup_thresholds['max_grace_periods']:
                if idle_time <= self.ttl + self.cleanup_thresholds['grace_period']:
                    self.grace_periods[conn.peer_id] = grace_used + 1
                    logger.debug(f"Connection {conn.peer_id} in grace period {grace_used + 1}")
                    return True
            return False

        # Reset grace period count when connection is active
        if conn.peer_id in self.grace_periods:
            del self.grace_periods[conn.peer_id]

        try:
            # Simple health check
            conn.writer.write(b"ping")
            await conn.writer.drain()
            return True
        except:
            conn.is_healthy = False
            return False

    async def _adaptive_cleanup(self):
        """Perform adaptive cleanup based on system and connection metrics"""
        try:
            memory_percent = psutil.virtual_memory().percent
            unhealthy_ratio = 1 - (sum(1 for c in self.connections.values() if c.is_healthy) / 
                                 max(1, len(self.connections)))
            
            # Adjust cleanup aggressiveness based on conditions
            if memory_percent > self.cleanup_thresholds['memory_percent']:
                idle_threshold = self.cleanup_thresholds['min_idle_time']
                logger.warning(f"High memory usage ({memory_percent:.1f}%), performing aggressive cleanup")
            elif unhealthy_ratio > self.cleanup_thresholds['unhealthy_ratio']:
                idle_threshold = self.cleanup_thresholds['min_idle_time'] * 2
                logger.warning(f"High unhealthy connection ratio ({unhealthy_ratio:.1f}), performing cleanup")
            else:
                idle_threshold = self.cleanup_thresholds['max_idle_time']

            current_time = time.time()
            for peer_id, conn in list(self.connections.items()):
                if current_time - conn.last_used > idle_threshold or not conn.is_healthy:
                    await self._close_connection(peer_id)

        except Exception as e:
            logger.error(f"Error during adaptive cleanup: {str(e)}")

    async def _cleanup_old_connections(self):
        current_time = time.time()
        for peer_id, conn in list(self.connections.items()):
            if current_time - conn.last_used > self.ttl or not conn.is_healthy:
                logger.info(f"Cleaning up old connection to {peer_id}")
                await self._close_connection(peer_id)

    async def _cleanup_all_connections(self):
        logger.info("Cleaning up all connections")
        for peer_id in list(self.connections.keys()):
            await self._close_connection(peer_id)

    def _calculate_dynamic_timeout(self) -> float:
        """Calculate dynamic timeout based on recent connection history and error patterns"""
        if not self._timeout_history:
            return self.default_timeouts["default"]
        
        # Use recent history to adjust timeout
        success_rate = sum(1 for _, success in self._timeout_history[-10:] if success) / 10
        base_timeout = self.default_timeouts["default"]
        
        if success_rate < 0.5:
            # Increase timeout when success rate is low
            return min(base_timeout * 2, self.default_timeouts["batch"])
        elif success_rate > 0.8:
            # Decrease timeout when success rate is high
            return max(base_timeout / 1.5, self.default_timeouts["emergency"])
        return base_timeout

    def _update_timeout_stats(self, timeout: float, success: bool):
        """Update timeout statistics"""
        self._timeout_history.append((timeout, success))
        self._timeout_history = self._timeout_history[-20:]  # Keep last 20 entries

    async def _monitor_connections(self):
        while not self._cleanup_event.is_set():
            try:
                metrics = {
                    "total_connections": len(self.connections),
                    "healthy_connections": sum(1 for c in self.connections.values() if c.is_healthy),
                    "total_bytes_sent": sum(c.metrics.total_bytes_sent for c in self.connections.values()),
                    "total_bytes_received": sum(c.metrics.total_bytes_received for c in self.connections.values())
                }
                logger.debug(f"Connection pool metrics: {metrics}")
                
                # Adjust monitoring interval based on connection health
                unhealthy_ratio = 1 - (metrics["healthy_connections"] / max(1, metrics["total_connections"]))
                if unhealthy_ratio > 0.3:
                    self.monitor_interval = max(15, self.monitor_interval / 2)  # More frequent monitoring
                else:
                    self.monitor_interval = min(300, self.monitor_interval * 1.5)  # Less frequent monitoring

                await self._adaptive_cleanup()
                await asyncio.sleep(self.monitor_interval)
                
            except Exception as e:
                logger.error(f"Error in connection monitor: {str(e)}")
                await asyncio.sleep(60)  # Fallback interval on error

    async def _close_connection(self, peer_id: str):
        if peer_id in self.connections:
            try:
                conn = self.connections[peer_id]
                if conn.writer:
                    conn.writer.close()
                    await conn.writer.wait_closed()
                del self.connections[peer_id]
                self.grace_periods.pop(peer_id, None)  # Clean up grace period tracking
                logger.info(f"Closed connection to {peer_id}")
            except Exception as e:
                logger.error(f"Error closing connection to {peer_id}: {str(e)}")
