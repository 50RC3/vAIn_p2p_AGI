import asyncio
from typing import Dict, Optional, Tuple, DefaultDict, Set, List
from collections import defaultdict
import time
import logging
from dataclasses import dataclass, field
import psutil
import errno
import socket
from core.interactive_utils import InteractiveSession, InteractiveConfig, InteractionLevel
from core.constants import INTERACTION_TIMEOUTS
from .connection_optimizations import TransportOptimizer, optimize_socket_buffers
from .circuit_breaker import CircuitBreaker

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
                 interactive: bool = True, default_timeouts: Optional[Dict[str, float]] = None,
                 max_per_peer: int = 5):
        self.connections: Dict[str, Connection] = {}
        self.max_connections = max_connections
        self.ttl = ttl
        self.interactive = interactive
        self.session: Optional[InteractiveSession] = None
        self._cleanup_event = asyncio.Event()
        self._interrupt_requested = False
        self._monitor_task = None
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            reset_timeout=30,
            interactive=interactive,
            service_name="connection_pool"
        )
        self.retry_config = {
            'max_retries': 3,
            'base_delay': 1,
            'max_delay': 10,
            'backoff_factor': 2
        }
        self.health_check_interval = 60
        self._last_health_check = 0
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
        self.max_per_peer = max_per_peer
        self.peer_connections: DefaultDict[str, Set[Connection]] = defaultdict(set)
        self.connection_queue: asyncio.Queue[Tuple[str, asyncio.Future]] = asyncio.Queue()
        self._queue_processor = None
        self.transport_optimizer = TransportOptimizer()

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
            self._queue_processor = asyncio.create_task(self._process_connection_queue())
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
            if not await self.circuit_breaker.allow_request_interactive():
                logger.warning(f"Circuit breaker preventing connection to {peer_id}")
                return None
                
            conn = await super().get_connection(peer_id, custom_timeouts)
            if not conn and self.interactive:
                # Try failover nodes
                for failover_id in self._get_failover_peers(peer_id):
                    logger.info(f"Attempting failover to {failover_id}")
                    if conn := await super().get_connection(failover_id, custom_timeouts):
                        break
                        
            if not conn:
                await self.circuit_breaker.record_failure_interactive()
                
            return conn

        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
            await self.circuit_breaker.record_failure_interactive() 
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
                
                # Optimize socket settings
                sock = writer.transport.get_extra_info('socket')
                if sock:
                    optimize_socket_buffers(sock)
                    self.transport_optimizer.optimize_tcp_connection(sock)
                
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

    async def _process_connection_queue(self):
        while not self._cleanup_event.is_set():
            try:
                peer_id, future = await self.connection_queue.get()
                if len(self.peer_connections[peer_id]) < self.max_per_peer:
                    conn = await self._create_connection(peer_id)
                    if conn:
                        self.peer_connections[peer_id].add(conn)
                        if not future.done():
                            future.set_result(conn)
                self.connection_queue.task_done()
            except Exception as e:
                logger.error(f"Error processing connection queue: {e}")
            await asyncio.sleep(0.1)

    async def _close_connection(self, peer_id: str):
        if peer_id in self.connections:
            try:
                conn = self.connections[peer_id]
                self.peer_connections[peer_id].discard(conn)
                if conn.writer:
                    conn.writer.close()
                    await conn.writer.wait_closed()
                del self.connections[peer_id]
                self.grace_periods.pop(peer_id, None)  # Clean up grace period tracking
                logger.info(f"Closed connection to {peer_id}")
                if not self.peer_connections[peer_id]:
                    del self.peer_connections[peer_id]
            except Exception as e:
                logger.error(f"Error closing connection to {peer_id}: {str(e)}")

    async def _check_pool_health(self) -> bool:
        """Periodic health check of connection pool"""
        try:
            current_time = time.time()
            if current_time - self._last_health_check < self.health_check_interval:
                return True

            self._last_health_check = current_time
            
            # Check connection states
            unhealthy = 0
            for conn in self.connections.values():
                if not await self._validate_connection(conn):
                    unhealthy += 1

            unhealthy_ratio = unhealthy / max(1, len(self.connections))
            if unhealthy_ratio > 0.5:  # Over 50% unhealthy
                logger.warning(f"Pool health check failed: {unhealthy_ratio:.1%} unhealthy")
                await self._adaptive_cleanup()
                return False

            return True

        except Exception as e:
            logger.error(f"Health check error: {str(e)}")
            return False

    def _get_failover_peers(self, failed_peer: str) -> List[str]:
        """Get list of failover peers for a failed peer"""
        # Simple round-robin failover
        return [p for p in self.connections.keys() if p != failed_peer]
