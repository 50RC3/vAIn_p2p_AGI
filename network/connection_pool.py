import asyncio
from typing import Dict, Optional, Tuple
import time
import logging
from dataclasses import dataclass
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

class Connection:
    def __init__(self, peer_id: str):
        self.peer_id = peer_id
        self.writer: Optional[asyncio.StreamWriter] = None
        self.reader: Optional[asyncio.StreamReader] = None
        self.last_used = time.time()
        self.metrics = ConnectionMetrics(
            created_at=time.time(),
            last_used=time.time()
        )
        self.is_healthy = True

    def update_usage(self):
        self.last_used = time.time()
        self.metrics.last_used = time.time()

class ConnectionPool:
    def __init__(self, max_connections: int = 100, ttl: int = 300, 
                 interactive: bool = True):
        self.connections: Dict[str, Connection] = {}
        self.max_connections = max_connections
        self.ttl = ttl
        self.interactive = interactive
        self.session: Optional[InteractiveSession] = None
        self._cleanup_event = asyncio.Event()
        self._interrupt_requested = False
        self._monitor_task = None

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

    async def get_connection(self, peer_id: str) -> Optional[Connection]:
        try:
            if peer_id in self.connections:
                conn = self.connections[peer_id]
                if not await self._validate_connection(conn):
                    await self._close_connection(peer_id)
                else:
                    conn.update_usage()
                    return conn
            return await self._create_connection(peer_id)
        except Exception as e:
            logger.error(f"Error getting connection to {peer_id}: {str(e)}")
            return None

    async def _create_connection(self, peer_id: str) -> Optional[Connection]:
        if len(self.connections) >= self.max_connections:
            if self.interactive and self.session:
                proceed = await self.session.confirm_with_timeout(
                    "Max connections reached. Clean up old connections?",
                    timeout=INTERACTION_TIMEOUTS["emergency"]
                )
                if proceed:
                    await self._cleanup_old_connections()
                else:
                    return None
            else:
                await self._cleanup_old_connections()

        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(peer_id, 8000),
                timeout=INTERACTION_TIMEOUTS["default"]
            )
            conn = Connection(peer_id)
            conn.reader = reader
            conn.writer = writer
            self.connections[peer_id] = conn
            logger.info(f"Created new connection to {peer_id}")
            return conn

        except asyncio.TimeoutError:
            logger.error(f"Connection timeout to {peer_id}")
            return None
        except Exception as e:
            logger.error(f"Failed to create connection to {peer_id}: {str(e)}")
            return None

    async def _validate_connection(self, conn: Connection) -> bool:
        if not conn.writer or conn.writer.is_closing():
            return False
            
        if time.time() - conn.last_used > self.ttl:
            return False

        try:
            # Simple health check
            conn.writer.write(b"ping")
            await conn.writer.drain()
            return True
        except:
            conn.is_healthy = False
            return False

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
                
                # Regular cleanup
                await self._cleanup_old_connections()
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"Error in connection monitor: {str(e)}")

    async def _close_connection(self, peer_id: str):
        if peer_id in self.connections:
            try:
                conn = self.connections[peer_id]
                if conn.writer:
                    conn.writer.close()
                    await conn.writer.wait_closed()
                del self.connections[peer_id]
                logger.info(f"Closed connection to {peer_id}")
            except Exception as e:
                logger.error(f"Error closing connection to {peer_id}: {str(e)}")
