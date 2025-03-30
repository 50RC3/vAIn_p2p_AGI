import asyncio
import logging
from typing import Dict, Optional
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Connection:
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    created_at: float
    last_used: float

class ConnectionPool:
    def __init__(self, max_connections: int = 100, ttl: int = 300):
        """Initialize connection pool
        
        Args:
            max_connections: Maximum number of connections to maintain
            ttl: Time-to-live for connections in seconds
        """
        self.max_connections = max_connections
        self.ttl = ttl
        self.connections: Dict[str, Connection] = {}
        self.lock = asyncio.Lock()
        self._cleanup_task = None
        self._interrupt_requested = False

    async def get_connection(self, peer_id: str) -> Optional[Connection]:
        """Get or create a connection to peer"""
        async with self.lock:
            if peer_id in self.connections:
                conn = self.connections[peer_id]
                conn.last_used = time.time()
                return conn
                
            # If connection doesn't exist, create a new one
            try:
                reader, writer = await asyncio.open_connection(peer_id, 8468)
                conn = Connection(
                    reader=reader,
                    writer=writer,
                    created_at=time.time(),
                    last_used=time.time()
                )
                self.connections[peer_id] = conn
                
                # If pool is full, remove oldest connection
                if len(self.connections) > self.max_connections:
                    oldest_peer = min(
                        self.connections.keys(),
                        key=lambda p: self.connections[p].last_used
                    )
                    await self._close_connection(oldest_peer)
                    
                return conn
                
            except Exception as e:
                logger.error(f"Failed to create connection to {peer_id}: {e}")
                return None

    async def _close_connection(self, peer_id: str):
        """Close a specific connection"""
        conn = self.connections.pop(peer_id, None)
        if conn:
            conn.writer.close()
            try:
                await conn.writer.wait_closed()
            except Exception:
                pass

    async def start_cleanup_task(self):
        """Start background task to clean up expired connections"""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self):
        """Periodically clean up expired connections"""
        while not self._interrupt_requested:
            try:
                # Wait for a while
                await asyncio.sleep(60)  # Check every minute
                
                # Clean up expired connections
                current_time = time.time()
                expired_peers = [
                    peer_id for peer_id, conn in self.connections.items()
                    if current_time - conn.last_used > self.ttl
                ]
                
                for peer_id in expired_peers:
                    await self._close_connection(peer_id)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in connection cleanup: {e}")

    async def cleanup(self):
        """Clean up all connections and tasks"""
        self._interrupt_requested = True
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        peer_ids = list(self.connections.keys())
        for peer_id in peer_ids:
            await self._close_connection(peer_id)
            
        logger.info("Connection pool cleanup completed")

    def force_close(self):
        """Force close all connections without waiting (for emergency shutdown)"""
        for conn in self.connections.values():
            try:
                conn.writer.close()
            except Exception:
                pass
        self.connections.clear()
        logger.warning("Force closed all connections")