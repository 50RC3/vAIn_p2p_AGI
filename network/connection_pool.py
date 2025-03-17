import asyncio
from typing import Dict, Optional
import time

class Connection:
    def __init__(self, peer_id: str):
        self.peer_id = peer_id
        self.writer: Optional[asyncio.StreamWriter] = None
        self.reader: Optional[asyncio.StreamReader] = None
        self.last_used = time.time()

class ConnectionPool:
    def __init__(self, max_connections: int = 100, ttl: int = 300):
        self.connections: Dict[str, Connection] = {}
        self.max_connections = max_connections
        self.ttl = ttl
        
    async def get_connection(self, peer_id: str) -> Optional[Connection]:
        if peer_id in self.connections:
            conn = self.connections[peer_id]
            if time.time() - conn.last_used > self.ttl:
                await self._close_connection(peer_id)
            else:
                return conn
        return await self._create_connection(peer_id)
        
    async def _create_connection(self, peer_id: str) -> Optional[Connection]:
        if len(self.connections) >= self.max_connections:
            await self._cleanup_old_connections()
            
        try:
            reader, writer = await asyncio.open_connection(peer_id, 8000)
            conn = Connection(peer_id)
            conn.reader = reader
            conn.writer = writer
            self.connections[peer_id] = conn
            return conn
        except:
            return None
            
    async def _cleanup_old_connections(self):
        current_time = time.time()
        for peer_id, conn in list(self.connections.items()):
            if current_time - conn.last_used > self.ttl:
                await self._close_connection(peer_id)
                
    async def _close_connection(self, peer_id: str):
        if peer_id in self.connections:
            conn = self.connections[peer_id]
            if conn.writer:
                conn.writer.close()
                await conn.writer.wait_closed()
            del self.connections[peer_id]
