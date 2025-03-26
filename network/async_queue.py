import asyncio
import time
from typing import Dict, Any, Optional, Set
from collections import defaultdict

class AsyncMessageQueue:
    def __init__(self, ttl: int = 300):
        self.pending_messages: Dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)
        self.seen_messages: Set[str] = set()
        self.message_timestamps: Dict[str, float] = {}
        self.ttl = ttl
        
    async def put(self, msg_id: str, message: Dict[str, Any]) -> None:
        """Add message to queue if not seen before"""
        if msg_id not in self.seen_messages:
            self.seen_messages.add(msg_id)
            self.message_timestamps[msg_id] = time.time()
            await self.pending_messages[msg_id].put(message)
            
    async def get(self) -> Optional[Dict[str, Any]]:
        """Get next message non-blockingly"""
        for msg_id, queue in self.pending_messages.items():
            if not queue.empty():
                message = queue.get_nowait()
                return message
        return None
        
    async def cleanup_old_messages(self):
        """Remove expired messages"""
        current_time = time.time()
        expired = [msg_id for msg_id, ts in self.message_timestamps.items() 
                  if current_time - ts > self.ttl]
        for msg_id in expired:
            self.seen_messages.discard(msg_id)
            self.message_timestamps.pop(msg_id)
            self.pending_messages.pop(msg_id)
