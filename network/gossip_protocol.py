import asyncio
import random
from typing import Dict, Set, List
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class GossipProtocol:
    """Protocol for efficient p2p message spreading using gossip algorithm"""
    
    def __init__(self, fanout: int = 3, rounds: int = 2, ttl: int = 300):
        """
        Initialize gossip protocol
        
        Args:
            fanout: Number of peers to forward messages to
            rounds: Maximum number of forwarding rounds
            ttl: Time-to-live for messages in seconds
        """
        self.fanout = fanout
        self.rounds = rounds
        self.ttl = ttl
        self.message_rounds: Dict[str, int] = {}
        self.seen_messages: Set[str] = set()
        self.message_timestamps: Dict[str, float] = {}
        self.propagation_stats = {
            'forwarded': 0,
            'dropped': 0,
            'expired': 0
        }
        self._cleanup_task = None
        
    def select_peers(self, peers: List[str], exclude: Set[str]) -> List[str]:
        """
        Select random subset of peers for gossip
        
        Args:
            peers: List of all available peers
            exclude: Set of peers to exclude (e.g., sender)
            
        Returns:
            List of selected peers for message forwarding
        """
        available = [p for p in peers if p not in exclude]
        count = min(self.fanout, len(available))
        if count == 0:
            return []
        return random.sample(available, count)
        
    async def should_propagate(self, msg_id: str) -> bool:
        """
        Determine if message should continue propagating
        
        Args:
            msg_id: Unique message identifier
            
        Returns:
            True if message should be propagated, False otherwise
        """
        # If already seen, don't propagate
        if msg_id in self.seen_messages:
            self.propagation_stats['dropped'] += 1
            return False
            
        # Track message
        self.seen_messages.add(msg_id)
        
        # Initialize round counter
        if msg_id not in self.message_rounds:
            self.message_rounds[msg_id] = 0
            self.message_timestamps[msg_id] = asyncio.get_event_loop().time()
            return True
            
        # Increment round counter
        self.message_rounds[msg_id] += 1
        
        # Check if maximum rounds reached
        if self.message_rounds[msg_id] < self.rounds:
            self.propagation_stats['forwarded'] += 1
            return True
        
        self.propagation_stats['dropped'] += 1
        return False
        
    async def cleanup_expired_messages(self):
        """Remove expired messages to prevent memory leaks"""
        now = asyncio.get_event_loop().time()
        expired_msgs = [
            msg_id for msg_id, timestamp in self.message_timestamps.items()
            if now - timestamp > self.ttl
        ]
        
        for msg_id in expired_msgs:
            self.seen_messages.discard(msg_id)
            self.message_rounds.pop(msg_id, None)
            self.message_timestamps.pop(msg_id, None)
            self.propagation_stats['expired'] += 1
            
    async def start_cleanup_task(self):
        """Start background task to clean up expired messages"""
        async def cleanup_worker():
            while True:
                try:
                    await asyncio.sleep(60)  # Clean up every minute
                    await self.cleanup_expired_messages()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in gossip cleanup: {str(e)}")
        
        self._cleanup_task = asyncio.create_task(cleanup_worker())
        
    async def stop(self):
        """Stop cleanup task"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            
    def get_metrics(self) -> Dict:
        """Get protocol metrics"""
        return {
            **self.propagation_stats,
            'tracked_messages': len(self.seen_messages),
            'active_rounds': len(self.message_rounds)
        }

# For backward compatibility
GossipManager = GossipProtocol  # Alias for existing code that uses GossipManager
