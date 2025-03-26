import asyncio
import random
from typing import Dict, Set, List

class GossipManager:
    def __init__(self, fanout: int = 3, rounds: int = 2):
        self.fanout = fanout
        self.rounds = rounds
        self.message_rounds: Dict[str, int] = {}
        self.seen_messages: Set[str] = set()
        
    def select_peers(self, peers: List[str], exclude: Set[str]) -> List[str]:
        """Select random subset of peers for gossip"""
        available = [p for p in peers if p not in exclude]
        count = min(self.fanout, len(available))
        return random.sample(available, count)
        
    async def should_propagate(self, msg_id: str) -> bool:
        """Determine if message should continue propagating"""
        if msg_id not in self.message_rounds:
            self.message_rounds[msg_id] = 0
            return True
            
        self.message_rounds[msg_id] += 1
        return self.message_rounds[msg_id] < self.rounds
