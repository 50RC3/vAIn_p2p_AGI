from typing import Set, Dict, List
import time
import random

class PeerExchange:
    def __init__(self, max_peers: int = 1000):
        self.peers: Dict[str, float] = {}  # peer_id -> last_seen
        self.max_peers = max_peers
        
    def add_peer(self, peer_id: str):
        self.peers[peer_id] = time.time()
        if len(self.peers) > self.max_peers:
            self._prune_old_peers()
            
    def get_peers(self, n: int = 10) -> List[str]:
        active_peers = [p for p, t in self.peers.items() 
                       if time.time() - t < 3600]  # Active in last hour
        return random.sample(active_peers, min(n, len(active_peers)))
        
    def _prune_old_peers(self, max_age: float = 3600 * 24):
        current_time = time.time()
        self.peers = {
            peer: last_seen for peer, last_seen in self.peers.items()
            if current_time - last_seen < max_age
        }
