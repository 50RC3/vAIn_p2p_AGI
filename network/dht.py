import asyncio
from kademlia.network import Server
from typing import Dict, Optional, List, Set
import logging
from collections import OrderedDict

class DHT:
    def __init__(self, node_id: str, config: Dict):
        self.node_id = node_id
        self.server = Server()
        self.bootstrap_nodes = config['bootstrap_nodes']
        self.port = config.get('port', 8468)
        self.logger = logging.getLogger('DHT')
        self.k_bucket_size = config.get('k_bucket_size', 20)
        self.routing_table: Dict[str, OrderedDict] = {}
        self.data_store: Dict[str, str] = {}
        self.geo_buckets: Dict[str, Set[str]] = {}  # Geographic routing
        self.last_seen: Dict[str, float] = {}
        self.peer_reputation: Dict[str, float] = {}
        
    async def start(self):
        """Start DHT node and join network."""
        self.logger.info(f"Starting DHT node {self.node_id}")
        await self.server.listen(self.port)
        await self._join_network()
        await self._bootstrap()
        
    async def stop(self):
        """Clean up DHT resources."""
        self.logger.info(f"Stopping DHT node {self.node_id}")
        self.server.stop()
        
    async def lookup(self, key: str) -> Optional[str]:
        """Look up node address in DHT."""
        if key in self.data_store:
            return self.data_store[key]
        return await self.server.get(key)
        
    async def store(self, key: str, value: str):
        """Store key-value pair in DHT."""
        self.data_store[key] = value
        await self.server.set(key, value)
        
    async def _join_network(self):
        """Join DHT network using bootstrap nodes."""
        for node in self.bootstrap_nodes:
            try:
                host, port = node.split(':')
                await self.server.bootstrap([(host, int(port))])
            except Exception as e:
                self.logger.error(f"Failed to bootstrap with node {node}: {e}")
                
    async def _bootstrap(self):
        """Bootstrap DHT node"""
        pass

    async def discover(self) -> Set[str]:
        """Enhanced peer discovery with reputation and geographic routing"""
        active_peers = set()
        
        # Get peers from each geographic region
        for region, peers in self.geo_buckets.items():
            regional_peers = await self._discover_region(region, peers)
            active_peers.update(regional_peers)
            
        # Filter by reputation
        return {
            peer for peer in active_peers
            if self.peer_reputation.get(peer, 0.5) >= 0.3
        }
        
    async def _discover_region(self, region: str, peers: Set[str]) -> Set[str]:
        """Discover peers in a specific geographic region"""
        active = set()
        for peer in peers:
            if await self._verify_peer_status(peer):
                active.add(peer)
                
        # Update routing table
        self.geo_buckets[region] = active
        return active
        
    async def _verify_peer_status(self, peer_id: str) -> bool:
        """Verify peer status with hardware attestation"""
        try:
            status = await self.server.get_peer_status(peer_id)
            if not status:
                return False
                
            # Verify hardware attestation
            if not await self._verify_hardware_attestation(peer_id, status):
                self.logger.warning(f"Failed hardware attestation for {peer_id}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Peer verification failed for {peer_id}: {e}")
            return False

    async def _ping_peer(self, peer_id: str) -> bool:
        """Check if peer is alive"""
        try:
            result = await self.server.ping(peer_id)
            if result:
                self.last_seen[peer_id] = asyncio.get_event_loop().time()
            return result
        except:
            return False

    async def _update_routing(self):
        """Update routing table based on reputation"""
        for peer, rep in self.peer_reputation.items():
            region = self._get_peer_region(peer)
            if rep > 0.5:  # Only route through reputable peers
                self.geo_buckets.setdefault(region, set()).add(peer)
