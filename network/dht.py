import asyncio
from kademlia.network import Server
from typing import Dict, Optional, List, Set, Tuple
import logging
from collections import OrderedDict
from tqdm import tqdm
from core.interactive_utils import InteractiveSession, InteractiveConfig, InteractionLevel
from core.constants import INTERACTION_TIMEOUTS

class DHT:
    def __init__(self, node_id: str, config: Dict, interactive: bool = True):
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
        self.interactive = interactive
        self.session: Optional[InteractiveSession] = None
        self._interrupt_requested = False
        self._cleanup_tasks: Set[asyncio.Task] = set()

    async def start(self):
        """Start DHT node and join network with interactive progress."""
        try:
            if self.interactive:
                self.session = InteractiveSession(
                    level=InteractionLevel.NORMAL,
                    config=InteractiveConfig(
                        timeout=INTERACTION_TIMEOUTS["default"],
                        persistent_state=True,
                        safe_mode=True
                    )
                )
                
            async with self.session if self.session else asyncio.nullcontext():
                self.logger.info(f"Starting DHT node {self.node_id}")
                
                if self.interactive:
                    print("\nInitializing DHT Node")
                    print("=" * 50)
                    
                await self.server.listen(self.port)
                await self._join_network_interactive()
                await self._bootstrap()
                
        except Exception as e:
            self.logger.error(f"Failed to start DHT node: {e}")
            raise
        finally:
            if self.session:
                await self.session.__aexit__(None, None, None)

    async def stop(self):
        """Clean up DHT resources with progress tracking."""
        try:
            self._interrupt_requested = True
            
            if self.interactive:
                print("\nStopping DHT Node")
                print("-" * 30)
            
            # Cancel pending tasks
            for task in self._cleanup_tasks:
                task.cancel()
            
            await asyncio.gather(*self._cleanup_tasks, return_exceptions=True)
            self.server.stop()
            
            self.logger.info(f"DHT node {self.node_id} stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error during DHT shutdown: {e}")
            raise

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
        """Enhanced peer discovery with interactive progress."""
        try:
            if self.interactive:
                self.session = InteractiveSession(
                    level=InteractionLevel.NORMAL,
                    config=InteractiveConfig(
                        timeout=INTERACTION_TIMEOUTS["default"],
                        persistent_state=True
                    )
                )

            async with self.session if self.session else asyncio.nullcontext():
                active_peers = set()
                regions = list(self.geo_buckets.keys())
                
                with tqdm(total=len(regions), desc="Discovering Peers") as pbar:
                    for region in regions:
                        if self._interrupt_requested:
                            break
                            
                        peers = await self._discover_region(region, self.geo_buckets[region])
                        active_peers.update(peers)
                        pbar.update(1)
                        
                        # Save progress periodically
                        if len(active_peers) % 10 == 0 and self.session:
                            await self._save_discovery_progress(active_peers)

                return {
                    peer for peer in active_peers 
                    if self.peer_reputation.get(peer, 0.5) >= 0.3
                }

        except Exception as e:
            self.logger.error(f"Peer discovery failed: {e}")
            raise
        finally:
            if self.session:
                await self.session.__aexit__(None, None, None)

    async def _save_discovery_progress(self, peers: Set[str]) -> None:
        """Save discovery progress for recovery."""
        if not self.session:
            return
            
        try:
            await self.session.save_progress({
                'discovered_peers': list(peers),
                'timestamp': asyncio.get_event_loop().time()
            })
        except Exception as e:
            self.logger.error(f"Failed to save discovery progress: {e}")

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

    async def _join_network_interactive(self):
        """Join DHT network with progress tracking and timeout handling."""
        if not self.bootstrap_nodes:
            self.logger.warning("No bootstrap nodes configured")
            return

        successful = 0
        with tqdm(total=len(self.bootstrap_nodes), desc="Joining Network") as pbar:
            for node in self.bootstrap_nodes:
                if self._interrupt_requested:
                    break
                    
                try:
                    async with asyncio.timeout(INTERACTION_TIMEOUTS["default"]):
                        host, port = node.split(':')
                        await self.server.bootstrap([(host, int(port))])
                        successful += 1
                        
                except asyncio.TimeoutError:
                    self.logger.warning(f"Bootstrap timeout for node {node}")
                except Exception as e:
                    self.logger.error(f"Failed to bootstrap with node {node}: {e}")
                finally:
                    pbar.update(1)

        if successful == 0:
            raise RuntimeError("Failed to connect to any bootstrap nodes")
