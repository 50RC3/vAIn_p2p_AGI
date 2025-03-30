import asyncio
from kademlia.network import Server
from typing import Dict, Optional, List, Set, Tuple, Any, DefaultDict
import logging
from collections import OrderedDict, defaultdict
from tqdm import tqdm
from core.interactive_utils import InteractiveSession, InteractiveConfig, InteractionLevel
from core.constants import INTERACTION_TIMEOUTS

# Custom exceptions
class DHTError(Exception):
    """Base exception for DHT errors"""
    pass

class BootstrapError(DHTError):
    """Error during bootstrap process"""
    pass

class PeerDiscoveryError(DHTError):
    """Error during peer discovery"""
    pass

class RoutingZone:
    def __init__(self, zone_id: str):
        self.zone_id = zone_id
        self.peers: Set[str] = set()
        self.relay_nodes: Set[str] = set()
        self.load_metrics: Dict[str, float] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.last_rebalance = 0.0

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
        # Add retry configuration
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 1.0)
        self.backoff_factor = config.get('backoff_factor', 1.5)
        
        # Add zone-based routing
        self.zones: Dict[str, RoutingZone] = {}
        self.zone_cache: OrderedDict = OrderedDict()  # LRU cache for zone lookups
        self.max_zone_cache_size = config.get('max_zone_cache_size', 1000)
        self.zone_rebalance_interval = config.get('zone_rebalance_interval', 300)  # 5 minutes
        self.load_thresholds = {
            'cpu': config.get('cpu_threshold', 80),
            'bandwidth': config.get('bandwidth_threshold', 90),
            'peers': config.get('peers_per_zone', 100)
        }
        self.zone_stats: DefaultDict[str, Dict] = defaultdict(dict)

    async def start(self):
        """Start DHT node and join network"""
        try:
            await self.server.listen(self.port)
            await self._join_network_interactive()
        except Exception as e:
            self.logger.error(f"Failed to start DHT: {e}")
            raise

    async def stop(self):
        """Stop DHT node"""
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
        """Enhanced peer discovery with interactive progress and error recovery."""
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
                active_peers = set()
                regions = list(self.geo_buckets.keys())
                failed_regions = []
                
                with tqdm(total=len(regions), desc="Discovering Peers") as pbar:
                    for region in regions:
                        if self._interrupt_requested:
                            break
                            
                        try:
                            async def discover_attempt():
                                peers = await self._discover_region(region, self.geo_buckets[region])
                                return peers

                            peers = await self._retry_with_backoff(discover_attempt)
                            active_peers.update(peers)
                        except Exception as e:
                            self.logger.error(f"Failed to discover peers in region {region}: {e}")
                            failed_regions.append((region, str(e)))
                        finally:
                            pbar.update(1)
                            
                        # Save progress periodically
                        if len(active_peers) % 10 == 0 and self.session:
                            await self._save_discovery_progress(active_peers)

                if not active_peers and failed_regions:
                    error_details = "\n".join(f"- {region}: {error}" for region, error in failed_regions)
                    raise PeerDiscoveryError(
                        f"Failed to discover peers in any region\nDetails:\n{error_details}"
                    )

                return {
                    peer for peer in active_peers 
                    if self.peer_reputation.get(peer, 0.5) >= 0.3
                }

        except Exception as e:
            self.logger.error(f"Peer discovery failed: {e}")
            raise PeerDiscoveryError(f"Peer discovery failed: {str(e)}") from e
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

    async def _verify_hardware_attestation(self, peer_id: str, status: Dict) -> bool:
        """Verify hardware attestation with multiple security measures."""
        try:
            # Get hardware signature and attestation data
            attestation = status.get('hardware_attestation')
            if not attestation:
                return False

            # Verify hardware signature
            if not await self._verify_signature(
                peer_id, 
                attestation.get('signature'),
                attestation.get('public_key')
            ):
                self.logger.warning(f"Invalid hardware signature from peer {peer_id}")
                return False

            # Verify TPM attestation if available
            if attestation.get('tpm_quote'):
                if not await self._verify_tpm_quote(
                    attestation['tpm_quote'],
                    attestation.get('pcr_values', {})
                ):
                    self.logger.warning(f"TPM verification failed for peer {peer_id}")
                    return False

            # Check hardware specifications
            hw_specs = attestation.get('specifications', {})
            if not self._verify_minimum_requirements(hw_specs):
                self.logger.warning(f"Peer {peer_id} does not meet minimum requirements")
                return False

            # Cache successful verification
            await self._cache_verification_result(peer_id, True)
            return True

        except Exception as e:
            self.logger.error(f"Hardware attestation failed for {peer_id}: {e}")
            await self._cache_verification_result(peer_id, False)
            return False

    async def _verify_signature(self, peer_id: str, signature: str, public_key: str) -> bool:
        """Verify hardware signature using public key."""
        try:
            # Add signature verification logic here
            sig_data = base64.b64decode(signature)
            key_data = base64.b64decode(public_key)
            # Implement actual signature verification
            return True  # Placeholder
        except Exception:
            return False

    async def _verify_tpm_quote(self, quote: str, pcr_values: Dict[str, str]) -> bool:
        """Verify TPM quote and PCR values."""
        try:
            # Verify PCR measurements against expected values
            expected_pcrs = {
                'PCR0': 'boot_secure',
                'PCR1': 'config_secure',
                'PCR2': 'kernel_secure'
            }
            return all(
                pcr_values.get(pcr) == expected_pcrs.get(pcr)
                for pcr in expected_pcrs
            )
        except Exception:
            return False

    def _verify_minimum_requirements(self, specs: Dict) -> bool:
        """Verify hardware meets minimum requirements."""
        try:
            min_requirements = {
                'cpu_cores': 4,
                'memory_gb': 8,
                'disk_gb': 100,
                'gpu_memory_gb': 4
            }
            return all(
                float(specs.get(key, 0)) >= value
                for key, value in min_requirements.items()
            )
        except Exception:
            return False

    async def _cache_verification_result(self, peer_id: str, result: bool) -> None:
        """Cache hardware verification result with expiration."""
        cache_key = f"hw_verify_{peer_id}"
        cache_data = {
            'result': result,
            'timestamp': time.time(),
            'expires': time.time() + 3600  # 1 hour cache
        }
        self.data_store[cache_key] = cache_data

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
        """Enhanced routing table update with zone management."""
        current_time = asyncio.get_event_loop().time()
        zones_to_rebalance = []

        for zone_id, zone in self.zones.items():
            # Check if zone needs rebalancing
            if current_time - zone.last_rebalance > self.zone_rebalance_interval:
                zones_to_rebalance.append(zone_id)
                
            # Update zone load metrics
            zone.load_metrics = await self._get_zone_load_metrics(zone)
            
            # Select relay nodes based on reputation and resources
            zone.relay_nodes = {
                peer for peer in zone.peers
                if self.peer_reputation.get(peer, 0) >= 0.8 
                and await self._check_relay_capability(peer)
            }

        # Rebalance zones if needed
        if zones_to_rebalance:
            await self._rebalance_zones(zones_to_rebalance)

    async def _join_network_interactive(self):
        """Join DHT network with progress tracking and timeout handling."""
        if not self.bootstrap_nodes:
            self.logger.warning("No bootstrap nodes configured")
            return

        successful = 0
        failed_nodes = []

        with tqdm(total=len(self.bootstrap_nodes), desc="Joining Network") as pbar:
            for node in self.bootstrap_nodes:
                if self._interrupt_requested:
                    break

                try:
                    async def bootstrap_attempt():
                        async with asyncio.timeout(INTERACTION_TIMEOUTS["default"]):
                            host, port = node.split(':')
                            await self.server.bootstrap([(host, int(port))])
                            return True

                    await self._retry_with_backoff(bootstrap_attempt)
                    successful += 1

                except asyncio.TimeoutError:
                    self.logger.warning(f"Bootstrap timeout for node {node}")
                    failed_nodes.append((node, "timeout"))
                except Exception as e:
                    self.logger.error(f"Failed to bootstrap with node {node}: {e}")
                    failed_nodes.append((node, str(e)))
                finally:
                    pbar.update(1)

        if successful == 0:
            error_details = "\n".join(f"- {node}: {reason}" for node, reason in failed_nodes)
            raise BootstrapError(
                f"Failed to connect to any bootstrap nodes\nDetails:\n{error_details}"
            )

    async def _retry_with_backoff(self, operation: Any, *args, **kwargs) -> Any:
        """Execute an operation with exponential backoff retry."""
        retries = 0
        last_exception = None
        delay = self.retry_delay

        while retries < self.max_retries:
            try:
                return await operation(*args, **kwargs)
            except Exception as e:
                last_exception = e
                retries += 1
                if retries == self.max_retries:
                    break

                self.logger.warning(
                    f"Operation failed (attempt {retries}/{self.max_retries}): {str(e)}"
                    f"\nRetrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)
                delay *= self.backoff_factor

        raise last_exception

    async def _get_zone_for_peer(self, peer_id: str) -> str:
        """Get or create routing zone for a peer based on network topology."""
        if peer_id in self.zone_cache:
            self.zones[self.zone_cache[peer_id]].cache_hits += 1
            return self.zone_cache[peer_id]

        region = self._get_peer_region(peer_id)
        latency = await self._measure_latency(peer_id)
        zone_id = f"{region}_{self._calculate_zone_suffix(latency)}"

        if zone_id not in self.zones:
            self.zones[zone_id] = RoutingZone(zone_id)
        
        # Update LRU cache
        self.zone_cache[peer_id] = zone_id
        if len(self.zone_cache) > self.max_zone_cache_size:
            self.zone_cache.popitem(last=False)
            
        self.zones[zone_id].cache_misses += 1
        return zone_id

    async def _rebalance_zones(self, zone_ids: List[str]):
        """Rebalance zones to maintain optimal performance."""
        for zone_id in zone_ids:
            zone = self.zones[zone_id]
            if len(zone.peers) > self.load_thresholds['peers']:
                await self._split_zone(zone)
            elif len(zone.peers) < self.load_thresholds['peers'] // 2:
                await self._merge_zones(zone_id)
            
            zone.last_rebalance = asyncio.get_event_loop().time()

    async def _get_zone_load_metrics(self, zone: RoutingZone) -> Dict[str, float]:
        """Get aggregate load metrics for a zone."""
        metrics = defaultdict(float)
        peer_count = len(zone.peers)
        if not peer_count:
            return metrics

        for peer in zone.peers:
            try:
                status = await self.server.get_peer_status(peer)
                if status:
                    for key, value in status.get('metrics', {}).items():
                        metrics[key] += value / peer_count
            except Exception as e:
                self.logger.warning(f"Failed to get metrics for peer {peer}: {e}")

        return dict(metrics)

    def _calculate_zone_suffix(self, latency: float) -> str:
        """Calculate zone suffix based on latency and network characteristics."""
        if latency < 50:
            return 'fast'
        elif latency < 150:
            return 'medium'
        return 'slow'

    async def _check_relay_capability(self, peer_id: str) -> bool:
        """Check if a peer can act as a relay node."""
        try:
            status = await self.server.get_peer_status(peer_id)
            if not status:
                return False

            metrics = status.get('metrics', {})
            return (
                metrics.get('cpu', 100) < self.load_thresholds['cpu'] and
                metrics.get('bandwidth', 100) < self.load_thresholds['bandwidth'] and
                status.get('uptime', 0) > 3600  # 1 hour minimum uptime
            )
        except Exception:
            return False

    async def join_network(self) -> bool:
        """Join the DHT network through bootstrap nodes"""
        try:
            # Fix the syntax error in the progress bar
            with tqdm(total=len(self.bootstrap_nodes), desc="Joining Network") as pbar:
                for node in self.bootstrap_nodes:
                    try:
                        await self.ping_node(node)
                        pbar.update(1)
                    except Exception as e:
                        self.logger.warning(f"Failed to connect to bootstrap node {node}: {e}")
            
            return len(self.routing_table) > 0
        
        except Exception as e:
            self.logger.error(f"Failed to join network: {e}")
            return False
