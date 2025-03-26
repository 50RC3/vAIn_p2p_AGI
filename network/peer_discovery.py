import socket
import json
import time
import asyncio
import logging
from typing import List, Dict, Optional, Set
from tqdm import tqdm
from core.constants import INTERACTION_TIMEOUTS, InteractionLevel
from core.interactive_utils import InteractiveSession, InteractiveConfig
from collections import OrderedDict, defaultdict
from functools import lru_cache
from operator import itemgetter
from .shard_manager import ShardManager
from .circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)

class PeerDiscovery:
    def __init__(self, port: int = 5000, interactive: bool = True, cooling_period: int = 60):
        self.port = port
        self.peers = {}  # Store peers with additional info
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.interactive = interactive
        self.session = None
        self._interrupt_requested = False
        self._cleanup_event = asyncio.Event()
        self.broadcast_interval = 30  # Seconds
        self.max_retries = 3
        self.retry_delay = 5  # Seconds
        self.cooling_period = cooling_period  # Cooling period in seconds
        self.max_inactive_time = 3600  # Remove peers inactive for 1 hour
        self.min_reputation = 0.0
        self.max_reputation = 5.0
        self.peer_id = self._generate_peer_id()  # Add peer ID generation
        
        # Add new fields for optimized peer management
        self._active_peers = OrderedDict()  # LRU cache of active peers
        self._peer_scores = {}  # Cache of peer scores for sorting
        self._prune_task = None
        self.max_cache_size = 1000
        self.prune_interval = 300  # 5 minutes
        self.batch_size = 50
        
        # Add peer-specific timeouts
        self.timeouts = {
            "discovery": INTERACTION_TIMEOUTS.get("discovery", 30),  # 30s for discovery operations
            "broadcast": INTERACTION_TIMEOUTS.get("broadcast", 15),  # 15s for broadcasts
            "response": INTERACTION_TIMEOUTS.get("response", 5),     # 5s for responses
            "cleanup": INTERACTION_TIMEOUTS.get("cleanup", 10)       # 10s for cleanup
        }
        
        self.shard_manager = ShardManager()
        
        # Add pheromone tracking
        self.pheromone_trails = defaultdict(dict)
        self.pheromone_decay = 0.95  # 5% decay per interval
        self.pheromone_strength = 1.0
        
        # Add circuit breaker
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            reset_timeout=60
        )
        
        # Add metrics
        self.discovery_metrics = {
            'attempts': 0,
            'successes': 0, 
            'failures': 0,
            'latencies': []
        }
        
    def broadcast_presence(self):
        """Broadcast presence with peer ID"""
        try:
            message = json.dumps({
                "type": "discovery", 
                "port": self.port,
                "id": self.peer_id  # Add peer_id to broadcast
            })
            self.sock.sendto(message.encode(), ('<broadcast>', self.port))
        except (json.JSONDecodeError, socket.error) as e:
            logger.error(f"Failed to broadcast presence: {str(e)}")
            raise

    async def start_interactive(self) -> None:
        """Start peer discovery with background pruning"""
        try:
            # Start periodic pruning task
            self._prune_task = asyncio.create_task(self._periodic_prune())
            
            if self.interactive:
                self.session = InteractiveSession(
                    level=InteractionLevel.NORMAL,
                    config=InteractiveConfig(
                        timeout=self.timeouts["discovery"],  # Use peer discovery timeout
                        persistent_state=True,
                        safe_mode=True
                    )
                )

            async with self.session:
                logger.info("Starting peer discovery service")
                await self._run_discovery_loop()

        except Exception as e:
            logger.error(f"Peer discovery failed: {str(e)}")
            raise
        finally:
            if self._prune_task:
                self._prune_task.cancel()
            await self._cleanup()

    async def _run_discovery_loop(self) -> None:
        """Enhanced discovery loop with pheromones and circuit breaker"""
        while not self._interrupt_requested:
            try:
                if not self.circuit_breaker.allow_request():
                    await asyncio.sleep(self.circuit_breaker.reset_timeout)
                    continue

                start_time = time.time()
                
                # Update pheromone trails
                self._decay_pheromones()
                discovered = await self._broadcast_safely()
                
                if discovered:
                    # Strengthen successful paths
                    for peer in discovered:
                        self._update_pheromone(peer)
                    
                    latency = time.time() - start_time
                    self.discovery_metrics['latencies'].append(latency)
                    self.discovery_metrics['successes'] += 1
                    self.circuit_breaker.record_success()
                
                self.discovery_metrics['attempts'] += 1
                
                # Use exponential backoff for retry delay
                retry_delay = min(
                    self.broadcast_interval * (2 ** self.circuit_breaker.failure_count),
                    300  # Max 5 minute delay
                )
                await asyncio.sleep(retry_delay)

            except Exception as e:
                self.discovery_metrics['failures'] += 1
                self.circuit_breaker.record_failure()
                logger.error(f"Discovery error: {str(e)}")
                await asyncio.sleep(self.retry_delay)

    def _decay_pheromones(self):
        """Decay pheromone trail strengths"""
        for peer_trails in self.pheromone_trails.values():
            for path, strength in peer_trails.items():
                peer_trails[path] *= self.pheromone_decay

    def _update_pheromone(self, peer_id: str, path: str = 'default'):
        """Update pheromone strength for successful discovery path"""
        self.pheromone_trails[peer_id][path] = min(
            self.pheromone_trails[peer_id].get(path, 0) + self.pheromone_strength,
            5.0  # Max strength cap
        )

    async def _broadcast_safely(self) -> None:
        """Send broadcast with error handling and confirmation for retries"""
        try:
            self.broadcast_presence()
            new_peers = await self._listen_for_responses()
            await self._update_peers(new_peers)
        except socket.error as e:
            logger.error(f"Network error during broadcast: {str(e)}")
            if self.interactive and self.session:
                retry = await self.session.confirm_with_timeout(
                    "\nNetwork error during broadcast. Retry?",
                    timeout=INTERACTION_TIMEOUTS["confirmation"],
                    default=True
                )
                if retry:
                    await asyncio.sleep(self.retry_delay)
                    await self._broadcast_safely()
                else:
                    raise

    async def update_peer_reputation(self, peer_id: str, delta: float, reason: str = ""):
        """Update peer reputation with interactive confirmation for large changes"""
        current_time = time.time()
        if peer_id not in self.peers:
            logger.warning(f"Attempted to update reputation for unknown peer {peer_id}")
            return False

        # Confirm large reputation changes
        if abs(delta) >= 1.0 and self.interactive and self.session:
            proceed = await self.session.confirm_with_timeout(
                f"\nLarge reputation change ({delta:+.2f}) for peer {peer_id}. Proceed?",
                timeout=INTERACTION_TIMEOUTS["confirmation"],
                default=False
            )
            if not proceed:
                logger.info("Reputation update cancelled by user")
                return False

        # ...existing reputation update code...
        peer_info = self.peers[peer_id]
        if current_time - peer_info["last_updated"] < self.cooling_period:
            logger.debug(
                f"Skipping reputation update for {peer_id} - in cooling period "
                f"({int(self.cooling_period - (current_time - peer_info['last_updated']))}s remaining)"
            )
            return False

        new_reputation = max(
            self.min_reputation,
            min(self.max_reputation, peer_info["reputation"] + delta)
        )
        
        self.peers[peer_id].update({
            "reputation": new_reputation,
            "last_updated": current_time,
            "last_reason": reason
        })
        
        logger.info(
            f"Updated peer {peer_id} reputation: {peer_info['reputation']:.2f} -> {new_reputation:.2f} "
            f"(Δ: {delta:+.2f}, reason: {reason})"
        )
        return True

    async def get_peer_status(self, peer_id: str) -> Optional[Dict]:
        """Get detailed peer status information"""
        if peer_id not in self.peers:
            return None
            
        peer_info = self.peers[peer_id]
        current_time = time.time()
        
        return {
            **peer_info,
            "age": current_time - peer_info["first_seen"],
            "cooling_remaining": max(0, self.cooling_period - (current_time - peer_info["last_updated"])),
            "is_cooling": (current_time - peer_info["last_updated"]) < self.cooling_period
        }

    async def cleanup_inactive_peers(self):
        """Remove inactive peers with confirmation"""
        current_time = time.time()
        inactive_peers = [
            peer_id for peer_id, info in self.peers.items()
            if current_time - info["last_updated"] > self.max_inactive_time
        ]
        
        if inactive_peers:
            if self.interactive and self.session:
                proceed = await self.session.confirm_with_timeout(
                    f"\nFound {len(inactive_peers)} inactive peers. Remove them?",
                    timeout=INTERACTION_TIMEOUTS["cleanup"],
                    default=True
                )
                if not proceed:
                    logger.info("Cleanup cancelled by user")
                    return

            for peer_id in inactive_peers:
                logger.info(f"Removing inactive peer {peer_id} (no activity for {self.max_inactive_time}s)")
                del self.peers[peer_id]

    async def _update_peers(self, new_peers: Set[str]):
        """Update peers with sharding support"""
        try:
            current_time = time.time()
            updates_skipped = 0
            updates_applied = 0
            
            # Process peers in batches with sharding
            for i in range(0, len(new_peers), self.batch_size):
                batch = list(new_peers)[i:i + self.batch_size]
                await self._process_peer_batch(batch)
                
            if self.interactive and (updates_applied > 0 or updates_skipped > 0):
                logger.info(f"Peer updates - Applied: {updates_applied}, Skipped: {updates_skipped}")

        except Exception as e:
            logger.error(f"Fatal error in peer update: {str(e)}")
            raise

    async def _process_peer_batch(self, peer_batch: List[str]):
        """Process a batch of peers with sharding support"""
        current_time = time.time()
        
        for peer in peer_batch:
            try:
                peer_info = await self.shard_manager.get_peer(peer)
                
                if peer_info and current_time - peer_info["last_updated"] > self.max_inactive_time:
                    logger.debug(f"Skipping expired peer {peer}")
                    continue

                if await self._validate_peer_async(peer):
                    await self.shard_manager.add_peer(peer, {
                        "reputation": 1.0,
                        "last_updated": current_time,
                        "first_seen": current_time,
                        "last_reason": "initial_discovery",
                        "address": peer,
                        "expires_at": current_time + self.max_inactive_time
                    })
                    
            except Exception as e:
                logger.warning(f"Error processing peer {peer}: {str(e)}")

    async def get_peers(self, n: int = 10, min_reputation: float = 0.0) -> List[str]:  # Make async 
        """Get peers with optimized caching and batching"""
        try:
            # Update peer scores periodically
            current_time = time.time()
            if not self._peer_scores or current_time - self._last_score_update > 60:
                await self._update_peer_scores()  # Add await here

            # Get top N peers by score
            top_peers = sorted(
                self._peer_scores.items(),
                key=itemgetter(1),
                reverse=True
            )[:n]

            return [peer_id for peer_id, _ in top_peers if peer_id in self.peers]

        except Exception as e:
            logger.error(f"Error retrieving peers: {str(e)}")
            return []

    async def _update_peer_scores(self):  # Make async
        """Update cached peer scores"""
        current_time = time.time()
        scores = {}
        for peer_id, info in self.peers.items():
            scores[peer_id] = await self._calculate_peer_score(info, current_time)  # Add await
        self._peer_scores = scores
        self._last_score_update = current_time

    async def _calculate_peer_score(self, peer_info: Dict, current_time: float) -> float:  # Make async
        """Calculate peer score based on multiple factors"""
        age = current_time - peer_info["first_seen"]
        activity = current_time - peer_info["last_updated"]
        
        score = peer_info["reputation"] * 0.5  # Base on reputation
        score += min(1.0, age / 86400) * 0.3  # Age factor (up to 1 day)
        score -= min(1.0, activity / 3600) * 0.2  # Activity penalty
        
        return max(0.0, score)

    async def _validate_peer_async(self, peer_info: Dict) -> bool:
        """Asynchronous peer validation"""
        # Basic structure validation
        if not isinstance(peer_info, dict):
            logger.debug("Invalid peer info format - not a dictionary")
            return False

        # Required fields validation with timeout
        async with asyncio.timeout(self.timeouts["response"]):
            required_fields = {"type", "port", "id"}
            if not all(field in peer_info for field in required_fields):
                logger.debug(f"Missing required fields: {required_fields - peer_info.keys()}")
                return False

            # Add other validation logic...
            return await self._validate_peer_id_async(peer_info["id"])

    async def _update_peers_atomically(self, updates: Dict):
        """Atomic peer updates with proper async handling"""
        async with asyncio.Lock():  # Ensure thread safety
            self.peers.update(updates)

    async def _listen_for_responses(self) -> Set[str]:
        """Listen for peer responses with enhanced error handling"""
        responses = set()
        try:
            self.sock.settimeout(self.timeouts["response"])
            start_time = time.time()
            
            while not self._interrupt_requested:
                try:
                    if time.time() - start_time > 30:  # Maximum listen time
                        break
                        
                    data, addr = self.sock.recvfrom(1024)
                    if not data or not addr:
                        continue
                        
                    try:
                        peer_info = json.loads(data.decode())
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON from {addr[0]}")
                        continue
                        
                    if await self._validate_peer_async(peer_info):
                        responses.add(addr[0])
                except socket.timeout:
                    break
                except Exception as e:
                    logger.warning(f"Error processing response: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Fatal error in response listening: {str(e)}")
            raise
        finally:
            self.sock.settimeout(None)
            
        return responses

    async def _validate_peer_id_async(self, peer_id: str) -> bool:
        """Validate peer ID format"""
        return (
            isinstance(peer_id, str) and
            8 <= len(peer_id) <= 64 and
            peer_id.isalnum()  # Only alphanumeric characters allowed
        )

    async def _is_different_address_async(self, peer_info: Dict) -> bool:
        """Check address mismatch with error handling"""
        try:
            existing = self.peers.get(peer_info["id"])
            if not existing:
                return False
            return existing.get("address") != peer_info.get("address")
        except Exception as e:
            logger.error(f"Error checking peer address: {str(e)}")
            return True  # Fail safe - treat as different

    async def _cleanup(self) -> None:
        """Clean up resources"""
        try:
            self._cleanup_event.set()
            self.sock.close()
            logger.info("Peer discovery cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def _generate_peer_id(self) -> str:
        """Generate a unique peer ID"""
        import uuid
        return str(uuid.uuid4()).replace('-', '')[:32]  # 32-char unique ID

    def request_shutdown(self) -> None:
        """Request graceful shutdown"""
        self._interrupt_requested = True
        logger.info("Shutdown requested for peer discovery")
