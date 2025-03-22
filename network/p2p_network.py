import asyncio
import logging
from typing import Dict, Set, Optional
from datetime import datetime, timedelta
from collections import defaultdict
from tqdm import tqdm
from core.constants import INTERACTION_TIMEOUTS, InteractionLevel
from core.interactive_utils import InteractiveSession, InteractiveConfig
from .dht import DHT
from .udp_broadcast import UDPBroadcast
from .pex import PeerExchange
from .connection_pool import ConnectionPool
from .message_protocol import SecureMessageProtocol
from security.auth import NodeAuthenticator
from network.consensus import ConsensusManager
from .rate_limiter import RateLimiter
from .circuit_breaker import CircuitBreaker

class P2PNetwork:
    def __init__(self, node_id: str, network_config: Dict, interactive: bool = True):
        self.node_id = node_id
        self.running = False
        self.peers: Set[str] = set()
        
        # Initialize P2P components
        self.dht = DHT(node_id, network_config['dht'])
        self.udp_broadcast = UDPBroadcast(network_config['udp'])
        self.auth = NodeAuthenticator(network_config.get('secret_key', ''))
        self.consensus = ConsensusManager()
        self.rate_limiter = RateLimiter(
            max_requests=100,  # 100 requests
            time_window=60     # per minute
        )
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            reset_timeout=60
        )
        self.error_counts = {}
        
        # Add new components
        self.pex = PeerExchange(max_peers=network_config.get('max_peers', 1000))
        self.connection_pool = ConnectionPool(
            max_connections=network_config.get('max_connections', 100),
            ttl=network_config.get('connection_ttl', 300)
        )
        self.message_protocol = SecureMessageProtocol(
            network_config['encryption_key'].encode()
        )
        
        # Setup logging
        self.logger = logging.getLogger('P2PNetwork')
        
        # Reputation and banning
        self.peer_reputation = {}
        self.banned_peers = set()
        self.suspicious_activity = defaultdict(list)
        self.reputation_threshold = network_config.get('reputation_threshold', 0.3)

        self.interactive = interactive
        self.session = None
        self._interrupt_requested = False
        self._progress_bar = None
        
    def start(self):
        """Start the P2P network node."""
        self.running = True
        self.logger.info(f"Starting P2P node {self.node_id}")
        asyncio.run(self._run_network())
        
    def stop(self):
        """Gracefully stop the P2P network node."""
        self.running = False
        self.logger.info(f"Stopping P2P node {self.node_id}")
        asyncio.run(self._cleanup())
        
    async def _run_network(self):
        """Run the main network loop."""
        # Start P2P services
        await self.udp_broadcast.start()
        await self.dht.start()
        
        while self.running:
            try:
                # Handle peer discovery and communication
                new_peers = await self._discover_peers()
                await self._handle_messages()
                self.peers.update(new_peers)
                await asyncio.sleep(1)
            except Exception as e:
                self.logger.error(f"Network error: {e}")
                
    async def _cleanup(self):
        """Clean up network resources."""
        await self.udp_broadcast.stop()
        await self.dht.stop()
        
    async def broadcast_message(self, message: Dict):
        """Broadcast a message to all peers."""
        await self.udp_broadcast.broadcast(message)
        
    async def lookup_node(self, node_id: str) -> Optional[str]:
        """Look up a node's address using DHT."""
        return await self.dht.lookup(node_id)

    async def _discover_peers(self) -> Set[str]:
        """Discover and authenticate new peers."""
        # Combine multiple discovery methods
        dht_peers = await self.dht.discover()
        broadcast_peers = set(await self.udp_broadcast.get_peers())
        pex_peers = set(self.pex.get_peers())
        
        all_peers = dht_peers.union(broadcast_peers, pex_peers)
        authenticated = set()
        
        for peer in all_peers:
            if peer not in self.peers:
                token = self.auth.generate_token(peer)
                if await self._authenticate_peer(peer, token):
                    authenticated.add(peer)
                    self.pex.add_peer(peer)
                    
        return authenticated
        
    async def _authenticate_peer(self, peer_id: str, token: str) -> bool:
        """Authenticate a new peer."""
        try:
            # Send authentication request
            auth_msg = {"type": "auth", "token": token}
            await self.node_comm.send_message(peer_id, auth_msg)
            return True
        except Exception as e:
            self.logger.error(f"Authentication failed for peer {peer_id}: {e}")
            return False
            
    async def _handle_messages(self):
        """Process incoming messages from peers."""
        while not self.node_comm.message_queue.empty():
            msg = await self.node_comm.message_queue.get()
            await self._handle_message(msg, msg.get('sender'))
            
    async def _handle_message(self, message: Dict, addr):
        try:
            if not await self.rate_limiter.allow_request(addr):
                raise RateLimitExceeded(f"Rate limit exceeded for {addr}")

            if not self.circuit_breaker.allow_request():
                raise CircuitBreakerOpen("Circuit breaker is open")

            await self._validate_message(message)
            await self._process_message(message)

        except RateLimitExceeded as e:
            self.logger.warning(f"Rate limit exceeded: {e}")
        except CircuitBreakerOpen as e:
            self.logger.error(f"Circuit breaker open: {e}")
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
            self.circuit_breaker.record_failure()

    async def _validate_message(self, message: Dict):
        """Enhanced message validation with reputation checks."""
        sender = message.get('sender')
        if not sender:
            raise ValueError("Missing sender information")
            
        if sender in self.banned_peers:
            raise ValueError(f"Message from banned peer {sender}")
            
        # Check message signatures and integrity
        if not await self._verify_message_integrity(message):
            await self._record_suspicious_activity(sender, "invalid_signature")
            raise ValueError("Invalid message signature")
            
        await self._update_peer_reputation(sender, 0.1)  # Reward valid messages

    async def _record_suspicious_activity(self, peer_id: str, activity_type: str):
        """Record suspicious activity for a peer"""
        self.suspicious_activity[peer_id].append({
            'type': activity_type,
            'timestamp': datetime.now()
        })
        
        # Check for repeated suspicious behavior
        recent_activities = [
            a for a in self.suspicious_activity[peer_id]
            if datetime.now() - a['timestamp'] < timedelta(hours=1)
        ]
        
        if len(recent_activities) >= 5:  # Threshold for suspicious activity
            await self._ban_peer(peer_id)
            
    async def _ban_peer(self, peer_id: str):
        """Ban a peer from the network"""
        self.banned_peers.add(peer_id)
        self.peers.discard(peer_id)
        self.peer_reputation.pop(peer_id, None)
        
        # Notify other peers about the banned node
        await self.broadcast_message({
            'type': 'peer_banned',
            'peer_id': peer_id,
            'reason': 'suspicious_activity'
        })
            
    async def _process_message(self, message: Dict):
        """Process a single message."""
        msg_type = message.get('type')
        if msg_type == 'auth':
            await self._handle_auth(message)
        elif msg_type == 'data':
            await self._handle_data(message)
            
    async def send_message(self, peer_id: str, message: Dict) -> bool:
        try:
            conn = await self.connection_pool.get_connection(peer_id)
            if not conn:
                return False
                
            encoded = self.message_protocol.encode_message(message)
            conn.writer.write(encoded)
            await conn.writer.drain()
            return True
        except Exception as e:
            self.logger.error(f"Failed to send message to {peer_id}: {e}")
            return False

    async def start_interactive(self):
        """Start the P2P network node with interactive controls"""
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

            async with self.session:
                self.running = True
                self.logger.info(f"Starting P2P node {self.node_id}")

                if self.interactive:
                    proceed = await self.session.confirm_with_timeout(
                        "\nStart P2P network node with current configuration?",
                        timeout=INTERACTION_TIMEOUTS["confirmation"]
                    )
                    if not proceed:
                        self.logger.info("Network start cancelled by user")
                        return

                await self._run_network_interactive()

        except Exception as e:
            self.logger.error(f"Failed to start network: {str(e)}")
            raise
        finally:
            if self.session:
                await self.session.__aexit__(None, None, None)

    async def _run_network_interactive(self):
        """Run the main network loop with interactive monitoring"""
        try:
            # Start P2P services with progress tracking
            if self.interactive:
                print("\nInitializing P2P Services")
                print("=" * 50)

            await self.udp_broadcast.start()
            await self.dht.start()

            while self.running and not self._interrupt_requested:
                try:
                    if self.interactive:
                        self._progress_bar = tqdm(total=100, desc="Network Operations")

                    # Enhanced peer discovery with progress updates
                    new_peers = await self._discover_peers_interactive()
                    self.peers.update(new_peers)

                    # Process messages with monitoring
                    await self._handle_messages_interactive()

                    if self._progress_bar:
                        self._progress_bar.update(100)
                        self._progress_bar.close()

                    await asyncio.sleep(1)

                except asyncio.CancelledError:
                    self.logger.info("Network loop interrupted")
                    break
                except Exception as e:
                    self.logger.error(f"Network loop error: {str(e)}")
                    if self.interactive:
                        retry = await self.session.confirm_with_timeout(
                            "\nError in network loop. Retry?",
                            timeout=INTERACTION_TIMEOUTS["emergency"]
                        )
                        if not retry:
                            break

        except Exception as e:
            self.logger.error(f"Fatal network error: {str(e)}")
            raise
        finally:
            await self._cleanup_interactive()

    async def _discover_peers_interactive(self) -> Set[str]:
        """Interactive peer discovery with progress tracking"""
        try:
            if self._progress_bar:
                self._progress_bar.set_description("Discovering Peers")

            dht_peers = await self.dht.discover()
            broadcast_peers = set(await self.udp_broadcast.get_peers())
            pex_peers = set(self.pex.get_peers())

            all_peers = dht_peers.union(broadcast_peers, pex_peers)
            authenticated = set()

            if self.interactive and all_peers:
                print(f"\nDiscovered {len(all_peers)} potential peers")

            for i, peer in enumerate(all_peers):
                if peer not in self.peers:
                    if self._progress_bar:
                        progress = (i + 1) * 100 // len(all_peers)
                        self._progress_bar.update(progress - self._progress_bar.n)

                    if await self._authenticate_peer_interactive(peer):
                        authenticated.add(peer)
                        self.pex.add_peer(peer)

            return authenticated

        except Exception as e:
            self.logger.error(f"Peer discovery error: {str(e)}")
            return set()

    async def _authenticate_peer_interactive(self, peer_id: str) -> bool:
        """Interactive peer authentication with safety checks"""
        try:
            token = self.auth.generate_token(peer_id)
            auth_msg = {"type": "auth", "token": token}

            if not await self.circuit_breaker.allow_request_interactive():
                return False

            return await self.send_message_interactive(peer_id, auth_msg)

        except Exception as e:
            self.logger.error(f"Authentication failed for peer {peer_id}: {str(e)}")
            return False

    async def send_message_interactive(self, peer_id: str, message: Dict) -> bool:
        """Send message with interactive monitoring and error handling"""
        try:
            if not await self.rate_limiter.allow_request(peer_id):
                if self.interactive:
                    self.logger.warning(f"Rate limit reached for peer {peer_id}")
                return False

            conn = await self.connection_pool.get_connection(peer_id)
            if not conn:
                return False

            encoded = await self.message_protocol.encode_message_interactive(message)
            if not encoded:
                return False

            conn.writer.write(encoded)
            await conn.writer.drain()
            return True

        except Exception as e:
            self.logger.error(f"Failed to send message to {peer_id}: {str(e)}")
            if self.interactive:
                await self.session.log_error(f"Message sending failed: {str(e)}")
            return False

    async def _cleanup_interactive(self):
        """Interactive cleanup with resource monitoring"""
        try:
            if self.interactive:
                print("\nCleaning up P2P Network")
                print("=" * 50)

            await self.udp_broadcast.stop()
            await self.dht.stop()
            
            if self._progress_bar:
                self._progress_bar.close()

            # Clear connection pool
            await self.connection_pool.cleanup()
            
            # Clear various collections
            self.peers.clear()
            self.banned_peers.clear()
            self.peer_reputation.clear()
            self.suspicious_activity.clear()

            self.logger.info("Network cleanup completed successfully")

        except Exception as e:
            self.logger.error(f"Cleanup error: {str(e)}")
            raise
        finally:
            self.running = False

    def request_shutdown(self):
        """Request graceful shutdown of the network"""
        self._interrupt_requested = True
        self.running = False
        self.logger.info("Shutdown requested for P2P network")
