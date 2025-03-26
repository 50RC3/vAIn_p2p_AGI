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
from security.secure_key_manager import SecureKeyManager
from .cluster_manager import ClusterManager
from .admin_commands import AdminCommands

class P2PNetworkError(Exception):
    """Base exception for P2P network errors"""
    pass

class PeerConnectionError(P2PNetworkError):
    """Raised when connection to a peer fails"""
    pass

class PeerAuthenticationError(P2PNetworkError):
    """Raised when peer authentication fails"""
    pass

class MessageTimeoutError(P2PNetworkError):
    """Raised when message sending times out"""
    pass

class NetworkTimeoutError(P2PNetworkError):
    """Raised when network operations timeout"""
    pass

class MessageValidationError(P2PNetworkError):
    """Raised when message validation fails"""
    pass

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
        
        # Add secure key manager
        self.key_manager = SecureKeyManager()
        
        # Add cluster manager
        self.cluster_manager = ClusterManager(network_config.get('cluster_config', {}))
        self.cluster_id = None
        self.peer_latencies = {}
        
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
        
        # Initialize reputation manager with revalidation
        self.reputation_manager = ReputationManager(
            decay_factor=network_config.get('reputation_decay', 0.95),
            min_reputation=network_config.get('min_reputation', -100),
            interactive=interactive
        )
        await self.reputation_manager.start()
        
        # Add consensus states
        self.pending_state_changes: Dict[str, Dict] = {}
        self.consensus_thresholds = {
            'peer_ban': 0.75,  # 75% agreement needed to ban peer
            'reputation_update': 0.65,  # 65% for reputation changes
            'cluster_reconfig': 0.80,  # 80% for cluster changes
            'network_param': 0.90  # 90% for network parameter changes
        }

        # Add admin commands
        self.admin = AdminCommands(self)

        self._active_tasks = set()
        self._cleanup_lock = asyncio.Lock()
        self._shutdown_timeout = network_config.get('shutdown_timeout', 30)  # 30 second timeout
        
        # Add load balancing and monitoring
        self.load_balancer = LoadBalancer()
        self.network_monitor = NetworkMonitor()
        
        # Add health check interval
        self.health_check_interval = network_config.get('health_check_interval', 60)
        self._last_health_check = 0

        self.debug_enabled = network_config.get('debug', False)
        
    def start(self):
        """Start the P2P network node."""
        self.running = True
        self.logger.info(f"Starting P2P node {self.node_id}")
        asyncio.run(self._run_network())
        
    def stop(self):
        """Gracefully stop the P2P network node with task cleanup."""
        self.running = False
        self.logger.info(f"Stopping P2P node {self.node_id}")
        
        try:
            # Create event loop if needed
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            # Run cleanup with timeout
            cleanup_task = self._cleanup()
            cleanup_future = asyncio.run_coroutine_threadsafe(cleanup_task, loop)
            cleanup_future.result(timeout=self._shutdown_timeout)
            
        except TimeoutError:
            self.logger.error("Shutdown timed out, forcing cleanup")
            self._force_cleanup()
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            raise
        finally:
            self._interrupt_requested = False
        
    async def _run_network(self):
        """Run the main network loop."""
        # Start P2P services
        await self.udp_broadcast.start()
        await self.dht.start()
        await self.reputation_manager.start_revalidation_loop()
        
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
        """Enhanced cleanup with active task handling."""
        async with self._cleanup_lock:
            try:
                # Cancel all active tasks
                active_tasks = list(self._active_tasks)
                if active_tasks:
                    self.logger.info(f"Cancelling {len(active_tasks)} active tasks")
                    for task in active_tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*active_tasks, return_exceptions=True)
                    self._active_tasks.clear()

                # Run existing cleanup
                await self._cleanup_interactive()

            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
                raise

    def _force_cleanup(self):
        """Force cleanup of resources when graceful shutdown fails."""
        try:
            # Clear all collections immediately
            self.peers.clear()
            self.banned_peers.clear()
            self.peer_reputation.clear()
            self.suspicious_activity.clear()
            self._active_tasks.clear()
            
            # Force close connections
            if hasattr(self, 'connection_pool'):
                self.connection_pool.force_close()
                
            self.logger.warning("Forced cleanup completed")
        except Exception as e:
            self.logger.error(f"Force cleanup error: {e}")

    async def register_task(self, task: asyncio.Task):
        """Register an active task for cleanup tracking."""
        self._active_tasks.add(task)
        try:
            await task
        except asyncio.CancelledError:
            pass
        finally:
            self._active_tasks.discard(task)
        
    async def broadcast_message(self, message: Dict):
        """Broadcast a message to all peers."""
        await self.udp_broadcast.broadcast(message)
        
    async def lookup_node(self, node_id: str) -> Optional[str]:
        """Look up a node's address using DHT."""
        return await self.dht.lookup(node_id)

    async def _discover_peers(self) -> Set[str]:
        """Enhanced peer discovery with coordinated methods"""
        discovered = set()
        
        # Run discovery methods concurrently
        discovery_tasks = [
            self.dht.discover(),
            self._discover_broadcast_peers(),
            self._discover_pex_peers()
        ]
        
        results = await asyncio.gather(*discovery_tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                self.logger.warning(f"Discovery method failed: {result}")
                continue
            discovered.update(result)
            
        # Filter and authenticate peers
        authenticated = set()
        for peer in discovered:
            if peer not in self.peers:
                token = self.auth.generate_token(peer)
                try:
                    if await self._authenticate_peer_with_timeout(peer, token):
                        authenticated.add(peer)
                        self.pex.add_peer(peer)
                        await self._measure_and_store_latency(peer)
                except Exception as e:
                    self.logger.warning(f"Failed to authenticate peer {peer}: {e}")
                    
        # Update cluster assignments
        if authenticated:
            await self._update_cluster_assignments(authenticated)
            
        return authenticated

    async def _authenticate_peer_with_timeout(self, peer_id: str, token: str) -> bool:
        """Authenticate peer with timeout"""
        try:
            async with asyncio.timeout(INTERACTION_TIMEOUTS["auth"]):
                return await self._authenticate_peer(peer_id, token)
        except asyncio.TimeoutError:
            self.logger.warning(f"Authentication timeout for peer {peer_id}")
            return False

    async def _measure_and_store_latency(self, peer_id: str):
        """Measure and store peer latency"""
        latency = await self._measure_latency(peer_id)
        self.peer_latencies[peer_id] = latency
        
        # Update metrics
        self.network_monitor.record_latency(peer_id, latency)

    async def _authenticate_peer(self, peer_id: str, token: str) -> bool:
        """Authenticate a new peer with E2EE session establishment"""
        try:
            # First authenticate using token
            auth_msg = {
                "type": "auth",
                "token": token,
                "public_key": self.key_manager.identity_public.public_bytes()
            }
            
            # Add timeout for authentication
            try:
                async with asyncio.timeout(10):
                    response = await self.node_comm.send_message(peer_id, auth_msg)
                    if not response or 'public_key' not in response:
                        return False
                        
                    # Establish E2EE session
                    if not await self.key_manager.establish_session(
                        peer_id, 
                        response['public_key']
                    ):
                        raise PeerAuthenticationError("Failed to establish secure session")
                        
                    return True
                    
            except asyncio.TimeoutError:
                raise PeerAuthenticationError(f"Authentication timeout for peer {peer_id}")
                
        except ConnectionError:
            raise PeerConnectionError(f"Failed to connect to peer {peer_id}")
        except Exception as e:
            raise PeerAuthenticationError(f"Authentication failed: {str(e)}")
            
    async def _handle_messages(self):
        """Process incoming messages from peers."""
        while not self.node_comm.message_queue.empty():
            msg = await self.node_comm.message_queue.get()
            await self._handle_message(msg, msg.get('sender'))
            
    async def _handle_message(self, message: Dict, addr):
        """Handle incoming encrypted messages"""
        try:
            if not await self.rate_limiter.allow_request(addr):
                raise RateLimitExceeded(f"Rate limit exceeded for {addr}")

            if not self.circuit_breaker.allow_request():
                raise CircuitBreakerOpen("Circuit breaker is open")

            try:
                async with asyncio.timeout(3):
                    # Decrypt message first
                    if 'nonce' not in message or 'ciphertext' not in message:
                        raise MessageValidationError("Invalid encrypted message format")
                        
                    nonce = base64.b64decode(message['nonce'])
                    ciphertext = base64.b64decode(message['ciphertext'])
                    
                    decrypted = self.key_manager.decrypt_message(
                        message.get('sender'),
                        nonce,
                        ciphertext
                    )
                    
                    if not decrypted:
                        raise MessageValidationError("Failed to decrypt message")
                    
                    # Then validate and process
                    await self._validate_message(decrypted)
                    await self._process_message(decrypted)
                    
            except asyncio.TimeoutError:
                raise MessageTimeoutError("Message processing timeout")

        except RateLimitExceeded as e:
            self.logger.warning(f"Rate limit exceeded: {e}")
        except CircuitBreakerOpen as e:
            self.logger.error(f"Circuit breaker open: {e}")
        except MessageValidationError as e:
            self.logger.error(f"Message validation error: {e}")
            await self._record_suspicious_activity(message.get('sender'), "validation_error")
        except MessageTimeoutError as e:
            self.logger.error(f"Message handling timeout: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error handling message: {e}")
            self.circuit_breaker.record_failure()

    async def _validate_message(self, message: Dict):
        """Enhanced message validation with cooled reputation tracking"""
        sender = message.get('sender')
        if not sender:
            raise ValueError("Missing sender information")
            
        if sender in self.banned_peers:
            raise ValueError(f"Message from banned peer {sender}")
            
        # Check message signatures and integrity
        if not await self._verify_message_integrity(message):
            await self._record_suspicious_activity(sender, "invalid_signature")
            raise ValueError("Invalid message signature")
            
        # Queue reputation adjustment with reason
        reputation_delta = await self._calculate_reputation_delta(message)
        await self.reputation_manager.update_reputation_interactive(
            sender, 
            reputation_delta,
            reason=f"message_{message.get('type', 'unknown')}"
        )

    async def _calculate_reputation_delta(self, message: Dict) -> float:
        """Calculate reputation change based on message quality and behavior"""
        base_delta = 0.1  # Base reputation increase for valid messages
        
        # Adjust based on message size and processing cost
        if len(str(message)) > 1024 * 1024:  # 1MB
            base_delta *= 0.5  # Penalty for large messages
            
        # Adjust based on message type and value
        if message.get('type') == 'resource_contribution':
            base_delta *= 1.5  # Bonus for contributing resources
            
        # Additional adjustments can be added based on other factors
        return base_delta

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
            
    async def _ban_peer(self, peer_id: str, reason: str = "suspicious_activity"):
        """Enhanced ban peer with reason logging"""
        self.banned_peers.add(peer_id)
        self.peers.discard(peer_id)
        self.peer_reputation.pop(peer_id, None)
        
        # Log ban reason
        self.logger.warning(f"Banned peer {peer_id}: {reason}")
        
        # Broadcast ban with reason
        await self.broadcast_message({
            'type': 'peer_banned',
            'peer_id': peer_id,
            'reason': reason,
            'timestamp': time.time()
        })
            
    async def _process_message(self, message: Dict):
        """Process messages with consensus handling"""
        msg_type = message.get('type')
        if msg_type == 'consensus_proposal':
            await self._handle_consensus_proposal(message)
        elif msg_type == 'consensus_vote':
            await self._handle_consensus_vote(message)
        elif msg_type == 'state_change':
            await self._handle_state_change(message)
        else:
            # ...existing message handling...
            msg_type = message.get('type')
            if msg_type == 'auth':
                await self._handle_auth(message)
            elif msg_type == 'data':
                await self._handle_data(message)
            
    async def _handle_consensus_proposal(self, message: Dict):
        """Handle incoming consensus proposals"""
        proposal_id = message.get('proposal_id')
        change_type = message.get('change_type')
        proposed_change = message.get('change')

        if not all([proposal_id, change_type, proposed_change]):
            return

        # Validate proposal based on type
        if not await self._validate_proposal(change_type, proposed_change):
            return

        # Get voting power for proposal
        voting_power = self.consensus.get_voting_power(self.node_id)
        if voting_power <= 0:
            return

        # Cast vote
        vote = await self._evaluate_proposal(change_type, proposed_change)
        vote_msg = {
            'type': 'consensus_vote',
            'proposal_id': proposal_id,
            'vote': vote,
            'voter': self.node_id,
            'voting_power': voting_power
        }

        # Broadcast vote
        await self.broadcast_message(vote_msg)

    async def _handle_consensus_vote(self, message: Dict):
        """Process consensus votes and apply changes when threshold met"""
        proposal_id = message.get('proposal_id')
        if not proposal_id or proposal_id not in self.pending_state_changes:
            return

        proposal = self.pending_state_changes[proposal_id]
        votes = proposal['votes']
        votes[message['voter']] = message['vote']

        # Calculate vote result
        total_power = sum(
            self.consensus.get_voting_power(voter)
            for voter in votes.keys()
        )
        approve_power = sum(
            self.consensus.get_voting_power(voter)
            for voter, vote in votes.items() if vote
        )

        threshold = self.consensus_thresholds[proposal['change_type']]
        if approve_power / total_power >= threshold:
            await self._apply_state_change(proposal)
            del self.pending_state_changes[proposal_id]

    async def propose_state_change(self, change_type: str, change: Dict) -> bool:
        """Propose network state change requiring consensus"""
        if change_type not in self.consensus_thresholds:
            raise ValueError(f"Invalid change type: {change_type}")

        proposal_id = f"{self.node_id}_{int(time.time())}"
        proposal = {
            'type': 'consensus_proposal',
            'proposal_id': proposal_id,
            'change_type': change_type,
            'change': change,
            'proposer': self.node_id,
            'timestamp': time.time()
        }

        # Store proposal
        self.pending_state_changes[proposal_id] = {
            'change_type': change_type,
            'change': change,
            'votes': {},
            'timestamp': time.time()
        }

        # Broadcast proposal
        await self.broadcast_message(proposal)
        return True

    async def _validate_proposal(self, change_type: str, change: Dict) -> bool:
        """Validate proposed change based on type"""
        try:
            if change_type == 'peer_ban':
                return await self._validate_ban_proposal(change)
            elif change_type == 'reputation_update':
                return await self._validate_reputation_proposal(change)
            elif change_type == 'cluster_reconfig':
                return await self._validate_cluster_proposal(change)
            elif change_type == 'network_param':
                return await self._validate_param_proposal(change)
            return False
        except Exception as e:
            self.logger.error(f"Proposal validation error: {str(e)}")
            return False

    async def _apply_state_change(self, proposal: Dict):
        """Apply approved state change"""
        change_type = proposal['change_type']
        change = proposal['change']

        try:
            if change_type == 'peer_ban':
                await self._ban_peer(change['peer_id'])
            elif change_type == 'reputation_update':
                await self.reputation_manager.update_reputation_interactive(
                    change['peer_id'],
                    change['delta'],
                    reason='consensus_approved'
                )
            elif change_type == 'cluster_reconfig':
                await self.cluster_manager.reconfigure_clusters(change)
            elif change_type == 'network_param':
                await self._update_network_params(change)

            self.logger.info(f"Applied {change_type} change: {change}")

        except Exception as e:
            self.logger.error(f"Failed to apply state change: {str(e)}")

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
        """Run network with monitoring and load balancing"""
        try:
            while self.running and not self._interrupt_requested:
                # Periodic health check
                current_time = time.time()
                if current_time - self._last_health_check >= self.health_check_interval:
                    health = await self.network_monitor.check_network_health()
                    if health and health.overall_health < 0.5:
                        logger.warning(f"Low network health: {health.overall_health:.2f}")
                        await self._optimize_network()
                    self._last_health_check = current_time

                # Update node capacity
                metrics = get_resource_metrics()
                await self.load_balancer.register_node(
                    self.node_id,
                    NodeCapacity(
                        cpu_available=100 - metrics.cpu_usage,
                        memory_available=100 - metrics.memory_usage,
                        bandwidth_available=100 - metrics.network_load,
                        current_tasks=len(self._active_tasks)
                    )
                )

                # Regular network operations
                await self._discover_peers_interactive()
                await self._handle_messages_interactive()

                # Collect debug metrics
                if self.debug_enabled:
                    metrics = debug_manager.get_metrics()
                    if metrics.error_count > 0:
                        logger.warning(f"Debug metrics: {metrics}")

                await asyncio.sleep(1)

        except Exception as e:
            debug_manager.track_error(e, {
                'node_id': self.node_id,
                'peer_count': len(self.peers),
                'active_tasks': len(self._active_tasks)
            })
            logger.error(f"Network error: {str(e)}")
            raise
        finally:
            await self._cleanup_interactive()

    async def _optimize_network(self):
        """Optimize network based on health metrics"""
        try:
            # Rebalance connections
            await self.cluster_manager._rebalance_clusters()
            
            # Clean up inactive peers
            await self.reputation_manager.cleanup()
            
            # Optimize resource usage
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Network optimization failed: {str(e)}")

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
                raise PeerConnectionError(f"No connection available for peer {peer_id}")

            try:
                async with asyncio.timeout(5):  # 5 second timeout
                    encoded = await self.message_protocol.encode_message_interactive(message)
                    if not encoded:
                        raise MessageValidationError("Failed to encode message")

                    conn.writer.write(encoded)
                    await conn.writer.drain()
                    return True
            except asyncio.TimeoutError:
                raise MessageTimeoutError(f"Message send timeout to peer {peer_id}")

        except PeerConnectionError as e:
            self.logger.error(f"Connection error: {str(e)}")
            if self.interactive:
                await self.session.log_error(f"Connection failed: {str(e)}")
            return False
        except MessageTimeoutError as e:
            self.logger.error(f"Timeout error: {str(e)}")
            if self.interactive:
                await self.session.log_error(f"Message timeout: {str(e)}")
            return False
        except MessageValidationError as e:
            self.logger.error(f"Validation error: {str(e)}")
            if self.interactive:
                await self.session.log_error(f"Message validation failed: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error sending message to {peer_id}: {str(e)}")
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

    async def start_admin_shell(self):
        """Start interactive admin shell"""
        if not self.interactive:
            raise RuntimeError("Admin shell requires interactive mode")
        await self.admin.start_interactive_shell()
