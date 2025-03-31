import asyncio
import logging
from typing import Dict, Set, Optional
from datetime import datetime, timedelta
from collections import defaultdict
from tqdm import tqdm
import torch
import time
import os

from core.constants import InteractionLevel
from core.interactive_utils import InteractiveSession, InteractiveConfig
from .connection_pool import ConnectionPool
from .udp_broadcast import UDPBroadcast
from .dht import DHT
from .pex import PeerExchange
from .message_protocol import SecureMessageProtocol
from security.auth import NodeAuthenticator
from network.consensus import ConsensusManager
from .rate_limiter import RateLimiter
from .circuit_breaker import CircuitBreaker
from security.secure_key_manager import SecureKeyManager
from .cluster_manager import ClusterManager
from .admin_commands import AdminCommands
from .reputation import ReputationManager
from .load_balancer import LoadBalancer, NodeCapacity
from .monitoring import NetworkMonitor, get_resource_metrics
from utils.debug_utils import DebugManager  # Import from the correct source

# Define INTERACTION_TIMEOUTS with default values
INTERACTION_TIMEOUTS = {
    "auth": 10,  # Timeout for authentication in seconds
    "default": 30,  # Default timeout for interactions
    "confirmation": 15  # Timeout for user confirmation
}

# Initialize logger
logger = logging.getLogger('P2PNetwork')

# Initialize debug manager
debug_manager = DebugManager()

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
        
        # Extract and set different ports for DHT and UDP
        udp_port = network_config['udp'].get('port', 8468)
        dht_port = network_config.get('dht', {}).get('port', 8469)  # Default to 8469 for DHT
        
        # Update DHT config with separate port if not explicitly set
        dht_config = network_config['dht']
        if 'port' not in dht_config:
            dht_config['port'] = dht_port
        
        # Initialize P2P components with separate ports
        self.dht = DHT(node_id, dht_config)
        self.udp_broadcast = UDPBroadcast({'port': udp_port})
        
        # Initialize other components
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
        
        # Initialize reputation manager with persistence
        reputation_path = network_config.get('reputation_path', None)
        if reputation_path:
            # If path is relative, make it relative to the config file location
            if not os.path.isabs(reputation_path):
                config_dir = os.path.dirname(os.path.abspath(
                    network_config.get('config_path', '.')
                ))
                reputation_path = os.path.join(config_dir, reputation_path)
                
            # Ensure directory exists
            os.makedirs(os.path.dirname(reputation_path), exist_ok=True)
        
        self.reputation_manager = ReputationManager(
            storage_path=reputation_path,
            persistence_interval=network_config.get('reputation_save_interval', 300)
        )
        
        # Add tracking for startup time
        self._start_time = time.time()

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
        
        # Initialize tracking dictionaries
        self._peer_violations = defaultdict(int)
        self._rate_limit_backoffs = {}
        self._circuit_reset_scheduled = False

        # Add message priority handling
        self.message_priorities = {
            'consensus_proposal': 10,  # Highest priority
            'consensus_vote': 9,
            'peer_banned': 8,
            'state_change': 7,
            'auth': 6,
            'resource_contribution': 5,
            'data': 4,
            'heartbeat': 1      # Lowest priority
        }
        
        # Add priority queues
        self.high_priority_queue = asyncio.PriorityQueue()
        self.normal_priority_queue = asyncio.PriorityQueue()
        self.low_priority_queue = asyncio.PriorityQueue()

        # Add enhanced diagnostics
        from collections import OrderedDict
        self._diagnostic_data = OrderedDict()
        self._diag_retention_time = network_config.get('diagnostic_retention', 3600)  # 1 hour
        self._last_diagnostics_cleanup = time.time()
        self.max_concurrent_tasks = network_config.get('max_concurrent_tasks', 100)

    async def start(self):
        """Start the P2P network node asynchronously."""
        self.running = True
        self.logger.info(f"Starting P2P node {self.node_id}")
        await self._run_network()

    def start_sync(self):
        """Synchronous wrapper for starting the network"""
        self.running = True
        self.logger.info(f"Starting P2P node {self.node_id}")
        asyncio.run(self._run_network())

    async def stop(self):
        """Gracefully stop the P2P network node with task cleanup."""
        self.running = False
        self.logger.info(f"Stopping P2P node {self.node_id}")
        
        try:
            # Create event loop if needed
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            # Define cleanup_task before using it
            cleanup_task = self._cleanup()
            
            # Run cleanup with timeout
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
        try:
            # Start P2P services
            await self.udp_broadcast.start()
            await self.dht.start()
            await self.reputation_manager.start()
            
            while self.running:
                # Handle peer discovery and communication
                new_peers = await self._discover_peers()
                await self._handle_messages()
                self.peers.update(new_peers)
                await asyncio.sleep(1)
        except Exception as e:
            self.logger.error(f"Network error: {str(e)}")
            raise
        finally:
            await self._cleanup()

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
        """Enhanced task registration with resource awareness"""
        # Check resource limits before taking new tasks
        if len(self._active_tasks) >= self.max_concurrent_tasks:
            # If we're at capacity, handle based on task priority
            task_info = getattr(task, 'task_info', {})
            priority = task_info.get('priority', 5) # Default medium priority
            
            if priority < 7:  # Not high priority
                self.logger.warning("Task capacity reached, rejecting non-critical task")
                task.cancel()
                return
            
            # For high priority tasks, try to make room by cancelling low priority tasks
            cancelled = await self._cancel_low_priority_tasks()
            if not cancelled and len(self._active_tasks) >= self.max_concurrent_tasks:
                self.logger.error("Critical task capacity reached, system overloaded")
                # Still accept the critical task
        
        # Add task tracking info    
        self._active_tasks.add(task)
        setattr(task, 'created_at', time.time())
        
        try:
            await task
        except asyncio.CancelledError:
            pass
        finally:
            self._active_tasks.discard(task)

    async def _cancel_low_priority_tasks(self) -> bool:
        """Cancel low priority tasks to make room for high priority ones"""
        # Find candidate tasks for cancellation
        low_priority_tasks = []
        
        for task in self._active_tasks:
            task_info = getattr(task, 'task_info', {})
            priority = task_info.get('priority', 5)
            
            # Consider low priority tasks that aren't near completion
            if priority <= 3 and not task.done():
                low_priority_tasks.append(task)
        
        # Cancel up to 2 tasks
        cancelled = 0
        for task in low_priority_tasks[:2]:
            task.cancel()
            cancelled += 1
        
        return cancelled > 0

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
            return await asyncio.wait_for(
                self._authenticate_peer(peer_id, token), 
                timeout=INTERACTION_TIMEOUTS["auth"]
            )
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
            response = await asyncio.wait_for(self.node_comm.send_message(peer_id, {"type": "auth"}), timeout=10)
            auth_msg = {
                "type": "auth",
                "response": response
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
        """Process incoming messages with priority"""
        # Process a batch of messages from each queue based on priority
        
        # Process all high priority messages
        while not self.high_priority_queue.empty():
            _, msg = await self.high_priority_queue.get()
            await self._handle_message(msg, msg.get('sender'))
            self.high_priority_queue.task_done()
        
        # Process up to 10 normal priority messages
        for _ in range(10):
            if self.normal_priority_queue.empty():
                break
            _, msg = await self.normal_priority_queue.get()
            await self._handle_message(msg, msg.get('sender'))
            self.normal_priority_queue.task_done()
        
        # Process up to 5 low priority messages if we're not too busy
        if len(self._active_tasks) < self.max_concurrent_tasks // 2:
            for _ in range(5):
                if self.low_priority_queue.empty():
                    break
                _, msg = await self.low_priority_queue.get()
                await self._handle_message(msg, msg.get('sender'))
                self.low_priority_queue.task_done()
        
        # Also process any messages from the original queue for backward compatibility
        while not self.node_comm.message_queue.empty():
            msg = await self.node_comm.message_queue.get()
            # Route to appropriate priority queue for next iteration
            await self._prioritize_message(msg)
            self.node_comm.message_queue.task_done()

    async def _prioritize_message(self, message: Dict):
        """Route message to appropriate priority queue"""
        msg_type = message.get('type', 'unknown')
        priority = self.message_priorities.get(msg_type, 3)  # Default medium priority
        
        # Adjust priority based on sender reputation
        sender = message.get('sender')
        if sender in self.peer_reputation:
            rep = self.peer_reputation[sender]
            if rep > 0.8:  # Trusted peers get priority boost
                priority += 1
            elif rep < 0.3:  # Low reputation peers get lowered priority
                priority -= 1
        
        # Route to appropriate queue
        if priority >= 7:
            await self.high_priority_queue.put((10-priority, message))
        elif priority >= 4:
            await self.normal_priority_queue.put((10-priority, message))
        else:
            await self.low_priority_queue.put((10-priority, message))

    async def _handle_message(self, message: Dict, addr):
        """Handle incoming encrypted messages with improved error recovery"""
        try:
            if not await self.rate_limiter.allow_request(addr):
                # Enhanced rate limiting with backoff strategy
                backoff_seconds = await self._calculate_backoff(addr)
                self.logger.warning(f"Rate limit exceeded for {addr}, applying backoff of {backoff_seconds}s")
                # Record the backoff in a tracking dictionary
                self._rate_limit_backoffs[addr] = (datetime.now(), backoff_seconds)
                return

            if not self.circuit_breaker.allow_request():
                # Circuit breaker with health check scheduling
                self.logger.error(f"Circuit breaker open for {addr}")
                if not self._circuit_reset_scheduled:
                    reset_task = asyncio.create_task(self._schedule_circuit_reset())
                    await self.register_task(reset_task)
                    self._circuit_reset_scheduled = True
                return

            # ... existing decryption and processing code ...
            
        except MessageValidationError as e:
            self.logger.error(f"Message validation error from {addr}: {e}")
            # Apply graduated response based on severity
            severity = self._assess_validation_error_severity(e)
            await self._record_suspicious_activity(message.get('sender'), f"validation_error:{severity}")
            
        except Exception as e:
            self.logger.error(f"Unexpected error handling message from {addr}: {e}")
            self.circuit_breaker.record_failure()
            # Add diagnostic info collection
            await self._collect_diagnostic_info(message, addr, e)

    async def _collect_diagnostic_info(self, message: Dict, addr, error):
        """Collect diagnostic information for errors"""
        now = time.time()
        
        # Create diagnostic entry
        diag_key = f"{addr}_{now}"
        self._diagnostic_data[diag_key] = {
            'timestamp': now,
            'peer': addr,
            'error': str(error),
            'error_type': type(error).__name__,
            'message_type': message.get('type', 'unknown'),
            'message_size': len(str(message)),
            'active_tasks': len(self._active_tasks),
            'system_load': get_resource_metrics()
        }
        
        # Clean up old diagnostic data periodically
        if now - self._last_diagnostics_cleanup > 300:  # Every 5 minutes
            self._cleanup_diagnostics()
            self._last_diagnostics_cleanup = now

    def _cleanup_diagnostics(self):
        """Clean up old diagnostic data"""
        now = time.time()
        cutoff = now - self._diag_retention_time
        
        # Remove old entries
        old_keys = []
        for key, value in list(self._diagnostic_data.items()):
            if value['timestamp'] < cutoff:
                old_keys.append(key)
                
        for key in old_keys:
            del self._diagnostic_data[key]
            
        self.logger.debug(f"Cleaned up {len(old_keys)} old diagnostic entries")

    async def get_diagnostics_report(self):
        """Generate diagnostics report for admin interface"""
        report = {
            'node_id': self.node_id,
            'uptime': time.time() - self._start_time,
            'peer_count': len(self.peers),
            'banned_peers': len(self.banned_peers),
            'active_tasks': len(self._active_tasks),
            'pending_consensus': len(self.pending_state_changes),
            'recent_errors': sum(1 for v in self._diagnostic_data.values() 
                                  if time.time() - v['timestamp'] < 600),  # Last 10 minutes
            'system_metrics': get_resource_metrics(),
            'circuit_breaker_state': self.circuit_breaker.get_state(),
            'top_peers_by_traffic': await self.network_monitor.get_top_peers_by_traffic(5),
            'cluster_health': await self.cluster_manager.get_cluster_health(),
        }
        
        return report

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
                    if health is None or not hasattr(health, 'overall_health'):
                        self.logger.error("Invalid health metrics received from network monitor")
                        continue  # Changed from return to continue to avoid premature exit
                    if health.overall_health < 0.5:
                        self.logger.warning(f"Low network health: {health.overall_health:.2f}")
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
                        self.logger.warning(f"Debug metrics: {metrics}")

                await asyncio.sleep(1)

        except Exception as e:
            debug_manager.track_error(e, {
                'node_id': self.node_id,
                'peer_count': len(self.peers),
                'active_tasks': len(self._active_tasks)
            })
            self.logger.error(f"Network error: {str(e)}")
            raise
        finally:
            await self._cleanup_interactive()

    async def _optimize_network(self):
        """Optimize network based on health metrics"""
        try:
            # Rebalance connections
            await self.cluster_manager._rebalance_clusters()
            
            # Clean up inactive peers
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            # Optimize resource usage
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

        except Exception as e:
            self.logger.error(f"Network optimization failed: {str(e)}")

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
            raise

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

        except (PeerConnectionError, MessageTimeoutError, MessageValidationError, Exception) as e:
            await self._log_error_interactive(e, peer_id)
            return False

    async def _log_error_interactive(self, error: Exception, peer_id: str):
        """Helper method to log errors and optionally log to the interactive session"""
        error_type = type(error).__name__
        self.logger.error(f"{error_type} error for peer {peer_id}: {str(error)}")
        if self.interactive and self.session:
            await self.session.log_error(f"{error_type} error: {str(error)}")

    async def _calculate_backoff(self, peer_id: str) -> float:
        """Calculate adaptive backoff time based on peer history"""
        # Exponential backoff with caps
        base_backoff = 1.0
        max_backoff = 300.0  # 5 minutes max

        # Get current violation count, default to 0
        violations = self._peer_violations.get(peer_id, 0)

        # Cap violations to a maximum value to prevent excessive memory usage
        max_violations = 10  # Reasonable cap for violations
        capped_violations = min(violations, max_violations)

        # Calculate exponential backoff
        backoff = min(base_backoff * (2 ** capped_violations), max_backoff)
        
        # Increment violation count
        self._peer_violations[peer_id] = violations + 1
        
        return backoff

    async def _schedule_circuit_reset(self, retry_count=0, max_retries=5):
        """Schedule circuit breaker reset with health check and retry limit"""
        try:
            if retry_count >= max_retries:
                self.logger.error("Maximum circuit breaker reset retries reached. Aborting further attempts.")
                return
            
            # Wait for reset timeout
            await asyncio.sleep(self.circuit_breaker.reset_timeout)
            
            # Check system health before resetting
            health = await self.network_monitor.check_network_health()
            
            if health and health.overall_health > 0.7:  # Only reset if health is good
                self.circuit_breaker.reset()
                self.logger.info("Circuit breaker reset after recovery period")
            else:
                self.logger.warning(f"Circuit remains open due to poor network health. Retry {retry_count + 1}/{max_retries}")
                # Reschedule another reset attempt
                reset_task = asyncio.create_task(self._schedule_circuit_reset(retry_count + 1, max_retries))
                await self.register_task(reset_task)
        finally:
            self._circuit_reset_scheduled = False

    async def _assess_validation_error_severity(self, error: Exception) -> int:
        """Assess the severity of a validation error
        
        Returns:
            int: Severity level (1-5) with 5 being most severe
        """
        error_str = str(error).lower()
        
        # Critical security issues
        if any(x in error_str for x in ['signature', 'tamper', 'forge']):
            return 5
        # Protocol violations
        elif any(x in error_str for x in ['protocol', 'invalid format']):
            return 4
        # Missing fields
        elif 'missing' in error_str:
            return 3
        # Type errors
        elif 'type' in error_str:
            return 2
        # Other validation issues
        else:
            return 1

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
