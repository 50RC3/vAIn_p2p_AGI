import asyncio
import logging
from typing import Dict, Set, Optional
from datetime import datetime, timedelta
from collections import defaultdict
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
    def __init__(self, node_id: str, network_config: Dict):
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
