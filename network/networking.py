import asyncio
import logging
import socket
import time
from typing import Dict, List, Any, Optional, Callable, Tuple, Set
from enum import Enum
import ipaddress
import random

from network.admin_commands import AdminCommands
from security.firewall_rules import FirewallManager
from security.auth_manager import AuthManager
from network.monitoring import ResourceMonitor
from utils.config import Config

logger = logging.getLogger(__name__)

class ConnectionStatus(Enum):
    DISCONNECTED = 0
    CONNECTING = 1
    CONNECTED = 2
    FAILED = 3
    BANNED = 4

class PeerType(Enum):
    UNKNOWN = 0
    FULL_NODE = 1
    LIGHT_NODE = 2
    VALIDATOR = 3
    RELAY = 4
    GATEWAY = 5

class PeerInfo:
    def __init__(
        self,
        peer_id: str,
        ip_address: str,
        port: int,
        peer_type: PeerType = PeerType.UNKNOWN,
        reputation: float = 0.5,
        last_seen: float = 0.0,
        connection_status: ConnectionStatus = ConnectionStatus.DISCONNECTED,
        metadata: Dict[str, Any] = None
    ):
        self.peer_id = peer_id
        self.ip_address = ip_address
        self.port = port
        self.peer_type = peer_type
        self.reputation = max(0.0, min(1.0, reputation))  # Constrain to [0,1]
        self.last_seen = last_seen if last_seen > 0 else time.time()
        self.connection_status = connection_status
        self.metadata = metadata or {}
        self.connection_attempts = 0
        self.connection_failures = 0
        self.bytes_received = 0
        self.bytes_sent = 0
        self.message_count_received = 0
        self.message_count_sent = 0
        self.latency_ms = 0
        self.features = set()
        self.protocol_version = "1.0.0"
        
    def __str__(self) -> str:
        status_str = self.connection_status.name
        return f"Peer {self.peer_id[:8]} | {self.ip_address}:{self.port} | {self.peer_type.name} | Rep: {self.reputation:.2f} | {status_str}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "peer_id": self.peer_id,
            "ip_address": self.ip_address,
            "port": self.port,
            "peer_type": self.peer_type.name,
            "reputation": self.reputation,
            "last_seen": self.last_seen,
            "status": self.connection_status.name,
            "latency_ms": self.latency_ms,
            "protocol_version": self.protocol_version,
            "features": list(self.features),
            **{f"metrics.{k}": v for k, v in self._get_metrics().items()}
        }
        
    def _get_metrics(self) -> Dict[str, Any]:
        return {
            "bytes_received": self.bytes_received,
            "bytes_sent": self.bytes_sent,
            "messages_received": self.message_count_received,
            "messages_sent": self.message_count_sent,
            "connection_attempts": self.connection_attempts,
            "connection_failures": self.connection_failures
        }
        
    def update_reputation(self, delta: float) -> None:
        """Update the peer's reputation score with the given delta."""
        self.reputation = max(0.0, min(1.0, self.reputation + delta))

class NetworkManager:
    """Manages network connections and peer-to-peer communication."""
    
    def __init__(self, config: Optional[Config] = None):
        self.peers: Dict[str, PeerInfo] = {}
        self.banned_peers: Dict[str, Tuple[float, str]] = {}  # peer_id -> (timestamp, reason)
        self.connected_peers: Set[str] = set()
        self.firewall = FirewallManager()
        self.auth_manager = AuthManager()
        self.resource_monitor = ResourceMonitor()
        self.config = config or Config()
        self.max_peers = self.config.get("network.max_peers", 50)
        self.min_reputation = self.config.get("network.min_reputation", 0.2)
        self.discovery_enabled = self.config.get("network.discovery_enabled", True)
        self.node_id = self._generate_node_id()
        self.listening_port = self.config.get("network.port", 9765)
        self._server = None
        self._discovery_task = None
        self._heartbeat_task = None
        self._running = False
        
    async def start(self) -> None:
        """Start the network manager and begin listening for connections."""
        if self._running:
            logger.warning("NetworkManager is already running")
            return
            
        self._running = True
        
        # Start listening for incoming connections
        try:
            self._server = await asyncio.start_server(
                self._handle_incoming_connection,
                '0.0.0.0',  # Listen on all interfaces
                self.listening_port
            )
            logger.info(f"Network manager listening on port {self.listening_port}")
            
            # Start background tasks
            self._discovery_task = asyncio.create_task(self._peer_discovery_loop())
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            # Connect to bootstrap nodes
            await self._connect_to_bootstrap_nodes()
            
            logger.info(f"Network manager started with node ID: {self.node_id}")
        except Exception as e:
            logger.error(f"Failed to start network manager: {str(e)}")
            self._running = False
            raise
        
    async def stop(self) -> None:
        """Stop the network manager and close all connections."""
        if not self._running:
            return
            
        logger.info("Stopping network manager...")
        self._running = False
        
        # Cancel background tasks
        if self._discovery_task and not self._discovery_task.done():
            self._discovery_task.cancel()
            
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            
        # Close all connections
        for peer_id in list(self.connected_peers):
            await self.disconnect_peer(peer_id, "NetworkManager shutting down")
            
        # Close server
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
            
        logger.info("Network manager stopped")
        
    async def connect_to_peer(self, peer_id: str) -> bool:
        """
        Connect to a specific peer by ID.
        
        Returns:
            bool: True if connection was successful, False otherwise
        """
        if peer_id not in self.peers:
            logger.error(f"Cannot connect to unknown peer: {peer_id}")
            return False
            
        peer = self.peers[peer_id]
        
        if peer.connection_status == ConnectionStatus.CONNECTED:
            logger.info(f"Already connected to peer {peer_id}")
            return True
            
        if peer.connection_status == ConnectionStatus.BANNED or peer_id in self.banned_peers:
            logger.warning(f"Cannot connect to banned peer {peer_id}")
            return False
            
        # Update connection metrics
        peer.connection_attempts += 1
        peer.connection_status = ConnectionStatus.CONNECTING
        
        try:
            # Check if peer is allowed by firewall
            if not await self._firewall_check(peer.ip_address, peer.port):
                logger.warning(f"Connection to {peer_id} blocked by firewall")
                peer.connection_status = ConnectionStatus.FAILED
                return False
            
            # Establish connection
            reader, writer = await asyncio.open_connection(peer.ip_address, peer.port)
            
            # Perform authentication handshake
            success = await self._authenticate_connection(reader, writer, peer)
            if not success:
                logger.warning(f"Authentication failed for peer {peer_id}")
                peer.connection_status = ConnectionStatus.FAILED
                peer.connection_failures += 1
                writer.close()
                await writer.wait_closed()
                return False
                
            # Start background task to handle messages from this peer
            asyncio.create_task(self._handle_peer_messages(peer_id, reader, writer))
            
            # Update peer info
            peer.connection_status = ConnectionStatus.CONNECTED
            peer.last_seen = time.time()
            self.connected_peers.add(peer_id)
            
            logger.info(f"Successfully connected to peer {peer_id}")
            return True
            
        except (ConnectionRefusedError, TimeoutError) as e:
            logger.warning(f"Failed to connect to peer {peer_id}: {str(e)}")
            peer.connection_status = ConnectionStatus.FAILED
            peer.connection_failures += 1
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to peer {peer_id}: {str(e)}")
            peer.connection_status = ConnectionStatus.FAILED
            peer.connection_failures += 1
            return False
            
    async def disconnect_peer(self, peer_id: str, reason: str = "Disconnected") -> bool:
        """Disconnect from a specific peer."""
        if peer_id not in self.peers or peer_id not in self.connected_peers:
            logger.warning(f"Cannot disconnect: Peer {peer_id} not connected")
            return False
            
        peer = self.peers[peer_id]
        
        # Update peer info
        peer.connection_status = ConnectionStatus.DISCONNECTED
        if peer_id in self.connected_peers:
            self.connected_peers.remove(peer_id)
            
        logger.info(f"Disconnected from peer {peer_id}: {reason}")
        return True
        
    async def broadcast_message(self, message_type: str, data: Any, exclude_peers: List[str] = None) -> int:
        """
        Broadcast a message to all connected peers.
        
        Args:
            message_type: Type of message to broadcast
            data: Message data to broadcast
            exclude_peers: List of peer IDs to exclude from broadcast
            
        Returns:
            int: Number of peers the message was sent to
        """
        exclude_peers = exclude_peers or []
        target_peers = [pid for pid in self.connected_peers if pid not in exclude_peers]
        
        send_count = 0
        for peer_id in target_peers:
            if await self.send_message(peer_id, message_type, data):
                send_count += 1
                
        return send_count
        
    async def send_message(self, peer_id: str, message_type: str, data: Any) -> bool:
        """
        Send a message to a specific peer.
        
        Args:
            peer_id: ID of the peer to send to
            message_type: Type of message to send
            data: Message data to send
            
        Returns:
            bool: True if message was sent successfully, False otherwise
        """
        if peer_id not in self.peers or peer_id not in self.connected_peers:
            logger.error(f"Cannot send message to disconnected peer {peer_id}")
            return False
            
        peer = self.peers[peer_id]
        
        try:
            # Here you would actually serialize and send the message
            # This is a placeholder for the actual message sending logic
            message = {
                "type": message_type,
                "sender": self.node_id,
                "timestamp": time.time(),
                "data": data
            }
            
            # Update statistics
            message_size = 1024  # Placeholder - would be actual size
            peer.bytes_sent += message_size
            peer.message_count_sent += 1
            
            logger.debug(f"Sent {message_type} message to {peer_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message to peer {peer_id}: {str(e)}")
            return False
            
    def add_peer(self, peer_info: PeerInfo) -> bool:
        """Add a new peer to the known peers list."""
        if peer_info.peer_id in self.peers:
            # Update existing peer info
            existing_peer = self.peers[peer_info.peer_id]
            existing_peer.ip_address = peer_info.ip_address
            existing_peer.port = peer_info.port
            existing_peer.peer_type = peer_info.peer_type
            existing_peer.last_seen = time.time()
            
            # Don't overwrite connection status if already connected
            if existing_peer.connection_status != ConnectionStatus.CONNECTED:
                existing_peer.connection_status = peer_info.connection_status
                
            # Merge metadata
            if peer_info.metadata:
                existing_peer.metadata.update(peer_info.metadata)
                
            logger.debug(f"Updated existing peer: {peer_info.peer_id}")
            return False
            
        # Check if the peer is banned
        if peer_info.peer_id in self.banned_peers:
            ban_time, reason = self.banned_peers[peer_info.peer_id]
            logger.warning(f"Ignoring banned peer {peer_info.peer_id}: {reason}")
            return False
            
        # Add the new peer
        self.peers[peer_info.peer_id] = peer_info
        logger.info(f"Added new peer: {peer_info}")
        return True
        
    def get_peer_info(self, peer_id: str) -> Optional[PeerInfo]:
        """Get information about a specific peer."""
        return self.peers.get(peer_id)
        
    def get_all_peers(self) -> Dict[str, PeerInfo]:
        """Get all known peers."""
        return self.peers.copy()
        
    def get_connected_peers(self) -> Dict[str, PeerInfo]:
        """Get all currently connected peers."""
        return {pid: self.peers[pid] for pid in self.connected_peers if pid in self.peers}
        
    async def ban_peer(self, peer_id: str, reason: str = "Banned by system") -> bool:
        """
        Ban a peer from connecting.
        
        Args:
            peer_id: The ID of the peer to ban
            reason: The reason for the ban
            
        Returns:
            bool: True if the peer was banned, False otherwise
        """
        if peer_id not in self.peers:
            logger.warning(f"Cannot ban unknown peer {peer_id}")