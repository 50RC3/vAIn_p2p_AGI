import asyncio
import json
import socket
import logging
import sys
import uuid
import aioice
from typing import Dict, Optional, Callable, Any, Tuple, List, Set
from dataclasses import dataclass
from datetime import datetime

@dataclass
class NetworkError:
    """Detailed network error information"""
    timestamp: datetime
    error_type: str
    message: str
    source: str
    details: Dict[str, Any]

@dataclass
class MessageFragment:
    """Fragment of a large message"""
    message_id: str
    fragment_id: int
    total_fragments: int
    data: bytes
    timestamp: float

@dataclass
class NATInfo:
    """NAT traversal information"""
    public_ip: str
    public_port: int
    nat_type: str
    mapped_address: Optional[Tuple[str, int]] = None
    relay_address: Optional[Tuple[str, int]] = None

class UDPBroadcastProtocol(asyncio.DatagramProtocol):
    def __init__(self, on_message: Callable):
        self.transport = None
        self.on_message = on_message
        
    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        try:
            message = json.loads(data.decode())
            asyncio.create_task(self.on_message(message, addr))
        except json.JSONDecodeError as e:
            error = NetworkError(
                timestamp=datetime.now(),
                error_type="decode_error",
                message=str(e),
                source=f"{addr[0]}:{addr[1]}",
                details={"raw_data": data[:100].hex()}  # First 100 bytes for debugging
            )
            logging.error(f"Message decode error: {error}")

class UDPBroadcast:
    """Handles UDP broadcast messaging for peer discovery and communication."""
    
    def __init__(self, port: int, logger: Optional[logging.Logger] = None, interactive: bool = True) -> None:
        """Initialize UDP broadcast handler.
        
        Args:
            port: UDP port number (1024-65535)
            logger: Optional logger instance
            interactive: Enable interactive mode
        """
        if not 1024 <= port <= 65535:
            raise ValueError("Port must be between 1024 and 65535")
        self.port = port
        self.logger = logger or logging.getLogger(__name__)
        self.broadcast_ip = '255.255.255.255'
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.running = False
        self.interactive = interactive
        self.session = None
        self._interrupt_requested = False
        self._cleanup_event = asyncio.Event()
        self._active_tasks = set()
        self._message_queue = asyncio.Queue()
        self._processing_queue = asyncio.Queue()
        self._worker_tasks = set()
        self._max_concurrent_messages = 5
        self._rate_limit = 100  # messages per second
        self._last_broadcast = 0.0
        self._worker_count = 0
        self._max_queue_size = 1000
        self._error_history = []
        self._max_error_history = 100
        self._socket_timeout = 5.0  # 5 second socket timeout

        # Fragmentation settings
        self._max_fragment_size = 512  # Bytes per fragment
        self._fragment_timeout = 30.0  # Seconds to wait for all fragments
        self._pending_fragments: Dict[str, List[MessageFragment]] = {}
        self._fragment_cleanup_task = None

        # NAT traversal settings
        self._stun_servers = [
            ('stun.l.google.com', 19302),
            ('stun1.l.google.com', 19302)
        ]
        self._turn_servers = []  # Configure your TURN servers
        self._nat_info: Optional[NATInfo] = None
        self._peer_nat_info: Dict[str, NATInfo] = {}
        self._relay_connections: Set[str] = set()
        self._hole_punch_timeout = 30.0
        self._max_punch_retries = 3

        # Add retry and timeout settings
        self._broadcast_retry_count = 3
        self._broadcast_retry_delay = 1.0  # seconds
        self._broadcast_timeout = 5.0  # seconds
        self._pending_messages: Dict[str, Dict] = {}
        self._message_handlers = {
            'discovery': self._handle_discovery_message,
            'status': self._handle_status_message,
            'data': self._handle_data_message,
            'presence': self._handle_presence_message,
            'error': self._handle_error_message
        }

        # Multicast settings
        self.multicast_group = '239.255.255.250'  # Standard multicast address
        self.multicast_ttl = 1  # Default TTL for local network
        self.reuse_addr = True
        self._multicast_interfaces = []  # Track active network interfaces
        self._joined_groups = set()

        # Update broadcast_ip to use multicast
        self.broadcast_ip = self.multicast_group
        
        # Initialize multicast socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if self.reuse_addr:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if hasattr(socket, 'SO_REUSEPORT'):  # Linux support
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        
        # Set multicast TTL
        self.socket.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, self.multicast_ttl)
        
        # Allow multicast on loopback
        self.socket.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)

    async def start(self):
        """Start UDP multicast service."""
        if self.running:
            return

        try:
            # Determine NAT type and get public endpoint
            self._nat_info = await self._discover_nat_info()
            
            # Initialize ICE for NAT traversal
            self._ice_gatherer = aioice.Gatherer(
                stun_servers=self._stun_servers,
                turn_servers=self._turn_servers
            )
            await self._ice_gatherer.gather()

            loop = asyncio.get_running_loop()
            self.protocol = UDPBroadcastProtocol(self._handle_message)
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            if self.reuse_addr:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                if hasattr(socket, 'SO_REUSEPORT'):
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)

            # Bind to all interfaces
            sock.bind(('', self.port))
            
            # Join multicast group on all interfaces
            await self._join_multicast_groups(sock)
            
            self.transport, _ = await loop.create_datagram_endpoint(
                lambda: self.protocol,
                sock=sock
            )
            
            # Start message processing workers
            for _ in range(self._max_concurrent_messages):
                worker = asyncio.create_task(self._message_processor())
                self._worker_tasks.add(worker)
                worker.add_done_callback(self._worker_tasks.discard)

            # Start fragment cleanup task
            self._fragment_cleanup_task = asyncio.create_task(self._cleanup_fragments())
            self._worker_tasks.add(self._fragment_cleanup_task)
            
            self.running = True
            self.logger.info(f"Started UDP broadcast with {self._max_concurrent_messages} workers")
            
        except socket.timeout as e:
            self._record_error("socket_timeout", str(e), "start")
            raise TimeoutError(f"Socket operation timed out during startup: {e}")
        except socket.error as e:
            self._record_error("socket_error", str(e), "start", {"errno": e.errno})
            raise RuntimeError(f"Socket error during startup: {e}")
        except Exception as e:
            self._record_error("startup_error", str(e), "start")
            raise

    async def _join_multicast_groups(self, sock: socket.socket) -> None:
        """Join multicast group on all network interfaces."""
        import netifaces
        
        try:
            for iface in netifaces.interfaces():
                addrs = netifaces.ifaddresses(iface)
                for addr in addrs.get(netifaces.AF_INET, []):
                    ip = addr['addr']
                    try:
                        # Add membership for this interface
                        mreq = socket.inet_aton(self.multicast_group) + socket.inet_aton(ip)
                        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
                        self._multicast_interfaces.append(ip)
                        self._joined_groups.add((self.multicast_group, ip))
                        self.logger.debug(f"Joined multicast group on interface {ip}")
                    except Exception as e:
                        self.logger.warning(f"Failed to join multicast on {ip}: {e}")

        except Exception as e:
            self.logger.error(f"Error setting up multicast: {e}")
            raise

    async def stop(self):
        """Stop UDP multicast service."""
        if not self.running:
            return

        try:
            # Cancel all worker tasks
            for task in self._worker_tasks:
                task.cancel()
            
            if self._worker_tasks:
                await asyncio.gather(*self._worker_tasks, return_exceptions=True)
            self._worker_tasks.clear()

            # Leave multicast groups
            for group, iface in self._joined_groups:
                try:
                    mreq = socket.inet_aton(group) + socket.inet_aton(iface)
                    self.socket.setsockopt(socket.IPPROTO_IP, socket.IP_DROP_MEMBERSHIP, mreq)
                except Exception as e:
                    self.logger.warning(f"Error leaving multicast group on {iface}: {e}")

            # Close transport and socket
            if self.transport:
                self.transport.close()
            self.running = False
            self.socket.close()
            
            self.logger.info("Stopped UDP broadcast service and workers")
            
        except Exception as e:
            self.logger.error(f"Error stopping UDP broadcast: {e}")
            raise

    async def broadcast(self, message: Dict):
        """Broadcast message with retries and timeout control."""
        if not self.running:
            raise RuntimeError("UDP broadcast service not running")

        msg_id = message.get('id', str(uuid.uuid4()))
        retry_count = 0

        while retry_count < self._broadcast_retry_count:
            try:
                # Local broadcast with timeout
                async with asyncio.timeout(self._broadcast_timeout):
                    await self._local_broadcast(message)

                # Cross-network peer communication
                peer_tasks = []
                for peer_id, nat_info in self._peer_nat_info.items():
                    task = self._send_to_peer(peer_id, message)
                    peer_tasks.append(task)

                if peer_tasks:
                    # Wait for all peer sends with timeout
                    async with asyncio.timeout(self._broadcast_timeout):
                        results = await asyncio.gather(*peer_tasks, return_exceptions=True)
                        
                        # Check for failures
                        failures = [i for i, r in enumerate(results) if isinstance(r, Exception)]
                        if failures:
                            raise RuntimeError(f"Failed to send to {len(failures)} peers")

                return True

            except Exception as e:
                retry_count += 1
                self.logger.warning(f"Broadcast attempt {retry_count} failed: {e}")
                
                if retry_count < self._broadcast_retry_count:
                    await asyncio.sleep(self._broadcast_retry_delay)
                else:
                    self._record_error("broadcast_error", str(e), "broadcast", 
                                     {"message_id": msg_id, "attempts": retry_count})
                    raise

    async def _send_to_peer(self, peer_id: str, message: Dict) -> None:
        """Send message to a specific peer with connection fallback."""
        try:
            # Try direct connection first
            if await self._try_direct_connection(peer_id):
                await self._send_direct(peer_id, message)
                return
                
            # Fall back to relay if direct connection fails
            if peer_id in self._relay_connections:
                await self._send_relay(peer_id, message)
            else:
                # Attempt hole punching
                if await self._hole_punch(peer_id):
                    await self._send_direct(peer_id, message)
                else:
                    # Set up relay connection
                    await self._setup_relay(peer_id)
                    await self._send_relay(peer_id, message)
                    self._relay_connections.add(peer_id)
                    
        except Exception as e:
            self.logger.error(f"Failed to send to peer {peer_id}: {e}")
            raise

    async def _hole_punch(self, peer_id: str) -> bool:
        """Attempt UDP hole punching with peer."""
        nat_info = self._peer_nat_info.get(peer_id)
        if not nat_info:
            return False
            
        try:
            for attempt in range(self._max_punch_retries):
                # Exchange connection info
                connection = await self._ice_gatherer.create_connection(
                    peer_address=(nat_info.public_ip, nat_info.public_port)
                )
                
                try:
                    async with asyncio.timeout(self._hole_punch_timeout):
                        await connection.connect()
                        return True
                except asyncio.TimeoutError:
                    continue
                    
            return False
            
        except Exception as e:
            self.logger.error(f"Hole punching failed with {peer_id}: {e}")
            return False

    async def _discover_nat_info(self) -> NATInfo:
        """Discover NAT type and public endpoint using STUN."""
        try:
            # Try multiple STUN servers
            for stun_server in self._stun_servers:
                try:
                    public_ip, public_port = await self._stun_request(stun_server)
                    nat_type = await self._determine_nat_type(stun_server)
                    
                    return NATInfo(
                        public_ip=public_ip,
                        public_port=public_port,
                        nat_type=nat_type
                    )
                except Exception as e:
                    self.logger.warning(f"STUN request failed for {stun_server}: {e}")
                    continue
                    
            raise RuntimeError("Failed to get NAT info from any STUN server")
            
        except Exception as e:
            self.logger.error(f"NAT discovery failed: {e}")
            raise

    async def _local_broadcast(self, message: Dict):
        """Send message using multicast."""
        try:
            data = json.dumps(message).encode()
            data_size = len(data)
            
            if data_size > self._max_fragment_size:
                # Handle fragmentation if needed
                message_id = str(uuid.uuid4())
                fragments = self._fragment_message(data, message_id)
                
                for fragment in fragments:
                    await self._send_fragment(fragment)
                return
                
            # Send to multicast group
            if self.protocol and self.protocol.transport:
                self.protocol.transport.sendto(data, (self.multicast_group, self.port))
            self.socket.sendto(data, (self.multicast_group, self.port))

        except (socket.error, socket.timeout) as e:
            self._record_error("network_error", str(e), "multicast", 
                             {"message_type": message.get("type")})
            raise

    def _fragment_message(self, data: bytes, message_id: str) -> List[MessageFragment]:
        """Split message into fragments."""
        fragments = []
        total_fragments = (len(data) + self._max_fragment_size - 1) // self._max_fragment_size
        
        for i in range(total_fragments):
            start = i * self._max_fragment_size
            end = start + self._max_fragment_size
            fragment_data = data[start:end]
            
            fragments.append(MessageFragment(
                message_id=message_id,
                fragment_id=i,
                total_fragments=total_fragments,
                data=fragment_data,
                timestamp=asyncio.get_event_loop().time()
            ))
        
        return fragments

    async def _send_fragment(self, fragment: MessageFragment):
        """Send a single message fragment."""
        fragment_data = {
            "type": "fragment",
            "message_id": fragment.message_id,
            "fragment_id": fragment.fragment_id,
            "total_fragments": fragment.total_fragments,
            "data": fragment.data.decode()
        }
        
        data = json.dumps(fragment_data).encode()
        if self.protocol and self.protocol.transport:
            self.protocol.transport.sendto(data, (self.multicast_group, self.port))
        self.socket.sendto(data, (self.multicast_group, self.port))

    async def _listen(self):
        while self.running:
            try:
                data, addr = self.socket.recvfrom(1024)
                message = json.loads(data.decode('utf-8'))
                await self._handle_broadcast(message, addr)
            except Exception as e:
                continue

    async def _handle_broadcast(self, message: Dict, addr: tuple):
        """Handle incoming broadcast messages."""
        pass

    async def _handle_message(self, message: Dict[str, Any], addr: Tuple[str, int]) -> None:
        """Handle incoming broadcast messages with routing and retries."""
        try:
            if not isinstance(message, dict):
                raise ValueError("Invalid message format")
                
            if message.get('type') == 'fragment':
                await self._handle_fragment(message, addr)
                return

            # Extract message metadata
            msg_type = message.get('type')
            msg_id = message.get('id', str(uuid.uuid4()))
            sender = message.get('sender')

            if not msg_type:
                raise ValueError("Message missing required 'type' field")

            # Log receipt and attempt processing
            self.logger.debug(f"Received {msg_type} message {msg_id} from {addr}")

            # Route to appropriate handler
            handler = self._message_handlers.get(msg_type)
            if handler:
                try:
                    async with asyncio.timeout(self._broadcast_timeout):
                        await handler(message, addr)
                except asyncio.TimeoutError:
                    self.logger.error(f"Handler timeout for message {msg_id}")
                    raise
            else:
                self.logger.warning(f"No handler for message type: {msg_type}")

            # Track successful processing
            self._pending_messages[msg_id] = {
                'status': 'processed',
                'timestamp': datetime.now(),
                'type': msg_type,
                'sender': sender
            }

        except (ValueError, TypeError) as e:
            self.logger.error(f"Invalid message from {addr}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error handling message from {addr}: {e}")
            raise RuntimeError(f"Message handling failed: {e}")

    async def _handle_fragment(self, fragment: Dict, addr: Tuple[str, int]):
        """Process received message fragment."""
        message_id = fragment['message_id']
        
        if message_id not in self._pending_fragments:
            self._pending_fragments[message_id] = []
            
        self._pending_fragments[message_id].append(MessageFragment(
            message_id=message_id,
            fragment_id=fragment['fragment_id'],
            total_fragments=fragment['total_fragments'],
            data=fragment['data'].encode(),
            timestamp=asyncio.get_event_loop().time()
        ))
        
        # Check if we have all fragments
        if self._is_message_complete(message_id):
            await self._reassemble_message(message_id, addr)

    async def _cleanup_fragments(self):
        """Remove expired message fragments."""
        while self.running:
            try:
                now = asyncio.get_event_loop().time()
                expired = [
                    msg_id for msg_id, fragments in self._pending_fragments.items()
                    if now - fragments[0].timestamp > self._fragment_timeout
                ]
                
                for msg_id in expired:
                    del self._pending_fragments[msg_id]
                    
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Fragment cleanup error: {e}")
                await asyncio.sleep(5)

    def _is_message_complete(self, message_id: str) -> bool:
        """Check if all fragments of a message have been received."""
        fragments = self._pending_fragments.get(message_id, [])
        if not fragments:
            return False
            
        total = fragments[0].total_fragments
        received = len(fragments)
        fragment_ids = {f.fragment_id for f in fragments}
        
        return received == total and len(fragment_ids) == total

    async def _reassemble_message(self, message_id: str, addr: Tuple[str, int]):
        """Reassemble complete message from fragments."""
        try:
            fragments = sorted(
                self._pending_fragments[message_id],
                key=lambda x: x.fragment_id
            )
            
            data = b''.join(f.data for f in fragments)
            message = json.loads(data.decode())
            
            # Process reassembled message
            await super()._handle_message(message, addr)
            
        finally:
            del self._pending_fragments[message_id]

    async def start_interactive(self):
        """Start UDP broadcast with interactive monitoring"""
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
                if self.interactive:
                    proceed = await self.session.confirm_with_timeout(
                        "\nStart UDP broadcast service with current configuration?",
                        timeout=INTERACTION_TIMEOUTS["confirmation"]
                    )
                    if not proceed:
                        self.logger.info("UDP broadcast start cancelled by user")
                        return

                await self.start()
                
                if self.interactive:
                    print("\nUDP Broadcast Service Active")
                    print("=" * 50)
                    print(f"Port: {self.port}")
                    print(f"Broadcast IP: {self.broadcast_ip}")
                    print("Press Ctrl+C to stop")

        except Exception as e:
            self.logger.error(f"Failed to start interactive UDP broadcast: {e}")
            raise
        finally:
            if self.session:
                await self.session.__aexit__(None, None, None)

    async def broadcast_interactive(self, message: Dict) -> bool:
        """Queue message for broadcasting with backpressure."""
        try:
            if not self.running:
                raise RuntimeError("Service not running")

            if self._message_queue.qsize() >= self._max_queue_size:
                self.logger.warning("Message queue full, applying backpressure")
                await asyncio.sleep(1)

            if self.interactive:
                self.logger.info(f"Queueing message: {message}")
                
            await self._message_queue.put(message)
            return True

        except Exception as e:
            self.logger.error(f"Failed to queue message: {e}")
            return False

    async def stop_interactive(self):
        """Stop service with graceful cleanup"""
        if not self.running:
            return

        try:
            self._interrupt_requested = True
            self._cleanup_event.set()

            if self.interactive:
                print("\nStopping UDP Broadcast Service...")
                print("Waiting for active tasks to complete...")

            # Wait for active tasks with timeout
            try:
                async with asyncio.timeout(5):  # 5 second timeout
                    await self._wait_for_tasks()
            except asyncio.TimeoutError:
                self.logger.warning("Some tasks did not complete gracefully")

            await self.stop()

            if self.interactive:
                print("UDP Broadcast Service stopped successfully")

        except Exception as e:
            self.logger.error(f"Error during interactive shutdown: {e}")
            raise
        finally:
            self._interrupt_requested = False
            self._cleanup_event.clear()

    async def _safe_broadcast(self, message: Dict) -> bool:
        """Send broadcast with safety checks"""
        try:
            if sys.getsizeof(message) > 1024:  # Max UDP datagram size check
                raise ValueError("Message too large for UDP broadcast")

            await self.broadcast(message)
            return True

        except Exception as e:
            self.logger.error(f"Safe broadcast failed: {e}")
            return False

    async def _wait_for_tasks(self):
        """Wait for active tasks to complete"""
        if self._active_tasks:
            await asyncio.gather(*self._active_tasks, return_exceptions=True)
            self._active_tasks.clear()

    async def _message_processor(self):
        """Process messages from queue with rate limiting."""
        try:
            while self.running:
                try:
                    message = await self._message_queue.get()
                    
                    # Apply rate limiting
                    now = asyncio.get_event_loop().time()
                    if now - self._last_broadcast < 1.0 / self._rate_limit:
                        await asyncio.sleep(1.0 / self._rate_limit)
                    
                    await self._safe_broadcast(message)
                    self._last_broadcast = asyncio.get_event_loop().time()
                    self._message_queue.task_done()
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Message processing error: {e}")
                    await asyncio.sleep(1)  # Backoff on error
                    
        except asyncio.CancelledError:
            self.logger.debug("Message processor cancelled")
        finally:
            self._worker_count -= 1

    def _record_error(self, error_type: str, message: str, source: str, 
                     details: Optional[Dict[str, Any]] = None) -> None:
        """Record detailed error information"""
        error = NetworkError(
            timestamp=datetime.now(),
            error_type=error_type,
            message=message,
            source=source,
            details=details or {}
        )
        self._error_history.append(error)
        if len(self._error_history) > self._max_error_history:
            self._error_history.pop(0)
        
        # Log the error with contextual information
        self.logger.error(
            f"UDP Broadcast error: {error_type} in {source} - {message} "
            f"Details: {details or 'none'}"
        )

    async def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics for monitoring"""
        if not self._error_history:
            return {"total_errors": 0}
            
        error_counts = {}
        for error in self._error_history:
            error_counts[error.error_type] = error_counts.get(error.error_type, 0) + 1
            
        return {
            "total_errors": len(self._error_history),
            "error_types": error_counts,
            "latest_error": vars(self._error_history[-1]),
            "timespan": {
                "first": self._error_history[0].timestamp,
                "last": self._error_history[-1].timestamp
            }
        }

    async def _handle_discovery_message(self, message: Dict, addr: Tuple[str, int]):
        """Handle peer discovery messages."""
        peer_id = message.get('id')
        if not peer_id:
            return
            
        # Update peer NAT info
        if message.get('nat_info'):
            self._peer_nat_info[peer_id] = NATInfo(**message['nat_info'])
            
        # Send response with our info
        response = {
            'type': 'discovery_response',
            'id': self.node_id,
            'nat_info': self._nat_info._asdict() if self._nat_info else None
        }
        await self._safe_broadcast(response)

    async def _handle_status_message(self, message: Dict, addr: Tuple[str, int]):
        """Handle peer status updates."""
        # Add status handling logic

    async def _handle_data_message(self, message: Dict, addr: Tuple[str, int]):
        """Handle data transfer messages."""
        # Add data handling logic

    async def _handle_presence_message(self, message: Dict, addr: Tuple[str, int]):
        """Handle peer presence notifications."""
        # Add presence handling logic

    async def _handle_error_message(self, message: Dict, addr: Tuple[str, int]):
        """Handle error notifications from peers."""
        # Add error handling logic

    def __del__(self):
        """Cleanup on deletion"""
        try:
            if self.running:
                asyncio.create_task(self.stop_interactive())
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
