import asyncio
import json
import socket
import logging
import sys
import uuid
from typing import Dict, Optional, Callable, Any, Tuple, List, Set
from dataclasses import dataclass
from datetime import datetime
from core.constants import INTERACTION_TIMEOUTS
from core.interactive_utils import InteractiveSession, InteractionLevel

logger = logging.getLogger(__name__)

@dataclass
class NetworkError:
    """Detailed network error information"""
    timestamp: datetime
    error_type: str
    message: str
    source: str
    details: Dict[str, Any] = None


@dataclass
class MessageFragment:
    """Fragment of a large message"""
    message_id: str
    fragment_id: int
    total_fragments: int
    data: bytes
    timestamp: float


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
    def __init__(self, config: Dict = None):
        """Initialize UDP broadcast handler"""
        config = config or {}
        # Extract configuration parameters
        self.port = config.get('port', 8468)
        # Ensure port is an integer
        if not isinstance(self.port, int):
            self.port = 8468  # Default to 8468 if not an integer
        
        self.broadcast_addr = config.get('broadcast_addr', '255.255.255.255')
        
        # Initialize socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # State tracking
        self.running = False
        self.logger = logging.getLogger("UDPBroadcast")
        self.peers: Set[str] = set()
        self.message_queue = asyncio.Queue()
        
        # Protocol components
        self.protocol = None
        self.transport = None
        
        # Error tracking
        self._error_history = []
        self._max_error_history = 100
        
        # Task management
        self._active_tasks = set()
        self._interrupt_requested = False
        self._listen_task = None

    async def start(self):
        """Start UDP broadcast service"""
        if self.running:
            self.logger.warning("UDP broadcast already running")
            return
            
        try:
            self.logger.info(f"Starting UDP broadcast service on port {self.port}")
            
            # Bind to port
            try:
                self.sock.bind(('0.0.0.0', self.port))
            except OSError as e:
                if e.errno == 10048:  # Address already in use
                    self.logger.warning(f"Port {self.port} already in use, attempting to rebind")
                    self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    if hasattr(socket, 'SO_REUSEPORT'):  # Not available on all platforms
                        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
                    self.sock.bind(('0.0.0.0', self.port))
                else:
                    raise
            
            # Create datagram endpoint
            loop = asyncio.get_event_loop()
            self.protocol = UDPBroadcastProtocol(self._handle_message)
            
            # Create a transport - this may fail if port is in use
            try:
                self.transport, _ = await loop.create_datagram_endpoint(
                    lambda: self.protocol,
                    sock=self.sock
                )
            except Exception:
                self.logger.warning("Failed to create transport with socket, using default creation")
                self.transport, _ = await loop.create_datagram_endpoint(
                    lambda: self.protocol,
                    local_addr=('0.0.0.0', self.port),
                    allow_broadcast=True
                )
            
            # Start listener task
            self._listen_task = asyncio.create_task(self._listen())
            self._active_tasks.add(self._listen_task)
            
            self.running = True
            self.logger.info(f"UDP broadcast service started successfully on port {self.port}")
            
        except Exception as e:
            self.logger.error(f"Failed to start UDP broadcast service: {e}")
            self._record_error("startup_error", str(e), "start")
            raise

    async def stop(self):
        """Stop UDP broadcast service"""
        if not self.running:
            self.logger.debug("UDP broadcast not running, nothing to stop")
            return
            
        try:
            self.logger.info("Stopping UDP broadcast service")
            self._interrupt_requested = True
            
            # Cancel all active tasks
            for task in list(self._active_tasks):
                if not task.done():
                    task.cancel()
            
            if self._active_tasks:
                await asyncio.gather(*self._active_tasks, return_exceptions=True)
            self._active_tasks.clear()
            
            # Close transport
            if self.transport:
                self.transport.close()
                self.transport = None
                
            # Close socket
            try:
                self.sock.close()
            except Exception as e:
                self.logger.warning(f"Error closing socket: {e}")
            
            self.running = False
            self.logger.info("UDP broadcast service stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping UDP broadcast: {e}")
            self._record_error("shutdown_error", str(e), "stop")
            raise

    async def broadcast(self, message: Dict):
        """Broadcast a message to all peers"""
        if not self.running:
            self.logger.warning("Cannot broadcast, UDP service not running")
            return
            
        try:
            # Add message ID if not present
            if 'id' not in message:
                message['id'] = str(uuid.uuid4())
                
            # Convert to JSON and broadcast
            data = json.dumps(message).encode()
            
            # Use transport if available (preferred method)
            if self.protocol and self.protocol.transport:
                self.protocol.transport.sendto(data, (self.broadcast_addr, self.port))
            else:
                # Fallback to direct socket
                self.sock.sendto(data, (self.broadcast_addr, self.port))
                
            self.logger.debug(f"Broadcast message: {message}")
            
        except Exception as e:
            self.logger.error(f"Failed to broadcast message: {e}")
            self._record_error("broadcast_error", str(e), "broadcast")
            # Don't re-raise to avoid crashing the caller

    async def _listen(self):
        """Listen for incoming broadcast messages"""
        try:
            self.logger.debug("Starting UDP listener loop")
            while not self._interrupt_requested:
                try:
                    # Use a socket with timeout to allow for graceful cancellation
                    self.sock.settimeout(1.0)
                    data, addr = self.sock.recvfrom(8192)
                    self.sock.settimeout(None)
                    
                    try:
                        message = json.loads(data.decode('utf-8'))
                        await self._handle_message(message, addr)
                    except json.JSONDecodeError:
                        self._record_error("decode_error", "Invalid JSON", addr[0])
                except socket.timeout:
                    # This is expected for the cancellation check
                    continue
                except ConnectionError:
                    if self._interrupt_requested:
                        break
                    self.logger.warning("Connection error in UDP listener, restarting socket")
                    await asyncio.sleep(1)
                    continue
                except Exception as e:
                    if self._interrupt_requested:
                        break
                    self.logger.error(f"Error receiving UDP data: {e}")
                    await asyncio.sleep(0.5)
            
            self.logger.debug("UDP listener loop ended")
            
        except asyncio.CancelledError:
            self.logger.info("UDP listener task cancelled")
            raise
        except Exception as e:
            self.logger.error(f"Fatal UDP listener error: {e}")
            self._record_error("listener_error", str(e), "listen")
            raise
        finally:
            self._active_tasks.discard(asyncio.current_task())

    async def _handle_message(self, message: Dict, addr: Tuple):
        """Process received broadcast message"""
        try:
            message_type = message.get('type')
            
            if message_type == 'discovery':
                # Add peer to known peers
                peer_id = message.get('id')
                if peer_id:
                    self.peers.add(peer_id)
                    self.logger.debug(f"Discovered peer {peer_id} at {addr}")
                    
            # Queue message for further processing
            await self.message_queue.put((message, addr))
            
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
            # Don't re-raise to avoid crashing the listener

    def _record_error(self, error_type: str, message: str, source: str, details: Dict = None):
        """Record error for troubleshooting"""
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

    async def get_peers(self) -> List[str]:
        """Get list of discovered peers"""
        return list(self.peers)
