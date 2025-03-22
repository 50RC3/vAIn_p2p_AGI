import asyncio
import json
import socket
import logging
from typing import Dict, Optional, Callable, Any, Tuple

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
            logging.error(f"Failed to decode message from {addr}: {e}")

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

    async def start(self):
        """Start UDP broadcast service."""
        if self.running:
            return

        try:
            loop = asyncio.get_running_loop()
            self.protocol = UDPBroadcastProtocol(self._handle_message)
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.bind(('', self.port))

            self.transport, _ = await loop.create_datagram_endpoint(
                lambda: self.protocol,
                sock=sock
            )
            
            self.running = True
            self.logger.info(f"Started UDP broadcast on port {self.port}")
            asyncio.create_task(self._listen())
            
        except Exception as e:
            self.logger.error(f"Failed to start UDP broadcast: {e}")
            raise

    async def stop(self):
        """Stop UDP broadcast service."""
        if not self.running:
            return

        try:
            if self.transport:
                self.transport.close()
            self.running = False
            self.logger.info("Stopped UDP broadcast service")
            self.socket.close()
            
        except Exception as e:
            self.logger.error(f"Error stopping UDP broadcast: {e}")
            raise

    async def broadcast(self, message: Dict):
        """Broadcast message to local network."""
        if not self.running:
            raise RuntimeError("UDP broadcast service not running")

        try:
            data = json.dumps(message).encode()
            if self.protocol and self.protocol.transport:
                self.protocol.transport.sendto(data, ('<broadcast>', self.port))
            self.socket.sendto(data, (self.broadcast_ip, self.port))
                
        except Exception as e:
            self.logger.error(f"Failed to broadcast message: {e}")
            raise

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
        """Handle incoming broadcast messages.
        
        Args:
            message: Dictionary containing the message data
            addr: Tuple of (host, port) for the sender
            
        Raises:
            ValueError: If message format is invalid
            RuntimeError: If message handling fails
        """
        try:
            if not isinstance(message, dict):
                raise ValueError("Invalid message format")
                
            if 'type' not in message:
                raise ValueError("Message missing required 'type' field")
                
            self.logger.debug(f"Received message from {addr}: {message}")
            # Add message handling logic here
            
        except (ValueError, TypeError) as e:
            self.logger.error(f"Invalid message from {addr}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error handling message from {addr}: {e}")
            raise RuntimeError(f"Message handling failed: {e}")

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
        """Send broadcast with progress tracking and error handling"""
        try:
            if not self.running:
                raise RuntimeError("Service not running")

            if self.interactive:
                self.logger.info(f"Broadcasting message: {message}")
                
            await self._message_queue.put(message)
            
            success = await self._safe_broadcast(message)
            
            if self.interactive:
                self.logger.info("Broadcast completed successfully" if success else "Broadcast failed")
                
            return success

        except Exception as e:
            self.logger.error(f"Interactive broadcast failed: {e}")
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

    def __del__(self):
        """Cleanup on deletion"""
        try:
            if self.running:
                asyncio.create_task(self.stop_interactive())
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
