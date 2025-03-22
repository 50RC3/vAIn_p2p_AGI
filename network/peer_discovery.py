import socket
import json
import asyncio
import logging
from typing import List, Dict, Optional, Set
from tqdm import tqdm
from core.constants import INTERACTION_TIMEOUTS, InteractionLevel
from core.interactive_utils import InteractiveSession, InteractiveConfig

logger = logging.getLogger(__name__)

class PeerDiscovery:
    def __init__(self, port: int = 5000, interactive: bool = True):
        self.port = port
        self.peers = set()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.interactive = interactive
        self.session = None
        self._interrupt_requested = False
        self._cleanup_event = asyncio.Event()
        self.broadcast_interval = 30  # Seconds
        self.max_retries = 3
        self.retry_delay = 5  # Seconds
        
    def broadcast_presence(self):
        message = json.dumps({"type": "discovery", "port": self.port})
        self.sock.sendto(message.encode(), ('<broadcast>', self.port))

    async def start_interactive(self) -> None:
        """Start peer discovery with interactive monitoring"""
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
                logger.info("Starting peer discovery service")
                await self._run_discovery_loop()

        except Exception as e:
            logger.error(f"Peer discovery failed: {str(e)}")
            raise
        finally:
            await self._cleanup()

    async def _run_discovery_loop(self) -> None:
        """Main discovery loop with progress tracking"""
        retry_count = 0
        
        while not self._interrupt_requested:
            try:
                if self.interactive:
                    with tqdm(total=100, desc="Discovering Peers") as pbar:
                        await self._broadcast_with_progress(pbar)
                else:
                    await self._broadcast_safely()

                retry_count = 0
                await asyncio.sleep(self.broadcast_interval)

            except Exception as e:
                retry_count += 1
                if retry_count >= self.max_retries:
                    logger.error("Max retries exceeded, stopping discovery")
                    break
                    
                logger.warning(f"Discovery error, retrying ({retry_count}/{self.max_retries}): {str(e)}")
                await asyncio.sleep(self.retry_delay)

    async def _broadcast_safely(self) -> None:
        """Send broadcast with error handling"""
        try:
            self.broadcast_presence()
            new_peers = await self._listen_for_responses()
            self.peers.update(new_peers)
        except socket.error as e:
            logger.error(f"Network error during broadcast: {str(e)}")
            raise

    async def _broadcast_with_progress(self, pbar: tqdm) -> None:
        """Broadcast with progress updates"""
        initial_peers = len(self.peers)
        await self._broadcast_safely()
        new_peers = len(self.peers) - initial_peers
        
        pbar.update(50)
        pbar.set_postfix({"new_peers": new_peers})
        await asyncio.sleep(0.5)
        pbar.update(50)

    async def _listen_for_responses(self) -> Set[str]:
        """Listen for peer responses with timeout"""
        responses = set()
        try:
            self.sock.settimeout(5)
            while not self._interrupt_requested:
                try:
                    data, addr = self.sock.recvfrom(1024)
                    peer_info = json.loads(data.decode())
                    if self._validate_peer(peer_info):
                        responses.add(addr[0])
                except socket.timeout:
                    break
        finally:
            self.sock.settimeout(None)
        return responses

    def _validate_peer(self, peer_info: Dict) -> bool:
        """Validate peer information"""
        required_fields = {"type", "port"}
        return (
            isinstance(peer_info, dict) and
            all(field in peer_info for field in required_fields) and 
            peer_info["type"] == "discovery" and
            isinstance(peer_info["port"], int) and
            1024 <= peer_info["port"] <= 65535
        )

    async def _cleanup(self) -> None:
        """Clean up resources"""
        try:
            self._cleanup_event.set()
            self.sock.close()
            logger.info("Peer discovery cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def request_shutdown(self) -> None:
        """Request graceful shutdown"""
        self._interrupt_requested = True
        logger.info("Shutdown requested for peer discovery")
