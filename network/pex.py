from typing import Set, Dict, List, Optional
import time
import random
import logging
import asyncio
from core.constants import INTERACTION_TIMEOUTS, InteractionLevel
from core.interactive_utils import InteractiveSession, InteractiveConfig

logger = logging.getLogger(__name__)

class PeerExchange:
    def __init__(self, max_peers: int = 1000, interactive: bool = True):
        self.peers: Dict[str, float] = {}  # peer_id -> last_seen
        self.max_peers = max_peers
        self.interactive = interactive
        self.session = None
        self._interrupt_requested = False
        self._cleanup_event = asyncio.Event()
        self.logger = logger
        
    def add_peer(self, peer_id: str):
        self.peers[peer_id] = time.time()
        if len(self.peers) > self.max_peers:
            self._prune_old_peers()
            
    def get_peers(self, n: int = 10) -> List[str]:
        active_peers = [p for p, t in self.peers.items() 
                       if time.time() - t < 3600]  # Active in last hour
        return random.sample(active_peers, min(n, len(active_peers)))
        
    def _prune_old_peers(self, max_age: float = 3600 * 24):
        current_time = time.time()
        self.peers = {
            peer: last_seen for peer, last_seen in self.peers.items()
            if current_time - last_seen < max_age
        }

    async def add_peer_interactive(self, peer_id: str) -> bool:
        """Add peer with interactive monitoring and validation"""
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
                if not self._validate_peer_id(peer_id):
                    self.logger.error(f"Invalid peer ID: {peer_id}")
                    return False

                self.add_peer(peer_id)
                
                # Monitor peer list size
                if len(self.peers) > self.max_peers * 0.9:  # 90% capacity warning
                    self.logger.warning("Peer list nearing capacity")
                    if self.interactive:
                        await self._cleanup_with_confirmation()

                return True

        except Exception as e:
            self.logger.error(f"Failed to add peer: {str(e)}")
            return False
        finally:
            if self.session:
                await self.session.__aexit__(None, None, None)

    async def get_peers_interactive(self, n: int = 10) -> Optional[List[str]]:
        """Get peers with interactive monitoring and safety checks"""
        try:
            if self.interactive:
                self.session = InteractiveSession(
                    level=InteractionLevel.NORMAL,
                    config=InteractiveConfig(
                        timeout=INTERACTION_TIMEOUTS["default"],
                        persistent_state=True
                    )
                )

            async with self.session:
                peers = self.get_peers(n)
                
                if not peers:
                    self.logger.warning("No active peers found")
                    return []

                if self.interactive:
                    print(f"\nFound {len(peers)} active peers")

                return peers

        except Exception as e:
            self.logger.error(f"Error getting peers: {str(e)}")
            return None
        finally:
            if self.session:
                await self.session.__aexit__(None, None, None)

    async def _cleanup_with_confirmation(self) -> None:
        """Interactive cleanup with user confirmation"""
        if self.interactive and self.session:
            proceed = await self.session.confirm_with_timeout(
                "\nPeer list nearing capacity. Perform cleanup?",
                timeout=INTERACTION_TIMEOUTS["cleanup"]
            )
            if proceed:
                self._prune_old_peers()
                self.logger.info("Cleanup completed")

    def _validate_peer_id(self, peer_id: str) -> bool:
        """Validate peer ID format"""
        return bool(peer_id and isinstance(peer_id, str) and 8 <= len(peer_id) <= 64)

    def request_shutdown(self) -> None:
        """Request graceful shutdown"""
        self._interrupt_requested = True
        self._cleanup_event.set()
        self.logger.info("Shutdown requested for peer exchange")

    async def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            await self._cleanup_event.wait()
            self._prune_old_peers(max_age=0)  # Clear all peers
            self.logger.info("Peer exchange cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
