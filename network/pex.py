from typing import Set, Dict, List, Optional
import time
import random
import logging
import asyncio
from collections import defaultdict
from core.constants import INTERACTION_TIMEOUTS, InteractionLevel
from core.interactive_utils import InteractiveSession, InteractiveConfig

logger = logging.getLogger(__name__)

class PeerExchangeError(Exception):
    """Base exception for peer exchange errors"""
    pass

class PeerValidationError(PeerExchangeError):
    """Raised when peer validation fails"""
    pass

class PeerRegistrationError(PeerExchangeError):
    """Raised when peer registration fails"""
    pass

class PeerCapacityError(PeerExchangeError):
    """Raised when peer list is at capacity"""
    pass

class PeerTimeoutError(PeerExchangeError):
    """Raised when peer operation times out"""
    pass

class PeerExchange:
    def __init__(self, max_peers: int = 1000, interactive: bool = True,
                 auto_cleanup: bool = False, cleanup_threshold: float = 0.9,
                 active_window: int = 3600, # 1 hour default
                 stale_window: int = 3600 * 24, # 24 hours default
                 timeouts: Optional[Dict[str, int]] = None):
        self.peers: Dict[str, float] = {}  # peer_id -> last_seen 
        self.max_peers = max_peers
        self.interactive = interactive
        self.auto_cleanup = auto_cleanup
        self.cleanup_threshold = min(max(0.5, cleanup_threshold), 0.95)  # Constrain between 0.5-0.95
        self.session = None
        self._interrupt_requested = False
        self._cleanup_event = asyncio.Event()
        self.logger = logger
        self.active_window = max(300, min(active_window, 86400))  # Between 5 min and 24 hours
        self.stale_window = max(3600, min(stale_window, 86400 * 7))  # Between 1 hour and 7 days
        
        # Enhanced timeout configuration with environment awareness
        base_timeouts = {
            "peer_add": self._calculate_timeout("peer_add"),
            "peer_get": self._calculate_timeout("peer_get"),
            "cleanup": self._calculate_timeout("cleanup"),
            "validation": self._calculate_timeout("validation")
        }
        self.timeouts = base_timeouts
        if timeouts:
            self.timeouts.update(timeouts)

        # Metrics for timeout adjustment
        self._operation_times = defaultdict(list)
        self._max_samples = 100  # Keep last 100 samples
        self._adjustment_interval = 300  # Adjust every 5 minutes
        self._last_adjustment = time.time()

        self._active_operations = set()
        self._shutdown_state = "running"  # running, shutting_down, shutdown
        self._operation_lock = asyncio.Lock()
        self._shutdown_timeout = 30  # seconds to wait for graceful shutdown
        self._active_tasks = set()
        self._peer_interactions = defaultdict(int)
        self._shutdown_stages = ["running", "saving", "disconnecting", "cleanup", "shutdown"]
        self._current_stage = "running"
        self._shutdown_timeouts = {
            "saving": 10,       # 10s for saving state
            "disconnecting": 20, # 20s for graceful peer disconnects  
            "cleanup": 15       # 15s for final cleanup
        }
        
    def add_peer(self, peer_id: str):
        """Add peer with optional auto-cleanup"""
        self.peers[peer_id] = time.time()
        
        # Check if we need to cleanup
        if len(self.peers) > self.max_peers:
            self._prune_old_peers()
        elif self.auto_cleanup and len(self.peers) > self.max_peers * self.cleanup_threshold:
            self._auto_cleanup()
            
    def get_peers(self, n: int = 10) -> List[str]:
        current_time = time.time()
        active_peers = [p for p, t in self.peers.items() 
                       if current_time - t < self.active_window]
        return random.sample(active_peers, min(n, len(active_peers)))
        
    def _prune_old_peers(self, max_age: Optional[float] = None):
        current_time = time.time()
        max_age = max_age or self.stale_window
        self.peers = {
            peer: last_seen for peer, last_seen in self.peers.items()
            if current_time - last_seen < max_age
        }

    def _auto_cleanup(self):
        """Perform automatic cleanup based on peer age"""
        try:
            # Calculate adaptive max age based on capacity pressure
            capacity_ratio = len(self.peers) / self.max_peers
            # Adjust max age down as we get closer to capacity
            adaptive_max_age = 3600 * 24 * (1.0 - (capacity_ratio - self.cleanup_threshold))
            
            self._prune_old_peers(max_age=adaptive_max_age)
            self.logger.info(f"Auto-cleanup completed. Remaining peers: {len(self.peers)}")
        except Exception as e:
            self.logger.error(f"Auto-cleanup failed: {str(e)}")

    def _validate_peer_id(self, peer_id: str) -> bool:
        """Validate peer ID format and uniqueness"""
        # Basic format validation
        if not (peer_id and isinstance(peer_id, str) and 8 <= len(peer_id) <= 64):
            self.logger.debug(f"Peer ID {peer_id} failed basic format validation")
            return False
            
        # Check for duplicate entries
        if peer_id in self.peers:
            self.logger.warning(f"Duplicate peer ID detected: {peer_id}")
            return False
            
        # Validate alphanumeric format with optional hyphens
        if not all(c.isalnum() or c == '-' for c in peer_id):
            self.logger.debug(f"Peer ID contains invalid characters: {peer_id}")
            return False
            
        # Additional validation - must contain at least one letter and one number
        has_letter = any(c.isalpha() for c in peer_id)
        has_number = any(c.isdigit() for c in peer_id)
        if not (has_letter and has_number):
            self.logger.debug(f"Peer ID must contain both letters and numbers: {peer_id}")
            return False
            
        return True

    async def _validate_peer_registration(self, peer_id: str) -> bool:
        """Validate peer registration with external systems"""
        try:
            # Add validation with external registration system here
            # For now just returning True as placeholder
            return True
        except Exception as e:
            self.logger.error(f"Peer registration validation failed: {str(e)}")
            return False

    async def _track_operation(self, operation_id: str):
        """Track an active peer operation"""
        async with self._operation_lock:
            self._active_operations.add(operation_id)

    async def _end_operation(self, operation_id: str):
        """Mark a peer operation as completed"""
        async with self._operation_lock:
            self._active_operations.discard(operation_id)

    def _calculate_timeout(self, operation: str) -> int:
        """Calculate appropriate timeout based on operation type and environment"""
        base_timeouts = {
            "peer_add": 30,
            "peer_get": 15,
            "cleanup": 45,
            "validation": 20
        }
        
        # Adjust for environment (e.g., slow network)
        network_factor = 1.0  # Could be adjusted based on network metrics
        return int(base_timeouts[operation] * network_factor)

    async def _track_operation_time(self, operation: str, duration: float):
        """Track operation duration for timeout adjustment"""
        async with self._operation_lock:
            self._operation_times[operation].append(duration)
            if len(self._operation_times[operation]) > self._max_samples:
                self._operation_times[operation].pop(0)

    async def _adjust_timeouts(self):
        """Adjust timeouts based on collected metrics"""
        now = time.time()
        if now - self._last_adjustment < self._adjustment_interval:
            return

        async with self._operation_lock:
            for operation, times in self._operation_times.items():
                if times:
                    # Use 95th percentile for timeout
                    p95 = sorted(times)[int(len(times) * 0.95)]
                    # Add 50% buffer
                    self.timeouts[operation] = int(p95 * 1.5)
            self._last_adjustment = now

    async def add_peer_interactive(self, peer_id: str) -> bool:
        """Add peer with enhanced error handling and timeout tracking"""
        operation_id = f"add_{peer_id}_{time.time()}"
        operation_start = time.time()
        
        try:
            if self._shutdown_state != "running":
                raise PeerExchangeError("Cannot add peer during shutdown")

            await self._track_operation(operation_id)

            if self.interactive:
                self.session = InteractiveSession(
                    level=InteractionLevel.NORMAL,
                    config=InteractiveConfig(
                        timeout=self.timeouts["peer_add"],
                        persistent_state=True,
                        safe_mode=True
                    )
                )

            async with self.session:
                # Enhanced validation with specific exceptions
                if not self._validate_peer_id(peer_id):
                    raise PeerValidationError(
                        f"Invalid peer ID format: {peer_id}. Must be 8-64 chars, "
                        "alphanumeric with hyphens, containing at least one letter and number"
                    )

                if not await self._validate_peer_registration(peer_id):
                    raise PeerRegistrationError(f"Peer registration failed: {peer_id}")

                if len(self.peers) >= self.max_peers:
                    raise PeerCapacityError(
                        f"Peer list at capacity ({self.max_peers}). "
                        f"Current peers: {len(self.peers)}"
                    )

                self.add_peer(peer_id)
                return True

        except asyncio.TimeoutError:
            raise PeerTimeoutError(
                f"Operation timed out after {self.timeouts['peer_add']}s: {peer_id}"
            )
        except PeerExchangeError:
            raise
        except Exception as e:
            raise PeerExchangeError(f"Unexpected error: {str(e)}") from e
        finally:
            duration = time.time() - operation_start
            await self._track_operation_time("peer_add", duration)
            await self._end_operation(operation_id)
            if self.session:
                await self.session.__aexit__(None, None, None)
                self.session = None
            await self._adjust_timeouts()

    async def get_peers_interactive(self, n: int = 10) -> Optional[List[str]]:
        """Get peers with interactive monitoring and safety checks"""
        operation_id = f"get_peers_{time.time()}"
        try:
            if self._shutdown_state != "running":
                return []

            await self._track_operation(operation_id)

            if self.interactive:
                self.session = InteractiveSession(
                    level=InteractionLevel.NORMAL,
                    config=InteractiveConfig(
                        timeout=self.timeouts["peer_get"],
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

        except asyncio.TimeoutError:
            self.logger.warning("Interactive session timed out, falling back to direct peer fetch")
            return self.get_peers(n)
        except Exception as e:
            self.logger.error(f"Error getting peers: {str(e)}")
            return None
        finally:
            await self._end_operation(operation_id)
            if self.session:
                await self.session.__aexit__(None, None, None)
                self.session = None

    async def track_peer_interaction(self, peer_id: str):
        """Track active peer interactions"""
        async with self._operation_lock:
            self._peer_interactions[peer_id] += 1

    async def end_peer_interaction(self, peer_id: str):
        """End tracked peer interaction"""
        async with self._operation_lock:
            self._peer_interactions[peer_id] = max(0, self._peer_interactions[peer_id] - 1)

    async def save_peer_state(self):
        """Save critical peer state before shutdown"""
        try:
            # Save peer list and metadata
            state = {
                'peers': dict(self.peers),
                'timestamp': time.time()
            }
            # Implementation of state saving goes here
            return True
        except Exception as e:
            self.logger.error(f"Failed to save peer state: {e}")
            return False

    async def request_shutdown(self):
        """Enhanced shutdown with staged cleanup and interaction tracking"""
        try:
            self._current_stage = "saving"
            self.logger.info("Starting peer exchange shutdown...")

            # Stage 1: Save State
            save_task = asyncio.create_task(self.save_peer_state())
            try:
                await asyncio.wait_for(save_task, timeout=self._shutdown_timeouts["saving"])
            except asyncio.TimeoutError:
                self.logger.warning("Peer state saving timed out")

            # Stage 2: Wait for active interactions
            self._current_stage = "disconnecting"
            disconnect_start = time.time()
            while sum(self._peer_interactions.values()) > 0:
                if time.time() - disconnect_start > self._shutdown_timeouts["disconnecting"]:
                    self.logger.warning("Force closing peer interactions")
                    break
                await asyncio.sleep(1)
                self.logger.info(f"Waiting for {sum(self._peer_interactions.values())} peer interactions")

            # Stage 3: Cleanup
            self._current_stage = "cleanup"
            cleanup_task = asyncio.create_task(self._cleanup_peers())
            try:
                await asyncio.wait_for(cleanup_task, timeout=self._shutdown_timeouts["cleanup"])
            except asyncio.TimeoutError:
                self.logger.warning("Peer cleanup timed out, forcing cleanup")
                self.peers.clear()

            self._current_stage = "shutdown"
            self._cleanup_event.set()
            self.logger.info("Peer exchange shutdown completed")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")
            # Force cleanup on error
            self.peers.clear()
            self._peer_interactions.clear()
            raise
        finally:
            self._interrupt_requested = True

    async def _cleanup_peers(self):
        """Graceful peer cleanup"""
        async with self._operation_lock:
            for peer_id in list(self.peers.keys()):
                if self._peer_interactions[peer_id] > 0:
                    self.logger.warning(f"Peer {peer_id} has {self._peer_interactions[peer_id]} pending interactions")
                del self.peers[peer_id]
                await asyncio.sleep(0)  # Yield to other tasks

    async def cleanup(self) -> None:
        """Enhanced cleanup with session handling"""
        try:
            if self._shutdown_state != "shutdown":
                await self.request_shutdown()

            # Clear all peers
            self._prune_old_peers(max_age=0)

            # Clean up any active sessions
            if self.session:
                try:
                    await asyncio.wait_for(
                        self.session.__aexit__(None, None, None),
                        timeout=5.0
                    )
                except asyncio.TimeoutError:
                    self.logger.warning("Session cleanup timed out")
                finally:
                    self.session = None

            self.logger.info("Peer exchange cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            raise
