from typing import Dict, Optional, Set, Tuple, List, NamedTuple, DefaultDict
from dataclasses import dataclass
import time
import logging
import asyncio
from collections import defaultdict
from time import monotonic
import os
import json
try:
    from atomicwrites import atomic_write
except ImportError:
    # Fallback if atomicwrites is not available
    atomic_write = None
    logging.warning("atomicwrites not available, using standard file operations")

logger = logging.getLogger(__name__)

@dataclass
class ReputationMetrics:
    score: float
    last_update: float
    total_contributions: int
    last_validation: float = 0.0
    validation_failures: int = 0
    total_validations: int = 0

@dataclass
class SuspiciousActivity:
    timestamp: float
    activity_type: str  # sybil, collusion, validation_failure, etc
    evidence: Dict[str, any]
    severity: float

class PendingChange(NamedTuple):
    score_delta: float
    timestamp: float
    reason: str

class ReputationManager:
    def __init__(self, storage_path: Optional[str] = None, persistence_interval: int = 300, 
                 significant_change_threshold: float = 0.1, interactive: bool = True):
        self.storage_path = storage_path
        self.persistence_interval = persistence_interval
        self.significant_change_threshold = significant_change_threshold
        self.logger = logging.getLogger('ReputationManager')
        self.stop_event = asyncio.Event()
        self._running = False
        self._last_save = 0
        self.reputations = {}
        self.reputation_metrics = {}
        self.pending_changes = defaultdict(list)
        self._validation_cache = {}
        self.cooling_period = 3600  # 1 hour cooling period for large changes
        self.interactive = interactive

    async def start(self):
        """Start reputation manager with persistence"""
        self._running = True
        self.stop_event.clear()
        
        # Load existing reputation data if available
        if self.storage_path:
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.reputations = data.get('reputations', {})
                    else:
                        self.logger.error("Invalid data format: Expected a dictionary")
                        self.reputations = {}
                    self.logger.info(f"Loaded {len(self.reputations)} reputation records")
            except FileNotFoundError:
                self.logger.warning(f"Reputation data file not found: {self.storage_path}")
            except Exception as e:
                self.logger.error(f"Failed to load reputation data: {e}")
        
        # Start persistence task if path is provided
        if self.storage_path:
            asyncio.create_task(self._persistence_task())
    
    async def _persistence_task(self):
        """Periodically save reputation data"""
        while self._running and not self.stop_event.is_set():
            try:
                await asyncio.sleep(self.persistence_interval)
                await self._save_reputations()
            except Exception as e:
                self.logger.error(f"Error during persistence task: {e}")
            except asyncio.CancelledError:
                break
        
        # Final save on task end
        if self._running:
            await self._save_reputations()

    async def _save_reputations(self):
        """Save reputation data to disk"""
        if not self.storage_path:
            return
        
        try:
            # Use atomicwrites for safer file operations if available
            if atomic_write:
                with atomic_write(self.storage_path, overwrite=True) as f:
                    json.dump({'reputations': self.reputations, 'timestamp': time.time()}, f, ensure_ascii=False)
            else:
                # Fallback to standard file write
                with open(self.storage_path, 'w') as f:
                    json.dump({'reputations': self.reputations, 'timestamp': time.time()}, f, ensure_ascii=False)
                
            self._last_save = monotonic()
            self.logger.debug(f"Saved {len(self.reputations)} reputation records")
        except Exception as e:
            self.logger.error(f"Failed to save reputation data: {e}")

    async def update_reputation_interactive(self, peer_id: str, delta: float, reason: str = ""):
        """Update peer reputation with optional reason"""
        current = self.reputations.get(peer_id, 0.5)  # Default neutral reputation
        
        # Apply bounded update
        new_value = max(0.0, min(1.0, current + delta))
        self.reputations[peer_id] = new_value
        
        # Log significant changes
        if abs(new_value - current) > self.significant_change_threshold:
            self.logger.info(f"Reputation change for {peer_id}: {current:.2f} → {new_value:.2f} ({reason})")
        
        # Save more frequently on significant negative changes
        if delta < -0.2 and self.storage_path and (monotonic() - self._last_save > 60):
            await self._save_reputations()
        
        return new_value

    async def _validate_peer(self, peer_id: str) -> bool:
        """Validate peer existence and status"""
        # Check cache first
        if peer_id in self._validation_cache:
            if time.time() - self._validation_cache[peer_id]['time'] < 300:  # 5 minute cache
                return self._validation_cache[peer_id]['result']
        
        # Simple validation for now (could be extended with actual checks)
        result = peer_id is not None and len(peer_id) > 0
        
        # Cache result
        self._validation_cache[peer_id] = {
            'time': time.time(),
            'result': result
        }
        
        return result

    def get_reputation(self, peer_id: str) -> float:
        """Get peer's current reputation score"""
        return self.reputations.get(peer_id, 0.0)

    async def cleanup(self):
        """Cleanup reputation data for peers that haven't been seen"""
        # Save before exiting
        if self._running and self.storage_path:
            try:
                await self._save_reputations()
            except Exception as e:
                self.logger.error(f"Error during cleanup save: {e}")
            finally:
                self._running = False
                self.stop_event.set()
                
    async def stop(self):
        """Stop reputation manager"""
        self._running = False
        self.stop_event.set()
        await self.cleanup()
