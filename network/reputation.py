from typing import Dict, Optional, Set
from dataclasses import dataclass
import time
import logging
import asyncio
from tqdm import tqdm
from core.constants import INTERACTION_TIMEOUTS, InteractionLevel
from core.interactive_utils import InteractiveSession, InteractiveConfig

@dataclass
class ReputationMetrics:
    score: float
    last_update: float
    total_contributions: int

class ReputationManager:
    def __init__(self, decay_factor: float = 0.95, min_reputation: float = -100, 
                 interactive: bool = True):
        self.reputation_scores: Dict[str, ReputationMetrics] = {}
        self.decay_factor = decay_factor
        self.min_reputation = min_reputation
        self.logger = logging.getLogger('ReputationManager')
        self.interactive = interactive
        self.session: Optional[InteractiveSession] = None
        self._interrupt_requested = False
        self._progress_file = "reputation_progress.json"
        
    def update_reputation(self, node_id: str, score_delta: float):
        """Update node's reputation score."""
        current_time = time.time()
        if node_id not in self.reputation_scores:
            self.reputation_scores[node_id] = ReputationMetrics(0.0, current_time, 0)
            
        metrics = self.reputation_scores[node_id]
        # Apply time decay
        time_diff = current_time - metrics.last_update
        decayed_score = metrics.score * (self.decay_factor ** (time_diff / 86400))  # Daily decay
        
        # Update metrics
        new_score = max(decayed_score + score_delta, self.min_reputation)
        self.reputation_scores[node_id] = ReputationMetrics(
            score=new_score,
            last_update=current_time,
            total_contributions=metrics.total_contributions + 1
        )
        
    def get_reputation(self, node_id: str) -> float:
        """Get current reputation score for a node."""
        if node_id not in self.reputation_scores:
            return 0.0
        return self.reputation_scores[node_id].score
        
    def get_top_nodes(self, n: int = 10) -> Dict[str, float]:
        """Get top N nodes by reputation score."""
        return dict(sorted(
            [(k, v.score) for k, v in self.reputation_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )[:n])

    async def update_reputation_interactive(self, node_id: str, 
                                         score_delta: float) -> bool:
        """Interactive reputation update with monitoring and safety checks"""
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
                # Monitor updates for safety
                if score_delta > 100 or score_delta < -100:
                    if self.interactive:
                        proceed = await self.session.confirm_with_timeout(
                            f"\nLarge reputation change ({score_delta}) detected. Continue?",
                            timeout=INTERACTION_TIMEOUTS["emergency"]
                        )
                        if not proceed:
                            self.logger.warning("Large reputation update cancelled")
                            return False

                self.update_reputation(node_id, score_delta)
                return True

        except Exception as e:
            self.logger.error(f"Interactive reputation update failed: {str(e)}")
            return False
        finally:
            if self.session:
                await self.session.__aexit__(None, None, None)

    async def bulk_update_interactive(self, updates: Dict[str, float]) -> bool:
        """Process multiple reputation updates with progress tracking"""
        if not updates:
            return True

        try:
            if self.interactive:
                self.session = InteractiveSession(
                    level=InteractionLevel.NORMAL,
                    config=InteractiveConfig(
                        timeout=INTERACTION_TIMEOUTS["batch"],
                        persistent_state=True,
                        safe_mode=True
                    )
                )

            async with self.session:
                # Restore progress if available
                processed: Set[str] = set()
                if self.interactive:
                    saved_progress = await self.session._load_progress()
                    if saved_progress:
                        processed = set(saved_progress.get("processed_nodes", []))
                        self.logger.info(f"Restored progress: {len(processed)} nodes")

                remaining_updates = {k: v for k, v in updates.items() 
                                  if k not in processed}

                if self.interactive:
                    pbar = tqdm(total=len(remaining_updates), 
                              desc="Processing Reputation Updates")

                for node_id, delta in remaining_updates.items():
                    if self._interrupt_requested:
                        break

                    success = await self.update_reputation_interactive(node_id, delta)
                    if success:
                        processed.add(node_id)
                        if self.interactive:
                            pbar.update(1)
                            await self.session._save_progress({
                                "processed_nodes": list(processed)
                            })

                if self.interactive:
                    pbar.close()

                return len(processed) == len(updates)

        except Exception as e:
            self.logger.error(f"Bulk reputation update failed: {str(e)}")
            return False
        finally:
            if self.session:
                await self.session.__aexit__(None, None, None)

    def request_shutdown(self):
        """Request graceful shutdown of reputation updates"""
        self._interrupt_requested = True
        self.logger.info("Shutdown requested for reputation manager")

    async def _cleanup(self):
        """Cleanup resources"""
        try:
            # Clean up any temporary files or resources
            if self.session:
                await self.session.__aexit__(None, None, None)
            self.session = None
        except Exception as e:
            self.logger.error(f"Cleanup failed: {str(e)}")
