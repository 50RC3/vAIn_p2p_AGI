from typing import List, Dict, Optional
from .reputation import ReputationManager
import numpy as np
import logging
import asyncio
from tqdm import tqdm
from core.constants import InteractionLevel, INTERACTION_TIMEOUTS 
from core.interactive_utils import InteractiveSession, InteractiveConfig

logger = logging.getLogger(__name__)

class ConsensusManager:
    def __init__(self, min_stake: float = 1000.0, min_reputation: float = 0.5, interactive: bool = True):
        self.min_stake = min_stake
        self.min_reputation = min_reputation
        self.node_stakes: Dict[str, float] = {}
        self.reputation_manager = ReputationManager()
        self.interactive = interactive
        self.session = None
        self._interrupt_requested = False
        self._cleanup_lock = asyncio.Lock()
        
    def is_valid_participant(self, node_id: str) -> bool:
        return (self.node_stakes.get(node_id, 0) >= self.min_stake and 
                self.reputation_manager.get_reputation(node_id) >= self.min_reputation)
                
    def get_voting_power(self, node_id: str) -> float:
        stake = self.node_stakes.get(node_id, 0.0)
        reputation = self.reputation_manager.get_reputation(node_id)
        return stake * reputation if stake >= self.min_stake else 0.0
        
    def reach_consensus(self, proposals: Dict[str, Dict]) -> Dict:
        if not proposals:
            raise ValueError("No proposals to evaluate")
            
        weighted_votes = {}
        for node_id, proposal in proposals.items():
            weight = self.get_voting_power(node_id)
            for key, value in proposal.items():
                if key not in weighted_votes:
                    weighted_votes[key] = []
                weighted_votes[key].append((value, weight))
                
        return self._aggregate_weighted_votes(weighted_votes)
        
    def _aggregate_weighted_votes(self, votes: Dict[str, List]) -> Dict:
        # Implement voting aggregation logic using stake-weighted consensus
        pass

    async def reach_consensus_interactive(self, proposals: Dict[str, Dict]) -> Optional[Dict]:
        """Interactive consensus reaching with progress tracking and error handling"""
        if not proposals:
            raise ValueError("No proposals to evaluate")

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
                # Try to restore previous progress
                saved_progress = await self.session._load_progress()
                if saved_progress:
                    logger.info("Restoring from previous consensus state")
                    weighted_votes = saved_progress.get('weighted_votes', {})
                else:
                    weighted_votes = {}

                print("\nProcessing Consensus Proposals")
                print("=" * 50)

                try:
                    with tqdm(total=len(proposals), desc="Processing Votes") as pbar:
                        for node_id, proposal in proposals.items():
                            if self._interrupt_requested:
                                logger.info("Consensus interrupted by user")
                                break

                            # Validate participant
                            if not self.is_valid_participant(node_id):
                                logger.warning(f"Invalid participant {node_id} - skipping")
                                continue

                            weight = self.get_voting_power(node_id)
                            for key, value in proposal.items():
                                if key not in weighted_votes:
                                    weighted_votes[key] = []
                                weighted_votes[key].append((value, weight))

                            # Save progress periodically
                            if len(weighted_votes) % 10 == 0:
                                await self._save_progress(weighted_votes)

                            pbar.update(1)

                    if not self._interrupt_requested and weighted_votes:
                        return await self._aggregate_weighted_votes_interactive(weighted_votes)
                    return None

                except Exception as e:
                    logger.error(f"Consensus processing error: {str(e)}")
                    if weighted_votes:
                        await self._save_progress(weighted_votes)
                    raise

        except Exception as e:
            logger.error(f"Interactive consensus failed: {str(e)}")
            raise
        finally:
            await self._cleanup()

    async def _aggregate_weighted_votes_interactive(self, votes: Dict[str, List]) -> Dict:
        """Aggregate votes with interactive monitoring"""
        try:
            results = {}
            with tqdm(total=len(votes), desc="Aggregating Results") as pbar:
                for key, weighted_values in votes.items():
                    if self._interrupt_requested:
                        break
                        
                    values, weights = zip(*weighted_values)
                    weights = np.array(weights)
                    
                    # Normalize weights
                    weights = weights / np.sum(weights)
                    
                    # For numeric values use weighted average
                    if all(isinstance(v, (int, float)) for v in values):
                        results[key] = float(np.average(values, weights=weights))
                    else:
                        # For non-numeric values use weighted mode
                        unique_values, value_counts = np.unique(values, return_counts=True)
                        weighted_counts = np.zeros_like(value_counts, dtype=float)
                        for i, value in enumerate(unique_values):
                            mask = np.array(values) == value
                            weighted_counts[i] = np.sum(weights[mask])
                        results[key] = unique_values[np.argmax(weighted_counts)]
                    
                    pbar.update(1)
                    
            return results

        except Exception as e:
            logger.error(f"Vote aggregation error: {str(e)}")
            raise

    async def _save_progress(self, weighted_votes: Dict) -> None:
        """Save consensus progress"""
        if self.session:
            await self.session.save_progress({
                'weighted_votes': weighted_votes,
                'timestamp': asyncio.get_event_loop().time()
            })

    async def _cleanup(self) -> None:
        """Cleanup resources"""
        async with self._cleanup_lock:
            try:
                self._interrupt_requested = False
                if self.session:
                    await self.session.__aexit__(None, None, None)
                    self.session = None
            except Exception as e:
                logger.error(f"Cleanup error: {str(e)}")

    def request_shutdown(self) -> None:
        """Request graceful shutdown"""
        self._interrupt_requested = True
        logger.info("Shutdown requested for consensus manager")
