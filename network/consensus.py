from typing import List, Dict, Optional, Tuple, Any, NamedTuple
from .reputation import ReputationManager
import numpy as np
import logging
import asyncio
from tqdm import tqdm
from core.constants import InteractionLevel, INTERACTION_TIMEOUTS 
from core.interactive_utils import InteractiveSession, InteractiveConfig
import backoff  # Add this import
from security.zk_validation import ZKProofValidator

logger = logging.getLogger(__name__)

class VoteResult(NamedTuple):
    value: Any
    confidence: float
    runner_up: Any
    margin: float
    total_weight: float

class ConsensusManager:
    def __init__(self, 
                 min_stake: float = 1000.0, 
                 min_reputation: float = 0.5,
                 pow_weight: float = 0.2,
                 stake_weight: float = 0.4,
                 contribution_weight: float = 0.4,
                 min_confidence: float = 0.6,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 partial_consensus_threshold: float = 0.7,
                 interactive: bool = True):
        self.min_stake = min_stake
        self.min_reputation = min_reputation
        self.pow_weight = pow_weight
        self.stake_weight = stake_weight
        self.contribution_weight = contribution_weight
        self.node_stakes: Dict[str, float] = {}
        self.node_pow_scores: Dict[str, float] = {}
        self.node_contributions: Dict[str, float] = {}
        self.reputation_manager = ReputationManager()
        self.interactive = interactive
        self.session = None
        self._interrupt_requested = False
        self._cleanup_lock = asyncio.Lock()
        self.min_confidence = min_confidence
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.partial_consensus_threshold = partial_consensus_threshold
        self.failed_nodes = set()
        self.zk_validator = ZKProofValidator()
        self.validation_rules = {
            'peer_ban': {
                'required_fields': ['peer_id', 'reason'],
                'min_reputation': 0.7,  # Higher threshold for bans
                'min_evidence': 3,
                'require_zk_proof': True
            },
            'reputation_update': {
                'required_fields': ['peer_id', 'delta'],
                'max_delta': 50.0,
                'min_reputation': 0.6
            },
            'cluster_reconfig': {
                'required_fields': ['cluster_id', 'changes'],
                'min_reputation': 0.8,
                'max_changes': 10
            }
        }

    def update_pow_score(self, node_id: str, pow_score: float) -> None:
        """Update node's PoW score based on computational work"""
        self.node_pow_scores[node_id] = pow_score

    def update_contribution_score(self, node_id: str, contribution_score: float) -> None:
        """Update node's contribution score based on resources and participation"""
        self.node_contributions[node_id] = contribution_score
        
    def is_valid_participant(self, node_id: str) -> bool:
        """Check if node meets minimum requirements for all consensus mechanisms"""
        has_min_stake = self.node_stakes.get(node_id, 0) >= self.min_stake
        has_min_reputation = self.reputation_manager.get_reputation(node_id) >= self.min_reputation
        has_min_pow = self.node_pow_scores.get(node_id, 0) > 0
        has_min_contribution = self.node_contributions.get(node_id, 0) > 0
        
        # Add ZK proof validation
        if self.validation_rules.get(node_id, {}).get('require_zk_proof'):
            proof = self._get_node_proof(node_id)
            if not self.zk_validator.verify_node_identity(proof):
                return False
        
        return all([has_min_stake, has_min_reputation, has_min_pow, has_min_contribution])
                
    def get_voting_power(self, node_id: str) -> float:
        """Calculate voting power using hybrid consensus weights"""
        if not self.is_valid_participant(node_id):
            return 0.0
            
        stake_power = self.node_stakes.get(node_id, 0.0) * self.stake_weight
        pow_power = self.node_pow_scores.get(node_id, 0.0) * self.pow_weight
        contribution_power = self.node_contributions.get(node_id, 0.0) * self.contribution_weight
        
        # Multiply by reputation as a global modifier
        reputation = self.reputation_manager.get_reputation(node_id)
        return (stake_power + pow_power + contribution_power) * reputation
        
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
        """
        Aggregate votes using weighted consensus based on vote type.
        Handles both numeric and categorical votes.
        """
        try:
            results = {}
            for key, weighted_values in votes.items():
                if not weighted_values:
                    continue
                    
                values, weights = zip(*weighted_values)
                weights = np.array(weights)
                
                # Skip if all weights are zero
                if not np.any(weights):
                    logger.warning(f"Skipping key {key} - all weights are zero")
                    continue
                    
                # Normalize weights
                weights = weights / np.sum(weights)
                
                # Determine vote type and aggregate accordingly
                if all(isinstance(v, (int, float)) for v in values):
                    # Numeric values - use weighted average
                    results[key] = self._aggregate_numeric_votes(values, weights)
                else:
                    # Categorical values - use weighted mode
                    results[key] = self._aggregate_categorical_votes(values, weights)
                    
                logger.debug(f"Aggregated result for {key}: {results[key]}")
                
            return results
            
        except Exception as e:
            logger.error(f"Vote aggregation failed: {str(e)}")
            raise

    def _aggregate_numeric_votes(self, values: tuple, weights: np.ndarray) -> float:
        """Aggregate numeric votes using weighted average"""
        try:
            # Handle potential outliers
            values_array = np.array(values)
            q1, q3 = np.percentile(values_array, [25, 75])
            iqr = q3 - q1
            outlier_mask = (values_array >= q1 - 1.5 * iqr) & (values_array <= q3 + 1.5 * iqr)
            
            if not np.all(outlier_mask):
                logger.warning(f"Removed {np.sum(~outlier_mask)} outlier votes")
                weights = weights[outlier_mask]
                values_array = values_array[outlier_mask]
                weights = weights / np.sum(weights)  # Renormalize weights
                
            return float(np.average(values_array, weights=weights))
            
        except Exception as e:
            logger.error(f"Numeric vote aggregation failed: {str(e)}")
            raise

    def _aggregate_categorical_votes(self, values: tuple, weights: np.ndarray) -> Any:
        """
        Aggregate categorical votes using weighted mode with confidence scoring
        and tie-breaking mechanisms.
        """
        try:
            unique_values, value_counts = np.unique(values, return_counts=True)
            weighted_counts = np.zeros_like(value_counts, dtype=float)
            total_weight = np.sum(weights)
            
            # Calculate weighted counts for each unique value
            for i, value in enumerate(unique_values):
                mask = np.array(values) == value
                weighted_counts[i] = np.sum(weights[mask])
            
            # Sort by weighted counts descending
            sorted_indices = np.argsort(weighted_counts)[::-1]
            sorted_values = unique_values[sorted_indices]
            sorted_counts = weighted_counts[sorted_indices]
            
            # Calculate confidence metrics
            winner_idx = sorted_indices[0]
            winner_value = unique_values[winner_idx]
            winner_weight = weighted_counts[winner_idx]
            
            confidence = winner_weight / total_weight
            
            # Handle potential ties
            if len(sorted_counts) > 1:
                margin = (sorted_counts[0] - sorted_counts[1]) / total_weight
                runner_up = sorted_values[1]
                
                # Check for tie (within 1%)
                if margin < 0.01:
                    # Tie-breaking strategies:
                    # 1. Check participation rate
                    winner_participants = np.sum(np.array(values) == winner_value)
                    runner_participants = np.sum(np.array(values) == runner_up)
                    
                    if winner_participants != runner_participants:
                        # Break tie by participation count
                        if runner_participants > winner_participants:
                            winner_value = runner_up
                            confidence = sorted_counts[1] / total_weight
                            margin = (runner_participants - winner_participants) / len(values)
                    else:
                        # If still tied, use timestamp as deterministic tie-breaker
                        timestamp_hash = hash(str(asyncio.get_event_loop().time()))
                        winner_value = sorted_values[timestamp_hash % 2]
                        
                        logger.warning(f"Tie broken deterministically for {winner_value}")
                        margin = 0.0
            else:
                runner_up = None
                margin = 1.0
            
            result = VoteResult(
                value=winner_value,
                confidence=confidence,
                runner_up=runner_up,
                margin=margin,
                total_weight=total_weight
            )
            
            # Log decision metrics
            logger.info(f"Vote Result: value={result.value}, confidence={result.confidence:.2f}, "
                       f"margin={result.margin:.2f}, total_weight={result.total_weight:.2f}")
            
            # Check minimum confidence threshold
            if confidence < self.min_confidence:
                logger.warning(f"Low confidence decision ({confidence:.2f} < {self.min_confidence})")
                
            return result.value
            
        except Exception as e:
            logger.error(f"Categorical vote aggregation failed: {str(e)}")
            raise

    @backoff.on_exception(
        backoff.expo,
        (ConnectionError, TimeoutError),
        max_tries=3,
        max_time=30
    )
    async def _collect_node_votes(self, node_id: str, proposal: Dict) -> Optional[Dict]:
        """Collect votes from a single node with automatic retries"""
        try:
            if node_id in self.failed_nodes:
                logger.warning(f"Skipping previously failed node {node_id}")
                return None

            weight = self.get_voting_power(node_id)
            if not weight:
                return None

            return {
                'node_id': node_id,
                'weight': weight,
                'votes': proposal
            }

        except Exception as e:
            logger.error(f"Node {node_id} vote collection failed: {str(e)}")
            self.failed_nodes.add(node_id)
            return None

    async def reach_consensus_interactive(self, proposals: Dict[str, Dict]) -> Optional[Dict]:
        """Interactive consensus reaching with enhanced fault handling"""
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
                saved_progress = await self.session._load_progress()
                weighted_votes = saved_progress.get('weighted_votes', {}) if saved_progress else {}
                
                total_nodes = len(proposals)
                min_required = int(total_nodes * self.partial_consensus_threshold)
                successful_nodes = set()

                print(f"\nProcessing Consensus Proposals (min required: {min_required})")
                print("=" * 50)

                try:
                    with tqdm(total=total_nodes, desc="Processing Votes") as pbar:
                        for node_id, proposal in proposals.items():
                            if self._interrupt_requested:
                                break

                            vote_result = await self._collect_node_votes(node_id, proposal)
                            if not vote_result:
                                continue

                            # Process votes
                            for key, value in vote_result['votes'].items():
                                if key not in weighted_votes:
                                    weighted_votes[key] = []
                                weighted_votes[key].append((value, vote_result['weight']))

                            successful_nodes.add(node_id)
                            
                            # Save progress periodically
                            if len(successful_nodes) % 10 == 0:
                                await self._save_progress({
                                    'weighted_votes': weighted_votes,
                                    'successful_nodes': list(successful_nodes),
                                    'failed_nodes': list(self.failed_nodes)
                                })

                            pbar.update(1)

                    # Check if we have enough participation
                    if len(successful_nodes) < min_required:
                        logger.error(f"Insufficient participation: {len(successful_nodes)}/{total_nodes} nodes")
                        if self.interactive:
                            proceed = await self.session.confirm_with_timeout(
                                f"\nOnly {len(successful_nodes)}/{total_nodes} nodes participated. Continue anyway?",
                                timeout=INTERACTION_TIMEOUTS["emergency"]
                            )
                            if not proceed:
                                return None

                    if not self._interrupt_requested and weighted_votes:
                        results = await self._aggregate_weighted_votes_interactive(weighted_votes)
                        
                        # Add enhanced metadata and validation results
                        results['_metadata'] = {
                            'total_nodes': total_nodes,
                            'successful_nodes': len(successful_nodes),
                            'failed_nodes': len(self.failed_nodes),
                            'participation_rate': len(successful_nodes) / total_nodes,
                            'consensus_threshold': self.partial_consensus_threshold,
                            'min_confidence': self.min_confidence,
                            'timestamp': asyncio.get_event_loop().time(),
                            'validation': {
                                'quorum_met': len(successful_nodes) >= min_required,
                                'integrity_check': all(key in weighted_votes for key in proposals[next(iter(proposals))]),
                                'weights_normalized': all(np.isclose(sum(w for _, w in votes), 1.0) 
                                                      for votes in weighted_votes.values())
                            }
                        }
                        
                        # Check data integrity before returning
                        if not results['_metadata']['validation']['integrity_check']:
                            logger.warning("Consensus results may be incomplete - missing vote keys detected")
                        
                        if not results['_metadata']['validation']['weights_normalized']:
                            logger.error("Weight normalization error detected")
                            if self.interactive and self.session:
                                proceed = await self.session.confirm_with_timeout(
                                    "Weight normalization error detected. Continue anyway?",
                                    timeout=INTERACTION_TIMEOUTS["emergency"]
                                )
                                if not proceed:
                                    return None
                        
                        return results
                    return None

                except Exception as e:
                    logger.error(f"Consensus processing error: {str(e)}")
                    if weighted_votes:
                        await self._save_progress({
                            'weighted_votes': weighted_votes,
                            'error': str(e),
                            'successful_nodes': list(successful_nodes),
                            'failed_nodes': list(self.failed_nodes)
                        })
                    raise

        except Exception as e:
            logger.error(f"Interactive consensus failed: {str(e)}")
            raise
        finally:
            self.failed_nodes.clear()
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

    async def validate_proposal(self, change_type: str, change: Dict, 
                              proposer_reputation: float) -> bool:
        """Validate proposal against consensus rules"""
        if change_type not in self.validation_rules:
            return False

        rules = self.validation_rules[change_type]
        
        # Check proposer reputation
        if proposer_reputation < rules['min_reputation']:
            return False

        # Validate required fields
        if not all(f in change for f in rules['required_fields']):
            return False

        # Type-specific validation
        if change_type == 'peer_ban':
            return len(change.get('evidence', [])) >= rules['min_evidence']
        elif change_type == 'reputation_update':
            return abs(change['delta']) <= rules['max_delta']
        elif change_type == 'cluster_reconfig':
            return len(change['changes']) <= rules['max_changes']

        return True
