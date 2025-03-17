from typing import Dict, Optional
from dataclasses import dataclass
import time
import logging

@dataclass
class ReputationMetrics:
    score: float
    last_update: float
    total_contributions: int

class ReputationManager:
    def __init__(self, decay_factor: float = 0.95, min_reputation: float = -100):
        self.reputation_scores: Dict[str, ReputationMetrics] = {}
        self.decay_factor = decay_factor
        self.min_reputation = min_reputation
        self.logger = logging.getLogger('ReputationManager')
        
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
