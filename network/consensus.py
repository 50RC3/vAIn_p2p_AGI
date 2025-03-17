from typing import List, Dict, Optional
from .reputation import ReputationManager
import numpy as np

class ConsensusManager:
    def __init__(self, min_stake: float = 1000.0, min_reputation: float = 0.5):
        self.min_stake = min_stake
        self.min_reputation = min_reputation
        self.node_stakes: Dict[str, float] = {}
        self.reputation_manager = ReputationManager()
        
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
