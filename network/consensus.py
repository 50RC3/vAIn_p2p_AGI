import logging
import asyncio
from typing import Dict, Any, Optional, Set, List

# Import ReputationManager lazily when needed to avoid circular import
# from .reputation import ReputationManager

logger = logging.getLogger(__name__)

class ConsensusManager:
    """Manages consensus operations in the P2P network."""
    
    def __init__(self):
        self.logger = logging.getLogger('ConsensusManager')
        self.voting_powers = {}
        self.proposals = {}
        self.votes = {}
        self._reputation_manager = None

    def get_voting_power(self, node_id: str) -> float:
        """Get voting power for a node based on reputation and stake."""
        if node_id in self.voting_powers:
            return self.voting_powers[node_id]
        
        # Default power if not set
        return 1.0
        
    def set_reputation_manager(self, reputation_manager):
        """Set the reputation manager for reputation-based voting."""
        self._reputation_manager = reputation_manager
        
    def _get_reputation(self, node_id: str) -> float:
        """Get node reputation score."""
        if self._reputation_manager:
            return self._reputation_manager.get_reputation(node_id)
        return 0.5  # Default neutral reputation

    async def submit_proposal(self, proposal_id: str, proposal_data: Dict[str, Any]) -> bool:
        """Submit a new proposal for consensus."""
        if proposal_id in self.proposals:
            return False
            
        self.proposals[proposal_id] = {
            'data': proposal_data,
            'timestamp': asyncio.get_event_loop().time(),
            'status': 'pending'
        }
        
        return True
        
    async def vote(self, proposal_id: str, voter_id: str, vote: bool) -> bool:
        """Record a vote for a proposal."""
        if proposal_id not in self.proposals:
            return False
            
        if proposal_id not in self.votes:
            self.votes[proposal_id] = {}
            
        voting_power = self.get_voting_power(voter_id)
        self.votes[proposal_id][voter_id] = {
            'vote': vote,
            'power': voting_power,
            'timestamp': asyncio.get_event_loop().time()
        }
        
        return True
        
    async def get_result(self, proposal_id: str) -> Optional[bool]:
        """Get the current result for a proposal."""
        if proposal_id not in self.proposals or proposal_id not in self.votes:
            return None
            
        # Calculate votes
        total_power_yes = 0
        total_power_no = 0
        
        for voter, vote_data in self.votes[proposal_id].items():
            if vote_data['vote']:
                total_power_yes += vote_data['power']
            else:
                total_power_no += vote_data['power']
                
        total_power = total_power_yes + total_power_no
        if total_power == 0:
            return None
            
        # Decide based on majority of voting power
        return total_power_yes > total_power_no
