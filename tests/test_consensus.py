import pytest
import numpy as np
from network.consensus import ConsensusManager, VoteResult

@pytest.fixture
def consensus_manager():
    return ConsensusManager(interactive=False)

def test_voting_power():
    cm = consensus_manager()
    node_id = "test_node"
    
    # Test invalid participant
    assert cm.get_voting_power(node_id) == 0.0
    
    # Test valid participant
    cm.node_stakes[node_id] = 2000.0  # Above min_stake
    cm.node_pow_scores[node_id] = 1.0
    cm.node_contributions[node_id] = 1.0
    assert cm.get_voting_power(node_id) > 0.0

def test_numeric_vote_aggregation():
    cm = consensus_manager()
    
    values = (1.0, 2.0, 3.0, 100.0)  # Include outlier
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    
    result = cm._aggregate_numeric_votes(values, weights)
    assert 1.0 <= result <= 3.0  # Outlier should be removed

@pytest.mark.asyncio
async def test_proposal_validation():
    cm = consensus_manager()
    
    # Test peer ban proposal
    ban_proposal = {
        'peer_id': 'malicious_node',
        'reason': 'spam',
        'evidence': ['event1', 'event2', 'event3']
    }
    assert await cm.validate_proposal('peer_ban', ban_proposal, 0.8)
    
    # Test invalid proposal
    invalid_proposal = {'peer_id': 'node1'}  # Missing required fields
    assert not await cm.validate_proposal('peer_ban', invalid_proposal, 0.8)
