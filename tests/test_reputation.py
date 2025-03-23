import pytest
import asyncio
from network.reputation import ReputationManager, ReputationMetrics, PendingChange
from time import monotonic

@pytest.fixture
def reputation_manager():
    return ReputationManager(interactive=False)

@pytest.mark.asyncio
async def test_reputation_update():
    rm = reputation_manager()
    node_id = "test_node"
    
    # Test initial reputation
    assert rm.get_reputation(node_id) == 0.0
    
    # Test basic update
    await rm.update_reputation_interactive(node_id, 1.0)
    assert rm.get_reputation(node_id) == 1.0
    
    # Test decay
    metrics = rm.reputation_scores[node_id]
    metrics.last_update -= 86400  # Simulate 1 day passing
    await rm.update_reputation_interactive(node_id, 0)
    assert rm.get_reputation(node_id) < 1.0

@pytest.mark.asyncio
async def test_cooling_period():
    rm = reputation_manager()
    node_id = "test_node"
    
    # Small changes should apply immediately
    await rm.update_reputation_interactive(node_id, 0.05)
    assert rm.get_reputation(node_id) == 0.05
    
    # Large changes should be queued
    await rm.update_reputation_interactive(node_id, 1.0)
    assert len(rm.pending_changes[node_id]) == 1
    assert rm.get_reputation(node_id) == 0.05  # Unchanged until cooling period

@pytest.mark.asyncio
async def test_validation():
    rm = reputation_manager()
    node_id = "test_node"
    
    # Test validation success
    result = await rm._validate_peer(node_id)
    assert result == True
    
    # Test validation caching
    cached_result = await rm._validate_peer(node_id)
    assert cached_result == True
    assert node_id in rm._validation_cache
