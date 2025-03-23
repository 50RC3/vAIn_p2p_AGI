import pytest
from network.dht import DHT, RoutingZone
from unittest.mock import Mock, patch

@pytest.fixture
def dht_config():
    return {
        'bootstrap_nodes': ['127.0.0.1:8468'],
        'port': 8468,
        'max_peers': 100
    }

@pytest.fixture
def dht_node(dht_config):
    return DHT('test_node', dht_config, interactive=False)

@pytest.mark.asyncio
async def test_peer_discovery():
    dht = dht_node()
    
    # Mock discovery methods
    async def mock_discover_region(*args):
        return {'peer1', 'peer2', 'peer3'}
    dht._discover_region = mock_discover_region
    
    discovered = await dht.discover()
    assert len(discovered) == 3
    assert 'peer1' in discovered

@pytest.mark.asyncio
async def test_zone_management():
    dht = dht_node()
    zone = RoutingZone('test_zone')
    
    # Test zone splitting
    zone.nodes = {'node1', 'node2', 'node3'}
    zone.load_metrics = {'cpu': 90, 'bandwidth': 85}
    
    await dht._rebalance_zones(['test_zone'])
    assert len(dht.zones) > 1  # Zone should split due to load
