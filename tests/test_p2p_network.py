import pytest
import asyncio
from network.p2p_network import P2PNetwork
from unittest.mock import Mock, patch

@pytest.fixture
def network_config():
    return {
        'dht': {'bootstrap_nodes': ['127.0.0.1:8468']},
        'udp': {'port': 8469},
        'secret_key': 'test_key',
        'encryption_key': 'test_enc_key'
    }

@pytest.fixture
def p2p_node(network_config):
    return P2PNetwork('test_node', network_config, interactive=False)

@pytest.mark.asyncio
async def test_message_handling():
    node = p2p_node()
    
    # Test message validation
    valid_msg = {
        'type': 'data',
        'sender': 'peer1',
        'content': 'test'
    }
    await node._validate_message(valid_msg)
    
    # Test consensus message
    consensus_msg = {
        'type': 'consensus_proposal',
        'proposal_id': 'test_1',
        'change_type': 'peer_ban',
        'change': {'peer_id': 'bad_peer', 'reason': 'spam'}
    }
    await node._process_message(consensus_msg)
    assert 'test_1' in node.pending_state_changes

@pytest.mark.asyncio
async def test_peer_authentication():
    node = p2p_node()
    
    # Test successful auth
    auth_result = await node._authenticate_peer_interactive('peer1')
    assert auth_result == True
    assert 'peer1' in node.peers
    
    # Test failed auth
    with pytest.raises(PeerAuthenticationError):
        await node._authenticate_peer_interactive('banned_peer')
