import pytest
import asyncio
from typing import Dict, Any
import logging
import torch
import torch.nn as nn
import os
import tempfile
from unittest.mock import MagicMock, patch
import sys

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pytest_setup")

# List of modules we expect might be missing
OPTIONAL_MODULES = [
    'ai_core.coordinator_factory',
    'ai_core.learning',
    'ai_core.chatbot',
    'ai_core.cognitive_evolution',
    'monitoring.node_status',
    'monitoring.system_monitor',
    'network.node_communication',
    'kademlia',
]

# Helper functions for tests
@pytest.fixture(scope="session")
def skip_missing_modules():
    """Skip tests that require modules that are not yet implemented."""
    missing = []
    for module in OPTIONAL_MODULES:
        try:
            __import__(module.split('.')[0])
        except ImportError:
            missing.append(module)
    
    return missing

def safe_import(module_name):
    """Import a module or return None if it's not available."""
    try:
        return __import__(module_name, fromlist=['*'])
    except ImportError:
        return None

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_node_metrics() -> Dict[str, Any]:
    """Mock node performance metrics"""
    return {
        'cpu': 50.0,
        'memory': 60.0,
        'bandwidth': 40.0,
        'disk': 30.0,
        'uptime': 3600,
        'error_rate': 0.01
    }

@pytest.fixture
def mock_signatures():
    """Mock cryptographic signatures for testing"""
    return {
        'valid_sig': 'valid_signature_bytes',
        'invalid_sig': 'invalid_signature_bytes',
        'public_key': 'mock_public_key_bytes'
    }

# New fixtures for advanced testing

@pytest.fixture
def simple_model():
    """Create a simple model for testing"""
    class SimpleTestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 2)
        
        def forward(self, x):
            return self.linear(x)
    
    return SimpleTestModel()

@pytest.fixture
def mock_federated_client():
    """Create a mock federated client"""
    client = MagicMock()
    client.train.return_value = {"linear.weight": torch.ones(2, 10), "linear.bias": torch.zeros(2)}
    return client

@pytest.fixture
def test_tensor_batch():
    """Create a batch of test tensors"""
    return [(torch.randn(8, 10), torch.randint(0, 2, (8,))) for _ in range(3)]

@pytest.fixture
def mock_p2p_network():
    """Create a mock P2P network"""
    network = MagicMock()
    network.broadcast.return_value = True
    network.request_from_peers.return_value = {"peer1": {"data": "test_data"}}
    return network

@pytest.fixture
def temp_model_file():
    """Create a temporary file for model saving/loading tests"""
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        file_path = f.name
    yield file_path
    if os.path.exists(file_path):
        os.remove(file_path)

@pytest.fixture(scope="session", autouse=True)
def add_project_root_to_path():
    """Add project root to Python path."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        logger.info(f"Added {project_root} to Python path")
