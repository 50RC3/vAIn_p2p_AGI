import pytest
import asyncio
from typing import Dict, Any
import logging

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)

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
