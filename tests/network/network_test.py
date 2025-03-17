import unittest
import asyncio
from network.p2p_network import P2PNetwork
from network.rate_limiter import RateLimiter
from network.circuit_breaker import CircuitBreaker

class TestP2PNetwork(unittest.TestCase):
    def setUp(self):
        self.config = {
            'dht': {'bootstrap_nodes': []},
            'udp': {'port': 8468},
            'secret_key': 'test_key'
        }
        self.network = P2PNetwork('test_node', self.config)
        
    async def test_rate_limiting(self):
        sender = "test_peer"
        limiter = RateLimiter(max_requests=2, time_window=1)
        
        # First two requests should pass
        self.assertTrue(await limiter.allow_request(sender))
        self.assertTrue(await limiter.allow_request(sender))
        # Third request should fail
        self.assertFalse(await limiter.allow_request(sender))
        
    def test_circuit_breaker(self):
        breaker = CircuitBreaker(failure_threshold=2, reset_timeout=1)
        
        # Should start closed
        self.assertTrue(breaker.allow_request())
        
        # Record failures
        breaker.record_failure()
        self.assertTrue(breaker.allow_request())
        breaker.record_failure()
        self.assertFalse(breaker.allow_request())

if __name__ == '__main__':
    unittest.main()
