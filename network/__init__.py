"""Network module for vAIn P2P AGI."""

# Import utility classes first to avoid circular imports
from .rate_limiter import RateLimiter
from .circuit_breaker import CircuitBreaker
from .pex import PeerExchange
from .monitoring import NetworkMonitor, get_resource_metrics
from .caching import CacheManager, CacheLevel, CachePolicy
from .message_protocol import SecureMessageProtocol, Message, KeyHistory

# Now we can import components that depend on the above
from .reputation import ReputationManager
from .consensus import ConsensusManager
from .load_balancer import LoadBalancer, NodeCapacity
from .cluster_manager import ClusterManager
from .gossip_protocol import GossipProtocol, GossipManager
from .admin_commands import AdminCommands

# Finally, import the main P2P network class
from .p2p_network import P2PNetwork

__all__ = [
    'P2PNetwork',
    'PeerExchange',
    'ConsensusManager',
    'AdminCommands',
    'RateLimiter', 
    'CircuitBreaker',
    'GossipProtocol',
    'GossipManager',
    'ReputationManager',
    'LoadBalancer',
    'NetworkMonitor',
    'get_resource_metrics',
    'SecureMessageProtocol',
    'Message',
    'KeyHistory',
    'CacheManager',
    'CacheLevel',
    'CachePolicy',
    'ClusterManager',
    'NodeCapacity'
]

# Any other package-level definitions or imports can go here
