"""Custom exceptions for P2P network operations"""

class NetworkError(Exception):
    """Base class for network-related errors"""
    pass

class PeerConnectionError(NetworkError):
    """Error when connecting to a peer"""
    pass

class PeerAuthenticationError(NetworkError):
    """Error when authenticating with a peer"""
    pass

class MessageFormatError(NetworkError):
    """Error in message format"""
    pass

class NetworkTimeoutError(NetworkError):
    """Network operation timed out"""
    pass