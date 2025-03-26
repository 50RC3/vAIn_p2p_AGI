import socket
import psutil
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

def optimize_socket_buffers(sock: socket.socket, is_udp: bool = False) -> None:
    """Optimize socket buffer sizes based on available memory"""
    try:
        # Get system memory info
        mem = psutil.virtual_memory()
        total_mem = mem.total
        
        # Calculate optimal buffer sizes (0.1% of available memory)
        optimal_buffer = min(max(65536, int(total_mem * 0.001)), 16777216)
        
        # Set socket buffer sizes
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, optimal_buffer)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, optimal_buffer)
        
        if not is_udp:
            # TCP specific optimizations
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # Disable Nagle's
            if hasattr(socket, 'TCP_QUICKACK'):
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_QUICKACK, 1)
                
    except Exception as e:
        logger.warning(f"Failed to optimize socket buffers: {e}")

class TransportOptimizer:
    def __init__(self):
        self.tcp_settings = {
            'keepalive': True,
            'keepidle': 60,  # Start probing after 60s idle
            'keepintvl': 10,  # Probe interval 10s
            'keepcnt': 3,    # 3 failed probes = dead connection
        }
        
    def optimize_tcp_connection(self, sock: socket.socket) -> None:
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            for opt in ['TCP_KEEPIDLE', 'TCP_KEEPINTVL', 'TCP_KEEPCNT']:
                if hasattr(socket, opt):
                    sock.setsockopt(socket.IPPROTO_TCP, getattr(socket, opt),
                                  self.tcp_settings[opt.lower()[4:]])
        except Exception as e:
            logger.warning(f"Failed to set TCP keepalive: {e}")
