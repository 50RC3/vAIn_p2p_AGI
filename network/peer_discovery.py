import socket
import json
from typing import List, Dict

class PeerDiscovery:
    def __init__(self, port: int = 5000):
        self.port = port
        self.peers = set()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        
    def broadcast_presence(self):
        message = json.dumps({"type": "discovery", "port": self.port})
        self.sock.sendto(message.encode(), ('<broadcast>', self.port))
