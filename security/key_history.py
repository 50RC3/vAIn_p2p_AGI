from typing import Dict, Optional, Tuple
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.asymmetric import ed25519
import time
import logging

logger = logging.getLogger(__name__)

class KeyHistory:
    def __init__(self, max_key_age: int = 180*86400): # 180 days
        self.encryption_keys: Dict[int, Tuple[bytes, float]] = {}  # version -> (key, timestamp)
        self.signing_keys: Dict[int, Tuple[ed25519.Ed25519PrivateKey, float]] = {}
        self.current_version = 0
        self.max_key_age = max_key_age

    def add_keys(self, encryption_key: bytes, signing_key: ed25519.Ed25519PrivateKey) -> int:
        """Add new key pair and return version number"""
        self.current_version += 1
        timestamp = time.time()
        self.encryption_keys[self.current_version] = (encryption_key, timestamp)
        self.signing_keys[self.current_version] = (signing_key, timestamp)
        self._cleanup_old_keys()
        return self.current_version

    def get_keys(self, version: int) -> Optional[Tuple[bytes, ed25519.Ed25519PrivateKey]]:
        """Retrieve key pair by version"""
        if version in self.encryption_keys and version in self.signing_keys:
            return (self.encryption_keys[version][0], self.signing_keys[version][0])
        return None

    def _cleanup_old_keys(self):
        """Remove keys older than max_key_age"""
        current_time = time.time()
        for version in list(self.encryption_keys.keys()):
            if current_time - self.encryption_keys[version][1] > self.max_key_age:
                del self.encryption_keys[version]
                del self.signing_keys[version]
