from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding, ec
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from typing import Dict, Optional, Tuple
import time

class AuthManager:
    def __init__(self):
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.public_key = self.private_key.public_key()
        self.sessions: Dict[str, bytes] = {}
        self.session_expiry: Dict[str, float] = {}
        self.session_duration = 3600  # 1 hour
        self.key_rotation_interval = 86400  # 24 hours
        self.last_rotation = time.time()
        
    def create_session(self, peer_id: str) -> Tuple[bytes, float]:
        """Create new session with expiry"""
        session_key = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b'session-key',
        ).derive(peer_id.encode())
        
        expiry = time.time() + self.session_duration
        self.sessions[peer_id] = session_key
        self.session_expiry[peer_id] = expiry
        
        # Rotate keys if needed
        if time.time() - self.last_rotation > self.key_rotation_interval:
            self._rotate_keys()
            
        return session_key, expiry

    def _rotate_keys(self):
        """Rotate asymmetric keys periodically"""
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.public_key = self.private_key.public_key()
        self.last_rotation = time.time()
        
    def verify_message(self, peer_id: str, message: bytes, signature: bytes) -> bool:
        """Verify signed message"""
        try:
            self.public_key.verify(
                signature,
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except:
            return False
