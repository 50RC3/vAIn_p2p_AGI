import base64
import logging
from typing import Dict, Optional, Tuple
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidKey

logger = logging.getLogger(__name__)

class SecureKeyManager:
    def __init__(self):
        # Generate node's long-term identity key
        self.identity_key = ec.generate_private_key(ec.SECP384R1())
        self.identity_public = self.identity_key.public_key()
        
        # Store session keys for peers
        self.peer_sessions: Dict[str, AESGCM] = {}
        self.peer_keys: Dict[str, bytes] = {}
        
    async def establish_session(self, peer_id: str, peer_public_bytes: bytes) -> bool:
        """Establish secure session with peer using ECDH"""
        try:
            # Convert peer's public key
            peer_public = ec.EllipticCurvePublicKey.from_encoded_point(
                ec.SECP384R1(),
                peer_public_bytes
            )
            
            # Generate shared secret using ECDH
            shared_key = self.identity_key.exchange(
                ec.ECDH(),
                peer_public
            )
            
            # Derive session key using HKDF
            derived_key = HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=None,
                info=b'session-key'
            ).derive(shared_key)
            
            # Create AEAD cipher for session
            self.peer_sessions[peer_id] = AESGCM(derived_key)
            self.peer_keys[peer_id] = derived_key
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to establish session with {peer_id}: {e}")
            return False
            
    def encrypt_message(self, peer_id: str, message: bytes) -> Optional[Tuple[bytes, bytes]]:
        """Encrypt message for peer with authenticated encryption"""
        if peer_id not in self.peer_sessions:
            return None
            
        try:
            # Generate random nonce
            nonce = AESGCM.generate_nonce()
            
            # Encrypt and authenticate message
            ciphertext = self.peer_sessions[peer_id].encrypt(
                nonce,
                message,
                None  # Additional authenticated data (optional)
            )
            
            return nonce, ciphertext
            
        except Exception as e:
            logger.error(f"Failed to encrypt message for {peer_id}: {e}")
            return None
            
    def decrypt_message(self, peer_id: str, nonce: bytes, ciphertext: bytes) -> Optional[bytes]:
        """Decrypt and verify message from peer"""
        if peer_id not in self.peer_sessions:
            return None
            
        try:
            return self.peer_sessions[peer_id].decrypt(
                nonce,
                ciphertext,
                None  # Additional authenticated data (optional) 
            )
        except InvalidKey:
            logger.error(f"Failed to decrypt/verify message from {peer_id}")
            return None
