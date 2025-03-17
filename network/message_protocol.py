import json
from typing import Dict, Any, Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ed25519
import base64

class SecureMessageProtocol:
    def __init__(self, encryption_key: bytes):
        self.fernet = Fernet(encryption_key)
        self.signing_key = ed25519.Ed25519PrivateKey.generate()
        self.verify_key = self.signing_key.public_key()
        
    def encode_message(self, message: Dict[str, Any]) -> Dict[str, bytes]:
        """Encode and sign a message"""
        json_data = json.dumps(message, sort_keys=True)
        encrypted = self.fernet.encrypt(json_data.encode())
        signature = self.signing_key.sign(encrypted)
        
        return {
            'data': base64.b64encode(encrypted),
            'signature': base64.b64encode(signature),
            'public_key': self.verify_key.public_bytes()
        }
        
    def decode_message(self, encoded_message: Dict[str, bytes]) -> Optional[Dict[str, Any]]:
        """Decode and verify a message"""
        try:
            data = base64.b64decode(encoded_message['data'])
            signature = base64.b64decode(encoded_message['signature'])
            public_key = ed25519.Ed25519PublicKey.from_public_bytes(
                encoded_message['public_key']
            )
            
            # Verify signature
            public_key.verify(signature, data)
            
            # Decrypt message
            decrypted = self.fernet.decrypt(data)
            return json.loads(decrypted.decode())
            
        except Exception as e:
            logging.error(f"Message verification failed: {e}")
            return None
