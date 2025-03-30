from typing import Dict, Any, Optional
import asyncio
from dataclasses import dataclass, field
from cryptography.fernet import Fernet
from datetime import datetime
import logging
import json
import time
from cryptography.hazmat.primitives.asymmetric import ed25519
import base64
from core.interactive_utils import InteractiveSession, InteractiveConfig, InteractionLevel
from core.constants import INTERACTION_TIMEOUTS
from cryptography.exceptions import InvalidKey, InvalidSignature
from cryptography.fernet import InvalidToken
from training.compression import AdaptiveCompression  # Fixed import path
from security.key_history import KeyHistory  # Import from security module

logger = logging.getLogger(__name__)

@dataclass
class Message:
    """Message class for network communication"""
    sender: str
    recipient: Optional[str]
    message_type: str
    content: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    message_id: Optional[str] = None
    is_encrypted: bool = False
    signature: Optional[bytes] = None
    ttl: int = 3600  # Time-to-live in seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format"""
        result = {
            "sender": self.sender,
            "type": self.message_type,
            "content": self.content,
            "timestamp": self.timestamp
        }
        if self.recipient:
            result["recipient"] = self.recipient
        if self.message_id:
            result["message_id"] = self.message_id
        if self.is_encrypted:
            result["encrypted"] = True
        if self.signature:
            result["signature"] = base64.b64encode(self.signature).decode('utf-8')
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message instance from dictionary"""
        signature = None
        if "signature" in data:
            signature = base64.b64decode(data["signature"])
        
        return cls(
            sender=data.get("sender", "unknown"),
            recipient=data.get("recipient"),
            message_type=data.get("type", "unknown"),
            content=data.get("content", {}),
            timestamp=data.get("timestamp", time.time()),
            message_id=data.get("message_id"),
            is_encrypted=data.get("encrypted", False),
            signature=signature,
            ttl=data.get("ttl", 3600)
        )

class SecureMessageProtocol:
    def __init__(self, encryption_key: bytes, interactive: bool = True):
        self.key_history = KeyHistory()  # Using the imported KeyHistory class
        self.fernet = Fernet(encryption_key)
        self.signing_key = ed25519.Ed25519PrivateKey.generate()
        self.verify_key = self.signing_key.public_key()
        self.interactive = interactive
        self.session = None
        self._interrupt_requested = False
        self._metrics_lock = asyncio.Lock()
        self.metrics = {
            'messages_processed': 0,
            'verification_failures': 0,
            'last_error': None,
            'key_rotations': 0,
            'last_key_rotation': time.time()
        }
        self.metrics.update({
            'pruned_sessions': 0,
            'invalid_attempts': 0
        })
        self.max_session_age = 3600 * 24  # 24 hours
        self.session_prune_interval = 3600  # 1 hour
        # Key rotation settings
        self.key_rotation_interval = 86400  # 24 hours in seconds
        self._lock = asyncio.Lock()
        self.current_key_version = 0
        # Initialize the key history with the initial keys
        self.current_key_version = self.key_history.add_keys(encryption_key, self.signing_key)

    async def rotate_keys(self) -> bool:
        """Rotate keys and maintain history"""
        try:
            async with self._lock:
                new_encryption_key = Fernet.generate_key()
                new_signing_key = ed25519.Ed25519PrivateKey.generate()
                
                # Store new keys in history
                self.current_key_version = self.key_history.add_keys(
                    new_encryption_key, new_signing_key
                )
                
                # Update current keys
                self.fernet = Fernet(new_encryption_key)
                self.signing_key = new_signing_key
                self.verify_key = new_signing_key.public_key()
                
                async with self._metrics_lock:
                    self.metrics['key_rotations'] += 1
                    self.metrics['last_key_rotation'] = time.time()
                
                logger.info("Successfully rotated encryption and signing keys")
                return True
        except Exception as e:
            logger.error(f"Key rotation failed: {str(e)}")
            async with self._metrics_lock:
                self.metrics['last_error'] = f"Key rotation failed: {str(e)}"
            return False

    async def check_key_rotation(self) -> None:
        """Check and perform key rotation if needed"""
        current_time = time.time()
        async with self._metrics_lock:
            if current_time - self.metrics['last_key_rotation'] >= self.key_rotation_interval:
                await self.rotate_keys()

    async def prune_old_sessions(self):
        """Remove old sessions to prevent memory leaks"""
        current_time = time.time()
        async with self._lock:
            for version in list(self.key_history.keys()):
                if current_time - version > self.max_session_age:
                    self.key_history.remove_version(version)
                    async with self._metrics_lock:
                        self.metrics['pruned_sessions'] += 1

    def encode_message(self, message: Dict[str, Any]) -> Dict[str, bytes]:
        """Encode and sign a message with version info"""
        json_data = json.dumps(message, sort_keys=True)
        encrypted = self.fernet.encrypt(json_data.encode())
        signature = self.signing_key.sign(encrypted)
        
        return {
            'data': base64.b64encode(encrypted),
            'signature': base64.b64encode(signature),
            'public_key': self.verify_key.public_bytes(),
            'key_version': self.current_key_version
        }

    async def encode_message_interactive(self, message: Dict[str, Any]) -> Optional[Dict[str, bytes]]:
        """Interactive message encoding with safety checks"""
        await self.check_key_rotation()
        try:
            if self.interactive:
                self.session = InteractiveSession(
                    level=InteractionLevel.NORMAL,
                    config=InteractiveConfig(
                        timeout=INTERACTION_TIMEOUTS["default"],
                        persistent_state=True,
                        safe_mode=True
                    )
                )

            async with self.session:
                # Validate message size
                msg_size = len(json.dumps(message))
                if msg_size > 1024 * 1024:  # 1MB limit
                    if self.interactive:
                        proceed = await self.session.confirm_with_timeout(
                            f"Large message size ({msg_size/1024:.1f}KB). Continue?",
                            timeout=INTERACTION_TIMEOUTS["emergency"]
                        )
                        if not proceed:
                            return None

                encoded = self.encode_message(message)
                async with self._metrics_lock:
                    self.metrics['messages_processed'] += 1
                return encoded

        except Exception as e:
            logger.error(f"Message encoding failed: {str(e)}")
            async with self._metrics_lock:
                self.metrics['last_error'] = str(e)
            raise
        finally:
            if self.session:
                await self.session.__aexit__(None, None, None)

    def decode_message(self, encoded_message: Dict[str, bytes]) -> Optional[Dict[str, Any]]:
        """Decode and verify a message with specific error handling"""
        try:
            data = base64.b64decode(encoded_message['data'])
            signature = base64.b64decode(encoded_message['signature'])
            public_key = ed25519.Ed25519PublicKey.from_public_bytes(
                encoded_message['public_key']
            )
            
            try:
                public_key.verify(signature, data)
            except InvalidSignature:
                logger.error("Message signature verification failed")
                return None
            
            try:
                decrypted = self.fernet.decrypt(data)
            except InvalidToken:
                logger.error("Message decryption failed - invalid token")
                return None
            except InvalidKey:
                logger.error("Message decryption failed - invalid key")
                return None
                
            return json.loads(decrypted.decode())
            
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error(f"Message format error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in message decoding: {e}")
            return None

    async def decode_message_interactive(self, encoded_message: Dict) -> Optional[Dict]:
        """Interactive message decoding with decompression support"""
        try:
            # Check if message is compressed
            if 'compressed' in encoded_message:
                encoded_message = await self._decompress_message(encoded_message)
                
            # Continue with normal decoding
            if self.interactive:
                self.session = InteractiveSession(
                    level=InteractionLevel.NORMAL,
                    config=InteractiveConfig(
                        timeout=INTERACTION_TIMEOUTS["default"],
                        persistent_state=True,
                        safe_mode=True
                    )
                )
            async with self.session:
                return await self._decode_message_secure(encoded_message)
                
        except Exception as e:
            logger.error(f"Interactive decoding failed: {str(e)}")
            return None
        finally:
            if self.session:
                await self.session.__aexit__(None, None, None)

    async def _decompress_message(self, message: Dict) -> Dict:
        """Decompress message using adaptive compression"""
        try:
            decompressor = AdaptiveCompression()
            decompressed = await decompressor.decompress_model_updates(message['compressed'])
            return self._tensor_to_dict(decompressed['message'])
        except Exception as e:
            logger.error(f"Decompression failed: {str(e)}")
            raise

    def _tensor_to_dict(self, tensor) -> Dict:
        """Convert tensor back to dictionary"""
        try:
            bytes_data = bytes([int(i) for i in tensor.tolist()])
            return json.loads(bytes_data.decode())
        except Exception as e:
            logger.error(f"Tensor conversion failed: {str(e)}")
            raise

    def request_shutdown(self):
        """Request graceful shutdown"""
        self._interrupt_requested = True
        logger.info("Shutdown requested for message protocol")

    async def get_metrics(self) -> Dict[str, Any]:
        """Get protocol metrics thread-safely"""
        async with self._metrics_lock:
            return self.metrics.copy()
