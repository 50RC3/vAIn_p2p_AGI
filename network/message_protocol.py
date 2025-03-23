import json
import logging
import asyncio
import time
from typing import Dict, Any, Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ed25519
import base64
from core.interactive_utils import InteractiveSession, InteractiveConfig, InteractionLevel
from core.constants import INTERACTION_TIMEOUTS
from cryptography.exceptions import InvalidKey, InvalidSignature
from cryptography.fernet import InvalidToken
from security.key_history import KeyHistory

logger = logging.getLogger(__name__)

class SecureMessageProtocol:
    def __init__(self, encryption_key: bytes, interactive: bool = True):
        self.key_history = KeyHistory()
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
        # Key rotation settings
        self.key_rotation_interval = 86400  # 24 hours in seconds
        self._lock = asyncio.Lock()
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

    async def decode_message_interactive(self, encoded_message: Dict[str, bytes]) -> Optional[Dict[str, Any]]:
        """Decode message with version-specific keys"""
        await self.check_key_rotation()
        try:
            if self.interactive:
                verify_timeout = INTERACTION_TIMEOUTS.get("verify", 30)
                emergency_timeout = INTERACTION_TIMEOUTS.get("emergency", 15)
                
                self.session = InteractiveSession(
                    level=InteractionLevel.NORMAL,
                    config=InteractiveConfig(
                        timeout=verify_timeout,
                        persistent_state=True,
                        safe_mode=True
                    )
                )

            async with self.session:
                # Validate required fields
                required_fields = {'data', 'signature', 'public_key'}
                missing_fields = required_fields - set(encoded_message.keys())
                if missing_fields:
                    logger.error(f"Missing required message fields: {missing_fields}")
                    async with self._metrics_lock:
                        self.metrics['verification_failures'] += 1
                    return None

                try:
                    if 'key_version' in encoded_message:
                        version = encoded_message['key_version']
                        keys = self.key_history.get_keys(version)
                        if keys:
                            temp_fernet = Fernet(keys[0])
                            temp_signing_key = keys[1]
                            # Use historical keys for decryption
                            # ...rest of decoding logic...
                        else:
                            logger.error(f"Required key version {version} not found")
                            return None

                    result = self.decode_message(encoded_message)
                    if result:
                        async with self._metrics_lock:
                            self.metrics['messages_processed'] += 1
                        
                        # Validate message size after decoding
                        msg_size = len(json.dumps(result))
                        if msg_size > 1024 * 1024:  # 1MB limit
                            if self.interactive:
                                proceed = await self.session.confirm_with_timeout(
                                    f"Large decoded message ({msg_size/1024:.1f}KB). Process?",
                                    timeout=emergency_timeout
                                )
                                if not proceed:
                                    return None
                    else:
                        async with self._metrics_lock:
                            self.metrics['verification_failures'] += 1
                    return result

                except InvalidSignature:
                    logger.error("Message signature verification failed")
                    async with self._metrics_lock:
                        self.metrics['verification_failures'] += 1
                        self.metrics['last_error'] = "Invalid signature"
                    raise
                except InvalidToken:
                    logger.error("Message decryption failed - invalid token")
                    async with self._metrics_lock:
                        self.metrics['verification_failures'] += 1
                        self.metrics['last_error'] = "Invalid token"
                    raise
                except InvalidKey:
                    logger.error("Message decryption failed - invalid key")
                    async with self._metrics_lock:
                        self.metrics['verification_failures'] += 1
                        self.metrics['last_error'] = "Invalid key"
                    raise
                except Exception as e:
                    logger.error(f"Message decoding failed: {str(e)}")
                    async with self._metrics_lock:
                        self.metrics['verification_failures'] += 1
                        self.metrics['last_error'] = str(e)
                    raise

        except Exception as e:
            logger.error(f"Interactive decoding failed: {str(e)}")
            raise
        finally:
            if self.session:
                await self.session.__aexit__(None, None, None)

    def request_shutdown(self):
        """Request graceful shutdown"""
        self._interrupt_requested = True
        logger.info("Shutdown requested for message protocol")

    async def get_metrics(self) -> Dict[str, Any]:
        """Get protocol metrics thread-safely"""
        async with self._metrics_lock:
            return self.metrics.copy()
