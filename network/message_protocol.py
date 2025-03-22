import json
import logging
import asyncio
from typing import Dict, Any, Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ed25519
import base64
from core.interactive_utils import InteractiveSession, InteractiveConfig, InteractionLevel
from core.constants import INTERACTION_TIMEOUTS

logger = logging.getLogger(__name__)

class SecureMessageProtocol:
    def __init__(self, encryption_key: bytes, interactive: bool = True):
        self.fernet = Fernet(encryption_key)
        self.signing_key = ed25519.Ed25519PrivateKey.generate()
        self.verify_key = self.signing_key.public_key()
        self.interactive = interactive
        self.session = None
        self._interrupt_requested = False
        self.metrics = {
            'messages_processed': 0,
            'verification_failures': 0,
            'last_error': None
        }

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

    async def encode_message_interactive(self, message: Dict[str, Any]) -> Optional[Dict[str, bytes]]:
        """Interactive message encoding with safety checks"""
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
                self.metrics['messages_processed'] += 1
                return encoded

        except Exception as e:
            logger.error(f"Message encoding failed: {str(e)}")
            self.metrics['last_error'] = str(e)
            raise
        finally:
            if self.session:
                await self.session.__aexit__(None, None, None)

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

    async def decode_message_interactive(self, encoded_message: Dict[str, bytes]) -> Optional[Dict[str, Any]]:
        """Interactive message decoding with validation"""
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
                # Validate required fields
                required_fields = {'data', 'signature', 'public_key'}
                if not all(field in encoded_message for field in required_fields):
                    logger.error("Missing required message fields")
                    self.metrics['verification_failures'] += 1
                    return None

                try:
                    result = self.decode_message(encoded_message)
                    if result:
                        self.metrics['messages_processed'] += 1
                    else:
                        self.metrics['verification_failures'] += 1
                    return result

                except Exception as e:
                    logger.error(f"Message decoding failed: {str(e)}")
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

    def get_metrics(self) -> Dict[str, Any]:
        """Get protocol metrics"""
        return self.metrics.copy()
