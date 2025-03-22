from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding, ec
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from typing import Dict, Optional, Tuple
import time
import logging
import asyncio
from tqdm import tqdm
from core.constants import INTERACTION_TIMEOUTS, InteractionLevel
from core.interactive_utils import InteractiveSession, InteractiveConfig

logger = logging.getLogger(__name__)

class AuthManager:
    def __init__(self, interactive: bool = True):
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
        self.interactive = interactive
        self.session: Optional[InteractiveSession] = None
        self._interrupt_requested = False
        self._stats = {
            'sessions_created': 0,
            'verifications': 0,
            'key_rotations': 0
        }

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

    async def create_session_interactive(self, peer_id: str) -> Optional[Tuple[bytes, float]]:
        """Interactive session creation with monitoring"""
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
                # Validate peer ID
                if not self._validate_peer_id(peer_id):
                    logger.error(f"Invalid peer ID: {peer_id}")
                    return None

                # Monitor active sessions
                if len(self.sessions) > 1000:  # Arbitrary limit
                    if self.interactive:
                        proceed = await self.session.confirm_with_timeout(
                            "\nHigh number of active sessions. Continue?",
                            timeout=INTERACTION_TIMEOUTS["emergency"]
                        )
                        if not proceed:
                            return None

                session_key, expiry = self.create_session(peer_id)
                self._stats['sessions_created'] += 1

                # Cleanup expired sessions periodically
                await self._cleanup_expired()

                return session_key, expiry

        except Exception as e:
            logger.error(f"Failed to create session: {str(e)}")
            return None
        finally:
            if self.session:
                await self.session.__aexit__(None, None, None)

    def _rotate_keys(self):
        """Rotate asymmetric keys periodically"""
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.public_key = self.private_key.public_key()
        self.last_rotation = time.time()

    async def verify_message_interactive(self, peer_id: str, message: bytes, 
                                      signature: bytes) -> bool:
        """Interactive message verification with safety controls"""
        try:
            if self.interactive:
                self.session = InteractiveSession(
                    level=InteractionLevel.NORMAL,
                    config=InteractiveConfig(
                        timeout=INTERACTION_TIMEOUTS["verify"],
                        persistent_state=True,
                        safe_mode=True
                    )
                )

            async with self.session:
                # Check session validity
                if peer_id not in self.sessions:
                    logger.warning(f"No active session for peer {peer_id}")
                    return False

                if time.time() > self.session_expiry[peer_id]:
                    logger.warning(f"Expired session for peer {peer_id}")
                    return False

                result = self.verify_message(peer_id, message, signature)
                self._stats['verifications'] += 1
                return result

        except Exception as e:
            logger.error(f"Message verification failed: {str(e)}")
            return False
        finally:
            if self.session:
                await self.session.__aexit__(None, None, None)

    async def _cleanup_expired(self):
        """Clean up expired sessions"""
        current_time = time.time()
        expired = [
            peer_id for peer_id, expiry in self.session_expiry.items()
            if current_time > expiry
        ]
        
        if expired:
            for peer_id in expired:
                del self.sessions[peer_id]
                del self.session_expiry[peer_id]
            logger.info(f"Cleaned up {len(expired)} expired sessions")

    async def rotate_keys_interactive(self) -> bool:
        """Interactive key rotation with validation"""
        try:
            if self.interactive:
                self.session = InteractiveSession(
                    level=InteractionLevel.NORMAL,
                    config=InteractiveConfig(
                        timeout=INTERACTION_TIMEOUTS["rotation"],
                        persistent_state=True,
                        safe_mode=True
                    )
                )

            async with self.session:
                if self.interactive:
                    proceed = await self.session.confirm_with_timeout(
                        "\nRotate authentication keys?",
                        timeout=INTERACTION_TIMEOUTS["confirmation"]
                    )
                    if not proceed:
                        return False

                self._rotate_keys()
                self._stats['key_rotations'] += 1
                return True

        except Exception as e:
            logger.error(f"Key rotation failed: {str(e)}")
            return False
        finally:
            if self.session:
                await self.session.__aexit__(None, None, None)

    def _validate_peer_id(self, peer_id: str) -> bool:
        """Validate peer ID format"""
        return bool(peer_id and isinstance(peer_id, str) and len(peer_id) <= 128)

    def request_shutdown(self):
        """Request graceful shutdown"""
        self._interrupt_requested = True
        logger.info("Shutdown requested for auth manager")

    def get_stats(self) -> Dict:
        """Get authentication statistics"""
        return self._stats.copy()
