import asyncio
import logging
from typing import Optional, Dict
from pathlib import Path
import psutil
from tqdm import tqdm

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.exceptions import InvalidKey
import base64

from core.constants import INTERACTION_TIMEOUTS, InteractionLevel
from core.interactive_utils import InteractiveSession, InteractiveConfig

logger = logging.getLogger(__name__)

class Encryption:
    def __init__(self, key: bytes = None, interactive: bool = True):
        self.interactive = interactive
        self.session = None
        self._interrupt_requested = False
        self.stats: Dict[str, int] = {
            'encryptions': 0,
            'decryptions': 0,
            'failures': 0
        }
        self._setup_key(key)

    def _setup_key(self, key: bytes = None) -> None:
        """Safely initialize encryption key"""
        try:
            self.key = key or Fernet.generate_key()
            self.cipher_suite = Fernet(self.key)
        except Exception as e:
            logger.error(f"Key setup failed: {str(e)}")
            raise

    async def encrypt_interactive(self, data: bytes, chunk_size: int = 1024*1024) -> Optional[bytes]:
        """Interactive encryption with progress tracking and monitoring"""
        if not data:
            logger.error("No data provided for encryption")
            return None

        try:
            if self.interactive:
                self.session = InteractiveSession(
                    level=InteractionLevel.NORMAL,
                    config=InteractiveConfig(
                        timeout=INTERACTION_TIMEOUTS["encryption"],
                        persistent_state=True,
                        safe_mode=True
                    )
                )

            async with self.session:
                # Monitor memory usage
                if not await self._check_resources():
                    return None

                total_chunks = (len(data) + chunk_size - 1) // chunk_size
                encrypted_chunks = []

                with tqdm(total=total_chunks, desc="Encrypting Data") as pbar:
                    for i in range(0, len(data), chunk_size):
                        if self._interrupt_requested:
                            logger.info("Encryption interrupted by user")
                            break

                        chunk = data[i:i + chunk_size]
                        try:
                            encrypted_chunk = self.cipher_suite.encrypt(chunk)
                            encrypted_chunks.append(encrypted_chunk)
                            pbar.update(1)
                        except Exception as e:
                            logger.error(f"Chunk encryption failed: {str(e)}")
                            self.stats['failures'] += 1
                            raise

                if encrypted_chunks:
                    self.stats['encryptions'] += 1
                    return b''.join(encrypted_chunks)
                return None

        except Exception as e:
            logger.error(f"Interactive encryption failed: {str(e)}")
            self.stats['failures'] += 1
            raise
        finally:
            if self.session:
                await self.session.__aexit__(None, None, None)

    async def decrypt_interactive(self, encrypted_data: bytes, chunk_size: int = 1024*1024) -> Optional[bytes]:
        """Interactive decryption with progress tracking and monitoring"""
        if not encrypted_data:
            logger.error("No data provided for decryption")
            return None

        try:
            if self.interactive:
                self.session = InteractiveSession(
                    level=InteractionLevel.NORMAL,
                    config=InteractiveConfig(
                        timeout=INTERACTION_TIMEOUTS["decryption"],
                        persistent_state=True,
                        safe_mode=True
                    )
                )

            async with self.session:
                # Monitor resources
                if not await self._check_resources():
                    return None

                total_chunks = (len(encrypted_data) + chunk_size - 1) // chunk_size
                decrypted_chunks = []

                with tqdm(total=total_chunks, desc="Decrypting Data") as pbar:
                    for i in range(0, len(encrypted_data), chunk_size):
                        if self._interrupt_requested:
                            logger.info("Decryption interrupted by user")
                            break

                        chunk = encrypted_data[i:i + chunk_size]
                        try:
                            decrypted_chunk = self.cipher_suite.decrypt(chunk)
                            decrypted_chunks.append(decrypted_chunk)
                            pbar.update(1)
                        except InvalidToken:
                            logger.error("Invalid token - data may be corrupted")
                            self.stats['failures'] += 1
                            raise
                        except Exception as e:
                            logger.error(f"Chunk decryption failed: {str(e)}")
                            self.stats['failures'] += 1
                            raise

                if decrypted_chunks:
                    self.stats['decryptions'] += 1
                    return b''.join(decrypted_chunks)
                return None

        except Exception as e:
            logger.error(f"Interactive decryption failed: {str(e)}")
            self.stats['failures'] += 1
            raise
        finally:
            if self.session:
                await self.session.__aexit__(None, None, None)

    async def _check_resources(self) -> bool:
        """Check system resources before processing"""
        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            if memory_usage > 1024:  # >1GB
                if self.interactive and self.session:
                    proceed = await self.session.confirm_with_timeout(
                        "\nHigh memory usage detected. Continue?",
                        timeout=INTERACTION_TIMEOUTS["emergency"]
                    )
                    if not proceed:
                        return False
                logger.warning(f"High memory usage: {memory_usage:.1f}MB")
            return True
        except Exception as e:
            logger.error(f"Resource check failed: {str(e)}")
            return False

    def request_shutdown(self):
        """Request graceful shutdown"""
        self._interrupt_requested = True
        logger.info("Shutdown requested for encryption module")

    # Legacy synchronous methods for backwards compatibility
    def encrypt(self, data: bytes) -> bytes:
        """Legacy synchronous encryption"""
        if not data:
            raise ValueError("No data provided for encryption")
        return self.cipher_suite.encrypt(data)

    def decrypt(self, encrypted_data: bytes) -> bytes:
        """Legacy synchronous decryption"""
        if not encrypted_data:
            raise ValueError("No data provided for decryption")
        return self.cipher_suite.decrypt(encrypted_data)
