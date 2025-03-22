import jwt
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
from core.constants import INTERACTION_TIMEOUTS, InteractionLevel
from core.interactive_utils import InteractiveSession, InteractiveConfig

logger = logging.getLogger(__name__)

class NodeAuthenticator:
    def __init__(self, secret_key: str, interactive: bool = True):
        self.secret_key = secret_key
        self.interactive = interactive
        self.session = None
        self._interrupt_requested = False
        self.stats = {'tokens_generated': 0, 'verifications': 0, 'failures': 0}
        
    async def generate_token_interactive(self, node_id: str) -> Optional[str]:
        """Interactive token generation with validation and monitoring"""
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
                if not self._validate_node_id(node_id):
                    logger.error(f"Invalid node ID: {node_id}")
                    return None

                try:
                    token = self.generate_token(node_id)
                    self.stats['tokens_generated'] += 1
                    return token

                except Exception as e:
                    logger.error(f"Token generation failed: {str(e)}")
                    self.stats['failures'] += 1
                    return None

        except Exception as e:
            logger.error(f"Interactive token generation failed: {str(e)}")
            raise
        finally:
            if self.session:
                await self.session.__aexit__(None, None, None)

    async def verify_token_interactive(self, token: str) -> Optional[Dict]:
        """Interactive token verification with timeouts and monitoring"""
        try:
            if self.interactive:
                self.session = InteractiveSession(
                    level=InteractionLevel.NORMAL,
                    config=InteractiveConfig(
                        timeout=INTERACTION_TIMEOUTS["verification"],
                        persistent_state=True,
                        safe_mode=True
                    )
                )

            async with self.session:
                try:
                    result = self.verify_token(token)
                    self.stats['verifications'] += 1
                    if not result:
                        self.stats['failures'] += 1
                    return result

                except Exception as e:
                    logger.error(f"Token verification failed: {str(e)}")
                    self.stats['failures'] += 1
                    return None

        except Exception as e:
            logger.error(f"Interactive token verification failed: {str(e)}")
            raise
        finally:
            if self.session:
                await self.session.__aexit__(None, None, None)

    def generate_token(self, node_id: str) -> str:
        payload = {
            'node_id': node_id,
            'exp': datetime.utcnow() + timedelta(days=1)
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
        
    def verify_token(self, token: str) -> Optional[Dict]:
        try:
            return jwt.decode(token, self.secret_key, algorithms=['HS256'])
        except jwt.InvalidTokenError:
            return None

    def _validate_node_id(self, node_id: str) -> bool:
        """Validate node ID format"""
        if not node_id or not isinstance(node_id, str):
            return False
        # Add additional validation as needed
        return True

    def request_shutdown(self):
        """Request graceful shutdown"""
        self._interrupt_requested = True
        logger.info("Shutdown requested for NodeAuthenticator")

    async def _cleanup(self):
        """Cleanup resources"""
        try:
            # Add cleanup logic here
            logger.info("NodeAuthenticator cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
