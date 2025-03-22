import asyncio
import aiohttp
import logging
from typing import Dict, Any, Optional
from core.constants import INTERACTION_TIMEOUTS, InteractionLevel
from core.interactive_utils import InteractiveSession, InteractiveConfig

logger = logging.getLogger(__name__)

class NodeCommunication:
    def __init__(self, node_id: str, interactive: bool = True):
        self.node_id = node_id
        self.message_queue = asyncio.Queue()
        self.interactive = interactive
        self.session = None
        self._interrupt_requested = False
        self._retry_count = 3
        self._timeout = INTERACTION_TIMEOUTS["default"]

    async def send_message_interactive(self, target_node: str, message: Dict[str, Any]) -> bool:
        """Send message with interactive monitoring and error handling"""
        try:
            if self.interactive:
                self.session = InteractiveSession(
                    level=InteractionLevel.NORMAL,
                    config=InteractiveConfig(
                        timeout=self._timeout,
                        persistent_state=True,
                        safe_mode=True
                    )
                )

            async with self.session:
                for attempt in range(self._retry_count):
                    if self._interrupt_requested:
                        logger.info("Message sending interrupted")
                        return False

                    try:
                        async with aiohttp.ClientSession() as http_session:
                            async with http_session.post(
                                f"http://{target_node}/message",
                                json=message,
                                timeout=self._timeout
                            ) as response:
                                if response.status == 200:
                                    logger.debug(f"Message sent to {target_node} successfully")
                                    return True
                                
                                logger.warning(f"Failed to send message: {response.status}")
                                if self.interactive and attempt < self._retry_count - 1:
                                    retry = await self.session.get_confirmation(
                                        f"Retry sending message? (Attempt {attempt + 1}/{self._retry_count})",
                                        timeout=INTERACTION_TIMEOUTS["emergency"]
                                    )
                                    if not retry:
                                        return False
                                    continue

                    except asyncio.TimeoutError:
                        logger.error(f"Timeout sending message to {target_node}")
                        if not self.interactive or attempt == self._retry_count - 1:
                            return False
                    except Exception as e:
                        logger.error(f"Error sending message: {str(e)}")
                        if not self.interactive or attempt == self._retry_count - 1:
                            return False

                return False

        except Exception as e:
            logger.error(f"Fatal error in send_message_interactive: {str(e)}")
            return False
        finally:
            if self.session:
                await self.session.__aexit__(None, None, None)

    async def send_message(self, target_node: str, message: Dict[str, Any]) -> bool:
        """Legacy method, redirects to interactive version"""
        return await self.send_message_interactive(target_node, message)

    def request_shutdown(self):
        """Request graceful shutdown of communication"""
        self._interrupt_requested = True

    async def cleanup(self):
        """Cleanup resources"""
        try:
            while not self.message_queue.empty():
                await self.message_queue.get()
            logger.info("Communication cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
