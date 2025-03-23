import asyncio
import aiohttp
import logging
from typing import Dict, Any, Optional
from core.constants import INTERACTION_TIMEOUTS, InteractionLevel
from core.interactive_utils import InteractiveSession, InteractiveConfig
from .rate_limiter import AdaptiveRateLimiter

logger = logging.getLogger(__name__)

class NodeCommunication:
    def __init__(self, node_id: str, interactive: bool = True):
        self.node_id = node_id
        self.message_queue = asyncio.Queue()
        self.interactive = interactive
        self.session = None
        self._interrupt_requested = False
        self._retry_count = 3
        self._interactive_timeout = INTERACTION_TIMEOUTS["batch"]
        self._non_interactive_timeout = INTERACTION_TIMEOUTS["default"]
        # Add session pool
        self._session_pool: Dict[str, aiohttp.ClientSession] = {}
        self._pool_lock = asyncio.Lock()
        self._max_pool_size = 10
        self._session_ttl = 300  # 5 minutes
        
        # Add rate limiting
        self._rate_limiter = AdaptiveRateLimiter(
            initial_rate=1000.0,  # 1KB/s initial rate
            window_size=60,       # 1 minute window
            interactive=interactive
        )
        self._pending_messages: Dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)
        self._worker_tasks: Set[asyncio.Task] = set()

    async def _get_client_session(self, target_node: str) -> aiohttp.ClientSession:
        """Get or create a ClientSession from the pool"""
        async with self._pool_lock:
            now = asyncio.get_event_loop().time()
            
            # Cleanup expired sessions
            expired = [node for node, (session, timestamp) in 
                      self._session_pool.items() if now - timestamp > self._session_ttl]
            for node in expired:
                session, _ = self._session_pool.pop(node)
                await session.close()

            # Return existing valid session
            if target_node in self._session_pool:
                session, timestamp = self._session_pool[target_node]
                if now - timestamp <= self._session_ttl:
                    self._session_pool[target_node] = (session, now)  # Update timestamp
                    return session

            # Create new session if pool not full
            if len(self._session_pool) >= self._max_pool_size:
                # Remove oldest session
                oldest_node = min(self._session_pool.keys(), 
                                key=lambda k: self._session_pool[k][1])
                old_session, _ = self._session_pool.pop(oldest_node)
                await old_session.close()

            session = aiohttp.ClientSession()
            self._session_pool[target_node] = (session, now)
            return session

    async def _message_worker(self, target_node: str):
        """Background worker to process messages for a target node"""
        try:
            while not self._interrupt_requested:
                message = await self._pending_messages[target_node].get()
                size = len(str(message))  # Approximate message size
                
                while not await self._rate_limiter.can_send_interactive(target_node, size):
                    if self._interrupt_requested:
                        return
                    await asyncio.sleep(1)
                
                await self._send_message_internal(target_node, message)
                self._pending_messages[target_node].task_done()
        except Exception as e:
            logger.error(f"Message worker error for {target_node}: {str(e)}")
        finally:
            self._worker_tasks.discard(asyncio.current_task())

    async def send_message_interactive(self, target_node: str, message: Dict[str, Any]) -> bool:
        """Queue message for sending with rate limiting"""
        try:
            # Ensure worker exists for target node
            if target_node not in self._worker_tasks:
                task = asyncio.create_task(self._message_worker(target_node))
                self._worker_tasks.add(task)
            
            # Queue message
            await self._pending_messages[target_node].put(message)
            return True

        except Exception as e:
            logger.error(f"Error queueing message: {str(e)}")
            return False

    async def _send_message_internal(self, target_node: str, message: Dict[str, Any]) -> bool:
        """Internal message sending implementation"""
        try:
            session_timeout = self._interactive_timeout if self.interactive else self._non_interactive_timeout
            
            if self.interactive:
                self.session = InteractiveSession(
                    level=InteractionLevel.NORMAL,
                    config=InteractiveConfig(
                        timeout=session_timeout,
                        persistent_state=True,
                        safe_mode=True
                    )
                )

            async with self.session:
                client_session = await self._get_client_session(target_node)
                
                for attempt in range(self._retry_count):
                    if self._interrupt_requested:
                        logger.info("Message sending interrupted")
                        return False

                    try:
                        async with client_session.post(
                            f"http://{target_node}/message",
                            json=message,
                            timeout=session_timeout
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
            logger.error(f"Send error: {str(e)}")
            return False
        finally:
            if self.session:
                await self.session.__aexit__(None, None, None)

    async def send_message(self, target_node: str, message: Dict[str, Any]) -> bool:
        """Legacy method, redirects to interactive version"""
        return await self.send_message_interactive(target_node, message)

    async def adjust_rate(self, target_node: str, congestion_level: float):
        """Adjust send rate based on observed congestion"""
        await self._rate_limiter.adjust_rate_interactive(congestion_level)

    def request_shutdown(self):
        """Request graceful shutdown of communication"""
        self._interrupt_requested = True

    async def cleanup(self):
        """Cleanup resources"""
        try:
            # Cancel all worker tasks
            for task in self._worker_tasks:
                task.cancel()
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
            
            # Cleanup rate limiter
            await self._rate_limiter.cleanup()
            
            # Cleanup session pool
            async with self._pool_lock:
                for session, _ in self._session_pool.values():
                    await session.close()
                self._session_pool.clear()
            
            while not self.message_queue.empty():
                await self.message_queue.get()
            logger.info("Communication cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
