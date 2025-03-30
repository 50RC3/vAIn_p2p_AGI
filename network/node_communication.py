import asyncio
import aiohttp
import logging
import time
import json
import base64
import torch
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass
from collections import defaultdict
from core.constants import INTERACTION_TIMEOUTS, InteractionLevel
from core.interactive_utils import InteractiveSession, InteractiveConfig
from .rate_limiter import AdaptiveRateLimiter
from .gossip_protocol import GossipManager
from training.compression import AdaptiveCompression

logger = logging.getLogger(__name__)

class NodeCommunicationError(Exception):
    """Base exception for node communication errors"""
    pass

class NodeCommunication:
    def __init__(self, node_id: str, interactive: bool = True):
        self.node_id = node_id
        self.interactive = interactive
        self.session: Optional[InteractiveSession] = None
        self.message_queue = asyncio.Queue()
        self._interrupt_requested = False
        self._retry_count = 3
        self._interactive_timeout = INTERACTION_TIMEOUTS["batch"]
        self._non_interactive_timeout = INTERACTION_TIMEOUTS["default"]
        self._session_pool: Dict[str, aiohttp.ClientSession] = {}
        self._pool_lock = asyncio.Lock()
        self._max_pool_size = 10
        self._session_ttl = 300  # 5 minutes
        
        self._rate_limiter = AdaptiveRateLimiter(
            initial_rate=1000.0,  # 1KB/s initial rate
            window_size=60,       # 1 minute window
            interactive=interactive
        )
        self._pending_messages: Dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)
        self._worker_tasks: Set[asyncio.Task] = set()
        
        self.compressor = AdaptiveCompression(
            base_compression_rate=0.1,  # Start with 10x compression
            min_rate=0.01,
            max_rate=0.3
        )
        self._compression_stats = {
            'total_original': 0,
            'total_compressed': 0,
            'compression_ratio': 1.0
        }
        self._compression_stats.update({
            'feature_reduction': 0.0,
            'bandwidth_saved': 0.0
        })
        self.msg_queue = AsyncMessageQueue()
        self.gossip = GossipManager()

    async def _get_client_session(self, target_node: str) -> aiohttp.ClientSession:
        """Get or create a ClientSession from the pool"""
        async with self._pool_lock:
            now = asyncio.get_event_loop().time()
            
            expired = [node for node, (session, timestamp) in 
                      self._session_pool.items() if now - timestamp > self._session_ttl]
            for node in expired:
                session, _ = self._session_pool.pop(node)
                await session.close()

            if target_node in self._session_pool:
                session, timestamp = self._session_pool[target_node]
                if now - timestamp <= self._session_ttl:
                    self._session_pool[target_node] = (session, now)
                    return session

            if len(self._session_pool) >= self._max_pool_size:
                oldest_node = min(self._session_pool.keys(), 
                                key=lambda k: self._session_pool[k][1])
                old_session, _ = self._session_pool.pop(oldest_node)
                await old_session.close()

            session = aiohttp.ClientSession()
            self._session_pool[target_node] = (session, now)
            return session

    async def _message_worker(self, target_node: str):
        """Background worker with gossip protocol"""
        try:
            while not self._interrupt_requested:
                message = await self.msg_queue.get()
                if message and await self.gossip.should_propagate(message['id']):
                    peers = self.gossip.select_peers(self._get_peers(), {target_node})
                    for peer in peers:
                        await self._send_message_internal(peer, message)
                await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"Worker error: {str(e)}")
            
    async def send_message_interactive(self, target_node: str, message: Dict[str, Any]) -> bool:
        """Queue message for asynchronous sending"""
        try:
            msg_id = f"{self.node_id}_{time.time()}_{target_node}"
            await self.msg_queue.put(msg_id, message)
            
            peers = self.gossip.select_peers(self._get_peers(), {target_node})
            
            propagation_tasks = [
                asyncio.create_task(self._send_message_internal(peer, message))
                for peer in peers
            ]
            
            asyncio.gather(*propagation_tasks, return_exceptions=True)
            return True
            
        except Exception as e:
            logger.error(f"Failed to queue message: {str(e)}")
            return False

    async def _compress_message(self, message: Dict[str, Any]) -> tuple[Dict, float]:
        """Compress message using edge computing when available"""
        try:
            if self.edge_service:
                compressed = await self.edge_service.offload_task(
                    "compression",
                    message
                )
                if compressed:
                    return compressed, compressed.get('ratio', 1.0)
            
            msg_tensor = self._dict_to_tensor(message)
            original_size = len(str(message))
            
            compressed, ratio = await self.compressor.compress_model_updates({
                'message': msg_tensor
            })
            
            compressed_size = len(str(compressed))
            
            self._compression_stats.update({
                'total_original': self._compression_stats['total_original'] + original_size,
                'total_compressed': self._compression_stats['total_compressed'] + compressed_size,
                'compression_ratio': ratio,
                'feature_reduction': 1.0 - (len(compressed) / len(message)),
                'bandwidth_saved': (original_size - compressed_size) / original_size
            })
            
            return compressed, ratio

        except Exception as e:
            logger.warning(f"Compression failed, sending uncompressed: {e}")
            return message, 1.0

    def _dict_to_tensor(self, d: Dict) -> torch.Tensor:
        """Convert dictionary to tensor for transmission"""
        base_size = len(str(d).encode())
        return base_size

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

    async def send_message_mobile(self, target_node: str, message: Dict[str, Any]) -> bool:
        """Send optimized message to mobile node"""
        try:
            if not hasattr(self, 'mobile_optimizer'):
                self.mobile_optimizer = MobileOptimizer()

            compressed_msg = self.mobile_optimizer.compress_for_mobile(message)
            
            compressed_msg['_mobile'] = True
            compressed_msg['_timestamp'] = time.time()
            
            metrics = {
                'original_size': len(str(message)),
                'compressed_size': len(str(compressed_msg)),
                'target_node': target_node
            }
            
            if metrics['compressed_size'] < metrics['original_size'] * 0.8:
                success = await self.send_message(target_node, compressed_msg)
                if success:
                    self.mobile_optimizer.update_history.append(metrics)
                return success
            
            return False

        except Exception as e:
            logger.error(f"Failed to send mobile message: {str(e)}")
            return False

    async def adjust_rate(self, target_node: str, congestion_level: float):
        """Adjust send rate based on observed congestion"""
        await self._rate_limiter.adjust_rate_interactive(congestion_level)

    def request_shutdown(self):
        """Request graceful shutdown of communication"""
        self._interrupt_requested = True

    async def cleanup(self):
        """Enhanced cleanup with message queue"""
        await self.msg_queue.cleanup_old_messages()
        try:
            for task in self._worker_tasks:
                task.cancel()
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
            
            await self._rate_limiter.cleanup()
            
            async with self._pool_lock:
                for session, _ in self._session_pool.values():
                    await session.close()
                self._session_pool.clear()
            
            while not self.message_queue.empty():
                await self.message_queue.get()
            logger.info("Communication cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    async def handle_mobile_connection(self, mobile_node: str):
        """Configure connection for mobile optimization"""
        self.compression.set_target_rate(0.1)
        self.batch_size = self._calculate_mobile_batch_size()

    def _calculate_mobile_batch_size(self) -> int:
        """Calculate optimal batch size for mobile connections"""
        base_size = 32
        if hasattr(self, 'network_monitor'):
            quality = self.network_monitor.get_quality_sync()
            return max(1, int(base_size * quality))
        return base_size
