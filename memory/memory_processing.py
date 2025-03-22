import torch
from typing import Dict, List, Optional, Tuple, NoReturn
from .memory_manager import MemoryManager
from core.interactive_utils import InteractiveSession, InteractionLevel, InteractionTimeout
from core.constants import INTERACTION_TIMEOUTS
import logging
import psutil
import asyncio
from pathlib import Path
import time

logger = logging.getLogger(__name__)

class MemoryProcessor:
    def __init__(self, memory_manager: MemoryManager, interactive: bool = True):
        self.memory_manager = memory_manager
        self.processing_queue = []
        self.interactive = interactive
        self.session = None
        self._interrupt_requested = False
        self._progress_path = Path("./progress/memory_processing.json")
        self._progress_path.parent.mkdir(parents=True, exist_ok=True)
        self._recovery_attempts = 0
        self._max_recovery_attempts = 3
        self._last_error_time = time.time()
        self._error_cooldown = 60  # 60 seconds between recovery attempts
        self._batch_size_limit = 1024 * 1024 * 1024  # 1GB

    async def process_batch_interactive(self, data: torch.Tensor) -> Optional[torch.Tensor]:
        """Process batch with interactive controls and progress tracking"""
        if not self._validate_input_tensor(data):
            logger.error("Invalid input tensor")
            return None

        try:
            self.session = InteractiveSession(
                level=InteractionLevel.NORMAL if self.interactive else InteractionLevel.NONE,
                config=InteractiveConfig(
                    timeout=INTERACTION_TIMEOUTS["batch"],
                    persistent_state=True,
                    safe_mode=True,
                    recovery_enabled=True,
                    max_cleanup_wait=30
                )
            )

            async with self.session:
                if not await self._check_resources():
                    return None

                try:
                    return await self._process_with_monitoring(data)
                except Exception as e:
                    return await self._handle_processing_error(e, data)

        except Exception as e:
            return await self._handle_fatal_error(e)

    def _validate_input_tensor(self, data: torch.Tensor) -> bool:
        """Validate input tensor before processing"""
        if not isinstance(data, torch.Tensor):
            return False
            
        # Check tensor size
        tensor_size = data.element_size() * data.nelement()
        if tensor_size > self._batch_size_limit:
            logger.error(f"Tensor size ({tensor_size} bytes) exceeds limit ({self._batch_size_limit} bytes)")
            return False
            
        # Basic sanity checks
        if data.nelement() == 0:
            return False
            
        return True

    async def _check_resources(self) -> bool:
        """Check system resources before processing"""
        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            if memory_usage > 0.9 * psutil.virtual_memory().total / 1024 / 1024:
                logger.error("Insufficient memory available")
                return False

            if hasattr(torch.cuda, 'is_available') and torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
                if gpu_memory > 0.9 * torch.cuda.get_device_properties(0).total_memory / 1024 / 1024:
                    logger.error("Insufficient GPU memory")
                    return False

            return True
        except Exception as e:
            logger.error(f"Resource check failed: {str(e)}")
            return False

    async def _process_with_monitoring(self, data: torch.Tensor) -> Optional[torch.Tensor]:
        """Process data with resource monitoring"""
        try:
            # Start monitoring task
            monitor_task = asyncio.create_task(self._monitor_resources())
            
            processed = await self._preprocess_with_validation(data)
            if processed is not None:
                await self._cache_result_safely(processed)
                
            return processed
        finally:
            # Cancel monitoring
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

    async def _cache_result_safely(self, tensor: torch.Tensor) -> None:
        """Safely cache processed tensor"""
        try:
            await self.memory_manager.cache_tensor_interactive('processed_batch', tensor)
        except Exception as e:
            logger.error(f"Failed to cache result: {str(e)}")
            self.memory_manager.clear_cache('processed_batch')

    async def _monitor_resources(self) -> None:
        """Monitor system resources during processing"""
        while True:
            try:
                await asyncio.sleep(1)
                if psutil.virtual_memory().percent > 95:
                    logger.warning("Critical memory usage detected")
                    self._interrupt_requested = True
            except asyncio.CancelledError:
                break

    async def _preprocess_with_validation(self, data: torch.Tensor) -> Optional[torch.Tensor]:
        """Preprocess with validation and progress tracking"""
        if not isinstance(data, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor")

        if data.isnan().any():
            logger.warning("Input contains NaN values")
            if self.interactive:
                if not await self.session.get_confirmation("Continue with NaN values?"):
                    return None

        try:
            processed = data.float().div(255)
            
            # Validate output
            if processed.isnan().any():
                raise ValueError("Processing resulted in NaN values")
                
            return processed

        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            if self.interactive:
                if await self.session.get_confirmation("Retry preprocessing?"):
                    return await self._preprocess_with_validation(data)
            raise

    async def _cleanup_timeout(self):
        """Handle cleanup after timeout"""
        if self.session:
            await self.session._save_progress()
        self.memory_manager.clear_cache('processed_batch')

    async def _handle_error(self, error: Exception):
        """Handle processing errors"""
        if self.session:
            await self.session._save_progress()
            
        if isinstance(error, RuntimeError) and "out of memory" in str(error):
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

    def process_batch(self, data: torch.Tensor) -> Optional[torch.Tensor]:
        """Synchronous wrapper for backwards compatibility"""
        return asyncio.run(self.process_batch_interactive(data))

    def _preprocess(self, data: torch.Tensor) -> torch.Tensor:
        # Add preprocessing logic
        return data.float().div(255)
