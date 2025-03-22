import torch
import gc
import logging
import asyncio
from typing import Dict, List, Optional, NoReturn
from dataclasses import dataclass
from pathlib import Path
from ..core.interactive_utils import InteractiveSession, InteractionLevel, InteractiveConfig
from ..core.constants import INTERACTION_TIMEOUTS

logger = logging.getLogger(__name__)

@dataclass
class MemoryStatus:
    total: int
    used: int
    free: int
    cached_tensors: int

class MemoryManager:
    def __init__(self, max_cache_size: int = 1024, interactive: bool = True):
        self.max_cache_size = max_cache_size
        self.tensor_cache = {}
        self.interactive = interactive
        self.session = None
        self._interrupt_requested = False
        self._progress_path = Path("./progress/memory_cache.json")
        self._progress_path.parent.mkdir(parents=True, exist_ok=True)
        self._recovery_attempts = 0
        self._max_recovery_attempts = 3
        self._last_error_time = 0
        self._error_cooldown = 60  # 60 seconds between recovery attempts
        
    async def cache_tensor_interactive(self, key: str, tensor: torch.Tensor) -> bool:
        """Interactive tensor caching with validation and progress tracking"""
        if not self._validate_tensor(tensor):
            logger.error("Invalid tensor provided")
            return False

        try:
            if self.interactive:
                self.session = InteractiveSession(
                    level=InteractionLevel.NORMAL,
                    config=InteractiveConfig(
                        timeout=INTERACTION_TIMEOUTS["save"],
                        persistent_state=True,
                        safe_mode=True,
                        recovery_enabled=True,
                        max_cleanup_wait=30
                    )
                )

                async with self.session:
                    if not await self._check_system_resources():
                        return False

                    try:
                        return await self._perform_caching(key, tensor)
                    except Exception as e:
                        return await self._handle_caching_error(e, key, tensor)
            else:
                return self._safe_cache_tensor(key, tensor)

        except Exception as e:
            return await self._handle_fatal_error(e)

    async def _check_system_resources(self) -> bool:
        """Validate system resources before caching"""
        status = self.get_memory_status()
        
        if status.used / status.total > 0.9:
            if not await self.session.get_confirmation(
                "WARNING: High memory usage (>90%). Continue caching? (y/n): "
            ):
                return False
                
            await self._force_cleanup()
            
        if psutil.virtual_memory().percent > 95:
            logger.error("System memory critically low")
            return False
            
        return True

    async def _perform_caching(self, key: str, tensor: torch.Tensor) -> bool:
        """Perform the actual caching operation with safety checks"""
        cache_size = self._get_cache_size()
        tensor_size = tensor.element_size() * tensor.nelement()
        
        if cache_size + tensor_size > self.max_cache_size:
            if await self.session.get_confirmation(
                f"Cache limit exceeded ({cache_size}/{self.max_cache_size} bytes). Evict old items? (y/n): "
            ):
                await self._evict_cache_interactive()
            else:
                return False
                
        try:
            self.tensor_cache[key] = tensor.detach().clone()
            await self.session._save_progress()
            return True
        except RuntimeError as e:
            if "out of memory" in str(e):
                await self._handle_oom_error()
            raise

    def _validate_tensor(self, tensor: torch.Tensor) -> bool:
        """Validate tensor before caching"""
        if not isinstance(tensor, torch.Tensor):
            return False
        if tensor.nelement() == 0:
            return False
        if tensor.isnan().any():
            return False
        return True

    async def _handle_caching_error(self, error: Exception, key: str, tensor: torch.Tensor) -> bool:
        """Handle recoverable caching errors"""
        logger.error(f"Caching error for key {key}: {str(error)}")
        
        if self._recovery_attempts >= self._max_recovery_attempts:
            logger.error("Max recovery attempts reached")
            return False
            
        self._recovery_attempts += 1
        
        try:
            await self._force_cleanup()
            return await self._perform_caching(key, tensor)
        finally:
            self._recovery_attempts = 0

    async def _handle_fatal_error(self, error: Exception) -> bool:
        """Handle unrecoverable errors"""
        logger.error(f"Fatal error in memory management: {str(error)}")
        await self._emergency_cleanup()
        return False

    async def _emergency_cleanup(self) -> None:
        """Emergency cleanup procedure"""
        try:
            # Clear all cached tensors
            self.tensor_cache.clear()
            
            # Force garbage collection
            gc.collect()
            
            # Clear CUDA cache if available
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
                
            # Save final state
            if self.session:
                await self.session._save_progress()
        except Exception as e:
            logger.critical(f"Emergency cleanup failed: {str(e)}")

    def cache_tensor(self, key: str, tensor: torch.Tensor) -> None:
        """Original synchronous caching method"""
        if self._get_cache_size() + tensor.element_size() * tensor.nelement() > self.max_cache_size:
            self._evict_cache()
        self.tensor_cache[key] = tensor
            
    async def _evict_cache_interactive(self) -> None:
        """Interactive cache eviction with progress tracking"""
        if not self.interactive or not self.session:
            self._evict_cache()
            return
            
        try:
            print("\nEvicting cache items:")
            for key in list(self.tensor_cache.keys()):
                if await self.session.get_confirmation(f"Evict {key}? (y/n): "):
                    del self.tensor_cache[key]
                    torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Cache eviction error: {str(e)}")
            self._evict_cache()  # Fallback to non-interactive eviction

    def _evict_cache(self) -> None:
        """Original cache eviction method"""
        while self.tensor_cache and self._get_cache_size() > self.max_cache_size:
            key = next(iter(self.tensor_cache))
            del self.tensor_cache[key]
        torch.cuda.empty_cache()

    async def _force_cleanup(self) -> None:
        """Force cleanup of GPU memory"""
        for key in list(self.tensor_cache.keys()):
            del self.tensor_cache[key]
        gc.collect()
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

    async def _handle_cache_error(self, error: Exception) -> None:
        """Handle caching errors with recovery attempts"""
        if self.session:
            logger.error(f"Cache error: {str(error)}")
            if await self.session.get_confirmation("Attempt recovery? (y/n): "):
                await self._force_cleanup()
                
    def _get_cache_size(self) -> int:
        """Calculate total size of cached tensors"""
        return sum(t.element_size() * t.nelement() for t in self.tensor_cache.values())

    def get_memory_status(self) -> MemoryStatus:
        torch.cuda.empty_cache()
        return MemoryStatus(
            total=torch.cuda.get_device_properties(0).total_memory,
            used=torch.cuda.memory_allocated(),
            free=torch.cuda.memory_reserved(),
            cached_tensors=len(self.tensor_cache)
        )
