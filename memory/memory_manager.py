import gc
import logging
import asyncio
import torch
from typing import Dict, Optional, Any
from pathlib import Path
import json
import time
import psutil
from dataclasses import dataclass

from core.interactive_utils import InteractiveSession, InteractiveConfig, InteractionLevel
from network.caching import CacheManager, CacheLevel, CachePolicy
from ai_core.model_storage import ModelStorage
from core.constants import INTERACTION_TIMEOUTS
from core.system_coordinator import get_coordinator

logger = logging.getLogger(__name__)

@dataclass
class MemoryStatus:
    total: int
    used: int
    free: int
    cached_tensors: int

class MemoryManager:
    def __init__(self, max_cache_size: int = 1024*1024*1024, interactive: bool = True):
        self.max_cache_size = max_cache_size
        self.tensor_cache: Dict[str, torch.Tensor] = {}
        self.interactive = interactive
        self.session: Optional[InteractiveSession] = None
        self._interrupt_requested = False
        self._progress_path = Path("./progress/memory_cache.json")
        self._progress_path.parent.mkdir(parents=True, exist_ok=True)
        self._recovery_attempts = 0
        self._max_recovery_attempts = 3
        self._last_error_time = 0
        self._error_cooldown = 60  # 60 seconds between recovery attempts
        self.model_storage = ModelStorage()
        self.storage_status: Dict[str, Any] = {}
        self.memory_systems: Dict[str, Any] = {}
        self.active_system = None
        self.cache_manager = CacheManager({
            CacheLevel.MEMORY: CachePolicy(
                max_size=max_cache_size,
                ttl=3600,
                level=CacheLevel.MEMORY
            )
        })
        self._cleanup_lock = asyncio.Lock()
        self._is_cleanup_pending = False
        
        # Get system coordinator for cross-module coordination
        try:
            self.system_coordinator = get_coordinator()
            if self.system_coordinator:
                # Register listeners for memory-related events
                self.system_coordinator.register_event_listener("high_memory", self._handle_high_memory_event)
                self.system_coordinator.register_event_listener("critical_memory", self._handle_critical_memory_event)
        except Exception as e:
            logger.warning(f"Failed to initialize system coordinator integration: {e}")
            self.system_coordinator = None
    
    async def register_memory_system(self, name: str, system: Any) -> bool:
        """Register a memory system with validation"""
        try:
            if not name or not system:
                logger.error("Invalid name or system provided")
                return False
                
            if name in self.memory_systems:
                logger.warning(f"Memory system {name} already registered")
                return True
                
            if hasattr(system, 'memory_size'):
                self.memory_systems[name] = system
                if not self.active_system:
                    self.active_system = name
                return True
                
            logger.error(f"Invalid memory system: {name}")
            return False
            
        except Exception as e:  
            logger.error(f"Failed to register memory system: {str(e)}")
            return False
        
    async def cache_tensor_interactive(self, key: str, tensor: torch.Tensor) -> bool:
        """Interactive tensor caching with validation and progress tracking"""
        if not self._validate_tensor(tensor):
            logger.error("Invalid tensor provided")
            return False

        try:
            metadata = {
                "shape": tensor.shape,
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "timestamp": time.time()
            }
            
            success = self.cache_manager.put(
                key,
                tensor.detach().clone(),
                metadata=metadata,
                level=CacheLevel.MEMORY
            )
            
            if success:
                model_hash, meta_hash = await self.model_storage.store_model_async(
                    tensor, metadata, interactive=self.interactive
                )
                
                self.storage_status[key] = {
                    "memory": True,
                    "ipfs": model_hash,
                    "metadata": meta_hash
                }
                
            return success

        except Exception as e:
            return await self._handle_caching_error(e, key, tensor)

    async def _check_system_resources(self) -> bool:
        """Validate system resources before caching"""
        status = self.get_memory_status()
        
        if status.used / status.total > 0.9:
            if self.session:
                if not await self.session.get_confirmation(
                    "WARNING: High memory usage (>90%). Continue caching? (y/n): "
                ):
                    return False
                
            await self._force_cleanup()
            
            # Notify system coordinator about high memory usage
            if self.system_coordinator:
                self.system_coordinator.dispatch_event("high_memory", {
                    "usage": status.used / status.total,
                    "source": "memory_manager"
                })
            
        if psutil.virtual_memory().percent > 95:
            logger.error("System memory critically low")
            
            # Notify system coordinator about critical memory
            if self.system_coordinator:
                self.system_coordinator.dispatch_event("critical_memory", {
                    "usage": psutil.virtual_memory().percent / 100,
                    "source": "memory_manager"
                })
                
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
        """Get current memory status"""
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
            
        total = 0
        used = 0
        free = 0
        
        try:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                total = torch.cuda.get_device_properties(device).total_memory
                used = torch.cuda.memory_allocated(device)
                free = torch.cuda.memory_reserved(device)
            else:
                # Fallback to system memory
                vm = psutil.virtual_memory()
                total = vm.total
                used = vm.used
                free = vm.free
        except Exception as e:
            logger.error(f"Error getting memory status: {e}")
            # Provide defaults
            total = 1
            used = 0
            free = 1
            
        return MemoryStatus(
            total=total,
            used=used,
            free=free,
            cached_tensors=len(self.tensor_cache)
        )

    async def _cleanup_storage(self):
        """Coordinate cleanup across storage systems"""
        try:
            # Clear expired memory cache
            self._evict_cache()
            
            # Clear expired IPFS cache
            self.model_storage.clear_expired_cache()
            
            # Update storage status
            for key in list(self.storage_status.keys()):
                if key not in self.tensor_cache:
                    del self.storage_status[key]
                    
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

    async def coordinate_memory_systems(self, model_id: str) -> bool:
        """Coordinate memory systems with enhanced safety"""
        if not model_id:
            logger.error("Invalid model_id provided")
            return False
            
        try:
            if model_id not in self.memory_systems:
                logger.error(f"Memory system {model_id} not registered")
                return False

            # Validate memory state
            memory_status = self.get_memory_status()
            if memory_status.used / memory_status.total > 0.8:
                await self._handle_high_memory_usage()
                
            # Coordinate with active system
            system = self.memory_systems[model_id]
            state = await self._safely_cache_state(system)
            if state:
                self.storage_status[f"{model_id}_state"] = {
                    "cached": True,
                    "timestamp": time.time(),
                    "size": state.get("size", 0)
                }
                
                # Notify system coordinator about state changes
                if self.system_coordinator:
                    self.system_coordinator.dispatch_event("memory_state_updated", {
                        "model_id": model_id,
                        "timestamp": time.time()
                    })
                
            # Notify component coordinator if available
            if hasattr(self, '_coordinator'):
                await self._coordinator.coordinate_state_updates()
            
            return True
            
        except Exception as e:
            logger.error(f"Memory coordination failed: {str(e)}")
            return False

    async def share_tensor(self, source_id: str, target_id: str, 
                         tensor_key: str) -> bool:
        """Share tensors between memory systems"""
        try:
            if source_id in self.memory_systems and tensor_key in self.tensor_cache:
                # Create view instead of copy when possible
                tensor = self.tensor_cache[tensor_key]
                target_key = f"{target_id}_{tensor_key}"
                self.tensor_cache[target_key] = tensor.detach()
                return True
        except Exception as e:
            logger.error(f"Tensor sharing failed: {e}")
        return False

    async def coordinate_systems_state(self) -> bool:
        """Coordinate state between memory systems"""
        if not self.memory_systems:
            return False

        try:
            # Collect states from all systems
            states = {}
            for name, system in self.memory_systems.items():
                if hasattr(system, 'cache_state'):
                    state = await system.cache_state()
                    if state:
                        states[name] = state

            # Validate collective state
            if await self._validate_collective_state(states):
                # Update storage status
                self.storage_status.update({
                    f"{name}_state": {"cached": True, "timestamp": time.time()}
                    for name in states.keys()
                })
                return True

            return False

        except Exception as e:
            logger.error(f"System state coordination failed: {e}")
            return False

    async def _validate_collective_state(self, states: Dict[str, Any]) -> bool:
        """Validate collective state of memory systems"""
        try:
            total_memory = sum(
                state.get('memory', torch.tensor(0)).numel() 
                for state in states.values()
            )
            
            if total_memory > self.max_cache_size:
                if self.interactive and self.session:
                    return await self.session.get_confirmation(
                        "Collective state exceeds cache size. Continue?",
                        timeout=INTERACTION_TIMEOUTS["emergency"]
                    )
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"State validation failed: {e}")
            return False

    async def _safely_cache_state(self, system: Any) -> Optional[Dict]:
        """Safely cache system state with error handling"""
        try:
            if hasattr(system, 'cache_state'):
                return await system.cache_state()
            return None
        except Exception as e:
            logger.error(f"State caching failed: {e}")
            return None

    async def _handle_high_memory_usage(self) -> None:
        """Handle high memory usage conditions"""
        if self.interactive and self.session:
            if await self.session.get_confirmation(
                "High memory usage detected. Perform cleanup?",
                timeout=INTERACTION_TIMEOUTS["emergency"]
            ):
                await self._cleanup_storage()
        else:
            await self._cleanup_storage()
            
        # Notify system coordinator
        if self.system_coordinator:
            self.system_coordinator.dispatch_event("memory_cleanup", {
                "timestamp": time.time(),
                "source": "memory_manager"
            })

    async def _handle_high_memory_event(self, data: Dict[str, Any]) -> None:
        """Handle high memory event from system coordinator."""
        if data.get("source") != "memory_manager":  # Avoid feedback loops
            logger.warning(f"High memory usage detected by {data.get('source', 'unknown')}: {data.get('usage', 0):.1%}")
            if data.get("usage", 0) > 0.85:
                await self._evict_cache_interactive()

    async def _handle_critical_memory_event(self, data: Dict[str, Any]) -> None:
        """Handle critical memory event from system coordinator."""
        if data.get("source") != "memory_manager":  # Avoid feedback loops
            logger.error(f"Critical memory usage detected by {data.get('source', 'unknown')}: {data.get('usage', 0):.1%}")
            await self._force_cleanup()
