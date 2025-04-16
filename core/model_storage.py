"""
Model Storage Wrapper - Redirects to ai_core.model_storage implementation
"""
import logging
import asyncio
from typing import Dict, Any, Optional, Union, Tuple
import torch

logger = logging.getLogger(__name__)

# Import the actual implementation with error handling
try:
    from ai_core.model_storage import ModelStorage as CoreModelStorage
    _core_implementation_available = True
except ImportError:
    logger.warning("ai_core.model_storage not available, using fallback implementation")
    _core_implementation_available = False
    
    # Define a fallback implementation
    class CoreModelStorage:
        """Fallback implementation when ai_core.model_storage is not available."""
        
        def __init__(self, **kwargs):
            self.models = {}
            self.metadata = {}
            logger.warning("Using fallback ModelStorage implementation")
            
        async def store_model_async(self, model, metadata=None, **kwargs):
            """Store model asynchronously."""
            import hashlib
            model_id = hashlib.md5(str(id(model)).encode()).hexdigest()
            meta_id = hashlib.md5(str(metadata).encode()).hexdigest() if metadata else None
            self.models[model_id] = model
            if metadata:
                self.metadata[meta_id] = metadata
            return model_id, meta_id
            
        def store_model(self, model, metadata=None, **kwargs):
            """Store model synchronously."""
            import hashlib
            model_id = hashlib.md5(str(id(model)).encode()).hexdigest()
            meta_id = hashlib.md5(str(metadata).encode()).hexdigest() if metadata else None
            self.models[model_id] = model
            if metadata:
                self.metadata[meta_id] = metadata
            return model_id, meta_id
            
        async def retrieve_model_async(self, model_id):
            """Retrieve model asynchronously."""
            return self.models.get(model_id)
            
        def retrieve_model(self, model_id):
            """Retrieve model synchronously."""
            return self.models.get(model_id)
            
        def clear_expired_cache(self):
            """Clear expired cache entries."""
            # No-op in fallback implementation
            pass


class ModelStorage:
    """Model storage wrapper that redirects to the core implementation."""
    
    def __init__(self, storage_dir: str = "./model_storage", **kwargs):
        """Initialize model storage.
        
        Args:
            storage_dir: Directory to store models
            **kwargs: Additional arguments passed to the core implementation
        """
        self._implementation = CoreModelStorage(storage_dir=storage_dir, **kwargs)
        logger.info(f"ModelStorage initialized with storage_dir: {storage_dir}")
        self._cache = {}
        
    async def store_model_async(self, model: Union[torch.nn.Module, torch.Tensor], 
                              metadata: Optional[Dict[str, Any]] = None, 
                              interactive: bool = False) -> Tuple[str, Optional[str]]:
        """Store model asynchronously.
        
        Args:
            model: PyTorch model or tensor to store
            metadata: Optional metadata associated with the model
            interactive: Whether to use interactive mode
            
        Returns:
            Tuple of (model_hash, metadata_hash)
        """
        try:
            logger.debug(f"Storing model with metadata: {metadata is not None}")
            return await self._implementation.store_model_async(
                model, metadata, interactive=interactive
            )
        except Exception as e:
            logger.error(f"Failed to store model asynchronously: {e}")
            raise
            
    def store_model(self, model: Union[torch.nn.Module, torch.Tensor], 
                  metadata: Optional[Dict[str, Any]] = None,
                  interactive: bool = False) -> Tuple[str, Optional[str]]:
        """Store model synchronously.
        
        Args:
            model: PyTorch model or tensor to store
            metadata: Optional metadata associated with the model
            interactive: Whether to use interactive mode
            
        Returns:
            Tuple of (model_hash, metadata_hash)
        """
        try:
            return self._implementation.store_model(
                model, metadata, interactive=interactive
            )
        except Exception as e:
            logger.error(f"Failed to store model: {e}")
            raise
            
    async def retrieve_model_async(self, model_id: str) -> Optional[Union[torch.nn.Module, torch.Tensor]]:
        """Retrieve model asynchronously.
        
        Args:
            model_id: ID of the model to retrieve
            
        Returns:
            The retrieved model or None if not found
        """
        try:
            # Check cache first
            if model_id in self._cache:
                logger.debug(f"Model {model_id} found in cache")
                return self._cache[model_id]
                
            # Retrieve from storage
            model = await self._implementation.retrieve_model_async(model_id)
            
            # Update cache
            if model is not None:
                self._cache[model_id] = model
                
            return model
        except Exception as e:
            logger.error(f"Failed to retrieve model {model_id}: {e}")
            return None
            
    def retrieve_model(self, model_id: str) -> Optional[Union[torch.nn.Module, torch.Tensor]]:
        """Retrieve model synchronously.
        
        Args:
            model_id: ID of the model to retrieve
            
        Returns:
            The retrieved model or None if not found
        """
        try:
            # Check cache first
            if model_id in self._cache:
                return self._cache[model_id]
                
            # Retrieve from storage
            model = self._implementation.retrieve_model(model_id)
            
            # Update cache
            if model is not None:
                self._cache[model_id] = model
                
            return model
        except Exception as e:
            logger.error(f"Failed to retrieve model {model_id}: {e}")
            return None
    
    async def update_state(self, model_id: str, state_update: Dict[str, Any]) -> bool:
        """Update model state.
        
        Args:
            model_id: ID of the model to update
            state_update: State updates to apply
            
        Returns:
            True if successful, False otherwise
        """
        if hasattr(self._implementation, "update_state"):
            try:
                return await self._implementation.update_state(model_id, state_update)
            except Exception as e:
                logger.error(f"Failed to update model state: {e}")
                return False
        else:
            logger.warning("update_state not supported by the implementation")
            return False
    
    def clear_expired_cache(self) -> None:
        """Clear expired cache entries."""
        # Clear local cache
        self._cache.clear()
        
        # Forward to implementation
        if hasattr(self._implementation, "clear_expired_cache"):
            self._implementation.clear_expired_cache()
