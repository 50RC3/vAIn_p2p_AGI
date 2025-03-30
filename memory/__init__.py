"""Memory management components for vAIn P2P AGI."""

# Use lazy imports to avoid circular dependencies
__all__ = ['MemoryManager', 'MemoryStatus']

# Dictionary of lazy-loaded memory classes
__lazy_modules = {}

def __getattr__(name):
    """Lazily load memory classes to prevent circular imports"""
    # Return cached module if available
    if name in __lazy_modules:
        return __lazy_modules[name]
    
    # MemoryManager and MemoryStatus
    if name in ['MemoryManager', 'MemoryStatus']:
        try:
            from .memory_manager import MemoryManager, MemoryStatus
            __lazy_modules['MemoryManager'] = MemoryManager
            __lazy_modules['MemoryStatus'] = MemoryStatus
            return __lazy_modules[name]
        except ImportError as e:
            import logging
            logging.getLogger(__name__).error(f"Error importing {name}: {e}")
            raise
    
    # Memory processor components
    elif name in ['MemoryProcessor']:
        try:
            from .memory_processing import MemoryProcessor
            __lazy_modules['MemoryProcessor'] = MemoryProcessor
            return __lazy_modules[name]
        except ImportError as e:
            import logging
            logging.getLogger(__name__).error(f"Error importing {name}: {e}")
            raise
    
    # If the name isn't recognized, raise AttributeError
    raise AttributeError(f"module 'memory' has no attribute '{name}'")
