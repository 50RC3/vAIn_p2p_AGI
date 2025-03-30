"""Models module for vAIn P2P AGI."""

# Import only the SimpleNN and AdvancedNN directly
# Avoid importing classes with complex dependencies
try:
    from .simple_nn import SimpleNN, AdvancedNN
    __has_simple_nn = True
except ImportError:
    __has_simple_nn = False
    import logging
    logging.getLogger(__name__).warning("Could not import SimpleNN and AdvancedNN")

# Define what gets imported with "from models import *"
__all__ = []

# Add SimpleNN and AdvancedNN to __all__ only if import succeeded
if __has_simple_nn:
    __all__.extend(['SimpleNN', 'AdvancedNN'])

# Dictionary of lazy-loaded model classes to avoid circular imports
__lazy_modules = {}

def __getattr__(name):
    """Lazily load model classes to avoid circular imports"""
    # Check if we've already loaded this module
    if name in __lazy_modules:
        return __lazy_modules[name]
    
    # Handle specific model types
    if name == 'HybridMemorySystem':
        try:
            # Import only when requested to avoid circular dependency
            from .hybrid_memory_system import HybridMemorySystem as cls
            __lazy_modules[name] = cls
            return cls
        except ImportError as e:
            import logging
            logging.getLogger(__name__).error(f"Error importing HybridMemorySystem: {e}")
            raise
    
    # Add other lazy imports as needed
    elif name == 'DNCController':
        try:
            from .dnc.dnc_controller import DNCController as cls
            __lazy_modules[name] = cls
            return cls
        except ImportError as e:
            import logging
            logging.getLogger(__name__).error(f"Error importing DNCController: {e}")
            raise
    
    # If the name isn't recognized, raise AttributeError
    raise AttributeError(f"module 'models' has no attribute '{name}'")
