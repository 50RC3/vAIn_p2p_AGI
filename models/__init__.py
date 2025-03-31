"""
Models package for vAIn P2P AGI system.
Contains model definitions, interfaces and related utilities.
"""

__version__ = "0.1.0"

# Import core model components
from .hybrid_memory_system import (
    HybridMemorySystem,
    MemoryAccessType,
    MemoryOperation
)

# Import lazy loading functions for use in other modules
from .hybrid_memory_system import (
    get_dnc_controller,
    get_memory_manager
)

# Import and expose key classes and functions from the types module
from .types import (
    ModelOutput, 
    ModelState, 
    ModelRole,
    get_resource_metrics
)

# Define package exports
__all__ = [
    'HybridMemorySystem',
    'MemoryAccessType',
    'MemoryOperation',
    'get_dnc_controller',
    'get_memory_manager',
    'ModelOutput',
    'ModelState',
    'ModelRole',
    'get_resource_metrics',
]

# Register with module registry if available
try:
    from ai_core.module_registry import ModuleRegistry
    import asyncio
    
    async def _register_models():
        registry = ModuleRegistry.get_instance()
        if not registry.is_initialized:
            await registry.initialize()
            
        # Register HybridMemorySystem
        await registry.register_module(
            "hybrid_memory",
            HybridMemorySystem,
            dependencies=["dnc_controller", "memory_manager"],
            config={
                "resource_requirements": {
                    "memory": "high", 
                    "cpu": "medium",
                    "gpu": "high",
                    "importance": "high"
                }
            }
        )
    
    # Run in an event loop if possible
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(_register_models())
        else:
            loop.run_until_complete(_register_models())
    except RuntimeError:
        # No event loop, models will need to be registered manually
        pass
    
except ImportError:
    # Module registry not available, models will need to be registered manually
    pass
