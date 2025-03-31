"""AI Core components for the vAIn P2P AGI system."""

__version__ = "0.1.0"

# Import core components
from .unified_model_system import UnifiedModelSystem
from .cognitive_evolution import CognitiveEvolution
from .model_coordinator import ModelCoordinator
from .module_registry import ModuleRegistry

# Setup exports
__all__ = [
    'UnifiedModelSystem',
    'CognitiveEvolution',
    'ModelCoordinator',
    'ModuleRegistry',
    'initialize_cognitive_system'
]

# Async initialization function
async def initialize_cognitive_system(memory_manager=None, resource_metrics=None, config=None, params=None):
    """
    Initialize the cognitive system components.
    
    Args:
        memory_manager: Optional memory manager instance
        resource_metrics: Optional resource metrics or parameters
        config: Optional configuration parameters (alternative to resource_metrics)
        params: Optional parameters (another alternative)
        
    Returns:
        Tuple of (UnifiedModelSystem, CognitiveEvolution)
    """
    from memory.memory_manager import MemoryManager
    import logging
    
    logger = logging.getLogger("ai_core")
    
    try:
        # Use whichever parameters are provided (config, params, or resource_metrics)
        # This makes the function more robust to different parameter names
        parameters = config or params or resource_metrics or {}
        
        # Initialize memory manager if not provided
        if memory_manager is None:
            from core.constants import MAX_CACHE_SIZE
            memory_manager = MemoryManager(max_cache_size=MAX_CACHE_SIZE)
        
        # Initialize unified model system with parameters
        unified_system = UnifiedModelSystem(memory_manager)
        
        # Initialize cognitive evolution
        cognitive_system = CognitiveEvolution(unified_system, memory_manager)
        
        # Initialize the cognitive network with parameters
        success = await cognitive_system.initialize_cognitive_network(**parameters if isinstance(parameters, dict) else {})
        if not success:
            logger.warning("Cognitive network initialization failed, some features may not work")
        
        return unified_system, cognitive_system
        
    except Exception as e:
        logger.error(f"Failed to initialize cognitive system: {e}")
        raise

# Auto-initialization for module registry
try:
    import asyncio
    
    async def _register_with_module_registry():
        registry = ModuleRegistry.get_instance()
        if not registry.is_initialized:
            await registry.initialize()
        
        await registry.register_module(
            "cognitive_evolution",
            CognitiveEvolution,
            dependencies=["unified_model_system", "memory_manager"],
            config={
                "resource_requirements": {
                    "memory": "high", 
                    "cpu": "high",
                    "gpu": "high",
                    "importance": "critical"
                }
            }
        )
    
    # Run in an event loop if possible
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(_register_with_module_registry())
        else:
            loop.run_until_complete(_register_with_module_registry())
    except RuntimeError:
        # No event loop, will be registered later
        pass
except ImportError:
    pass  # ModuleRegistry not available
