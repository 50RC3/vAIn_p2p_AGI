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

# Import multi-agent system
from .multi_agent_system import MultiAgentSystem

# Import neurotransmitter system
from .neurotransmitter import NeurotransmitterSystem

# Import cognitive evolution
from .cognitive_evolution import CognitiveEvolution

# Import federated learning components
from .federated_learning import LocalTrainer, IFederatedLearning

# Import learning coordinator
from .learning_coordinator import LearningCoordinator, LearningStats

# Import reputation system
from .reputation import ClientReputationSystem, IReputationSystem

# Import model coordinator
from .coordinator import ModelCoordinator

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
    'MultiAgentSystem',
    'NeurotransmitterSystem',
    'CognitiveEvolution',
    'LocalTrainer',
    'IFederatedLearning',
    'ClientReputationSystem',
    'IReputationSystem',
    'ModelCoordinator',
    'LearningCoordinator',
    'LearningStats',
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
        
        # Register NeurotransmitterSystem
        await registry.register_module(
            "neurotransmitter",
            NeurotransmitterSystem,
            dependencies=[],
            config={
                "resource_requirements": {
                    "memory": "low",
                    "cpu": "low",
                    "gpu": "none",
                    "importance": "high"
                }
            }
        )
        
        # Register CognitiveEvolution
        await registry.register_module(
            "cognitive_evolution",
            CognitiveEvolution,
            dependencies=["memory_manager", "unified_model_system"],
            config={
                "resource_requirements": {
                    "memory": "medium",
                    "cpu": "high",
                    "gpu": "medium",
                    "importance": "high"
                }
            }
        )
        
        # Register LocalTrainer
        await registry.register_module(
            "local_trainer",
            LocalTrainer,
            dependencies=["neurotransmitter"],
            config={
                "resource_requirements": {
                    "memory": "medium",
                    "cpu": "high",
                    "gpu": "high",
                    "importance": "high"
                }
            }
        )
        
        # Register ClientReputationSystem
        await registry.register_module(
            "reputation_system",
            ClientReputationSystem,
            dependencies=[],
            config={
                "resource_requirements": {
                    "memory": "low",
                    "cpu": "low",
                    "gpu": "none",
                    "importance": "medium"
                }
            }
        )
        
        # Register ModelCoordinator
        await registry.register_module(
            "model_coordinator",
            ModelCoordinator,
            dependencies=[],
            config={
                "resource_requirements": {
                    "memory": "medium",
                    "cpu": "medium",
                    "gpu": "low",
                    "importance": "high"
                }
            }
        )
        
        # Register LearningCoordinator
        await registry.register_module(
            "learning_coordinator",
            LearningCoordinator,
            dependencies=["rl_trainer", "unsupervised_module", "self_supervised_module"],
            config={
                "resource_requirements": {
                    "memory": "medium",
                    "cpu": "high",
                    "gpu": "medium",
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
