"""
Core modules for the vAIn P2P AGI system.
"""

from .model_evaluation import ModelEvaluator
from .version_control import ModelVersionControl
from .training_coordinator import TrainingCoordinator
from .resource_management import ResourceManager
from .module_registry import (
    ModuleRegistry, ModuleRegistryError, ConfigManager, 
    DependencyResolver, LifecycleManager, MetricsTracker, CallbackManager
)

__all__ = [
    'ModelEvaluator',
    'ModelVersionControl',
    'TrainingCoordinator',
    'ResourceManager',
    'ModuleRegistry',
    'ModuleRegistryError',
    'ConfigManager',
    'DependencyResolver',
    'LifecycleManager',
    'MetricsTracker',
    'CallbackManager',
]
