"""
Core components of the vAIn P2P AGI system.
"""

try:
    from .model_evaluation import ModelEvaluator
except ImportError:
    ModelEvaluator = None

try:
    from .version_control import ModelVersionControl
except ImportError:
    ModelVersionControl = None

try:
    from .training_coordinator import TrainingCoordinator
except ImportError:
    TrainingCoordinator = None

__all__ = [
    'ModelEvaluator',
    'ModelVersionControl',
    'TrainingCoordinator'
]
