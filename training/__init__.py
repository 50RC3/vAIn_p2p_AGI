"""vAIn training module for production environment."""

__version__ = '0.2.1'

try:
    from .federated_training import FederatedTraining
    from .federated_client import FederatedClient
    from .contrastive_loss import ContrastiveLoss
    from .clustering_loss import ClusteringLoss
    from .meta_reptile import MetaReptile
    from .distillation import DistillationTrainer
    from .local_trainer import LocalTrainer
except ImportError as e:
    raise ImportError(f"Failed to import required training components: {e}")

__all__ = [
    'FederatedTraining',
    'FederatedClient', 
    'ContrastiveLoss',
    'ClusteringLoss',
    'MetaReptile',
    'DistillationTrainer',
    'LocalTrainer'
]
