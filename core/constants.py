from enum import Enum

class ModelStatus(Enum):
    PENDING = "pending"
    TRAINING = "training"
    VALIDATED = "validated"
    REJECTED = "rejected"

class TrainingMode(Enum):
    FEDERATED = "federated"
    DISTRIBUTED = "distributed"
    HYBRID = "hybrid"

MINIMUM_STAKE = 1000  # Minimum stake required for training participation
MAX_BATCH_SIZE = 128  # Maximum batch size for training
DEFAULT_TIMEOUT = 300  # Default timeout for training rounds in seconds
