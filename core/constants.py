from enum import Enum
from typing import Dict, Any, Tuple

class ModelStatus(Enum):
    PENDING = "pending"
    TRAINING = "training"
    VALIDATED = "validated"
    REJECTED = "rejected"

class TrainingMode(Enum):
    FEDERATED = "federated"
    DISTRIBUTED = "distributed"
    HYBRID = "hybrid"

class InteractionLevel(Enum):
    """Controls level of interactive prompts and feedback"""
    NONE = "none"       # No interactive prompts
    MINIMAL = "minimal" # Only critical prompts
    NORMAL = "normal"   # Standard interactive mode
    VERBOSE = "verbose" # Detailed interactive mode

# System constants
MINIMUM_STAKE = 1000  # Minimum stake required for training participation
MAX_BATCH_SIZE = 128  # Maximum batch size for training
DEFAULT_TIMEOUT = 300  # Default timeout for training rounds in seconds

# Interactive configuration boundaries
CONFIG_BOUNDS: Dict[str, Tuple[Any, Any]] = {
    "batch_size": (1, MAX_BATCH_SIZE),
    "stake_amount": (MINIMUM_STAKE, 1_000_000),
    "timeout": (30, 3600),  # 30s to 1h
    "retry_attempts": (1, 5),
    "confirmation_blocks": (1, 12),
    "max_retries": (1, 10),
    "max_input_length": (128, 4096)
}

# Interactive timeouts (seconds)
INTERACTION_TIMEOUTS = {
    "input": 300,       # 5 minutes for user input
    "confirmation": 60, # 1 minute for confirmations
    "validation": 120,  # 2 minutes for validation steps
    "resume": 180,     # 3 minutes for resume confirmation
    "emergency": 15,    # 15 seconds for emergency operations
    "default": 30,
    "batch": 300,
    "save": 60,
    "shutdown": 30,     # 30 seconds for shutdown procedures
    "cleanup": 10,      # 10 seconds for cleanup tasks
    "recovery": 60,     # 60 seconds for recovery attempts
    "backup": 30,       # 30 seconds for backup operations
    "restore": 45,      # 45 seconds for state restoration
    "resource_check": 30, # 30 seconds for resource verification
    "heartbeat": 60,    # 60 seconds between heartbeats
    "graceful_shutdown": 30, # 30 seconds for graceful shutdown
}

# Input validation settings
INPUT_VALIDATION = {
    "max_length": 1024,
    "min_length": 1,
    "retry_delay": 1.0,
    "max_retries": 3
}

# Retry configurations
RETRY_CONFIG = {
    "max_retries": 3,
    "backoff_factor": 1.5,
    "jitter": 0.1,
    "max_timeout": 600  # 10 minutes absolute maximum
}

# Error recovery settings
RECOVERY_CONFIG = {
    "max_attempts": 3,
    "backoff_factor": 1.5,
    "initial_delay": 1.0,
    "max_delay": 30.0
}

# Safety thresholds
SAFETY_THRESHOLDS = {
    "max_memory_percent": 90,
    "max_input_length": 1024,
    "max_retry_attempts": 3,
    "min_cleanup_interval": 10,
    "max_batch_size": 1000,
}

# Progress update intervals (seconds) 
PROGRESS_INTERVALS = {
    "training": 5,    # Training progress updates
    "validation": 10, # Validation progress updates  
    "sync": 30       # Sync status updates
}

def validate_bounds(value: Any, param: str) -> bool:
    """Validate a parameter value is within defined bounds"""
    if param not in CONFIG_BOUNDS:
        raise ValueError(f"Unknown parameter: {param}")
        
    min_val, max_val = CONFIG_BOUNDS[param]
    return isinstance(value, type(min_val)) and min_val <= value <= max_val
