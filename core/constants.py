from enum import Enum
from typing import Dict, Any, Tuple
import platform
import os
from pathlib import Path

class ModelStatus(Enum):
    PENDING = "pending"
    TRAINING = "training"
    VALIDATED = "validated"
    REJECTED = "rejected"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"
    UPDATING = "updating"

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

class TaskPriority(Enum):
    """Priority levels for tasks"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

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

INTERACTION_TIMEOUTS.update({
    "discovery": 30,      # 30 seconds for peer discovery operations
    "broadcast": 15,      # 15 seconds for broadcast operations
    "response": 5,        # 5 seconds for peer responses
    "peer_cleanup": 10,   # 10 seconds for peer cleanup operations
    "peer_update": 20,    # 20 seconds for peer updates
})

# Enhanced timeouts for NORMAL mode
INTERACTION_TIMEOUTS.update({
    "normal_op": 60,        # Standard operation timeout
    "normal_confirm": 45,   # Normal confirmation timeout
    "normal_retry": 30,     # Retry operation timeout
    "normal_sync": 90,      # Synchronization timeout
    "normal_recovery": 120,  # Recovery operation timeout
    "normal_backup": 75,    # Backup operation timeout
})

# NORMAL mode specific constants
NORMAL_MODE_SETTINGS = {
    'max_concurrent_ops': 5,
    'min_progress_interval': 1,
    'max_recovery_attempts': 3,
    'resource_check_interval': 2,
    'status_update_interval': 5
}

# Sharding configuration
SHARD_CONFIG = {
    "max_peers_per_shard": 1000,
    "num_shards": 16,  # Start with 16 shards
    "replication_factor": 3,  # Each shard replicated 3 times
    "shard_cleanup_interval": 300,  # 5 minutes
    "reshard_threshold": 0.8,  # Trigger resharding at 80% capacity
}

# Load balancing configuration
LOAD_BALANCING = {
    "max_load": 0.8,  # 80% max load per node
    "min_load": 0.2,  # 20% min load before rebalance
    "check_interval": 60,  # Check every 60 seconds
    "rebalance_threshold": 0.3  # 30% load difference triggers rebalance
}

# Traffic shaping configuration
TRAFFIC_SHAPING = {
    "critical_allocation": 0.4,
    "high_allocation": 0.3, 
    "medium_allocation": 0.2,
    "low_allocation": 0.1,
    "burst_multiplier": 2.0,  # Allow burst for critical traffic
    "window_size": 1.0  # 1 second measurement window
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

# Platform-specific settings
IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"
IS_MACOS = platform.system() == "Darwin"

# Interaction timeout constants (in seconds)
INTERACTION_TIMEOUTS = {
    "input": 60,         # Basic user input timeout
    "confirmation": 30,  # Confirmation dialog timeout
    "critical": 120,     # Critical decision timeout
    "file_operation": 60,  # File read/write operation timeout
    "network": 45,       # Network operation timeout
    "computation": 300,  # Long computation timeout
    "connection": 30,    # Connection establishment timeout
    "initialization": 90,  # System initialization timeout
    "shutdown": 30,      # Graceful shutdown timeout
    "backup": 120,       # Data backup timeout
    "restore": 180,      # Data restoration timeout
}

# Resource threshold constants
RESOURCE_THRESHOLDS = {
    "memory_critical": 95,    # Memory usage percentage considered critical
    "memory_warning": 85,     # Memory usage percentage that triggers warnings
    "cpu_critical": 95,       # CPU usage percentage considered critical
    "cpu_warning": 80,        # CPU usage percentage that triggers warnings
    "disk_critical": 95,      # Disk usage percentage considered critical
    "disk_warning": 85,       # Disk usage percentage that triggers warnings
    "network_warning": 80,    # Network utilization that triggers warnings
    "battery_critical": 10,   # Battery percentage considered critical (mobile devices)
    "battery_warning": 20,    # Battery percentage that triggers warnings
}

# Backoff strategy constants
BACKOFF = {
    "base_delay": 1.0,        # Base delay for retries (seconds)
    "max_delay": 60.0,        # Maximum delay between retries (seconds)
    "factor": 2.0,            # Multiplicative factor for backoff
    "jitter": 0.1,            # Randomness factor to avoid thundering herd
    "max_retries": 5,         # Maximum number of retry attempts
}

# Path constants
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEMP_DIR = BASE_DIR / "temp"
DATA_DIR = BASE_DIR / "data"
CONFIG_DIR = BASE_DIR / "config"
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"
CACHE_DIR = BASE_DIR / "cache"
SESSION_SAVE_PATH = BASE_DIR / "sessions"

# Ensure critical directories exist
for directory in [TEMP_DIR, DATA_DIR, CONFIG_DIR, LOGS_DIR, MODELS_DIR, CACHE_DIR, SESSION_SAVE_PATH]:
    directory.mkdir(exist_ok=True, parents=True)

# Feature flags
FEATURES = {
    "use_compression": True,
    "enable_recovery": True,
    "debug_logging": False,
    "use_encryption": True,
    "enable_monitoring": True,
}

# Default operation modes
DEFAULT_MODE = "standard"  # Options: minimal, standard, verbose, debug
DEFAULT_VERBOSITY = 1      # 0=quiet, 1=normal, 2=verbose, 3=debug

# Default file encoding
DEFAULT_ENCODING = "utf-8"

# Memory management
MEMORY_CHUNK_SIZE = 1024 * 1024  # 1MB
MAX_CACHE_SIZE = 1024 * 1024 * 1024 * 2  # 2GB
MEMORY_CLEANUP_THRESHOLD = 0.85  # Cleanup when 85% of max cache is used

# Network settings
DEFAULT_PORT = 8000
DEFAULT_HOST = "127.0.0.1"
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY = 2

# Model parameters
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 10

# Federated learning parameters
AGGREGATION_MIN_CLIENTS = 3
AGGREGATION_TIMEOUT = 120
CLIENT_SELECTION_THRESHOLD = 0.7
