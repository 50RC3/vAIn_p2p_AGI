from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any
import os

class InteractionLevel(Enum:
    NONE = "none"
    MINIMAL = "minimal" 
    NORMAL = "normal"
    VERBOSE = "verbose"

@dataclass
class InteractiveConfig:
    timeout: int = int(os.getenv('INTERACTIVE_TIMEOUT', 300))
    max_retries: int = int(os.getenv('INTERACTIVE_MAX_RETRIES', 3))
    safe_mode: bool = os.getenv('INTERACTIVE_SAFE_MODE', 'true').lower() == 'true'
    persistent_state: bool = os.getenv('INTERACTIVE_PERSISTENT_STATE', 'true').lower() == 'true'
    auto_recovery: bool = os.getenv('INTERACTIVE_AUTO_RECOVERY', 'true').lower() == 'true'
    cleanup_timeout: int = int(os.getenv('INTERACTIVE_CLEANUP_TIMEOUT', 30))
    memory_threshold: float = float(os.getenv('INTERACTIVE_MEMORY_THRESHOLD', 0.9))
    log_interactions: bool = os.getenv('INTERACTIVE_LOG_INTERACTIONS', 'true').lower() == 'true'
    heartbeat_interval: int = int(os.getenv('INTERACTIVE_HEARTBEAT_INTERVAL', 30))

    def validate(self) -> bool:
        if not os.getenv('NODE_ENV') == 'production':
            return True

        # Strict validation for production
        if not 0 < self.timeout < 3600:
            raise ValueError("Production timeout must be between 0 and 3600 seconds")
        if not 0 < self.max_retries < 10:
            raise ValueError("Production max retries must be between 0 and 10")
        if not 0 < self.memory_threshold < 1:
            raise ValueError("Production memory threshold must be between 0 and 1")
        if not 0 < self.cleanup_timeout < 300:
            raise ValueError("Production cleanup timeout must be between 0 and 300 seconds")
        if not 0 < self.heartbeat_interval < 300:
            raise ValueError("Production heartbeat interval must be between 0 and 300 seconds")
        return True

# Enhanced timeouts for production
INTERACTION_TIMEOUTS = {
    "default": int(os.getenv('INTERACTIVE_TIMEOUT', 300)),
    "confirmation": min(60, int(os.getenv('INTERACTIVE_TIMEOUT', 300)) // 5),
    "emergency": min(30, int(os.getenv('INTERACTIVE_TIMEOUT', 300)) // 10),
    "batch": min(600, int(os.getenv('INTERACTIVE_TIMEOUT', 300)) * 2),
    "resource_check": min(15, int(os.getenv('INTERACTIVE_TIMEOUT', 300)) // 20),
    "cleanup": int(os.getenv('INTERACTIVE_CLEANUP_TIMEOUT', 30)),
    "config": int(os.getenv('INTERACTIVE_TIMEOUT', 300)),
    "rotation": min(120, int(os.getenv('INTERACTIVE_TIMEOUT', 300)) // 2.5)
}

def get_production_config() -> InteractiveConfig:
    """Get production-optimized interactive config"""
    if os.getenv('NODE_ENV') != 'production':
        raise ValueError("Not in production environment")

    return InteractiveConfig(
        timeout=min(300, int(os.getenv('INTERACTIVE_TIMEOUT', 300))),
        max_retries=min(3, int(os.getenv('INTERACTIVE_MAX_RETRIES', 3))),
        safe_mode=True,  # Always true in production
        persistent_state=True,  # Always true in production
        auto_recovery=True,  # Always true in production
        cleanup_timeout=min(30, int(os.getenv('INTERACTIVE_CLEANUP_TIMEOUT', 30))),
        memory_threshold=max(0.85, float(os.getenv('INTERACTIVE_MEMORY_THRESHOLD', 0.9))),
        log_interactions=True,  # Always true in production
        heartbeat_interval=min(30, int(os.getenv('INTERACTIVE_HEARTBEAT_INTERVAL', 30)))
    )
