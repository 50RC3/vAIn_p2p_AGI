"""
System-wide configuration for vAIn_p2p_AGI

This module manages global system configuration including logging, resource limits,
metrics collection, and interaction settings.
"""

import os
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

# Try importing constants, fallback if not available
try:
    from core.constants import INTERACTION_LEVELS, InteractionLevel, BASE_DIR
except ImportError:
    from enum import Enum
    class InteractionLevel(Enum):
        NONE = "none"
        MINIMAL = "minimal"
        NORMAL = "normal"
        VERBOSE = "verbose"
    
    INTERACTION_LEVELS = {
        "none": InteractionLevel.NONE,
        "minimal": InteractionLevel.MINIMAL,
        "normal": InteractionLevel.NORMAL,
        "verbose": InteractionLevel.VERBOSE
    }
    
    BASE_DIR = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)

@dataclass
class ResourceLimits:
    """Resource limits configuration."""
    cpu_percent: float = 90.0
    memory_percent: float = 85.0
    disk_percent: float = 90.0
    connections_per_host: int = 10
    max_worker_threads: int = 4
    timeout_seconds: int = 30

@dataclass
class MetricsConfig:
    """Metrics collection configuration."""
    enabled: bool = True
    storage_path: str = str(BASE_DIR / "logs" / "metrics")
    collection_interval: float = 60.0  # seconds
    retention_days: int = 7
    alert_threshold_cpu: float = 85.0
    alert_threshold_memory: float = 80.0
    alert_threshold_disk: float = 85.0
    alert_threshold_latency: float = 500.0  # ms

@dataclass
class AlertConfig:
    """Alert configuration."""
    enabled: bool = True
    notification_channels: List[str] = field(default_factory=lambda: ["console"])
    minimum_severity: str = "warning"
    throttle_period: int = 300  # seconds
    alert_history_size: int = 100
    webhook_url: Optional[str] = None
    email_recipients: List[str] = field(default_factory=list)

@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    file_logging: bool = True
    log_dir: str = str(BASE_DIR / "logs")
    max_size: int = 10  # MB
    backup_count: int = 5
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    
    def get_level(self) -> int:
        """Get the numeric logging level."""
        return {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }.get(self.level.upper(), logging.INFO)

@dataclass
class SystemConfig:
    """Global system configuration."""
    version: str = "0.1.0"
    node_id: str = os.getenv("NODE_ID", "")
    node_type: str = "default"
    interactive: bool = True
    interaction_level: str = os.getenv("INTERACTION_LEVEL", "NORMAL")
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    advanced_mode: bool = False
    
    def __post_init__(self):
        """Initialize after creation."""
        # Create a unique node_id if not provided
        if not self.node_id:
            import uuid
            self.node_id = str(uuid.uuid4())[:8]
        
        # Ensure interaction_level is uppercase
        self.interaction_level = self.interaction_level.upper()
        
        # Create necessary directories
        self._create_dirs()
        
    def _create_dirs(self):
        """Create necessary directories."""
        try:
            # Create logs directory
            log_dir = Path(self.logging.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Create metrics directory if enabled
            if self.metrics.enabled:
                metrics_dir = Path(self.metrics.storage_path)
                metrics_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create directories: {e}")
    
    def get_interaction_level(self) -> InteractionLevel:
        """Get the interaction level enum."""
        level_str = self.interaction_level.lower()
        return INTERACTION_LEVELS.get(level_str, InteractionLevel.NORMAL)
    
    def validate(self) -> bool:
        """Validate the configuration."""
        try:
            # Validate interaction level
            if self.interaction_level.upper() not in ["NONE", "MINIMAL", "NORMAL", "VERBOSE"]:
                logger.error(f"Invalid interaction level: {self.interaction_level}")
                return False
                
            # Validate resource limits
            for attr_name in ["cpu_percent", "memory_percent", "disk_percent"]:
                value = getattr(self.resource_limits, attr_name)
                if not 0 <= value <= 100:
                    logger.error(f"Invalid {attr_name}: {value}, must be 0-100")
                    return False
                    
            # Validate logging level
            if self.logging.level.upper() not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
                logger.error(f"Invalid logging level: {self.logging.level}")
                return False
                
            # Check directories are writable
            try:
                log_path = Path(self.logging.log_dir) / "test_write.tmp"
                with open(log_path, "w") as f:
                    f.write("test")
                log_path.unlink()
            except Exception as e:
                logger.error(f"Log directory is not writable: {e}")
                return False
                
            if self.metrics.enabled:
                try:
                    metrics_path = Path(self.metrics.storage_path) / "test_write.tmp"
                    with open(metrics_path, "w") as f:
                        f.write("test")
                    metrics_path.unlink()
                except Exception as e:
                    logger.error(f"Metrics directory is not writable: {e}")
                    return False
                    
            return True
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "node_id": self.node_id,
            "node_type": self.node_type,
            "interactive": self.interactive,
            "interaction_level": self.interaction_level,
            "resource_limits": asdict(self.resource_limits),
            "metrics": asdict(self.metrics),
            "alerts": asdict(self.alerts),
            "logging": asdict(self.logging),
            "advanced_mode": self.advanced_mode
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemConfig':
        """Create a SystemConfig from a dictionary."""
        # Handle nested objects
        if "resource_limits" in data and isinstance(data["resource_limits"], dict):
            data["resource_limits"] = ResourceLimits(**data["resource_limits"])
            
        if "metrics" in data and isinstance(data["metrics"], dict):
            data["metrics"] = MetricsConfig(**data["metrics"])
            
        if "alerts" in data and isinstance(data["alerts"], dict):
            data["alerts"] = AlertConfig(**data["alerts"])
            
        if "logging" in data and isinstance(data["logging"], dict):
            data["logging"] = LoggingConfig(**data["logging"])
            
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})
    
    @classmethod
    def load(cls, config_path: Optional[Union[str, Path]] = None) -> 'SystemConfig':
        """Load configuration from file."""
        path = Path(config_path) if config_path else Path(BASE_DIR) / "config" / "system.json"
        
        if not path.exists():
            logger.warning(f"Config file not found: {path}, using defaults")
            return cls()
            
        try:
            with open(path, "r") as f:
                data = json.load(f)
                return cls.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load config from {path}: {e}")
            return cls()
    
    def save(self, config_path: Optional[Union[str, Path]] = None) -> bool:
        """Save configuration to file."""
        path = Path(config_path) if config_path else Path(BASE_DIR) / "config" / "system.json"
        
        try:
            # Create directory if it doesn't exist
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
                
            logger.info(f"Saved configuration to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save config to {path}: {e}")
            return False
    
    def update_interactive(self) -> bool:
        """Update configuration interactively."""
        # Import tools here to avoid circular imports
        try:
            from tools.config_manager import ConfigManager
            
            # Create a temporary JSON file for the config manager to work with
            temp_path = Path(BASE_DIR) / "config" / "system_temp.json"
            with open(temp_path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
            
            # Use config manager to update the file
            config_manager = ConfigManager()
            success = config_manager.update_config_interactive("system")
            
            if success:
                # Reload from the updated file
                updated = self.load(temp_path)
                
                # Update self with new values
                for key, value in asdict(updated).items():
                    setattr(self, key, value)
                
                # Save to the proper location and remove temp file
                self.save()
                temp_path.unlink(missing_ok=True)
                
                logger.info("System configuration updated successfully")
                return True
            else:
                temp_path.unlink(missing_ok=True)
                logger.warning("System configuration update cancelled")
                return False
                
        except ImportError:
            # Fallback to basic interactive update
            print("\nCurrent System Configuration:")
            
            print(f"Interaction Level [{self.interaction_level}]: ", end="")
            new_level = input() or self.interaction_level
            if new_level.upper() in ["NONE", "MINIMAL", "NORMAL", "VERBOSE"]:
                self.interaction_level = new_level.upper()
            
            print(f"Logging Level [{self.logging.level}]: ", end="")
            new_log_level = input() or self.logging.level
            if new_log_level.upper() in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
                self.logging.level = new_log_level.upper()
            
            # CPU limit
            print(f"CPU Usage Limit (%) [{self.resource_limits.cpu_percent}]: ", end="")
            try:
                new_cpu = input() or str(self.resource_limits.cpu_percent)
                self.resource_limits.cpu_percent = float(new_cpu)
            except ValueError:
                print("Invalid input, keeping current value")
            
            # Memory limit
            print(f"Memory Usage Limit (%) [{self.resource_limits.memory_percent}]: ", end="")
            try:
                new_memory = input() or str(self.resource_limits.memory_percent)
                self.resource_limits.memory_percent = float(new_memory)
            except ValueError:
                print("Invalid input, keeping current value")
            
            # Save updated config
            return self.save()

def get_system_config(config_path: Optional[Union[str, Path]] = None) -> SystemConfig:
    """Get the system configuration singleton."""
    if not hasattr(get_system_config, "_instance"):
        get_system_config._instance = SystemConfig.load(config_path)
    return get_system_config._instance
