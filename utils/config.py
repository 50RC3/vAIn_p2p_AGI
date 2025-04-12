import os
import json
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import shutil

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "system": {
        "name": "vAIn P2P AGI",
        "version": "0.2.1",
        "interactive": True,
        "debug": False,
        "log_level": "info"
    },
    "resources": {
        "memory_warning_threshold": 80.0,
        "memory_critical_threshold": 90.0,
        "cpu_warning_threshold": 80.0,
        "cpu_critical_threshold": 90.0,
        "disk_warning_threshold": 85.0,
        "disk_critical_threshold": 95.0,
        "check_interval": 60,
        "enable_memory_optimizations": True,
        "resource_log_path": "logs/resources.log"
    },
    "network": {
        "peer_connection_timeout": 30,
        "max_peers": 50,
        "min_peers": 5,
        "heartbeat_interval": 60,
        "discovery_interval": 300
    },
    "storage": {
        "data_dir": "data",
        "models_dir": "models",
        "checkpoints_dir": "checkpoints",
        "max_storage_gb": 10,
        "backup_interval": 1800
    },
    "ai": {
        "default_model": "local",
        "max_token_length": 4096,
        "max_history_items": 100,
        "interactive_mode": True,
        "safety_filters": True
    },
    "metrics": {
        "enabled": True,
        "collection_interval": 60,
        "retention_days": 7,
        "metrics_dir": "metrics"
    }
}

CONFIG_PATH = "config/config.json"
CONFIG_BACKUP_PATH = "config/config.backup.json"


class Config:
    """Configuration management with defaults, validation, and persistence"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration
        
        Args:
            config_path: Path to configuration file. If None, uses default path.
        """
        self.config_path = Path(config_path or CONFIG_PATH)
        self.config: Dict[str, Any] = {}
        self.loaded = False
        
    def load(self, fallback_to_defaults: bool = True) -> bool:
        """Load configuration from disk
        
        Args:
            fallback_to_defaults: If True, use defaults if config file not found
            
        Returns:
            Success status
        """
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
                logger.info(f"Configuration loaded from {self.config_path}")
                self.loaded = True
                return True
            elif fallback_to_defaults:
                logger.warning(f"Configuration file not found at {self.config_path}, using defaults")
                self.config = DEFAULT_CONFIG.copy()
                self.loaded = True
                return True
            else:
                logger.error(f"Configuration file not found at {self.config_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            if fallback_to_defaults:
                logger.warning("Using default configuration due to loading error")
                self.config = DEFAULT_CONFIG.copy()
                self.loaded = True
                return True
            return False
    
    def save(self, create_backup: bool = True) -> bool:
        """Save configuration to disk
        
        Args:
            create_backup: If True, create a backup of existing config
            
        Returns:
            Success status
        """
        try:
            # Create directory if it doesn't exist
            self.config_path.parent.mkdir(exist_ok=True, parents=True)
            
            # Create backup if requested and file exists
            if create_backup and self.config_path.exists():
                backup_path = Path(CONFIG_BACKUP_PATH)
                shutil.copy2(self.config_path, backup_path)
                logger.debug(f"Backup created at {backup_path}")
            
            # Write current config
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"Configuration saved to {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def get(self, path: str, default: Any = None) -> Any:
        """Get configuration value by path
        
        Args:
            path: Dot-separated path to config value (e.g. "system.debug")
            default: Default value if path not found
            
        Returns:
            Configuration value
        """
        if not self.loaded:
            self.load()
            
        parts = path.split('.')
        current = self.config
        
        try:
            for part in parts:
                current = current[part]
            return current
        except (KeyError, TypeError):
            return default
    
    def set(self, path: str, value: Any) -> bool:
        """Set configuration value by path
        
        Args:
            path: Dot-separated path to config value
            value: Value to set
            
        Returns:
            Success status
        """
        if not self.loaded:
            self.load()
            
        parts = path.split('.')
        current = self.config
        
        try:
            # Navigate to the parent of the target
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Set the value
            current[parts[-1]] = value
            return True
        except Exception as e:
            logger.error(f"Error setting config value at {path}: {e}")
            return False
    
    def update(self, config_dict: Dict[str, Any]) -> bool:
        """Update configuration with new values
        
        Args:
            config_dict: Dictionary of configuration values to update
            
        Returns:
            Success status
        """
        if not self.loaded:
            self.load()
            
        try:
            self._deep_update(self.config, config_dict)
            return True
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
            return False
    
    def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Recursively update target dictionary with values from source"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value
    
    def reset_to_defaults(self) -> bool:
        """Reset configuration to defaults
        
        Returns:
            Success status
        """
        try:
            self.config = DEFAULT_CONFIG.copy()
            self.loaded = True
            return True
        except Exception as e:
            logger.error(f"Error resetting configuration: {e}")
            return False
    
    def validate(self) -> List[str]:
        """Validate configuration
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Validate required sections exist
        required_sections = ["system", "resources", "storage", "ai"]
        for section in required_sections:
            if section not in self.config:
                errors.append(f"Required configuration section '{section}' is missing")
        
        # Validate specific settings
        try:
            # Resource thresholds should be between 0-100
            for threshold in ["memory_warning_threshold", "memory_critical_threshold", 
                             "cpu_warning_threshold", "cpu_critical_threshold"]:
                if section := self.config.get("resources"):
                    value = section.get(threshold, 0)
                    if not 0 <= value <= 100:
                        errors.append(f"Resource {threshold} must be between 0-100, got {value}")
            
            # Warning threshold should be less than critical
            if section := self.config.get("resources"):
                if section.get("memory_warning_threshold", 0) >= section.get("memory_critical_threshold", 100):
                    errors.append("Memory warning threshold must be less than critical threshold")
                if section.get("cpu_warning_threshold", 0) >= section.get("cpu_critical_threshold", 100):
                    errors.append("CPU warning threshold must be less than critical threshold")
        except Exception as e:
            errors.append(f"Error during configuration validation: {e}")
        
        return errors


# Create a singleton instance for easy import
default_config = Config()

def get_config(path: str = None, default: Any = None) -> Any:
    """Convenient function to get configuration value"""
    if not default_config.loaded:
        default_config.load()
    
    if path is None:
        return default_config.config
    return default_config.get(path, default)

def set_config(path: str, value: Any) -> bool:
    """Convenient function to set configuration value"""
    return default_config.set(path, value)
