import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Set
import logging
from dataclasses import dataclass, field

from utils.unified_logger import get_logger

logger = get_logger("debug_config")

@dataclass
class DebugConfig:
    """Centralized debug configuration"""
    # General settings
    enabled: bool = True
    level: str = "INFO"
    log_directory: str = "logs/debug"
    save_metrics: bool = True
    metrics_interval: int = 60  # seconds
    
    # Ports and services
    debug_port: int = 5678
    web_debug_port: int = 8765
    monitoring_port: int = 9090
    
    # Feature flags
    profiling_enabled: bool = False
    memory_profiling: bool = False
    network_monitoring: bool = True
    log_monitoring: bool = True
    
    # Component-specific debug settings
    component_debug_levels: Dict[str, str] = field(default_factory=dict)
    
    # Security
    require_auth: bool = True
    allowed_ips: Set[str] = field(default_factory=lambda: {"127.0.0.1"})
    
    # Performance
    max_history_records: int = 1000
    max_debug_file_size_mb: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = {k: v for k, v in vars(self).items()}
        # Convert sets to lists for JSON serialization
        if "allowed_ips" in result:
            result["allowed_ips"] = list(result["allowed_ips"])
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DebugConfig":
        """Create instance from dictionary"""
        # Convert lists back to sets where needed
        if "allowed_ips" in data and isinstance(data["allowed_ips"], list):
            data["allowed_ips"] = set(data["allowed_ips"])
        
        # Create instance with filtered data matching fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

class DebugConfigManager:
    """Manages debug configuration across the system"""
    
    _instance = None
    
    @classmethod
    def get_instance(cls) -> "DebugConfigManager":
        """Get or create singleton instance"""
        if cls._instance is None:
            cls._instance = DebugConfigManager()
        return cls._instance
    
    def __init__(self, config_path: str = "config/debug.json"):
        """Initialize debug configuration manager"""
        self.config_path = config_path
        self.config = self._load_config()
        self._setup_directories()
        
    def _load_config(self) -> DebugConfig:
        """Load configuration from file or use defaults"""
        config = DebugConfig()
        
        # Try to load from file
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    data = json.load(f)
                config = DebugConfig.from_dict(data)
                logger.info(f"Loaded debug configuration from {self.config_path}")
            except Exception as e:
                logger.error(f"Error loading debug config: {e}", exc_info=True)
                
        # Override with environment variables
        self._apply_env_overrides(config)
        
        return config
        
    def _apply_env_overrides(self, config: DebugConfig) -> None:
        """Apply environment variable overrides to configuration"""
        env_prefix = "VAIN_DEBUG_"
        
        # Map environment variables to config attributes
        for key in vars(config):
            env_key = f"{env_prefix}{key.upper()}"
            if env_key in os.environ:
                env_value = os.environ[env_key]
                
                # Type conversion
                if isinstance(getattr(config, key), bool):
                    value = env_value.lower() == "true"
                elif isinstance(getattr(config, key), int):
                    value = int(env_value)
                elif isinstance(getattr(config, key), set):
                    value = set(env_value.split(","))
                elif isinstance(getattr(config, key), dict):
                    # Format for dicts: KEY1:VALUE1,KEY2:VALUE2
                    try:
                        pairs = env_value.split(",")
                        value = {}
                        for pair in pairs:
                            if ":" in pair:
                                k, v = pair.split(":", 1)
                                value[k.strip()] = v.strip()
                    except Exception:
                        logger.warning(f"Invalid dict format for {env_key}, skipping")
                        continue
                else:
                    value = env_value
                
                setattr(config, key, value)
                logger.debug(f"Applied debug config override from environment: {key}={value}")
                
    def _setup_directories(self) -> None:
        """Create necessary directories"""
        Path(self.config.log_directory).mkdir(exist_ok=True, parents=True)
        
    def save_config(self) -> bool:
        """Save current configuration to file"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            # Save config
            with open(self.config_path, "w") as f:
                json.dump(self.config.to_dict(), f, indent=2)
            logger.info(f"Saved debug configuration to {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving debug config: {e}", exc_info=True)
            return False
            
    def get_component_debug_level(self, component_name: str) -> str:
        """Get debug level for a specific component"""
        return self.config.component_debug_levels.get(component_name, self.config.level)
        
    def set_component_debug_level(self, component_name: str, level: str) -> None:
        """Set debug level for a specific component"""
        self.config.component_debug_levels[component_name] = level
        
    def is_debug_enabled_for(self, component_name: str) -> bool:
        """Check if debugging is enabled for a component"""
        if not self.config.enabled:
            return False
            
        component_level = self.get_component_debug_level(component_name)
        return self._is_debug_level(component_level)
        
    def _is_debug_level(self, level: str) -> bool:
        """Check if a level name is at or below DEBUG"""
        level_value = getattr(logging, level.upper(), logging.INFO)
        return level_value <= logging.DEBUG