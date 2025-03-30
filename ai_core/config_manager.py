import os
import json
import asyncio
import logging
from typing import Dict, Any, Optional, List, Set, Callable
import time

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Centralized configuration management for all AI modules.
    
    This class handles:
    - Loading and saving configuration
    - Configuration validation
    - Default values
    - Runtime configuration updates
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = ConfigManager()
        return cls._instance
    
    def __init__(self):
        if self.__class__._instance is not None:
            raise RuntimeError("ConfigManager is a singleton. Use ConfigManager.get_instance()")
            
        self.configs: Dict[str, Dict[str, Any]] = {}
        self.defaults: Dict[str, Dict[str, Any]] = {}
        self.config_dir = os.path.join("config")
        self.lock = asyncio.Lock()
        self.last_modified_times: Dict[str, float] = {}
        self.watcher_task = None
        self.callbacks: Dict[str, Set[Callable]] = {
            "config_updated": set(),
            "config_loaded": set(),
            "error": set()
        }
        
        # Create config directory if it doesn't exist
        os.makedirs(self.config_dir, exist_ok=True)
    
    async def initialize(self):
        """Initialize configuration manager"""
        try:
            logger.info("Initializing configuration manager")
            
            # Load all configurations from the config directory
            await self._load_all_configs()
            
            # Start config file watcher
            self.watcher_task = asyncio.create_task(self._watch_config_files())
            
            logger.info("Configuration manager initialized")
            return True
        
        except Exception as e:
            logger.error(f"Failed to initialize configuration manager: {e}")
            return False
    
    async def _load_all_configs(self):
        """Load all configuration files from the config directory"""
        try:
            files = os.listdir(self.config_dir)
            for file in files:
                if file.endswith(".json"):
                    config_name = os.path.splitext(file)[0]
                    await self.load_config(config_name)
        except Exception as e:
            logger.error(f"Failed to load configurations: {e}")
    
    async def load_config(self, config_name: str) -> Optional[Dict[str, Any]]:
        """Load a specific configuration file"""
        async with self.lock:
            try:
                file_path = os.path.join(self.config_dir, f"{config_name}.json")
                
                # Check if file exists
                if not os.path.isfile(file_path):
                    logger.warning(f"Configuration file {file_path} not found")
                    return None
                
                # Load configuration from file
                with open(file_path, 'r') as f:
                    config = json.load(f)
                
                # Store configuration
                self.configs[config_name] = config
                
                # Store last modified time
                self.last_modified_times[config_name] = os.path.getmtime(file_path)
                
                logger.info(f"Loaded configuration '{config_name}'")
                
                # Apply defaults if available
                if config_name in self.defaults:
                    await self._apply_defaults(config_name)
                
                # Notify callbacks
                await self._notify_callbacks("config_loaded", {
                    "config_name": config_name,
                    "config": config
                })
                
                return config
            
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in configuration file {config_name}: {e}")
                await self._notify_callbacks("error", {
                    "config_name": config_name,
                    "error": f"Invalid JSON: {e}",
                    "operation": "load"
                })
                return None
                
            except Exception as e:
                logger.error(f"Failed to load configuration '{config_name}': {e}")
                await self._notify_callbacks("error", {
                    "config_name": config_name,
                    "error": str(e),
                    "operation": "load"
                })
                return None
    
    async def save_config(self, config_name: str, config: Dict[str, Any]) -> bool:
        """Save a configuration to file"""
        async with self.lock:
            try:
                file_path = os.path.join(self.config_dir, f"{config_name}.json")
                
                # Create a backup of the existing file if it exists
                if os.path.isfile(file_path):
                    backup_path = f"{file_path}.bak"
                    os.replace(file_path, backup_path)
                
                # Write new configuration to file
                with open(file_path, 'w') as f:
                    json.dump(config, f, indent=2)
                
                # Update in-memory configuration
                self.configs[config_name] = config
                
                # Update last modified time
                self.last_modified_times[config_name] = os.path.getmtime(file_path)
                
                logger.info(f"Saved configuration '{config_name}'")
                
                # Notify callbacks
                await self._notify_callbacks("config_updated", {
                    "config_name": config_name,
                    "config": config
                })
                
                return True
            
            except Exception as e:
                logger.error(f"Failed to save configuration '{config_name}': {e}")
                await self._notify_callbacks("error", {
                    "config_name": config_name,
                    "error": str(e),
                    "operation": "save"
                })
                return False
    
    async def register_defaults(self, config_name: str, defaults: Dict[str, Any]) -> None:
        """Register default values for a configuration"""
        self.defaults[config_name] = defaults
        
        # Apply defaults if config already exists
        if config_name in self.configs:
            await self._apply_defaults(config_name)
    
    async def _apply_defaults(self, config_name: str) -> None:
        """Apply default values to a configuration"""
        if config_name not in self.configs or config_name not in self.defaults:
            return
            
        config = self.configs[config_name]
        defaults = self.defaults[config_name]
        
        # Apply defaults recursively
        updated_config = self._apply_defaults_recursive(config, defaults)
        
        # Update config if changes were made
        if updated_config != config:
            self.configs[config_name] = updated_config
            
            # Save updated config
            await self.save_config(config_name, updated_config)
    
    def _apply_defaults_recursive(self, config: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
        """Apply default values recursively to a configuration"""
        result = config.copy()
        
        for key, default_value in defaults.items():
            if key not in result:
                # Key not in config, use default
                result[key] = default_value
            elif isinstance(default_value, dict) and isinstance(result[key], dict):
                # Both values are dicts, apply defaults recursively
                result[key] = self._apply_defaults_recursive(result[key], default_value)
                
        return result
    
    async def _watch_config_files(self) -> None:
        """Watch for changes in configuration files"""
        try:
            while True:
                await asyncio.sleep(5)  # Check every 5 seconds
                
                for config_name in self.configs.keys():
                    file_path = os.path.join(self.config_dir, f"{config_name}.json")
                    
                    if not os.path.isfile(file_path):
                        continue
                        
                    # Check if file has been modified
                    mod_time = os.path.getmtime(file_path)
                    last_mod_time = self.last_modified_times.get(config_name, 0)
                    
                    if mod_time > last_mod_time:
                        logger.info(f"Configuration '{config_name}' has been modified, reloading")
                        await self.load_config(config_name)
        
        except asyncio.CancelledError:
            logger.info("Config file watcher stopped")
        except Exception as e:
            logger.error(f"Error in config file watcher: {e}")
    
    async def get_config(self, config_name: str) -> Optional[Dict[str, Any]]:
        """Get a configuration"""
        return self.configs.get(config_name)
    
    async def get_config_value(self, config_name: str, path: str, default: Any = None) -> Any:
        """Get a specific value from a configuration using dot notation path"""
        config = self.configs.get(config_name)
        if not config:
            return default
            
        # Split path into components
        components = path.split(".")
        
        # Navigate through components
        current = config
        for component in components:
            if not isinstance(current, dict) or component not in current:
                return default
            current = current[component]
            
        return current
    
    async def set_config_value(self, config_name: str, path: str, value: Any) -> bool:
        """Set a specific value in a configuration using dot notation path"""
        config = self.configs.get(config_name)
        if not config:
            return False
            
        # Split path into components
        components = path.split(".")
        
        # Navigate through components
        current = config
        for i, component in enumerate(components[:-1]):
            if not isinstance(current, dict):
                return False
                
            if component not in current:
                # Create missing components
                current[component] = {}
                
            current = current[component]
            
        # Set the value
        last_component = components[-1]
        if not isinstance(current, dict):
            return False
            
        current[last_component] = value
        
        # Save updated config
        await self.save_config(config_name, config)
        
        return True
    
    async def register_callback(self, event_type: str, callback: Callable) -> None:
        """Register a callback for configuration events"""
        if event_type in self.callbacks:
            self.callbacks[event_type].add(callback)
    
    async def _notify_callbacks(self, event_type: str, data: Dict[str, Any]) -> None:
        """Notify registered callbacks"""
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    logger.error(f"Error in configuration callback: {e}")
    
    async def shutdown(self) -> None:
        """Shut down the configuration manager"""
        try:
            logger.info("Shutting down configuration manager")
            
            # Stop config file watcher
            if self.watcher_task:
                self.watcher_task.cancel()
                try:
                    await self.watcher_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Configuration manager shut down successfully")
        
        except Exception as e:
            logger.error(f"Error shutting down configuration manager: {e}")
