"""
Network configuration for vAIn P2P AGI system
"""
import os
import json
import logging
import secrets
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class NetworkConfig:
    """Configuration for P2P network settings"""
    
    def __init__(self):
        # Default UDP configuration
        self.udp = {
            'port': 8468,  # Default UDP port
            'broadcast': True,
            'discovery_interval': 60,  # Seconds
            'buffer_size': 4096
        }
        
        # Default TCP configuration
        self.tcp = {
            'port': 8469,  # Default TCP port
            'max_connections': 50,
            'timeout': 30  # Seconds
        }
        
        # Default DHT configuration
        self.dht = {
            'bootstrap_nodes': [],
            'port': 8470,
            'id_bits': 160,
            'bucket_size': 20,
            'refresh_interval': 3600  # 1 hour in seconds
        }
        
        # Discovery settings
        self.discovery = {
            'enabled': True,
            'interval': 60,  # Seconds
            'ping_timeout': 5  # Seconds
        }
        
        # Security settings
        self.security = {
            'encryption': True,
            'authentication': True,
            'key_rotation': 86400,  # 24 hours in seconds
            'encryption_key': secrets.token_hex(16)  # Add a default encryption key
        }
    
    @classmethod
    def from_env(cls) -> 'NetworkConfig':
        """Create network configuration from environment variables"""
        config = cls()
        
        # Try to load from environment variable first
        config_path = os.environ.get('NETWORK_CONFIG_PATH')
        
        if config_path and Path(config_path).exists():
            logger.info(f"Loading network configuration from {config_path}")
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    
                # Update configuration with loaded data
                for key, value in config_data.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
            except Exception as e:
                logger.error(f"Failed to load network configuration: {e}")
        else:
            # Load individual settings from environment variables
            try:
                if os.environ.get('UDP_PORT'):
                    config.udp['port'] = int(os.environ.get('UDP_PORT'))
                
                if os.environ.get('TCP_PORT'):
                    config.tcp['port'] = int(os.environ.get('TCP_PORT'))
                
                if os.environ.get('DISCOVERY_ENABLED') is not None:
                    config.discovery['enabled'] = os.environ.get('DISCOVERY_ENABLED').lower() == 'true'
                
                if os.environ.get('SECURITY_ENCRYPTION') is not None:
                    config.security['encryption'] = os.environ.get('SECURITY_ENCRYPTION').lower() == 'true'
            except Exception as e:
                logger.error(f"Error parsing environment variables: {e}")
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'udp': self.udp,
            'tcp': self.tcp,
            'dht': self.dht,  # Added DHT to the dictionary
            'discovery': self.discovery,
            'security': self.security
        }
    
    def save(self, path: Optional[str] = None) -> None:
        """Save configuration to file"""
        if path is None:
            # Default to config directory
            path = os.path.join(
                os.environ.get('PROJECT_ROOT', '.'), 
                'config', 
                'network_config.json'
            )
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Write config to file
            with open(path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            
            logger.info(f"Network configuration saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save network configuration: {e}")
