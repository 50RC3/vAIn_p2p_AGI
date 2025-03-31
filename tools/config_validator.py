"""
Configuration Validator for vAIn_p2p_AGI

This module provides centralized validation for all configuration types
with detailed reporting of validation errors.
"""

import os
import ssl
import logging
import inspect
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Union, Callable

logger = logging.getLogger(__name__)

class ValidationResult:
    """Represents the result of a configuration validation."""
    
    def __init__(self):
        self.valid = True
        self.errors = []
        self.warnings = []
        self.field_errors = {}
        
    def add_error(self, message: str, field: Optional[str] = None) -> None:
        """Add an error message."""
        self.valid = False
        self.errors.append(message)
        if field:
            if field not in self.field_errors:
                self.field_errors[field] = []
            self.field_errors[field].append(message)
            
    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)
        
    def __bool__(self) -> bool:
        """Boolean representation of validation result."""
        return self.valid
        
    def __str__(self) -> str:
        """String representation of validation result."""
        if self.valid:
            if self.warnings:
                return f"Valid but with {len(self.warnings)} warnings"
            return "Valid"
        return f"Invalid with {len(self.errors)} errors"
        
    def get_report(self) -> str:
        """Generate a detailed report of validation results."""
        lines = []
        if not self.valid:
            lines.append("Configuration validation failed:")
            for i, error in enumerate(self.errors, 1):
                lines.append(f"  {i}. {error}")
        
        if self.warnings:
            lines.append("Warnings:")
            for i, warning in enumerate(self.warnings, 1):
                lines.append(f"  {i}. {warning}")
                
        return "\n".join(lines) if lines else "Validation passed successfully."


class ConfigValidator:
    """Validates configuration against defined schemas and constraints."""
    
    @staticmethod
    def validate_network_config(config: Dict[str, Any]) -> ValidationResult:
        """Validate network configuration."""
        result = ValidationResult()
        
        # Check required fields
        required_fields = ["node_env", "port", "database_url"]
        for field in required_fields:
            if field not in config:
                result.add_error(f"Missing required field: {field}", field)
        
        # Validate node_env
        if "node_env" in config:
            valid_envs = ['development', 'staging', 'production']
            if config["node_env"] not in valid_envs:
                result.add_error(
                    f"node_env must be one of {valid_envs}, got: {config['node_env']}", 
                    "node_env"
                )
        
        # Validate port ranges
        if "port" in config:
            port = config["port"]
            if not isinstance(port, int):
                result.add_error(f"Port must be an integer, got: {type(port).__name__}", "port")
            elif not (1024 <= port <= 65535):
                result.add_error(f"Port must be between 1024 and 65535, got: {port}", "port")
        
        # Check for available port
        if "port" in config and isinstance(config["port"], int):
            import socket
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    if sock.connect_ex(('localhost', config['port'])) == 0:
                        result.add_warning(f"Port {config['port']} is already in use")
            except Exception:
                pass
        
        # If cert_path is specified, check that it exists
        if "cert_path" in config and config["cert_path"]:
            cert_path = Path(config["cert_path"])
            if not cert_path.exists():
                result.add_error(f"Certificate file not found: {cert_path}", "cert_path")
            else:
                # Validate certificate format
                try:
                    ssl_context = ssl.SSLContext()
                    ssl_context.load_cert_chain(cert_path)
                except (ssl.SSLError, ValueError) as e:
                    result.add_error(f"Invalid certificate: {e}", "cert_path")
        
        # If key_path is specified, check that it exists
        if "key_path" in config and config["key_path"]:
            key_path = Path(config["key_path"])
            if not key_path.exists():
                result.add_error(f"Key file not found: {key_path}", "key_path")
        
        # Connection limits
        if "max_connections" in config:
            if not isinstance(config["max_connections"], int):
                result.add_error("max_connections must be an integer", "max_connections")
            elif config["max_connections"] <= 0:
                result.add_error("max_connections must be positive", "max_connections")
            elif config["max_connections"] > 1000:
                result.add_warning("max_connections is set very high (>1000)")
        
        # Peer timeout
        if "peer_timeout" in config:
            if not isinstance(config["peer_timeout"], (int, float)):
                result.add_error("peer_timeout must be a number", "peer_timeout")
            elif config["peer_timeout"] <= 0:
                result.add_error("peer_timeout must be positive", "peer_timeout")
                
        return result
    
    @staticmethod
    def validate_training_config(config: Dict[str, Any]) -> ValidationResult:
        """Validate training configuration."""
        result = ValidationResult()
        
        # Validate batch_size
        if "batch_size" in config:
            if not isinstance(config["batch_size"], int):
                result.add_error(
                    f"batch_size must be an integer, got: {type(config['batch_size']).__name__}", 
                    "batch_size"
                )
            elif config["batch_size"] <= 0:
                result.add_error(f"batch_size must be positive, got: {config['batch_size']}", 
                                "batch_size")
            elif config["batch_size"] > 1024:
                result.add_warning("Very large batch size may cause memory issues")
        
        # Validate learning_rate
        if "learning_rate" in config:
            if not isinstance(config["learning_rate"], (int, float)):
                result.add_error(
                    f"learning_rate must be a number, got: {type(config['learning_rate']).__name__}", 
                    "learning_rate"
                )
            elif not (0 < config["learning_rate"] < 1):
                result.add_error(
                    f"learning_rate must be between 0 and 1, got: {config['learning_rate']}", 
                    "learning_rate"
                )
        
        # Validate num_epochs
        if "num_epochs" in config:
            if not isinstance(config["num_epochs"], int):
                result.add_error(
                    f"num_epochs must be an integer, got: {type(config['num_epochs']).__name__}", 
                    "num_epochs"
                )
            elif config["num_epochs"] <= 0:
                result.add_error(f"num_epochs must be positive, got: {config['num_epochs']}", 
                                "num_epochs")
            elif config["num_epochs"] > 1000:
                result.add_warning("Very large number of epochs may cause training to take too long")
        
        # Validate hidden_size
        if "hidden_size" in config:
            if not isinstance(config["hidden_size"], int):
                result.add_error(
                    f"hidden_size must be an integer, got: {type(config['hidden_size']).__name__}", 
                    "hidden_size"
                )
            elif config["hidden_size"] <= 0:
                result.add_error(f"hidden_size must be positive, got: {config['hidden_size']}", 
                                "hidden_size")
                
        # Validate num_layers
        if "num_layers" in config:
            if not isinstance(config["num_layers"], int):
                result.add_error(
                    f"num_layers must be an integer, got: {type(config['num_layers']).__name__}", 
                    "num_layers"
                )
            elif config["num_layers"] <= 0:
                result.add_error(f"num_layers must be positive, got: {config['num_layers']}", 
                                "num_layers")
        
        # Validate clients_per_round and min_clients
        if "clients_per_round" in config and "min_clients" in config:
            if config["min_clients"] > config["clients_per_round"]:
                result.add_error(
                    "min_clients cannot be greater than clients_per_round", 
                    "min_clients"
                )
        
        # Check checkpoint directory if enabled
        if config.get("enable_checkpointing") and "checkpoint_dir" in config:
            checkpoint_dir = Path(config["checkpoint_dir"])
            if not checkpoint_dir.exists():
                try:
                    checkpoint_dir.mkdir(parents=True)
                except Exception as e:
                    result.add_error(
                        f"Cannot create checkpoint directory: {e}", 
                        "checkpoint_dir"
                    )
            elif not os.access(checkpoint_dir, os.W_OK):
                result.add_error(
                    f"Checkpoint directory is not writable: {checkpoint_dir}", 
                    "checkpoint_dir"
                )
                
        return result
    
    @staticmethod
    def validate_blockchain_config(config: Dict[str, Any]) -> ValidationResult:
        """Validate blockchain configuration."""
        result = ValidationResult()
        
        # Validate network
        valid_networks = ["mainnet", "polygon", "development", "localhost", "testnet"]
        if "network" in config:
            if not isinstance(config["network"], str):
                result.add_error(
                    f"network must be a string, got: {type(config['network']).__name__}", 
                    "network"
                )
            elif config["network"] not in valid_networks:
                result.add_error(
                    f"network must be one of {valid_networks}, got: {config['network']}", 
                    "network"
                )
        
        # Validate gas_price_gwei
        if "gas_price_gwei" in config:
            if not isinstance(config["gas_price_gwei"], (int, float)):
                result.add_error(
                    f"gas_price_gwei must be a number, got: {type(config['gas_price_gwei']).__name__}", 
                    "gas_price_gwei"
                )
            elif config["gas_price_gwei"] <= 0:
                result.add_error(
                    f"gas_price_gwei must be positive, got: {config['gas_price_gwei']}", 
                    "gas_price_gwei"
                )
            elif config["network"] == "mainnet" and config["gas_price_gwei"] > 200:
                result.add_warning("gas_price_gwei is very high (>200) for mainnet")
        
        # Validate gas_limit
        if "gas_limit" in config:
            if not isinstance(config["gas_limit"], int):
                result.add_error(
                    f"gas_limit must be an integer, got: {type(config['gas_limit']).__name__}", 
                    "gas_limit"
                )
            elif config["gas_limit"] < 21000:
                result.add_error(
                    f"gas_limit must be at least 21000, got: {config['gas_limit']}", 
                    "gas_limit"
                )
        
        # Only validate private_key for non-development networks
        if "network" in config and config["network"] not in ["development", "localhost"]:
            if "private_key" not in config or not config["private_key"]:
                result.add_error(
                    "private_key is required for non-development networks", 
                    "private_key"
                )
            elif len(config["private_key"]) != 64:
                result.add_error(
                    f"private_key must be 64 characters long, got: {len(config['private_key'])} characters", 
                    "private_key"
                )
        
        # Check if local node is available when using localhost
        if config.get("network") == "localhost":
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.connect(("127.0.0.1", 8545))
                sock.close()
            except Exception:
                result.add_warning("Local blockchain node not reachable at 127.0.0.1:8545")
                
        return result

    @staticmethod
    def validate_system_config(config: Dict[str, Any]) -> ValidationResult:
        """Validate system configuration."""
        result = ValidationResult()
        
        # Validate log_level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if "log_level" in config:
            if not isinstance(config["log_level"], str):
                result.add_error(
                    f"log_level must be a string, got: {type(config['log_level']).__name__}", 
                    "log_level"
                )
            elif config["log_level"].upper() not in valid_log_levels:
                result.add_error(
                    f"log_level must be one of {valid_log_levels}, got: {config['log_level']}", 
                    "log_level"
                )
        
        # Validate interaction_level
        valid_interaction_levels = ["NONE", "MINIMAL", "NORMAL", "VERBOSE"]
        if "interaction_level" in config:
            if not isinstance(config["interaction_level"], str):
                result.add_error(
                    f"interaction_level must be a string, got: {type(config['interaction_level']).__name__}", 
                    "interaction_level"
                )
            elif config["interaction_level"].upper() not in valid_interaction_levels:
                result.add_error(
                    f"interaction_level must be one of {valid_interaction_levels}, got: {config['interaction_level']}", 
                    "interaction_level"
                )
        
        # Validate metrics settings
        if "metrics" in config:
            metrics_config = config["metrics"]
            
            if "enabled" in metrics_config and not isinstance(metrics_config["enabled"], bool):
                result.add_error(
                    f"metrics.enabled must be a boolean, got: {type(metrics_config['enabled']).__name__}", 
                    "metrics.enabled"
                )
                
            if "storage_path" in metrics_config:
                path = Path(metrics_config["storage_path"])
                if not path.exists():
                    try:
                        path.mkdir(parents=True, exist_ok=True)
                    except Exception as e:
                        result.add_error(
                            f"Cannot create metrics storage path: {e}", 
                            "metrics.storage_path"
                        )
                
            if "collection_interval" in metrics_config:
                if not isinstance(metrics_config["collection_interval"], (int, float)):
                    result.add_error(
                        f"metrics.collection_interval must be a number, got: {type(metrics_config['collection_interval']).__name__}", 
                        "metrics.collection_interval"
                    )
                elif metrics_config["collection_interval"] <= 0:
                    result.add_error(
                        f"metrics.collection_interval must be positive, got: {metrics_config['collection_interval']}", 
                        "metrics.collection_interval"
                    )
                
        # Validate resource_limits
        if "resource_limits" in config:
            limits = config["resource_limits"]
            
            for limit_name in ["cpu_percent", "memory_percent", "disk_percent"]:
                if limit_name in limits:
                    if not isinstance(limits[limit_name], (int, float)):
                        result.add_error(
                            f"resource_limits.{limit_name} must be a number", 
                            f"resource_limits.{limit_name}"
                        )
                    elif not (0 < limits[limit_name] <= 100):
                        result.add_error(
                            f"resource_limits.{limit_name} must be between 0 and 100", 
                            f"resource_limits.{limit_name}"
                        )
        
        return result

    @staticmethod
    def validate_general_config(config: Dict[str, Any]) -> ValidationResult:
        """General validation that applies to any configuration."""
        result = ValidationResult()
        
        for key, value in config.items():
            if isinstance(value, dict) and not key.startswith('_'):
                # Check for empty nested dictionaries
                if not value:
                    result.add_warning(f"Empty nested configuration section: {key}")
            elif isinstance(value, str) and not value and not key.startswith('_'):
                # Check for empty strings that might be unintentional
                result.add_warning(f"Empty string value for key: {key}")
                
        return result

    @classmethod
    def get_all_validators(cls) -> Dict[str, Callable]:
        """Get all validator methods in the class."""
        validators = {}
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if name.startswith('validate_') and name != 'validate_general_config':
                config_type = name.replace('validate_', '', 1).replace('_config', '')
                validators[config_type] = method
        return validators

    @classmethod
    def validate(cls, config_type: str, config: Dict[str, Any]) -> ValidationResult:
        """Validate a configuration by type."""
        validators = cls.get_all_validators()
        
        # Apply general validation
        result = cls.validate_general_config(config)
        
        # Apply specific validation if available
        if config_type in validators:
            specific_result = validators[config_type](config)
            
            # Merge results
            result.valid = result.valid and specific_result.valid
            result.errors.extend(specific_result.errors)
            result.warnings.extend(specific_result.warnings)
            result.field_errors.update(specific_result.field_errors)
        else:
            logger.warning(f"No specific validator available for '{config_type}' configuration")
            
        return result
