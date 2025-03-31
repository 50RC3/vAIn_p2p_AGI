#!/usr/bin/env python3
"""
Configuration Manager for vAIn_p2p_AGI

This tool provides unified management of all configuration files
across the system with validation, interactive updates, and backup.
"""

import os
import sys
import json
import logging
import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import asyncio

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from core.constants import CONFIG_DIR, BASE_DIR, InteractionLevel
except ImportError:
    # Fallback if constants aren't available
    BASE_DIR = Path(__file__).resolve().parent.parent
    CONFIG_DIR = BASE_DIR / "config"
    
    from enum import Enum
    class InteractionLevel(Enum):
        NONE = "none"
        MINIMAL = "minimal"
        NORMAL = "normal"
        VERBOSE = "verbose"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("config_manager")

class ConfigValidator:
    """Validates configurations against schemas and constraints."""
    
    @staticmethod
    def validate_network_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate network configuration."""
        errors = []
        
        # Check required fields
        required_fields = ["node_env", "port"]
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        # Validate port ranges
        if "port" in config:
            port = config["port"]
            if not isinstance(port, int) or not (1024 <= port <= 65535):
                errors.append(f"Port must be an integer between 1024 and 65535, got: {port}")
        
        # If cert_path is specified, check that it exists
        if "cert_path" in config and config["cert_path"]:
            cert_path = Path(config["cert_path"])
            if not cert_path.exists():
                errors.append(f"Certificate file not found: {cert_path}")
        
        # If key_path is specified, check that it exists
        if "key_path" in config and config["key_path"]:
            key_path = Path(config["key_path"])
            if not key_path.exists():
                errors.append(f"Key file not found: {key_path}")
                
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_training_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate training configuration."""
        errors = []
        
        # Validate batch_size
        if "batch_size" in config:
            if not isinstance(config["batch_size"], int) or config["batch_size"] <= 0:
                errors.append(f"batch_size must be a positive integer, got: {config['batch_size']}")
        
        # Validate learning_rate
        if "learning_rate" in config:
            if not isinstance(config["learning_rate"], (int, float)) or not (0 < config["learning_rate"] < 1):
                errors.append(f"learning_rate must be between 0 and 1, got: {config['learning_rate']}")
        
        # Validate num_epochs
        if "num_epochs" in config:
            if not isinstance(config["num_epochs"], int) or config["num_epochs"] <= 0:
                errors.append(f"num_epochs must be a positive integer, got: {config['num_epochs']}")
        
        # Validate hidden_size
        if "hidden_size" in config:
            if not isinstance(config["hidden_size"], int) or config["hidden_size"] <= 0:
                errors.append(f"hidden_size must be a positive integer, got: {config['hidden_size']}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_blockchain_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate blockchain configuration."""
        errors = []
        
        # Validate network
        valid_networks = ["mainnet", "polygon", "development"]
        if "network" in config and config["network"] not in valid_networks:
            errors.append(f"network must be one of {valid_networks}, got: {config['network']}")
        
        # Validate gas_price_gwei
        if "gas_price_gwei" in config:
            if not isinstance(config["gas_price_gwei"], int) or config["gas_price_gwei"] <= 0:
                errors.append(f"gas_price_gwei must be a positive integer, got: {config['gas_price_gwei']}")
        
        # Validate gas_limit
        if "gas_limit" in config:
            if not isinstance(config["gas_limit"], int) or config["gas_limit"] < 21000:
                errors.append(f"gas_limit must be at least 21000, got: {config['gas_limit']}")
        
        # Only validate private_key for non-development networks
        if "network" in config and config["network"] != "development":
            if "private_key" not in config or not config["private_key"]:
                errors.append("private_key is required for non-development networks")
            elif len(config["private_key"]) != 64:
                errors.append("private_key must be 64 characters long")
        
        return len(errors) == 0, errors

class ConfigManager:
    """Manages configuration files with validation and interactive updates."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize the configuration manager."""
        self.config_dir = config_dir or CONFIG_DIR
        self.config_dir.mkdir(exist_ok=True, parents=True)
        self.backup_dir = self.config_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True, parents=True)
        
        # Map of config types to validators
        self.validators = {
            "network": ConfigValidator.validate_network_config,
            "training": ConfigValidator.validate_training_config,
            "blockchain": ConfigValidator.validate_blockchain_config,
        }
        
        # Map of config types to interactive updaters
        self.interactive_updaters = {
            "network": self._update_network_config_interactive,
            "training": self._update_training_config_interactive,
            "blockchain": self._update_blockchain_config_interactive,
        }
    
    def list_configs(self) -> List[str]:
        """List available config files."""
        configs = []
        for file in self.config_dir.glob("*.json"):
            if file.name.startswith(".") or file.name.startswith("backup_"):
                continue
            configs.append(file.stem)
        return configs
    
    def read_config(self, config_name: str) -> Dict[str, Any]:
        """Read a configuration file."""
        config_path = self.config_dir / f"{config_name}.json"
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return {}
        
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in config file: {config_path}")
            return {}
        except Exception as e:
            logger.error(f"Error reading config file: {e}")
            return {}
    
    def write_config(self, config_name: str, config: Dict[str, Any]) -> bool:
        """Write a configuration file."""
        config_path = self.config_dir / f"{config_name}.json"
        
        # Create a backup before writing
        self._backup_config(config_name)
        
        try:
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            logger.info(f"Wrote config file: {config_path}")
            return True
        except Exception as e:
            logger.error(f"Error writing config file: {e}")
            return False
    
    def _backup_config(self, config_name: str) -> bool:
        """Create a backup of a configuration file."""
        config_path = self.config_dir / f"{config_name}.json"
        if not config_path.exists():
            return False
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"{config_name}_{timestamp}.json"
        
        try:
            shutil.copy2(config_path, backup_path)
            logger.info(f"Created backup: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return False
    
    def validate_config(self, config_name: str, config: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[str]]:
        """Validate a configuration against its schema."""
        if config is None:
            config = self.read_config(config_name)
        
        if not config:
            return False, ["Empty or missing configuration"]
        
        # Use specific validator if available
        if config_name in self.validators:
            return self.validators[config_name](config)
        
        # Default validation (just check for non-empty dict)
        return True, []
    
    def _update_network_config_interactive(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Interactively update network configuration."""
        print("\nCurrent Network Configuration:")
        for key, value in config.items():
            print(f"{key}: {value}")
        
        if input("\nUpdate network configuration? (y/n): ").lower() != 'y':
            return config
        
        updated_config = config.copy()
        
        # Environment
        valid_envs = ['development', 'staging', 'production']
        while True:
            env = input(f"Enter environment {valid_envs} [{config.get('node_env', 'development')}]: ") or config.get('node_env', 'development')
            if env in valid_envs:
                updated_config['node_env'] = env
                break
            print(f"Invalid environment. Must be one of: {valid_envs}")
        
        # Port
        while True:
            try:
                port_input = input(f"Enter port (1024-65535) [{config.get('port', 8000)}]: ")
                if not port_input:
                    updated_config['port'] = config.get('port', 8000)
                    break
                
                port = int(port_input)
                if 1024 <= port <= 65535:
                    updated_config['port'] = port
                    break
                print("Port must be between 1024 and 65535")
            except ValueError:
                print("Port must be a number")
        
        # Database URL
        updated_config['database_url'] = input(f"Enter database URL [{config.get('database_url', 'sqlite:///db.sqlite3')}]: ") or config.get('database_url', 'sqlite:///db.sqlite3')
        
        # SSL Configuration
        if input("Configure SSL? (y/n): ").lower() == 'y':
            cert_path_str = input(f"Enter SSL certificate path [{config.get('cert_path', '')}]: ").strip() or config.get('cert_path', '')
            key_path_str = input(f"Enter SSL key path [{config.get('key_path', '')}]: ").strip() or config.get('key_path', '')
            
            if cert_path_str:
                updated_config['cert_path'] = cert_path_str
                if not os.path.exists(cert_path_str):
                    print(f"Warning: Certificate file not found: {cert_path_str}")
            
            if key_path_str:
                updated_config['key_path'] = key_path_str
                if not os.path.exists(key_path_str):
                    print(f"Warning: Key file not found: {key_path_str}")
        else:
            # Remove SSL config if not needed
            updated_config.pop('cert_path', None)
            updated_config.pop('key_path', None)
        
        # Connection limits
        try:
            max_connections = input(f"Enter max connections [{config.get('max_connections', 10)}]: ")
            if max_connections:
                updated_config['max_connections'] = int(max_connections)
        except ValueError:
            print("Max connections must be a number, using previous value")
        
        try:
            peer_timeout = input(f"Enter peer timeout in seconds [{config.get('peer_timeout', 30)}]: ")
            if peer_timeout:
                updated_config['peer_timeout'] = int(peer_timeout)
        except ValueError:
            print("Peer timeout must be a number, using previous value")
        
        return updated_config
    
    def _update_training_config_interactive(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Interactively update training configuration."""
        print("\nCurrent Training Configuration:")
        for key, value in config.items():
            print(f"{key}: {value}")
        
        if input("\nUpdate training configuration? (y/n): ").lower() != 'y':
            return config
        
        updated_config = config.copy()
        
        # Batch size
        try:
            batch_size = input(f"Enter batch size [{config.get('batch_size', 32)}]: ")
            if batch_size:
                updated_config['batch_size'] = int(batch_size)
        except ValueError:
            print("Batch size must be a number, using previous value")
        
        # Learning rate
        try:
            learning_rate = input(f"Enter learning rate [{config.get('learning_rate', 0.001)}]: ")
            if learning_rate:
                updated_config['learning_rate'] = float(learning_rate)
        except ValueError:
            print("Learning rate must be a number, using previous value")
        
        # Epochs
        try:
            num_epochs = input(f"Enter number of epochs [{config.get('num_epochs', 10)}]: ")
            if num_epochs:
                updated_config['num_epochs'] = int(num_epochs)
        except ValueError:
            print("Number of epochs must be a number, using previous value")
        
        # Hidden size
        try:
            hidden_size = input(f"Enter hidden size [{config.get('hidden_size', 256)}]: ")
            if hidden_size:
                updated_config['hidden_size'] = int(hidden_size)
        except ValueError:
            print("Hidden size must be a number, using previous value")
        
        # Number of layers
        try:
            num_layers = input(f"Enter number of layers [{config.get('num_layers', 2)}]: ")
            if num_layers:
                updated_config['num_layers'] = int(num_layers)
        except ValueError:
            print("Number of layers must be a number, using previous value")
        
        # Federated learning parameters
        print("\nFederated Learning Parameters:")
        
        try:
            num_rounds = input(f"Enter number of rounds [{config.get('num_rounds', 5)}]: ")
            if num_rounds:
                updated_config['num_rounds'] = int(num_rounds)
        except ValueError:
            print("Number of rounds must be a number, using previous value")
        
        try:
            min_clients = input(f"Enter minimum clients [{config.get('min_clients', 2)}]: ")
            if min_clients:
                updated_config['min_clients'] = int(min_clients)
        except ValueError:
            print("Minimum clients must be a number, using previous value")
        
        try:
            clients_per_round = input(f"Enter clients per round [{config.get('clients_per_round', 2)}]: ")
            if clients_per_round:
                updated_config['clients_per_round'] = int(clients_per_round)
        except ValueError:
            print("Clients per round must be a number, using previous value")
        
        return updated_config
    
    def _update_blockchain_config_interactive(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Interactively update blockchain configuration."""
        print("\nCurrent Blockchain Configuration:")
        for key, value in config.items():
            # Don't display private key
            if key == "private_key":
                print(f"private_key: {'*' * 8}")
            else:
                print(f"{key}: {value}")
        
        if input("\nUpdate blockchain configuration? (y/n): ").lower() != 'y':
            return config
        
        updated_config = config.copy()
        
        # Network
        valid_networks = ["mainnet", "polygon", "development"]
        while True:
            network = input(f"Enter network {valid_networks} [{config.get('network', 'development')}]: ") or config.get('network', 'development')
            if network in valid_networks:
                updated_config['network'] = network
                break
            print(f"Invalid network. Must be one of: {valid_networks}")
        
        # Only prompt for private_key if not in development mode
        if updated_config['network'] != 'development':
            private_key = input("Enter private key (64 chars, or leave blank to keep current): ")
            if private_key:
                if len(private_key) != 64:
                    print("Warning: Private key should be 64 characters long")
                updated_config['private_key'] = private_key
            elif 'private_key' not in updated_config:
                print("Warning: No private key set for non-development network")
            
            # Infura project ID
            infura_id = input(f"Enter Infura project ID [{config.get('infura_project_id', '')}]: ")
            if infura_id:
                updated_config['infura_project_id'] = infura_id
        
        # Gas price
        try:
            gas_price = input(f"Enter gas price in gwei [{config.get('gas_price_gwei', 50)}]: ")
            if gas_price:
                updated_config['gas_price_gwei'] = int(gas_price)
        except ValueError:
            print("Gas price must be a number, using previous value")
        
        # Gas limit
        try:
            gas_limit = input(f"Enter gas limit [{config.get('gas_limit', 6000000)}]: ")
            if gas_limit:
                gas_limit_int = int(gas_limit)
                if gas_limit_int < 21000:
                    print("Gas limit must be at least 21000")
                else:
                    updated_config['gas_limit'] = gas_limit_int
        except ValueError:
            print("Gas limit must be a number, using previous value")
        
        return updated_config
    
    def update_config_interactive(self, config_name: str) -> bool:
        """Interactively update a configuration file."""
        config = self.read_config(config_name)
        
        # Use specific updater if available
        if config_name in self.interactive_updaters:
            updated_config = self.interactive_updaters[config_name](config)
        else:
            # Generic updater
            print(f"\nNo specific updater available for {config_name} configuration.")
            print("Current configuration:")
            for key, value in config.items():
                print(f"{key}: {value}")
            
            if input("\nUpdate this configuration manually? (y/n): ").lower() != 'y':
                return False
            
            print("Manual editing not implemented. Please edit the file directly.")
            return False
        
        # Validate updated config
        valid, errors = self.validate_config(config_name, updated_config)
        if not valid:
            print(f"Invalid configuration:")
            for error in errors:
                print(f"- {error}")
            
            if input("Save anyway? (y/n): ").lower() != 'y':
                return False
        
        # Write updated config
        return self.write_config(config_name, updated_config)
    
    def create_default_configs(self, overwrite: bool = False) -> None:
        """Create default configuration files if they don't exist."""
        # Default network config
        network_config = {
            "node_env": "development",
            "port": 8000,
            "database_url": "sqlite:///db.sqlite3",
            "max_connections": 10,
            "peer_timeout": 30
        }
        
        # Default training config
        training_config = {
            "batch_size": 32,
            "learning_rate": 0.001,
            "num_epochs": 10,
            "hidden_size": 256,
            "num_layers": 2,
            "num_rounds": 5,
            "min_clients": 2,
            "clients_per_round": 2
        }
        
        # Default blockchain config
        blockchain_config = {
            "network": "development",
            "gas_limit": 6000000,
            "gas_price_gwei": 50,
            "private_key": "",
            "infura_project_id": "",
            "retry_attempts": 3,
            "retry_delay": 5
        }
        
        default_configs = {
            "network": network_config,
            "training": training_config,
            "blockchain": blockchain_config,
        }
        
        for name, config in default_configs.items():
            config_path = self.config_dir / f"{name}.json"
            if overwrite or not config_path.exists():
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)
                logger.info(f"Created default config: {name}")
    
    def unified_config_update(self, interactive_level: InteractionLevel = InteractionLevel.NORMAL) -> bool:
        """Update all configurations with appropriate interactive level."""
        print("\n=============================================")
        print("    vAIn P2P AGI Configuration Manager")
        print("=============================================\n")
        
        all_success = True
        
        # Handle different interaction levels
        if interactive_level == InteractionLevel.NONE:
            print("Running in non-interactive mode, skipping configuration updates.")
            return True
        
        if interactive_level == InteractionLevel.MINIMAL:
            # Only update critical configs in minimal mode
            configs_to_update = ["network"]
        elif interactive_level == InteractionLevel.VERBOSE:
            # Update all configs in verbose mode and create defaults if missing
            self.create_default_configs(overwrite=False)
            configs_to_update = self.list_configs()
        else:
            # Default to standard configs for normal mode
            configs_to_update = ["network", "training", "blockchain"]
        
        # Ensure default configs exist
        self.create_default_configs(overwrite=False)
        
        # Update each config interactively
        for config_name in configs_to_update:
            print(f"\nUpdating {config_name} configuration:")
            success = self.update_config_interactive(config_name)
            all_success = all_success and success
            
            if not success:
                print(f"Failed to update {config_name} configuration.")
            
            # In minimal mode, stop after first config
            if interactive_level == InteractionLevel.MINIMAL and config_name == "network":
                break
        
        if all_success:
            print("\nAll configurations updated successfully.")
        else:
            print("\nSome configuration updates failed. See above for details.")
        
        return all_success

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="vAIn P2P AGI Configuration Manager")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available configurations")
    
    # Show command
    show_parser = subparsers.add_parser("show", help="Show a specific configuration")
    show_parser.add_argument("config_name", help="Name of the configuration to show")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate a specific configuration")
    validate_parser.add_argument("config_name", help="Name of the configuration to validate")
    
    # Update command
    update_parser = subparsers.add_parser("update", help="Update a specific configuration")
    update_parser.add_argument("config_name", help="Name of the configuration to update")
    
    # Update all command
    update_all_parser = subparsers.add_parser("update-all", help="Update all configurations")
    update_all_parser.add_argument("--level", choices=["none", "minimal", "normal", "verbose"],
                                  default="normal", help="Interaction level")
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create default configurations")
    create_parser.add_argument("--overwrite", action="store_true", 
                             help="Overwrite existing configurations")
    
    return parser.parse_args()

async def main():
    """Main entry point for the configuration manager."""
    args = parse_args()
    
    config_manager = ConfigManager()
    
    if args.command == "list":
        configs = config_manager.list_configs()
        print("\nAvailable configurations:")
        for config in configs:
            print(f"- {config}")
    
    elif args.command == "show":
        config = config_manager.read_config(args.config_name)
        if config:
            print(f"\n{args.config_name} configuration:")
            print(json.dumps(config, indent=2))
        else:
            print(f"Configuration '{args.config_name}' not found or empty.")
    
    elif args.command == "validate":
        valid, errors = config_manager.validate_config(args.config_name)
        if valid:
            print(f"Configuration '{args.config_name}' is valid.")
        else:
            print(f"Configuration '{args.config_name}' is invalid:")
            for error in errors:
                print(f"- {error}")
    
    elif args.command == "update":
        success = config_manager.update_config_interactive(args.config_name)
        if success:
            print(f"Configuration '{args.config_name}' updated successfully.")
        else:
            print(f"Configuration update failed or was cancelled.")
    
    elif args.command == "update-all":
        level_map = {
            "none": InteractionLevel.NONE,
            "minimal": InteractionLevel.MINIMAL,
            "normal": InteractionLevel.NORMAL,
            "verbose": InteractionLevel.VERBOSE
        }
        level = level_map.get(args.level, InteractionLevel.NORMAL)
        config_manager.unified_config_update(interactive_level=level)
    
    elif args.command == "create":
        config_manager.create_default_configs(overwrite=args.overwrite)
        print("Default configurations created.")
    
    else:
        # No command provided, show help
        print("Please specify a command. Use --help for available commands.")

if __name__ == "__main__":
    asyncio.run(main())
