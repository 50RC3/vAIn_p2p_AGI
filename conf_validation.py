#!/usr/bin/env python3
"""
Configuration validation and fix tool for vAIn_p2p_AGI
"""
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from tools.config_manager import ConfigManager
    from tools.config_validator import ValidationResult
except ImportError:
    print("Could not import configuration tools. Make sure you're running from project root.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("config_validation")

def validate_and_fix_configs():
    """Validate and fix all configuration files."""
    logger.info("🔍 Validating all configuration files...")
    
    # Initialize the config manager
    config_manager = ConfigManager()
    
    # Ensure default configs exist
    config_manager.create_default_configs(overwrite=False)
    
    # List available configurations
    configs = config_manager.list_configs()
    logger.info("Found %d configuration files: %s", len(configs), ', '.join(configs))
    
    # Track if any changes were made
    changes_made = False
    
    # Validate each configuration
    for config_name in configs:
        logger.info("\nValidating %s configuration...", config_name)
        config = config_manager.read_config(config_name)
        
        if not config:
            logger.warning("Config %s is empty or invalid JSON", config_name)
            continue
            
        valid, errors = config_manager.validate_config(config_name, config)
        
        if valid:
            logger.info("✅ %s: Valid", config_name)
        else:
            logger.warning("❌ %s: Invalid with %d errors", config_name, len(errors))
            for error in errors:
                logger.warning("   - %s", error)
                
            # Fix common issues
            fixed_config = fix_common_config_issues(config_name, config, errors)
            
            if fixed_config != config:
                logger.info("🔧 Fixed issues in %s", config_name)
                config_manager.write_config(config_name, fixed_config)
                changes_made = True
                
                # Revalidate after fixes
                valid, new_errors = config_manager.validate_config(config_name, fixed_config)
                if valid:
                    logger.info("✅ %s: Valid after fixes", config_name)
                else:
                    logger.warning("⚠️ %s: Still has %d errors after fixes", config_name, len(new_errors))
                    for error in new_errors:
                        logger.warning("   - %s", error)
    
    if changes_made:
        logger.info("\n✅ Fixed configuration issues")
    else:
        logger.info("\n✅ All configurations are valid. No changes needed.")
def fix_common_config_issues(config_name: str, config: Dict[str, Any], errors: List[str]) -> Dict[str, Any]:
    """Fix common configuration issues based on validation errors."""
    fixed_config = config.copy()
    
    # Network configuration fixes
    if config_name == "network":
        # Fix port issues
        if any("port" in error.lower() for error in errors):
            if "port" not in fixed_config or not isinstance(fixed_config["port"], int):
                fixed_config["port"] = 8000
            elif fixed_config["port"] < 1024 or fixed_config["port"] > 65535:
                fixed_config["port"] = 8000
        
        # Fix node_env issues
        if any("node_env" in error.lower() for error in errors):
            valid_envs = ['development', 'staging', 'production']
            if "node_env" not in fixed_config or fixed_config["node_env"] not in valid_envs:
                fixed_config["node_env"] = "development"
    
    # Blockchain configuration fixes
    elif config_name == "blockchain":
        # Fix network issues
        if any("network" in error.lower() for error in errors):
            valid_networks = ["mainnet", "polygon", "development"]
            if "network" not in fixed_config or fixed_config["network"] not in valid_networks:
                fixed_config["network"] = "development"
        
        # Fix gas price issues
        if any("gas_price" in error.lower() for error in errors):
            if "gas_price_gwei" not in fixed_config or not isinstance(fixed_config["gas_price_gwei"], int):
                fixed_config["gas_price_gwei"] = 50
    
    # Training configuration fixes
    elif config_name == "training":
        # Add common training config fixes if needed
        if "batch_size" not in fixed_config or not isinstance(fixed_config["batch_size"], int):
            fixed_config["batch_size"] = 32
    
    return fixed_config

if __name__ == "__main__":
    validate_and_fix_configs()