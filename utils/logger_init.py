import logging
import logging.handlers
import json
from pathlib import Path
from typing import Optional, Dict
import os

def init_logger(
    name: str = "vAIn",
    config_path: Optional[str] = None,
    log_dir: str = "logs",
    default_level: int = logging.INFO
) -> logging.Logger:
    """Initialize logger with optional configuration."""
    # Load config if provided
    config = load_logging_config(config_path) if config_path else {}
    
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Get logger
    logger = logging.getLogger(name)
    logger.setLevel(config.get('level', default_level))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Formatter
    formatter = logging.Formatter(
        config.get('format', '%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    )
    
    # Console Handler
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    # File Handler with rotation
    if config.get('file_logging', True):
        file_handler = logging.handlers.RotatingFileHandler(
            log_path / f"{name}.log",
            maxBytes=config.get('max_bytes', 10485760),  # 10MB default
            backupCount=config.get('backup_count', 5)
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Set debug mode if environment variable is set
    if os.getenv('VAIN_DEBUG', '').lower() == 'true':
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    return logger

def load_logging_config(config_path: str) -> Dict:
    """Load logging configuration from JSON file."""
    try:
        with open(config_path) as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading logging config: {e}")
        return {}

# Example usage:
if __name__ == "__main__":
    logger = init_logger(
        name="vAIn",
        config_path="config/logging.json",
        log_dir="logs"
    )
    logger.info("Logger initialized")
