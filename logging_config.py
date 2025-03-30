"""
Configure logging for the vAIn_p2p_AGI application.
This ensures consistent logging across all modules.
"""
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

def configure_logging(level=logging.INFO, log_dir="logs"):
    """
    Configure global logging settings.
    
    Args:
        level: The logging level for the application
        log_dir: Directory to store log files
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    
    # File handler for general logs
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "vain.log"),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    
    # Error file handler
    error_handler = RotatingFileHandler(
        os.path.join(log_dir, "errors.log"),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    
    # Add handlers to root logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)
    
    # Specific module logging levels
    logging.getLogger('ai_core.nlp.nltk_utils').setLevel(logging.DEBUG)
    logging.getLogger('ai_core.chatbot.interface').setLevel(logging.DEBUG)
    logging.getLogger('models.simple_nn').setLevel(logging.DEBUG)
    
    # Suppress verbose logs from third-party libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    logger.info("Logging configuration complete")
    return logger
