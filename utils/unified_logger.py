import logging
import logging.handlers
import os
import time
from typing import Dict, Any, Optional
from pathlib import Path
import json
from dataclasses import dataclass, field, asdict
import sys
from datetime import datetime

@dataclass
class LogRecord:
    """Standardized log record format for all system components"""
    timestamp: float = field(default_factory=time.time)
    level: str = "INFO"
    component: str = "system"
    message: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    source_file: str = ""
    line_number: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        record = asdict(self)
        record["datetime"] = datetime.fromtimestamp(self.timestamp).isoformat()
        return record

@dataclass
class LoggerConfig:
    """Centralized configuration for all loggers"""
    level: str = "INFO"
    file_logging: bool = True
    console_logging: bool = True
    log_dir: str = "logs"
    component_name: str = "vAIn"
    max_bytes: int = 10485760  # 10MB
    backup_count: int = 5
    structured_logging: bool = False
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    log_component_metrics: bool = False
    add_trace_info: bool = True

class UnifiedLogger:
    """Unified logging system for vAIn P2P AGI"""
    
    _instances: Dict[str, "UnifiedLogger"] = {}
    
    @classmethod
    def get_logger(cls, name: str = "vAIn", config_path: Optional[str] = None) -> "UnifiedLogger":
        """Get or create a logger instance"""
        if name not in cls._instances:
            cls._instances[name] = UnifiedLogger(name, config_path)
        return cls._instances[name]
    
    def __init__(self, name: str = "vAIn", config_path: Optional[str] = None):
        """Initialize the unified logger with configuration"""
        self.name = name
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()
        
    def _load_config(self, config_path: Optional[str]) -> LoggerConfig:
        """Load logger configuration from file or environment"""
        config = LoggerConfig()
        
        # Try to load from config file if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config_data = json.load(f)
                    for key, value in config_data.items():
                        if hasattr(config, key):
                            setattr(config, key, value)
            except Exception as e:
                print(f"Error loading logger config from {config_path}: {e}")
                
        # Override with environment variables if set
        env_prefix = "VAIN_LOG_"
        for key in vars(config):
            env_var = f"{env_prefix}{key.upper()}"
            if env_var in os.environ:
                env_value = os.environ[env_var]
                # Convert booleans
                if env_value.lower() in ("true", "false"):
                    env_value = env_value.lower() == "true"
                # Convert integers
                elif env_value.isdigit():
                    env_value = int(env_value)
                setattr(config, key, env_value)
                
        return config
    
    def _setup_logger(self) -> logging.Logger:
        """Set up the logger with configured handlers"""
        # Get the logger
        logger = logging.getLogger(f"{self.name}")
        
        # Set level based on configuration
        log_level = getattr(logging, self.config.level, logging.INFO)
        logger.setLevel(log_level)
        
        # Clear any existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            
        # Create formatter
        formatter = logging.Formatter(
            self.config.format,
            datefmt=self.config.date_format
        )
        
        # Add console handler if enabled
        if self.config.console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
        # Add file handlers if enabled
        if self.config.file_logging:
            # Create log directory if it doesn't exist
            log_dir = Path(self.config.log_dir)
            log_dir.mkdir(exist_ok=True, parents=True)
            
            # Main log file with rotation
            main_log_file = log_dir / f"{self.name}.log"
            file_handler = logging.handlers.RotatingFileHandler(
                main_log_file,
                maxBytes=self.config.max_bytes,
                backupCount=self.config.backup_count
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            # Error-specific log file with rotation
            error_log_file = log_dir / f"{self.name}_errors.log"
            error_handler = logging.handlers.RotatingFileHandler(
                error_log_file,
                maxBytes=self.config.max_bytes,
                backupCount=self.config.backup_count
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatter)
            logger.addHandler(error_handler)
            
        return logger
        
    def debug(self, message: str, **context) -> None:
        """Log debug message with optional context"""
        self._log(message, logging.DEBUG, context)
        
    def info(self, message: str, **context) -> None:
        """Log info message with optional context"""
        self._log(message, logging.INFO, context)
        
    def warning(self, message: str, **context) -> None:
        """Log warning message with optional context"""
        self._log(message, logging.WARNING, context)
        
    def error(self, message: str, **context) -> None:
        """Log error message with optional context"""
        self._log(message, logging.ERROR, context)
        
    def critical(self, message: str, **context) -> None:
        """Log critical message with optional context"""
        self._log(message, logging.CRITICAL, context)
    
    def _log(self, message: str, level: int, context: Dict[str, Any]) -> None:
        """Log a message with the specified level and context"""
        # Get caller information if trace info is enabled
        source_file = ""
        line_number = 0
        
        if self.config.add_trace_info:
            try:
                # Get the caller's frame (2 levels up from this function)
                frame = sys._getframe(2)
                source_file = frame.f_code.co_filename
                line_number = frame.f_lineno
            except Exception:
                pass
        
        # Create structured record if enabled
        if self.config.structured_logging:
            record = LogRecord(
                timestamp=time.time(),
                level=logging.getLevelName(level),
                component=self.name,
                message=message,
                context=context,
                source_file=source_file,
                line_number=line_number
            )
            
            # Log as JSON
            self.logger.log(level, json.dumps(record.to_dict()))
        else:
            # Add context to message if present
            if context:
                context_str = " ".join([f"{k}={v}" for k, v in context.items()])
                message = f"{message} [{context_str}]"
                
            # Log with standard logger
            self.logger.log(level, message)

# Convenience function to get logger
def get_logger(name: str = "vAIn", config_path: Optional[str] = None) -> UnifiedLogger:
    """Get a configured logger instance"""
    return UnifiedLogger.get_logger(name, config_path)