import logging
import sys
import psutil
import torch
from typing import Dict, Any, Optional
from dataclasses import dataclass
import time
from pathlib import Path

@dataclass
class DebugMetrics:
    memory_usage: float
    cpu_usage: float 
    gpu_usage: Optional[float]
    error_count: int
    last_error: Optional[str]
    latency_ms: float

class DebugManager:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self._setup_logging()
        self.error_counts = {}
        self.start_time = time.time()
        
    def _setup_logging(self):
        debug_log = self.log_dir / "debug.log"
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(debug_log),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
    def get_metrics(self) -> DebugMetrics:
        """Get current system metrics"""
        gpu_usage = None
        if torch.cuda.is_available():
            gpu_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            
        return DebugMetrics(
            memory_usage=psutil.virtual_memory().percent,
            cpu_usage=psutil.cpu_percent(),
            gpu_usage=gpu_usage,
            error_count=sum(self.error_counts.values()),
            last_error=next(iter(self.error_counts.keys()), None),
            latency_ms=0
        )
        
    def track_error(self, error: Exception, context: Dict[str, Any] = None):
        """Track error with context for debugging"""
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        logger = logging.getLogger("debug")
        logger.error(f"Error: {str(error)}", exc_info=True, extra={
            "context": context,
            "metrics": self.get_metrics().__dict__
        })

debug_manager = DebugManager()  # Global instance
