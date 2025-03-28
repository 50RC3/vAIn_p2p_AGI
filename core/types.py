from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from enum import Enum

class ModelState(Enum):
    """Model states"""
    READY = "ready"
    TRAINING = "training"
    ERROR = "error"
    
class NetworkError(Exception):
    """Base network error"""
    pass

@dataclass
class Task:
    """Base task definition"""
    id: str
    priority: int
    data: Dict[str, Any]
    
@dataclass
class Resources:
    """System resource metrics"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_status: str
