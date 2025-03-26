import torch
from .simple_nn import SimpleNN, AdvancedNN
from .hybrid_memory_system import HybridMemorySystem
from .reptile_meta.reptile_model import ReptileModel
from .dnc.dnc_controller import DNCController
from .vain_transformer.transformer import vAInTransformer
from dataclasses import dataclass
from typing import Dict, Any, Optional
import psutil

@dataclass
class ModelOutput:
    output: torch.Tensor
    attention: Optional[torch.Tensor] = None
    memory_state: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass 
class ResourceMetrics:
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float] = None

@dataclass 
class CognitiveState:
    """Track cognitive system state"""
    current_focus: torch.Tensor
    memory_state: Dict[str, Any]
    attention_patterns: Optional[torch.Tensor] = None
    metacognitive_state: Optional[Dict] = None

@dataclass
class EvolutionMetrics:
    """Metrics for cognitive evolution"""
    learning_rate: float
    adaptation_score: float
    cognitive_complexity: float
    response_quality: float
    
@dataclass
class ModelState:
    model_hash: str
    version: int
    timestamp: float
    metrics: ResourceMetrics

@dataclass
class ModelRole:
    """Define model roles in unified system"""
    MEMORY = "memory"      # Memory-focused models (DNC, HMS)
    PROCESSING = "processing"  # Processing models (Transformer)
    META = "meta"         # Meta-learning models (Reptile)

@dataclass
class UnifiedModelConfig:
    """Configuration for unified model system"""
    role: str
    memory_share: float = 0.3  # Portion of memory allocated
    priority: int = 1
    can_offload: bool = True

__all__ = [
    'SimpleNN', 'AdvancedNN', 'HybridMemorySystem', 
    'ReptileModel', 'DNCController', 'vAInTransformer',
    'ModelOutput', 'ModelRole', 'UnifiedModelConfig',
    'CognitiveState', 'EvolutionMetrics'
]

# Add shared functionality
def get_resource_metrics() -> ResourceMetrics:
    metrics = ResourceMetrics(
        cpu_usage=psutil.cpu_percent(),
        memory_usage=psutil.virtual_memory().percent
    )
    if torch.cuda.is_available():
        metrics.gpu_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
    return metrics
