from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
import torch

@dataclass
class ModelOutput:
    """
    Standard output format for model predictions and responses.
    Used for consistent handling of model outputs across the system.
    """
    logits: Optional[torch.Tensor] = None
    predictions: Optional[torch.Tensor] = None
    embeddings: Optional[torch.Tensor] = None
    text: Optional[str] = None
    confidence: float = 0.0
    latency: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    attention_weights: Optional[torch.Tensor] = None
    memory_usage: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        result = {
            "confidence": self.confidence,
            "latency": self.latency,
            "metadata": self.metadata,
        }
        
        if self.text is not None:
            result["text"] = self.text
            
        # Include tensor data as lists if present
        if self.predictions is not None:
            result["predictions"] = self.predictions.tolist()
            
        if self.logits is not None:
            result["logits"] = self.logits.tolist()
            
        if self.embeddings is not None:
            result["embeddings"] = self.embeddings.tolist()
            
        if self.attention_weights is not None:
            result["attention_weights"] = self.attention_weights.tolist()
            
        if self.memory_usage is not None:
            result["memory_usage"] = self.memory_usage
            
        return result

@dataclass
class ModelState:
    """
    Represents the current state of a model including metrics and metadata.
    """
    id: str
    score: float = 0.0
    created_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    

@dataclass
class ModelRole:
    """
    Defines the role and capabilities of a model in the system.
    """
    name: str
    capabilities: List[str] = field(default_factory=list)
    priority: int = 0
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    

def get_resource_metrics() -> Dict[str, float]:
    """
    Get current system resource metrics (CPU, memory, etc.).
    """
    import psutil
    
    return {
        "cpu_usage": psutil.cpu_percent(),
        "memory_usage": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent,
    }
