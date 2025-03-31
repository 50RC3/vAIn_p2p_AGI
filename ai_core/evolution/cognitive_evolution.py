"""
Cognitive Evolution module for tracking and managing AI cognitive capabilities growth.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import time
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class EvolutionMetrics:
    """Tracks metrics related to cognitive evolution"""
    global_intelligence_score: float = 0.0
    reasoning_capability: float = 0.0
    learning_efficiency: float = 0.0
    knowledge_breadth: float = 0.0
    adaptability: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            "global_intelligence_score": self.global_intelligence_score,
            "reasoning_capability": self.reasoning_capability,
            "learning_efficiency": self.learning_efficiency,
            "knowledge_breadth": self.knowledge_breadth,
            "adaptability": self.adaptability,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvolutionMetrics':
        """Create metrics from dictionary"""
        return cls(
            global_intelligence_score=data.get("global_intelligence_score", 0.0),
            reasoning_capability=data.get("reasoning_capability", 0.0),
            learning_efficiency=data.get("learning_efficiency", 0.0),
            knowledge_breadth=data.get("knowledge_breadth", 0.0),
            adaptability=data.get("adaptability", 0.0),
            timestamp=data.get("timestamp", time.time())
        )


@dataclass
class EvolutionConfig:
    """Configuration for cognitive evolution"""
    baseline_threshold: float = 0.6  # Baseline threshold for capabilities
    evolution_rate: float = 0.05     # Rate of evolution per learning cycle
    metrics_history_size: int = 100  # Number of historical metrics to keep
    save_interval: int = 10          # Save evolution state every N updates
    save_path: str = "models/evolution"  # Path to save evolution data
    enable_meta_learning: bool = True    # Enable meta-learning capabilities


class CognitiveEvolution:
    """Manages the cognitive evolution of the system"""
    
    def __init__(self, config: Optional[EvolutionConfig] = None):
        """Initialize the cognitive evolution system"""
        self.config = config or EvolutionConfig()
        self.metrics = EvolutionMetrics()
        self.metrics_history: List[EvolutionMetrics] = []
        self.meta_learning_state: Dict[str, Any] = {}
        self.update_counter = 0
        
        # Create save directory if it doesn't exist
        save_path = Path(self.config.save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Try to load previous evolution state
        self._load_state()
    
    async def update_metrics(self, learning_metrics: Dict[str, Any]) -> EvolutionMetrics:
        """Update evolution metrics based on learning metrics"""
        try:
            # Extract relevant metrics from learning data
            self.metrics.learning_efficiency = self._calculate_learning_efficiency(learning_metrics)
            self.metrics.reasoning_capability = self._calculate_reasoning(learning_metrics)
            self.metrics.knowledge_breadth = self._calculate_knowledge_breadth(learning_metrics) 
            self.metrics.adaptability = self._calculate_adaptability(learning_metrics)
            
            # Calculate global intelligence score (weighted average)
            self.metrics.global_intelligence_score = (
                0.25 * self.metrics.learning_efficiency +
                0.30 * self.metrics.reasoning_capability +
                0.25 * self.metrics.knowledge_breadth +
                0.20 * self.metrics.adaptability
            )
            
            # Update timestamp
            self.metrics.timestamp = time.time()
            
            # Add metrics to history
            self.metrics_history.append(EvolutionMetrics(**self.metrics.to_dict()))
            if len(self.metrics_history) > self.config.metrics_history_size:
                self.metrics_history = self.metrics_history[-self.config.metrics_history_size:]
            
            # Save state periodically
            self.update_counter += 1
            if self.update_counter >= self.config.save_interval:
                self._save_state()
                self.update_counter = 0
            
            return self.metrics
            
        except Exception as e:
            logger.error(f"Error updating evolution metrics: {e}")
            return self.metrics
    
    def _calculate_learning_efficiency(self, metrics: Dict[str, Any]) -> float:
        """Calculate learning efficiency metric"""
        try:
            # Extract relevant values from learning metrics
            ss_loss = metrics.get("self_supervised", {}).get("avg_loss", 0)
            rl_reward = metrics.get("reinforcement", {}).get("avg_reward", 0)
            
            # Inverse loss (lower loss is better)
            if ss_loss > 0:
                ss_efficiency = max(0, 1 - min(ss_loss, 1))
            else:
                ss_efficiency = 0.5  # Default if no data
                
            # Normalize reward to 0-1 range
            rl_efficiency = min(1, max(0, rl_reward / 10))
            
            # Combine metrics with weights
            efficiency = 0.5 * ss_efficiency + 0.5 * rl_efficiency
            
            # Smooth with previous value for stability
            current = getattr(self.metrics, "learning_efficiency", 0.5)
            return 0.7 * current + 0.3 * efficiency
            
        except Exception as e:
            logger.warning(f"Error calculating learning efficiency: {e}")
            return getattr(self.metrics, "learning_efficiency", 0.5)
    
    def _calculate_reasoning(self, metrics: Dict[str, Any]) -> float:
        """Calculate reasoning capability metric"""
        try:
            # Extract relevant values from metrics
            rl_metrics = metrics.get("reinforcement", {})
            consistency = rl_metrics.get("consistency", 0.5)
            accuracy = rl_metrics.get("accuracy", 0.5)
            
            # Calculate reasoning score
            reasoning = 0.6 * accuracy + 0.4 * consistency
            
            # Smooth with previous value
            current = getattr(self.metrics, "reasoning_capability", 0.5)
            return 0.8 * current + 0.2 * reasoning
            
        except Exception as e:
            logger.warning(f"Error calculating reasoning capability: {e}")
            return getattr(self.metrics, "reasoning_capability", 0.5)
    
    def _calculate_knowledge_breadth(self, metrics: Dict[str, Any]) -> float:
        """Calculate knowledge breadth metric"""
        try:
            # Extract relevant values
            unsupervised = metrics.get("unsupervised", {})
            clusters = unsupervised.get("clusters", 0)
            samples = unsupervised.get("samples_processed", 0)
            
            # Calculate normalized cluster diversity
            if samples > 0:
                diversity = min(1.0, clusters / max(10, samples / 50))
            else:
                diversity = 0.5
                
            # Smooth with previous value
            current = getattr(self.metrics, "knowledge_breadth", 0.5)
            return 0.9 * current + 0.1 * diversity
            
        except Exception as e:
            logger.warning(f"Error calculating knowledge breadth: {e}")
            return getattr(self.metrics, "knowledge_breadth", 0.5)
    
    def _calculate_adaptability(self, metrics: Dict[str, Any]) -> float:
        """Calculate adaptability metric"""
        try:
            # Extract relevant values
            if not self.metrics_history:
                return 0.5
            
            # Calculate rate of improvement over last few updates
            if len(self.metrics_history) >= 3:
                recent = self.metrics_history[-3:]
                improvements = [
                    m.global_intelligence_score - p.global_intelligence_score
                    for m, p in zip(recent[1:], recent[:-1])
                ]
                avg_improvement = sum(improvements) / len(improvements)
                
                # Normalize improvement rate
                adaptability = min(1.0, max(0.0, 0.5 + avg_improvement * 5))
            else:
                adaptability = 0.5
                
            # Smooth with previous value
            current = getattr(self.metrics, "adaptability", 0.5)
            return 0.8 * current + 0.2 * adaptability
            
        except Exception as e:
            logger.warning(f"Error calculating adaptability: {e}")
            return getattr(self.metrics, "adaptability", 0.5)
    
    def get_current_metrics(self) -> EvolutionMetrics:
        """Get current evolution metrics"""
        return self.metrics
    
    def get_evolution_history(self) -> List[Dict[str, Any]]:
        """Get history of evolution metrics"""
        return [m.to_dict() for m in self.metrics_history]
    
    def _save_state(self) -> None:
        """Save evolution state to disk"""
        try:
            save_path = Path(self.config.save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            save_data = {
                "current_metrics": self.metrics.to_dict(),
                "metrics_history": [m.to_dict() for m in self.metrics_history],
                "meta_learning_state": self.meta_learning_state
            }
            
            with open(save_path / "evolution_state.json", "w") as f:
                json.dump(save_data, f)
                
            logger.info("Saved cognitive evolution state")
            
        except Exception as e:
            logger.error(f"Failed to save evolution state: {e}")
    
    def _load_state(self) -> None:
        """Load evolution state from disk"""
        try:
            save_path = Path(self.config.save_path)
            state_file = save_path / "evolution_state.json"
            
            if state_file.exists():
                with open(state_file, "r") as f:
                    save_data = json.load(f)
                
                self.metrics = EvolutionMetrics.from_dict(save_data.get("current_metrics", {}))
                
                history_data = save_data.get("metrics_history", [])
                self.metrics_history = [
                    EvolutionMetrics.from_dict(m) for m in history_data
                ]
                
                self.meta_learning_state = save_data.get("meta_learning_state", {})
                
                logger.info("Loaded cognitive evolution state")
                
        except Exception as e:
            logger.error(f"Failed to load evolution state: {e}")
