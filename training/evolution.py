import time
import logging
from typing import Dict, List, Any, Optional
import json
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class EvolutionTracker:
    """Tracks evolutionary progress of agent capabilities and cognitive development"""
    
    def __init__(self, save_directory: str = "./evolution_data"):
        self.evolution_history = []
        self.generations = 0
        self.started_at = time.time()
        self.save_dir = Path(save_directory)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = {
            "cognitive_complexity": 0.0,
            "adaptability": 0.0,
            "knowledge_breadth": 0.0,
            "learning_rate": 0.0,
            "problem_solving": 0.0
        }
        logger.info("Evolution tracker initialized")
        
    def record_generation(self, metrics: Dict[str, Any]):
        """Record a new generation of evolution
        
        Args:
            metrics: Dictionary of evolution metrics
        """
        self.generations += 1
        
        # Update metrics with running average
        for key, value in metrics.items():
            if key in self.metrics:
                # Weighted average favoring recent values
                self.metrics[key] = 0.7 * value + 0.3 * self.metrics[key]
                
        # Create record
        record = {
            "generation": self.generations,
            "timestamp": time.time(),
            "elapsed_time": time.time() - self.started_at,
            "metrics": dict(self.metrics)
        }
        
        self.evolution_history.append(record)
        
        # Periodically save to disk
        if self.generations % 10 == 0:
            self.save_history()
            
        return record
        
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get a summary of evolutionary progress
        
        Returns:
            Dict with progress metrics
        """
        if not self.evolution_history:
            return {"generations": 0, "progress": 0.0}
            
        first_gen = self.evolution_history[0]["metrics"]
        current_gen = self.metrics
        
        progress = {}
        total_improvement = 0.0
        num_metrics = 0
        
        for key in current_gen:
            if key in first_gen:
                # Calculate improvement percentage
                if first_gen[key] > 0:
                    improvement = (current_gen[key] - first_gen[key]) / first_gen[key]
                else:
                    improvement = current_gen[key]
                    
                progress[key] = improvement
                total_improvement += improvement
                num_metrics += 1
                
        avg_improvement = total_improvement / max(1, num_metrics)
        
        return {
            "generations": self.generations,
            "elapsed_time": time.time() - self.started_at,
            "avg_improvement": avg_improvement,
            "metrics": dict(self.metrics),
            "progress_by_metric": progress
        }
        
    def save_history(self) -> bool:
        """Save evolution history to disk
        
        Returns:
            bool: Success status
        """
        try:
            file_path = self.save_dir / f"evolution_history_{int(time.time())}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.evolution_history, f, indent=2)
            logger.debug("Saved evolution history to %s", file_path)
            return True
        except Exception as e:
            logger.error("Failed to save evolution history: %s", e)
            return False