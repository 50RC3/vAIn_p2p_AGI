"""
Evolution tracker for monitoring cognitive and learning progress across agents.
"""
from typing import Dict, List, Any, Optional
import numpy as np
import logging
import time
from datetime import datetime
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class EvolutionTracker:
    """
    Tracks the evolution and learning progress of the multi-agent system.
    Records metrics, cognitive improvements and collective intelligence measurements.
    """
    
    def __init__(self, save_dir: Optional[str] = None):
        """
        Initialize the evolution tracker.
        
        Args:
            save_dir: Optional directory path to save evolution data
        """
        self.metrics_history: List[Dict[str, Any]] = []
        self.evolution_stages: List[Dict[str, Any]] = []
        self.evolution_start_time = time.time()
        self.last_checkpoint_time = time.time()
        
        # Metrics tracking
        self.collective_intelligence: float = 0.0
        self.cognitive_complexity: float = 0.0
        self.knowledge_breadth: float = 0.0
        self.adaptation_rate: float = 0.0
        
        # Directory for saving evolution data
        if save_dir:
            self.save_dir = Path(save_dir)
        else:
            self.save_dir = Path("./data/evolution")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Evolution tracker initialized")
    
    def track_evolution_step(self, metrics: Dict[str, Any]) -> None:
        """
        Track a single evolution step.
        
        Args:
            metrics: Dictionary of metrics from this evolution step
        """
        timestamp = time.time()
        metrics["timestamp"] = timestamp
        
        # Add to history
        self.metrics_history.append(metrics)
        
        # Update current metrics
        if "collective_intelligence" in metrics:
            self.collective_intelligence = metrics["collective_intelligence"]
        
        if "cognitive_complexity" in metrics:
            self.cognitive_complexity = metrics["cognitive_complexity"]
            
        # Enforce history size limit to prevent memory bloat
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
            
        # Save checkpoint periodically
        if timestamp - self.last_checkpoint_time > 3600:  # Every hour
            self._save_checkpoint()
            self.last_checkpoint_time = timestamp
    
    def mark_evolution_milestone(self, stage_name: str, description: str, 
                              metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Mark a significant milestone in the system's evolution.
        
        Args:
            stage_name: Name of the evolution stage
            description: Description of the milestone
            metrics: Optional metrics at this milestone
        """
        milestone = {
            "stage": stage_name,
            "description": description,
            "timestamp": time.time(),
            "elapsed_days": (time.time() - self.evolution_start_time) / (60 * 60 * 24),
            "metrics": metrics or {}
        }
        
        self.evolution_stages.append(milestone)
        logger.info(f"Evolution milestone reached: {stage_name}")
        
        # Always save milestones immediately
        self._save_milestone(milestone)
    
    def calculate_growth_rate(self, metric_name: str, window_size: int = 10) -> float:
        """
        Calculate growth rate for a specific metric.
        
        Args:
            metric_name: Name of the metric to calculate growth for
            window_size: Window size for calculating growth
            
        Returns:
            float: Growth rate as a percentage
        """
        if len(self.metrics_history) < window_size:
            return 0.0
        
        recent_metrics = self.metrics_history[-window_size:]
        if not all(metric_name in metrics for metrics in recent_metrics):
            return 0.0
        
        start_value = recent_metrics[0].get(metric_name, 0)
        end_value = recent_metrics[-1].get(metric_name, 0)
        
        if start_value == 0:
            return 0.0
            
        growth_rate = (end_value - start_value) / start_value
        return growth_rate * 100  # as percentage
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current evolution state.
        
        Returns:
            Dict[str, Any]: Summary of the evolution state
        """
        return {
            "collective_intelligence": self.collective_intelligence,
            "cognitive_complexity": self.cognitive_complexity,
            "knowledge_breadth": self.knowledge_breadth,
            "adaptation_rate": self.adaptation_rate,
            "evolution_time_days": (time.time() - self.evolution_start_time) / (60 * 60 * 24),
            "milestone_count": len(self.evolution_stages),
            "last_milestone": self.evolution_stages[-1]["stage"] if self.evolution_stages else None
        }
    
    def _save_checkpoint(self) -> None:
        """Save a checkpoint of the current evolution data"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = self.save_dir / f"evolution_checkpoint_{timestamp}.json"
            
            data = {
                "timestamp": time.time(),
                "metrics": self.metrics_history[-100:],  # Last 100 metrics entries
                "collective_intelligence": self.collective_intelligence,
                "cognitive_complexity": self.cognitive_complexity,
                "knowledge_breadth": self.knowledge_breadth,
                "adaptation_rate": self.adaptation_rate,
                "milestones": self.evolution_stages
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
                
            logger.debug(f"Saved evolution checkpoint to {file_path}")
        except Exception as e:
            logger.error(f"Error saving evolution checkpoint: {e}")
    
    def _save_milestone(self, milestone: Dict[str, Any]) -> None:
        """Save a single milestone to a dedicated file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stage_name = milestone["stage"].replace(" ", "_").lower()
            file_path = self.save_dir / f"milestone_{stage_name}_{timestamp}.json"
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(milestone, f, indent=2)
                
            logger.debug(f"Saved evolution milestone to {file_path}")
        except Exception as e:
            logger.error(f"Error saving evolution milestone: {e}")