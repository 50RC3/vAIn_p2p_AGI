from dataclasses import dataclass, field
from typing import Dict, List
from statistics import mean

@dataclass
class ModelMetrics:
    """Tracks training and aggregation metrics"""
    
    # Aggregation metrics
    aggregation_times: List[float] = field(default_factory=list)
    update_counts: List[int] = field(default_factory=list)
    memory_usage: List[int] = field(default_factory=list)
    
    # Training metrics
    training_losses: List[float] = field(default_factory=list)
    epoch_counts: List[int] = field(default_factory=list)
    batch_counts: List[int] = field(default_factory=list)

    def update_aggregation_stats(self, num_updates: int, time_taken: float, memory_used: int):
        """Record aggregation performance metrics"""
        self.aggregation_times.append(time_taken)
        self.update_counts.append(num_updates)
        self.memory_usage.append(memory_used)

    def update_training_stats(self, avg_loss: float, num_epochs: int, num_batches: int):
        """Record training performance metrics"""
        self.training_losses.append(avg_loss)
        self.epoch_counts.append(num_epochs)
        self.batch_counts.append(num_batches)

    def get_summary(self) -> Dict:
        """
        Return summary statistics.
        Returns:
            Dict containing:
            - avg_aggregation_time (float): Average time for aggregation.
            - avg_updates_per_round (float): Average number of updates per round.
            - avg_memory_usage (float): Average memory usage.
            - avg_training_loss (float): Average training loss.
            - total_epochs (int): Total number of epochs.
            - total_batches (int): Total number of batches.
        """
        return {
            "avg_aggregation_time": mean(self.aggregation_times) if self.aggregation_times else 0,
            "avg_updates_per_round": mean(self.update_counts) if self.update_counts else 0,
            "avg_memory_usage": mean(self.memory_usage) if self.memory_usage else 0,
            "avg_training_loss": mean(self.training_losses) if self.training_losses else 0,
            "total_epochs": sum(self.epoch_counts),
            "total_batches": sum(self.batch_counts)
        }
