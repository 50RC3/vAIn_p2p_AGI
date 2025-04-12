"""
Neurotransmitter System Module for vAIn P2P AGI.
Provides dynamic modulation of learning parameters.
"""

import numpy as np
from typing import Dict, Any, Optional

class NeurotransmitterSystem:
    """Adjusts neurotransmitter levels for dynamic modulation."""
    
    def __init__(self, initial_levels: Optional[Dict[str, float]] = None):
        if initial_levels is None:
            initial_levels = {
                "dopamine": 0.5,
                "serotonin": 0.5,
                "acetylcholine": 0.5
            }
        self.levels = initial_levels
    
    def adjust_levels(self, reward_signal: float, cognitive_load: float):
        """Adjust neurotransmitter levels based on external signals."""
        self.levels["dopamine"] = np.clip(self.levels["dopamine"] + 0.1 * reward_signal, 0, 1)
        self.levels["serotonin"] = np.clip(self.levels["serotonin"] - 0.05 * cognitive_load, 0, 1)
        self.levels["acetylcholine"] = np.clip(self.levels["acetylcholine"] + 0.05 * (1 - cognitive_load), 0, 1)
    
    def modulate_parameters(self) -> Dict[str, Any]:
        """Generate parameters for learning systems based on neurotransmitter levels."""
        return {
            "learning_rate": 0.001 + 0.005 * self.levels["dopamine"],
            "exploration": 0.1 + 0.2 * (1 - self.levels["serotonin"]),
            "attention_span": 5 + int(10 * self.levels["acetylcholine"])
        }
