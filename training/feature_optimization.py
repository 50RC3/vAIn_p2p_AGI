import torch
import numpy as np
from typing import Dict, List, Tuple
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import logging

logger = logging.getLogger(__name__)

class FeatureOptimizer:
    def __init__(self, max_features: int = 100, min_importance: float = 0.01):
        self.max_features = max_features
        self.min_importance = min_importance
        self.feature_importance: Dict[str, float] = {}
        self.selected_features: List[str] = []
        self._history: List[Dict[str, float]] = []
        
    def select_features(self, data: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        """Select most important features to transmit"""
        try:
            # Convert tensors to numpy for analysis
            numpy_data = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v 
                         for k, v in data.items()}
            
            # Calculate feature importance scores
            importance_scores = {}
            for key, tensor in numpy_data.items():
                if isinstance(tensor, np.ndarray) and tensor.ndim > 1:
                    # Use mutual information for feature selection
                    selector = SelectKBest(score_func=mutual_info_classif, k='all')
                    if tensor.shape[1] > 1:  # Only if we have multiple features
                        features = tensor.reshape(tensor.shape[0], -1)
                        pseudo_target = np.mean(features, axis=1)  # Create proxy target
                        selector.fit(features, pseudo_target)
                        scores = selector.scores_
                        importance_scores[key] = np.mean(scores)
            
            # Update feature importance tracking
            self._update_importance(importance_scores)
            
            # Select features above importance threshold
            selected = {k: v for k, v in data.items() 
                       if k in self.selected_features}
            
            return selected, self.selected_features
            
        except Exception as e:
            logger.error(f"Feature selection failed: {str(e)}")
            return data, list(data.keys())
    
    def _update_importance(self, new_scores: Dict[str, float]) -> None:
        """Update feature importance scores with exponential decay"""
        decay = 0.95  # Exponential decay factor
        
        # Update importance scores
        for feature, score in new_scores.items():
            if feature in self.feature_importance:
                self.feature_importance[feature] = (
                    decay * self.feature_importance[feature] + 
                    (1 - decay) * score
                )
            else:
                self.feature_importance[feature] = score
        
        # Select features above threshold
        important_features = {k: v for k, v in self.feature_importance.items()
                            if v >= self.min_importance}
        
        # Limit to max features
        sorted_features = sorted(important_features.items(), 
                               key=lambda x: x[1], reverse=True)
        self.selected_features = [k for k, _ in sorted_features[:self.max_features]]
        
        # Track history
        self._history.append(self.feature_importance.copy())
        if len(self._history) > 100:
            self._history.pop(0)
