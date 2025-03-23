import torch
import torch.nn as nn
import logging
from typing import Tuple, Optional, Dict, List
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class ClusteringLoss(nn.Module):
    def __init__(self, num_clusters: int, feature_dim: int, 
                 temperature: float = 1.0, eps: float = 1e-6):
        super().__init__()
        
        # Validate inputs
        if not isinstance(num_clusters, int) or num_clusters < 1:
            raise ValueError("num_clusters must be a positive integer")
        if not isinstance(feature_dim, int) or feature_dim < 1:
            raise ValueError("feature_dim must be a positive integer")
            
        self.num_clusters = num_clusters
        self.feature_dim = feature_dim
        self.temperature = temperature
        self.eps = eps
        
        # Initialize centroids with Xavier/Glorot initialization
        self.centroids = nn.Parameter(
            torch.empty(num_clusters, feature_dim).normal_(std=1.0 / (feature_dim ** 0.5))
        )
        
        self.history: List[Dict] = []
        self.cluster_stats = {
            'sizes': torch.zeros(num_clusters),
            'distances': torch.zeros(num_clusters)
        }

    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """Normalize feature vectors for numerical stability"""
        return features / (torch.norm(features, dim=1, keepdim=True) + self.eps)
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute clustering loss and assignments
        
        Args:
            features: Input features of shape (batch_size, feature_dim)
            
        Returns:
            Tuple of (loss, cluster_assignments)
        """
        if features.dim() != 2 or features.size(1) != self.feature_dim:
            raise ValueError(
                f"Expected features shape (B, {self.feature_dim}), "
                f"got {tuple(features.shape)}"
            )
            
        # Move centroids to input device if needed
        if features.device != self.centroids.device:
            self.to(features.device)
            
        # Normalize features and centroids
        features_norm = self._normalize_features(features)
        centroids_norm = self._normalize_features(self.centroids)
        
        # Compute cosine distances efficiently
        distances = 1 - torch.mm(features_norm, centroids_norm.t())
        
        # Apply temperature scaling for better gradients
        distances = distances / self.temperature
        
        # Get cluster assignments
        cluster_assignments = torch.argmin(distances, dim=1)
        
        # Compute loss with numerical stability
        min_distances = distances.min(dim=1)[0]
        loss = torch.mean(min_distances)
        
        # Check for potential issues
        if torch.isnan(loss) or torch.isinf(loss):
            logging.warning("Detected NaN/Inf in clustering loss")
            loss = torch.zeros_like(loss, requires_grad=True)
        
        # Update cluster statistics
        for i in range(self.num_clusters):
            mask = cluster_assignments == i
            self.cluster_stats['sizes'][i] = mask.sum().item()
            if mask.any():
                self.cluster_stats['distances'][i] = distances[mask, i].mean().item()
        
        # Track history
        self.history.append({
            'loss': loss.item(),
            'cluster_sizes': self.cluster_stats['sizes'].tolist(),
            'avg_distances': self.cluster_stats['distances'].tolist()
        })
        
        return loss, cluster_assignments
    
    def get_cluster_centers(self) -> torch.Tensor:
        """Return normalized cluster centroids"""
        with torch.no_grad():
            return self._normalize_features(self.centroids)
            
    def to(self, device) -> 'ClusteringLoss':
        """Override to() to handle device transfers"""
        super().to(device)
        self.centroids = nn.Parameter(self.centroids.to(device))
        return self

    def save_state(self, path: str) -> None:
        """Save model state and training history"""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'centroids': self.centroids.data.cpu().numpy().tolist(),
            'history': self.history,
            'cluster_stats': {
                k: v.cpu().numpy().tolist() for k, v in self.cluster_stats.items()
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(state, f)
            
    def load_state(self, path: str) -> None:
        """Load model state and training history"""
        with open(path, 'r') as f:
            state = json.load(f)
            
        self.centroids = nn.Parameter(torch.tensor(state['centroids']))
        self.history = state['history']
        self.cluster_stats = {
            k: torch.tensor(v) for k, v in state['cluster_stats'].items()
        }
