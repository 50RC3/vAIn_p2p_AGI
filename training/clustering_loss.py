import torch
import torch.nn as nn

class ClusteringLoss(nn.Module):
    def __init__(self, num_clusters, feature_dim):
        super().__init__()
        self.num_clusters = num_clusters
        self.centroids = nn.Parameter(torch.randn(num_clusters, feature_dim))
        
    def forward(self, features):
        # Compute distances to centroids
        distances = torch.cdist(features, self.centroids)
        cluster_assignments = torch.argmin(distances, dim=1)
        
        # Compute clustering loss
        loss = distances.min(dim=1)[0].mean()
        return loss, cluster_assignments
