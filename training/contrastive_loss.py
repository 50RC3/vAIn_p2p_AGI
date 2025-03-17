import torch
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T)
        similarity_matrix = similarity_matrix / self.temperature
        
        # Create masks for positive and negative pairs
        labels_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
        positive_mask = labels_matrix.float()
        negative_mask = (~labels_matrix).float()
        
        return -(positive_mask * torch.log(similarity_matrix.exp() / 
                (similarity_matrix.exp().sum(1, keepdim=True)))).mean()
