import torch
from .compression import compress_gradients, decompress_gradients

class FederatedLearning:
    def __init__(self, model, data_loader, optimizer):
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer

    async def train(self):
        for data, target in self.data_loader:
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self._compute_loss(output, target)
            loss.backward()

            # Convert model parameters to a list before compression
            params = list(self.model.parameters())
            compressed_gradients = compress_gradients(
                [param.grad for param in params], 
                compression_ratio=0.1
            )
            
            # Get device from model parameters
            device = next(self.model.parameters()).device
            decompressed_gradients = decompress_gradients(compressed_gradients, device=device)

            self._apply_gradients(decompressed_gradients)
            self.optimizer.step()

    def _compute_loss(self, output, target):
        return torch.nn.functional.cross_entropy(output, target)

    def _apply_gradients(self, gradients):
        for param, grad in zip(self.model.parameters(), gradients):
            if grad is not None:
                param.grad = grad