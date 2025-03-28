import torch
import numpy as np
from typing import Dict

def compute_accuracy(model, data_loader, device='cuda') -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    return correct / total

def compute_loss(model, data_loader, criterion=None, device='cuda') -> float:
    """
    Compute average loss for a model on the provided data loader
    
    Args:
        model: The PyTorch model to evaluate
        data_loader: DataLoader containing the evaluation data
        criterion: Loss function to use (defaults to CrossEntropyLoss if None)
        device: Device to run the computation on
        
    Returns:
        Average loss value across all batches
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    # Default to CrossEntropyLoss if no criterion is provided
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0
