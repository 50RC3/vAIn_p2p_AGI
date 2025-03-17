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
