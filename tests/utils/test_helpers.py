import pytest
import torch
import torch.nn as nn
import os
import tempfile
from utils.helpers import save_checkpoint, load_checkpoint

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.linear(x)

@pytest.fixture
def model():
    return SimpleModel()

@pytest.fixture 
def optimizer(model):
    return torch.optim.Adam(model.parameters())

@pytest.fixture
def temp_checkpoint():
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        temp_path = f.name
    yield temp_path
    if os.path.exists(temp_path):
        os.remove(temp_path)

def test_save_load_checkpoint(model, optimizer, temp_checkpoint):
    epoch = 5
    save_checkpoint(model, optimizer, epoch, temp_checkpoint)
    
    # Create new model and optimizer for loading
    new_model = SimpleModel()
    new_optimizer = torch.optim.Adam(new_model.parameters())
    
    checkpoint = load_checkpoint(temp_checkpoint, new_model, new_optimizer)
    
    assert checkpoint['epoch'] == epoch
    assert torch.equal(model.linear.weight, new_model.linear.weight)
    assert torch.equal(model.linear.bias, new_model.linear.bias)
