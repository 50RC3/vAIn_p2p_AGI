import pytest
import torch
import torch.nn as nn
from typing import Dict, List
from unittest.mock import MagicMock, patch
import sys
import os

# Add the project root to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ai.federated_learning import FederatedLearner
from ai.exceptions import AggregationError

class TestFederatedLearner:
    @pytest.fixture
    def federated_learner(self, simple_model):
        """Create a FederatedLearner instance for testing"""
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        return FederatedLearner(
            model=simple_model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device="cpu",
            error_feedback=True,
            compression_rate=0.1
        )
    
    def test_initialization(self, federated_learner):
        """Test that the FederatedLearner initializes correctly"""
        assert federated_learner.model is not None
        assert federated_learner.optimizer is not None
        assert federated_learner.error_feedback is True
        assert federated_learner.compression_rate == 0.1
    
    def test_get_set_model_parameters(self, federated_learner):
        """Test getting and setting model parameters"""
        # Get original parameters
        orig_params = federated_learner.get_model_parameters()
        
        # Create new parameters (all ones)
        new_params = {}
        for name, param in federated_learner.model.named_parameters():
            new_params[name] = torch.ones_like(param)
        
        # Set new parameters
        federated_learner.set_model_parameters(new_params)
        
        # Get parameters again
        updated_params = federated_learner.get_model_parameters()
        
        # Check that parameters were updated
        for name, param in updated_params.items():
            assert torch.all(param == torch.ones_like(param))
        
        # Restore original parameters
        federated_learner.set_model_parameters(orig_params)
    
    def test_apply_error_feedback(self, federated_learner):
        """Test error feedback mechanism"""
        # Create sample gradients
        gradients = {}
        for name, param in federated_learner.model.named_parameters():
            gradients[name] = torch.ones_like(param) * 0.1
        
        # Apply error feedback (first time, no residuals yet)
        updated_grads = federated_learner.apply_error_feedback(gradients)
        
        # Check that gradients are unchanged on first pass
        for name, grad in updated_grads.items():
            assert torch.all(torch.isclose(grad, gradients[name]))
        
        # Set some error residuals
        federated_learner.error_residuals = {}
        for name, param in federated_learner.model.named_parameters():
            federated_learner.error_residuals[name] = torch.ones_like(param) * 0.05
        
        # Apply error feedback again
        updated_grads = federated_learner.apply_error_feedback(gradients)
        
        # Check that gradients now include residuals (0.1 + 0.05 = 0.15)
        for name, grad in updated_grads.items():
            assert torch.all(torch.isclose(grad, torch.ones_like(grad) * 0.15))
    
    @pytest.mark.asyncio
    async def test_aggregate_models(self, federated_learner):
        """Test model aggregation"""
        # Create model updates
        model_updates = []
        for i in range(3):  # Three clients
            update = {}
            for name, param in federated_learner.model.named_parameters():
                # Each client has slightly different updates
                update[name] = torch.ones_like(param) * (i + 1) * 0.1
            model_updates.append(update)
        
        # Aggregate models
        aggregated = await federated_learner.aggregate_models(model_updates)
        
        # Check result (should be average: (0.1 + 0.2 + 0.3) / 3 = 0.2)
        for name, param in aggregated.items():
            assert torch.all(torch.isclose(param, torch.ones_like(param) * 0.2))
    
    @pytest.mark.asyncio
    async def test_aggregate_models_empty(self, federated_learner):
        """Test aggregation with too few models"""
        with pytest.raises(AggregationError):
            await federated_learner.aggregate_models([])
    
    def test_validate_update(self, federated_learner):
        """Test update validation"""
        update = {}
        for name, param in federated_learner.model.named_parameters():
            update[name] = torch.ones_like(param) * 0.1
        
        # Should return True for valid update
        assert federated_learner._validate_update(update) is True
