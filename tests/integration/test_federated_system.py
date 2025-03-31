import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from unittest.mock import MagicMock, patch
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from training.federated import FederatedLearning
from training.federated_client import FederatedClient
from core.constants import ModelStatus

class TestFederatedSystem:
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration"""
        config = MagicMock()
        config.min_clients = 2
        config.clients_per_round = 2
        config.num_rounds = 2
        config.client_fraction = 1.0
        config.learning_rate = 0.01
        config.local_epochs = 1
        config.interaction_level = "NORMAL"
        return config
    
    @pytest.fixture
    def mock_data_loader(self):
        """Create a mock data loader with a few batches"""
        data = [
            (torch.randn(8, 10), torch.randint(0, 2, (8,))),
            (torch.randn(8, 10), torch.randint(0, 2, (8,)))
        ]
        return data
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing"""
        return nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
    
    @pytest.fixture
    def federated_system(self, mock_config, simple_model, mock_data_loader):
        """Create a federated learning system with clients"""
        federated = FederatedLearning(mock_config)
        federated.global_model = simple_model
        
        # Create and register clients
        for i in range(3):
            client = FederatedClient(
                model=simple_model,
                data_loader=mock_data_loader,
                config=mock_config
            )
            # Mock the train method to return model state_dict
            client.train = MagicMock()
            client.train.return_value = simple_model.state_dict()
            
            federated.register_client(client)
            
        return federated
    
    @pytest.mark.asyncio
    async def test_federated_training_flow(self, federated_system, simple_model):
        """Test the complete federated training flow"""
        # Mock the _aggregate method to return the model
        federated_system._aggregate = MagicMock(return_value=simple_model)
        
        # Also mock _track_training_progress to avoid side effects
        federated_system._track_training_progress = MagicMock()
        
        # Run federated training
        result_model = federated_system.train()
        
        # Check that training completed
        assert result_model is not None
        assert federated_system.status == ModelStatus.VALIDATED
        
        # Check that clients were selected
        selected_clients = federated_system.select_clients()
        assert len(selected_clients) == federated_system.config.clients_per_round
        
        # Verify that _track_training_progress was called
        assert federated_system._track_training_progress.call_count > 0
    
    def test_client_update(self, federated_system, mock_data_loader):
        """Test client model update"""
        # Mock _compress_update to return the update directly
        federated_system._compress_update = MagicMock(side_effect=lambda x: x)
        
        # Run client update
        update = federated_system.client_update(mock_data_loader)
        
        # Check that update has the expected structure
        assert isinstance(update, dict)
        
        # Check that all expected keys are present (based on model structure)
        expected_keys = set(['0.weight', '0.bias', '2.weight', '2.bias'])
        assert set(update.keys()) == expected_keys
    
    @pytest.mark.asyncio
    async def test_global_intelligence_update(self, federated_system, simple_model):
        """Test updating global intelligence score"""
        # Create a list of models
        models = [simple_model for _ in range(3)]
        
        # Mock _evaluate_cognitive_abilities to return predictable scores
        federated_system._evaluate_cognitive_abilities = MagicMock(return_value=0.7)
        
        # Initial score should be 0
        assert federated_system.global_intelligence_score == 0.0
        
        # Update global intelligence
        new_score = await federated_system.update_global_intelligence(models)
        
        # Check new score
        assert new_score == 0.7
        assert federated_system.global_intelligence_score == 0.7
        
        # Check that evolution history was updated
        assert len(federated_system.evolution_history) == 1
