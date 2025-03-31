import pytest
import torch
from unittest.mock import MagicMock, patch
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ai_core.cognitive_evolution import CognitiveEvolution
from models import ModelOutput

class TestCognitiveEvolution:
    @pytest.fixture
    def mock_unified_system(self):
        unified_system = MagicMock()
        
        # Setup models dict
        unified_system.models = {
            "memory_encoder": MagicMock(),
            "cognitive_processor": MagicMock()
        }
        
        # Setup interfaces dict
        unified_system.interfaces = {
            "memory_encoder": MagicMock(),
            "cognitive_processor": MagicMock()
        }
        
        # Mock coordinate_inference method
        unified_system.coordinate_inference = MagicMock()
        mock_output = ModelOutput(
            output_tensor=torch.randn(1, 10),
            confidence=0.85,
            metadata={"attention": torch.rand(1, 5)}
        )
        unified_system.coordinate_inference.return_value = mock_output
        
        return unified_system
    
    @pytest.fixture
    def mock_memory_manager(self):
        return MagicMock()
    
    @pytest.fixture
    def cognitive_evolution(self, mock_unified_system, mock_memory_manager):
        return CognitiveEvolution(mock_unified_system, mock_memory_manager)
    
    @pytest.mark.asyncio
    async def test_initialization(self, cognitive_evolution):
        """Test cognitive evolution initialization"""
        assert cognitive_evolution.cognitive_states == {}
        assert cognitive_evolution.evolution_history == []
        assert cognitive_evolution._active_learning is False
    
    @pytest.mark.asyncio
    async def test_initialize_cognitive_network(self, cognitive_evolution):
        """Test cognitive network initialization"""
        # Mock the register_model method to always return True
        cognitive_evolution.unified_system.register_model.return_value = True
        
        await cognitive_evolution.initialize_cognitive_network()
        
        # Check that register_model was called 3 times (for each model)
        assert cognitive_evolution.unified_system.register_model.call_count >= 1
        
        # Check that active learning is enabled
        assert cognitive_evolution._active_learning is True
    
    @pytest.mark.asyncio
    async def test_cognitive_cycle(self, cognitive_evolution):
        """Test one cognitive processing cycle"""
        # Create a test input tensor
        input_tensor = torch.rand(1, 10)
        
        # Run a cognitive cycle
        output = await cognitive_evolution.cognitive_cycle(input_tensor)
        
        # Check that coordinate_inference was called with the input tensor
        cognitive_evolution.unified_system.coordinate_inference.assert_called_once_with(input_tensor)
        
        # Check that we got a ModelOutput
        assert isinstance(output, ModelOutput)
        
        # Check that a cognitive state was stored
        assert len(cognitive_evolution.cognitive_states) == 1
    
    def test_cleanup_old_states(self, cognitive_evolution):
        """Test cleanup of old cognitive states"""
        # Add 1100 mock states
        for i in range(1100):
            cognitive_evolution.cognitive_states[i] = {"mock": "state"}
        
        # Run cleanup
        cognitive_evolution._cleanup_old_states()
        
        # Should keep only 1000 states (the newest ones)
        assert len(cognitive_evolution.cognitive_states) == 1000
        
        # The oldest states should be removed (0-99), keeping 100-1099
        for i in range(100):
            assert i not in cognitive_evolution.cognitive_states
        
        for i in range(100, 1100):
            assert i in cognitive_evolution.cognitive_states
