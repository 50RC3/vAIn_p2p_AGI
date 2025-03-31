import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from models.genetic_algo.evolution import GeneticOptimizer

class TestGeneticOptimizer:
    @pytest.fixture
    def population(self):
        """Create a small population of simple models"""
        population_size = 4
        models = []
        
        for _ in range(population_size):
            model = nn.Sequential(
                nn.Linear(10, 5),
                nn.ReLU(),
                nn.Linear(5, 2)
            )
            models.append(model)
            
        return models
    
    @pytest.fixture
    def fitness_scores(self):
        """Create fitness scores for the population"""
        return torch.tensor([0.7, 0.3, 0.5, 0.2])
    
    @pytest.fixture
    def genetic_optimizer(self):
        """Create a GeneticOptimizer instance"""
        return GeneticOptimizer(population_size=4, mutation_rate=0.1, interactive=False)
    
    def test_initialization(self, genetic_optimizer):
        """Test that the GeneticOptimizer initializes correctly"""
        assert genetic_optimizer.population_size == 4
        assert genetic_optimizer.mutation_rate == 0.1
        assert genetic_optimizer.interactive is False
        assert genetic_optimizer._interrupt_requested is False
    
    def test_validate_inputs(self, genetic_optimizer, population, fitness_scores):
        """Test input validation"""
        # Valid inputs
        assert genetic_optimizer._validate_inputs(population, fitness_scores) is True
        
        # Invalid population size
        with pytest.raises(ValueError):
            genetic_optimizer._validate_inputs(population[:2], fitness_scores)
        
        # Invalid fitness scores length
        with pytest.raises(ValueError):
            genetic_optimizer._validate_inputs(population, fitness_scores[:-1])
        
        # Invalid population type
        with pytest.raises(TypeError):
            genetic_optimizer._validate_inputs(["not", "a", "model", "list"], fitness_scores)
    
    @pytest.mark.asyncio
    async def test_save_progress(self, genetic_optimizer, population):
        """Test saving evolution progress"""
        # Mock session with _save_progress method
        genetic_optimizer.session = MagicMock()
        genetic_optimizer.session.config.persistent_state = True
        genetic_optimizer.session._save_progress = MagicMock()
        
        await genetic_optimizer._save_progress(population)
        
        # Check that population was stored
        assert "population" in genetic_optimizer._progress
        assert genetic_optimizer._progress["population"] == population
        
        # Check that _save_progress was called
        genetic_optimizer.session._save_progress.assert_called_once()
    
    def test_cleanup(self, genetic_optimizer):
        """Test cleanup of resources"""
        # Set some state
        genetic_optimizer._interrupt_requested = True
        genetic_optimizer._progress = {"some": "data"}
        
        # Mock torch.cuda.empty_cache
        with patch("torch.cuda.empty_cache") as mock_empty_cache:
            genetic_optimizer.cleanup()
            
            # Check state was reset
            assert genetic_optimizer._interrupt_requested is False
            assert genetic_optimizer._progress == {}
            
            # Check CUDA cache was cleared
            mock_empty_cache.assert_called_once()
