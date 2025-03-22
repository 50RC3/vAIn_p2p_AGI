import pytest
import torch
from ai_core.coordinator_factory import create_validation_coordinator
from ai_core.model_storage import ModelStorage
from ai_core.model_evaluation import ModelEvaluator

@pytest.fixture
def storage():
    return ModelStorage()  # Configure with appropriate params

@pytest.fixture
def evaluator():
    return ModelEvaluator()  # Configure with appropriate params

@pytest.fixture
def coordinator(storage, evaluator):
    return create_validation_coordinator(
        storage=storage,
        evaluator=evaluator,
        validation_threshold=0.85,
        validation_timeout=60
    )

async def test_validation_coordinator(coordinator):
    # Test validation workflow
    model = torch.nn.Linear(10, 2)  # Example model
    validators = ["node1", "node2", "node3"]
    
    result = await coordinator.coordinate_validation(model, validators)
    assert "model_hash" in result
    assert "consensus" in result
    assert "results" in result
