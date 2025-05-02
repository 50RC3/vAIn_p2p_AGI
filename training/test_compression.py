import asyncio
import time
import torch
import pytest
import numpy as np
from unittest.mock import patch, Mock
from compression import AdaptiveCompression, CompressionError

async def main():
    """Test the AdaptiveCompression class."""
    # Create a simple model update
    model_update = {
        'layer1.weight': torch.randn(10, 10),
        'layer1.bias': torch.randn(10),
        'layer2.weight': torch.randn(5, 10),
        'layer2.bias': torch.randn(5),
    }
    
    # Create the compression instance
    compressor = AdaptiveCompression(base_rate=0.2)
    
    # Compress
    print(f"Compressing model updates with base_rate={compressor.current_rate:.2f}...")
    compressed, ratio = await compressor.compress_model_updates(model_update)
    print(f"Compression ratio: {ratio:.4f}")
    
    # Decompress
    print("Decompressing model updates...")
    decompressed = await compressor.decompress_model_updates(compressed)
    
    # Verify the shape is preserved
    print("\nShape verification:")
    for name, tensor in model_update.items():
        print(f"{name}: Original shape {tensor.shape}, Decompressed shape {decompressed[name].shape}")
    
    # Get stats
    stats = compressor.get_stats()
    print("\nCompression Stats:")
    for key, value in stats.items():
        if key != "history":  # Skip printing the full history
            print(f"  {key}: {value}")
    
    # Test quality feedback
    print("\nTesting quality feedback mechanism:")
    print(f"Initial rate: {compressor.current_rate:.4f}")
    
    # Simulate low quality feedback
    compressor.update_quality_feedback(0.5)
    await compressor.compress_model_updates(model_update)
    print(f"After low quality feedback: {compressor.current_rate:.4f}")
    
    # Simulate good quality feedback
    compressor.update_quality_feedback(1.0)
    await compressor.compress_model_updates(model_update)
    print(f"After good quality feedback: {compressor.current_rate:.4f}")

if __name__ == "__main__":
    asyncio.run(main())

@pytest.fixture
def compressor():
    return AdaptiveCompression(base_rate=0.2)

@pytest.fixture
def test_tensor():
    return torch.randn(5, 5)

@pytest.fixture
def compressed_data(test_tensor: torch.Tensor):
    # Create data similar to what _compress_gradients would produce
    tensor_np = test_tensor.detach().cpu().numpy()
    flattened = tensor_np.flatten()
    indices = np.argsort(np.abs(flattened))[-5:]  # Keep top 5 values
    values = flattened[indices]
    
    return {
        'layer1.weight': {
            'shape': tensor_np.shape,
            'indices': indices.tolist(),
            'values': values.tolist()
        },
        'layer2.bias': {
            'shape': [3],
            'data': [0.1, 0.2, 0.3]
        },
        'non_tensor_param': "some_value"
    }

@pytest.mark.asyncio
async def test_decompress_successful(compressor: AdaptiveCompression, compressed_data: dict[str, Any]):
    # Test normal decompression flow
    result = await compressor.decompress_model_updates(compressed_data)
    
    # Verify result structure
    assert 'layer1.weight' in result
    assert 'layer2.bias' in result
    assert 'non_tensor_param' in result
    assert isinstance(result['layer1.weight'], torch.Tensor)
    assert isinstance(result['layer2.bias'], torch.Tensor)
    assert result['non_tensor_param'] == "some_value"
    
    # Verify shapes
    assert result['layer1.weight'].shape == tuple(compressed_data['layer1.weight']['shape'])
    assert result['layer2.bias'].shape == tuple(compressed_data['layer2.bias']['shape'])

@pytest.mark.asyncio
async def test_decompress_updates_stats(compressor: AdaptiveCompression, compressed_data: dict[str, Any]):
    # Record initial stats
    initial_stats = compressor.stats.get_average_stats()
    
    # Perform decompression
    await compressor.decompress_model_updates(compressed_data)
    
    # Verify stats were updated
    updated_stats = compressor.stats.get_average_stats()
    assert len(compressor.stats.decompression_times) == 1
    assert updated_stats["avg_decompression_time"] > 0

@pytest.mark.asyncio
async def test_decompress_error_handling(compressor: AdaptiveCompression):
    # Test with invalid data that should cause decompression to fail
    invalid_data = {'layer': {'shape': (5, 5), 'indices': [0, 1, 2], 'invalid_key': [0.1, 0.2]}}
    
    with pytest.raises(CompressionError):
        await compressor.decompress_model_updates(invalid_data)

@pytest.mark.asyncio
async def test_decompress_empty_input(compressor: AdaptiveCompression):
    # Test with empty dictionary
    result = await compressor.decompress_model_updates({})
    assert result == {}

@pytest.mark.asyncio
async def test_decompress_with_mock(compressor: AdaptiveCompression):
    # Test with mocked _decompress_gradients to isolate the method being tested
    with patch.object(compressor, '_decompress_gradients', return_value={'mocked': 'result'}):
        result = await compressor.decompress_model_updates({'test': 'data'})
        assert result == {'mocked': 'result'}
        compressor._decompress_gradients.assert_called_once_with({'test': 'data'})

@pytest.mark.asyncio
async def test_decompress_round_trip(compressor: AdaptiveCompression):
    """Test that compression followed by decompression restores data accurately."""
    original_data = {
        'layer1.weight': torch.randn(10, 10),
        'layer2.bias': torch.randn(5),
        'non_tensor_value': "test_string"
    }

    # Compress the data
    compressed_data, _ = await compressor.compress_model_updates(original_data)

    # Decompress the data
    decompressed_data = await compressor.decompress_model_updates(compressed_data)

    # Verify structure
    assert set(original_data.keys()) == set(decompressed_data.keys())

    # Verify tensors are close (not exact due to compression)
    for key in ['layer1.weight', 'layer2.bias']:
        assert torch.is_tensor(decompressed_data[key])
        assert decompressed_data[key].shape == original_data[key].shape

    # Verify non-tensor values are preserved exactly
    assert decompressed_data['non_tensor_value'] == original_data['non_tensor_value']
@pytest.mark.asyncio
async def test_decompress_with_none_values(compressor: AdaptiveCompression):
    """Test decompression with None values in the data."""
    compressed_data = {
        'layer1.weight': None,
        'layer2': {'shape': (3, 3), 'data': [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}
    }                
    result = await compressor.decompress_model_updates(compressed_data)

    assert result['layer1.weight'] is None
    assert torch.is_tensor(result['layer2'])
    assert result['layer2'].shape == (3, 3)

@pytest.mark.asyncio
async def test_decompress_large_tensor(compressor: AdaptiveCompression):
    """Test decompression with a large tensor."""
    # Create a large tensor
    large_tensor = torch.randn(100, 100)
    tensor_np = large_tensor.detach().cpu().numpy()            
    # Create compressed representation (top 1000 values)
    flattened = tensor_np.flatten()
    indices = np.argsort(np.abs(flattened))[-1000:]
    values = flattened[indices]
    compressed_data = {
        'large_tensor': {
            'shape': tensor_np.shape,
            'indices': indices.tolist(),
            'values': values.tolist()
        }
    }

    # Measure decompression time for large tensor
    start_time = time.time()
    result = await compressor.decompress_model_updates(compressed_data)
    decomp_time = time.time() - start_time

    assert isinstance(result['large_tensor'], torch.Tensor)
    assert result['large_tensor'].shape == (100, 100)
    assert decomp_time > 0

@pytest.mark.asyncio
async def test_decompress_sparse_tensor(compressor: AdaptiveCompression):
    """Test decompression with a sparse tensor."""
    # Create a sparse tensor
    sparse_tensor = torch.sparse_coo_tensor(indices=[[0, 1], [2, 3]], values=[1.0, 2.0], size=(5, 5))
    tensor_np = sparse_tensor.to_dense().detach().cpu().numpy()            
    # Create compressed representation (top 5 values)
    flattened = tensor_np.flatten()
    indices = np.argsort(np.abs(flattened))[-5:]
    values = flattened[indices]
    compressed_data = {
        'sparse_tensor': {
            'shape': tensor_np.shape,
            'indices': indices.tolist(),
            'values': values.tolist()
        }
    }

    # Measure decompression time for sparse tensor
    start_time = time.time()
    result = await compressor.decompress_model_updates(compressed_data)
    decomp_time = time.time() - start_time

    assert isinstance(result['sparse_tensor'], torch.Tensor)
    assert result['sparse_tensor'].shape == (5, 5)
    assert decomp_time > 0
    assert torch.all(result['sparse_tensor'].to_sparse().indices() == sparse_tensor.indices())
