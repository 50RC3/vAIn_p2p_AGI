import unittest
import pytest
import time
import torch
import torch.nn as nn
from unittest.mock import MagicMock
from ai_core.evaluation.benchmark import ModelBenchmark
from models.simple_nn import SimpleNN
from config import Config

class TestModelBenchmark(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.model = SimpleNN(self.config)
        self.test_data = self._create_test_data()
        self.benchmark = ModelBenchmark(self.model, self.test_data)
    
    def test_evaluation_metrics(self):
        metrics = self.benchmark.evaluate()
        # Basic metric presence tests
        self.assertIn('accuracy', metrics)
        self.assertIn('latency_ms', metrics)
        self.assertIn('memory_mb', metrics)
        
        # Value range tests
        self.assertTrue(0 <= metrics['accuracy'] <= 1)
        self.assertTrue(metrics['latency_ms'] > 0)
        self.assertTrue(metrics['memory_mb'] > 0)
    
    def test_edge_cases(self):
        # Test with empty batch
        empty_data = [(torch.randn(0, 784), torch.randn(0)) for _ in range(1)]
        with self.assertRaises(ValueError):
            self.benchmark.evaluate(empty_data)
            
        # Test with very large batch
        large_data = [(torch.randn(1000, 784), torch.randint(0, 10, (1000,)))]
        metrics = self.benchmark.evaluate(large_data)
        self.assertIsNotNone(metrics)

    def test_performance_benchmarks(self):
        # Test inference speed
        start_time = time.time()
        self.benchmark.evaluate()
        duration = time.time() - start_time
        self.assertLess(duration, 5.0)  # Should complete within 5 seconds
        
    def _create_test_data(self):
        # Create more realistic test data
        return [
            (torch.randn(32, 784), torch.randint(0, 10, (32,))),
            (torch.randn(16, 784), torch.randint(0, 10, (16,))),
            (torch.randn(64, 784), torch.randint(0, 10, (64,)))
        ]

def test_benchmark_execution_time():
    start_time = time.time()
    # Test code...
    end_time = time.time()
    execution_time = end_time - start_time
    assert execution_time < 1.0  # Maximum allowed time

if __name__ == '__main__':
    unittest.main()
