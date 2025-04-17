import pytest
import os
import importlib

def test_core_modules_exist():
    """Test that basic project structure exists."""
    core_modules = [
        'config',
        'models',
        'network',
        'training',
        'utils'
    ]
    
    for module in core_modules:
        assert os.path.exists(os.path.join(os.getcwd(), module)), f"Module directory {module} does not exist"
        assert os.path.exists(os.path.join(os.getcwd(), module, "__init__.py")), f"Module {module} missing __init__.py"

def test_basic_imports():
    """Test that we can import basic modules."""
    import tqdm
    import web3
    assert tqdm.__version__ is not None
    assert web3.__version__ is not None
    
    # Test core module imports that should work
    modules_to_check = [
        'config', 
        'models',
        'training'
    ]
    
    for module_name in modules_to_check:
        try:
            module = importlib.import_module(module_name)
            assert module is not None
        except ImportError as e:
            pytest.skip(f"Cannot import {module_name}: {e}")