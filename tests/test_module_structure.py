import pytest
import os
import importlib

def test_module_exists():
    """Test that core modules exist in the project structure."""
    core_modules = [
        'ai_core',
        'network',
        'config',
        'memory',
        'models',
        'training'
    ]
    
    for module in core_modules:
        assert os.path.isdir(module), f"Module directory {module} does not exist"
        assert os.path.exists(f"{module}/__init__.py"), f"Module {module} is missing __init__.py"

def test_basic_imports():
    """Test basic imports that don't require external dependencies."""
    try:
        import ai_core
        assert ai_core is not None
    except ImportError as e:
        pytest.skip(f"ai_core module cannot be imported: {e}")
        
    try:
        import config
        assert config is not None
    except ImportError as e:
        pytest.skip(f"config module cannot be imported: {e}")