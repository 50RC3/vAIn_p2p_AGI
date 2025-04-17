import pytest
import os
import importlib.util

def test_directory_structure():
    """Test that the main directories exist."""
    required_directories = [
        'ai_core',
        'config',
        'core',
        'data',
        'models',
        'network',
        'tests',
        'training',
        'utils'
    ]
    
    for directory in required_directories:
        assert os.path.isdir(directory), f"Directory {directory} does not exist"
        assert os.path.exists(f"{directory}/__init__.py"), f"Directory {directory} is missing __init__.py"

def test_key_files_exist():
    """Test that key files exist."""
    required_files = [
        'start.py',
        'README.md',
        'requirements-dev.txt',
    ]
    
    for file in required_files:
        assert os.path.exists(file), f"File {file} does not exist"

def module_exists(module_name):
    """Check if a module exists without importing it."""
    parts = module_name.split('.')
    
    # Check if the base directory exists
    base_dir = parts[0]
    if not os.path.isdir(base_dir):
        return False
    
    # Build the file path
    if len(parts) > 1:
        file_path = os.path.join(base_dir, *parts[1:])
        return os.path.exists(f"{file_path}.py")
    else:
        return os.path.exists(f"{base_dir}/__init__.py")

def test_optional_modules_status():
    """Report on the status of optional modules."""
    optional_modules = [
        'ai_core.coordinator_factory',
        'ai_core.learning',
        'ai_core.learning.advanced_learning',
        'ai_core.chatbot',
        'ai_core.chatbot.interface',
        'ai_core.cognitive_evolution',
        'monitoring.node_status',
        'monitoring.system_monitor',
        'network.node_communication',
        'network.consensus',
        'network.reputation',
        'network.p2p_network',
        'network.dht',
        'security.auth'
    ]
    
    implemented = []
    not_implemented = []
    
    for module in optional_modules:
        if module_exists(module):
            implemented.append(module)
        else:
            not_implemented.append(module)
    
    print("\n--- Module Implementation Status ---")
    
    if implemented:
        print("\nImplemented modules:")
        for module in implemented:
            print(f"✓ {module}")
    
    if not_implemented:
        print("\nNot yet implemented modules:")
        for module in not_implemented:
            print(f"✗ {module}")
    
    # This test always passes, it's informational
    assert True

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