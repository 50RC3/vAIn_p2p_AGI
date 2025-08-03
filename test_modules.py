#!/usr/bin/env python3
"""
Module testing utility for vAIn P2P AGI system.
Tests import of all core modules and reports any issues.
"""

import sys
import traceback
import importlib
from pathlib import Path
from typing import List, Dict, Any


def test_module_import(module_name: str) -> Dict[str, Any]:
    """Test importing a single module and return results."""
    result = {
        'module': module_name,
        'success': False,
        'error': None,
        'error_type': None
    }
    
    try:
        importlib.import_module(module_name)
        result['success'] = True
        print(f"✓ {module_name}")
    except Exception as e:
        result['error'] = str(e)
        result['error_type'] = type(e).__name__
        print(f"✗ {module_name} - {result['error_type']}: {result['error']}")
    
    return result


def get_core_modules() -> List[str]:
    """Get list of core modules to test."""
    return [
        # Core modules
        'config',
        'config.blockchain_config',
        'config.network_config',
        'core',
        'core.constants',
        'core.cross_domain_transfer',
        'core.metrics_collector',
        'core.interactive_utils',
        
        # AI modules
        'ai_core',
        'models',
        
        # Network modules
        'network',
        
        # Utilities
        'utils',
        'utils.metrics',
        'utils.debug_launcher',
        'utils.debug_port_manager',
        
        # UI modules
        'ui',
        'ui.terminal_ui',
        'ui.interface_manager',
        
        # Memory modules
        'memory',
        'memory.memory_manager',
    ]


def test_all_modules() -> Dict[str, Any]:
    """Test all core modules and return summary."""
    modules = get_core_modules()
    results = []
    
    print("Testing module imports...")
    print("=" * 50)
    
    for module in modules:
        result = test_module_import(module)
        results.append(result)
    
    print("\n" + "=" * 50)
    
    # Summary
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    summary = {
        'total': len(results),
        'successful': len(successful),
        'failed': len(failed),
        'success_rate': len(successful) / len(results) * 100,
        'results': results,
        'failed_modules': failed
    }
    
    print(f"Summary: {summary['successful']}/{summary['total']} modules imported successfully ({summary['success_rate']:.1f}%)")
    
    if failed:
        print(f"\nFailed modules ({len(failed)}):")
        for result in failed:
            print(f"  - {result['module']}: {result['error_type']}")
    
    return summary


def test_entry_points() -> Dict[str, Any]:
    """Test main entry points."""
    print("\nTesting entry points...")
    print("=" * 50)
    
    entry_points = {
        'main.py': 'main',
        'start.py': 'start', 
        'debug.py': 'debug',
        'config.py': 'config'
    }
    
    results = {}
    
    for script, module in entry_points.items():
        try:
            spec = importlib.util.spec_from_file_location(module, script)
            if spec and spec.loader:
                # Don't actually execute, just test if it can be loaded
                print(f"✓ {script} - loadable")
                results[script] = {'success': True, 'error': None}
            else:
                print(f"✗ {script} - not loadable")
                results[script] = {'success': False, 'error': 'Could not create spec'}
        except Exception as e:
            print(f"✗ {script} - {type(e).__name__}: {e}")
            results[script] = {'success': False, 'error': str(e)}
    
    return results


def check_optional_dependencies():
    """Check status of optional dependencies."""
    print("\nChecking optional dependencies...")
    print("=" * 50)
    
    optional_deps = {
        'torch': 'PyTorch for deep learning',
        'web3': 'Web3 for blockchain functionality', 
        'psutil': 'System monitoring',
        'numpy': 'Numerical computing',
        'tqdm': 'Progress bars',
        'debugpy': 'Debug adapter protocol'
    }
    
    available = {}
    
    for dep, description in optional_deps.items():
        try:
            importlib.import_module(dep)
            print(f"✓ {dep} - {description}")
            available[dep] = True
        except ImportError:
            print(f"✗ {dep} - {description} (optional)")
            available[dep] = False
    
    return available


def main():
    """Main testing function."""
    print("vAIn P2P AGI Module Testing Utility")
    print("=" * 50)
    
    # Test core modules
    module_results = test_all_modules()
    
    # Test entry points
    entry_results = test_entry_points()
    
    # Check optional dependencies
    dep_status = check_optional_dependencies()
    
    # Final summary
    print("\n" + "=" * 50)
    print("FINAL SUMMARY")
    print("=" * 50)
    
    print(f"Core modules: {module_results['successful']}/{module_results['total']} working")
    working_entries = sum(1 for r in entry_results.values() if r['success'])
    print(f"Entry points: {working_entries}/{len(entry_results)} working")
    available_deps = sum(1 for v in dep_status.values() if v)
    print(f"Optional dependencies: {available_deps}/{len(dep_status)} available")
    
    if module_results['success_rate'] >= 80:
        print("\n🎉 System is mostly functional!")
        return 0
    else:
        print("\n⚠️  System has significant issues that need attention.")
        return 1


if __name__ == "__main__":
    import importlib.util
    sys.exit(main())