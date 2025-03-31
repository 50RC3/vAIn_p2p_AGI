#!/usr/bin/env python3
"""
Verification utility for vAIn P2P AGI dependencies.
Provides simplified interface to dependency_checker.py.
"""

import sys
import logging
import argparse
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

# Import functions from dependency_checker
from utils.dependency_checker import (
    check_dependencies,
    check_module_dependencies,
    install_dependencies,
    check_system_dependencies,
    generate_requirements_txt,
    save_dependencies_config,
    load_dependencies_config
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Check if colorama is available for colored output
try:
    from colorama import init, Fore, Style
    init()  # Initialize colorama
    COLOR_AVAILABLE = True
except ImportError:
    COLOR_AVAILABLE = False
    logger.debug("Colorama not found. Running without colored output.")


def print_colored(message: str, color=None, bold=False) -> None:
    """Print colored output if colorama is available."""
    if not COLOR_AVAILABLE or color is None:
        print(message)
        return
    
    color_code = getattr(Fore, color.upper(), "")
    style_code = Style.BRIGHT if bold else ""
    print(f"{style_code}{color_code}{message}{Style.RESET_ALL}")


def print_dependency_status(missing: List[Dict[str, Any]], all_deps: bool = False) -> None:
    """Format and print dependency status in a user-friendly way."""
    core_issues = [item for item in missing if item.get('category') == 'core']
    optional_issues = [item for item in missing if item.get('category') != 'core']
    
    if not missing:
        print_colored("✓ All dependencies are installed and up to date!", "GREEN", bold=True)
        return
    
    # Print core dependency issues
    if core_issues:
        print_colored("\n❌ CORE DEPENDENCY ISSUES:", "RED", bold=True)
        for item in core_issues:
            if item['status'] == 'missing':
                print_colored(f"  • Missing: {item['name']} {item['required'] or ''}", "YELLOW")
            elif item['status'] == 'incompatible':
                print_colored(f"  • Incompatible: {item['name']}, required: {item['required']}, installed: {item['installed']}", "YELLOW")
    else:
        print_colored("✓ All core dependencies are installed and compatible.", "GREEN")
    
    # Print optional dependency issues if requested
    if all_deps and optional_issues:
        print_colored("\n⚠ OPTIONAL DEPENDENCY ISSUES:", "YELLOW", bold=True)
        by_category = {}
        for item in optional_issues:
            category = item.get('category', 'unknown')
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(item)
        
        for category, items in by_category.items():
            print_colored(f"  Category: {category}", "CYAN")
            for item in items:
                if item['status'] == 'missing':
                    print(f"    • Missing: {item['name']} {item['required'] or ''}")
                elif item['status'] == 'incompatible':
                    print(f"    • Incompatible: {item['name']}, required: {item['required']}, installed: {item['installed']}")


def print_system_info(sys_info: Dict[str, Any]) -> None:
    """Format and print system information in a user-friendly way."""
    print_colored("\n===== SYSTEM INFORMATION =====", "CYAN", bold=True)
    print(f"Python Version: {sys_info['python_version']}")
    
    # PyTorch info
    if sys_info['torch_version']:
        if sys_info['cuda_available']:
            print_colored(f"✓ PyTorch: {sys_info['torch_version']} with CUDA {sys_info['cuda_version']}", "GREEN")
        else:
            print_colored(f"⚠ PyTorch: {sys_info['torch_version']} (without CUDA - GPU acceleration not available)", "YELLOW")
    else:
        print_colored("❌ PyTorch: Not installed", "RED")
    
    # Node.js info
    if sys_info['node_js_version']:
        print_colored(f"✓ Node.js: {sys_info['node_js_version']}", "GREEN")
    else:
        print_colored("⚠ Node.js: Not found or not correctly installed", "YELLOW")
    
    # IPFS info
    if sys_info['ipfs_available']:
        print_colored(f"✓ IPFS: {sys_info['ipfs_version']}", "GREEN")
    else:
        print_colored("⚠ IPFS: Not found or not correctly installed", "YELLOW")
    
    # List any issues
    if sys_info['issues']:
        print_colored("\nSystem Issues:", "YELLOW")
        for issue in sys_info['issues']:
            print(f" • {issue}")


def verify_dependencies(core_only: bool = False, module: Optional[str] = None, 
                        install: bool = False, show_system: bool = True) -> bool:
    """
    Main function to verify dependencies.
    
    Args:
        core_only: If True, only check core dependencies
        module: Specific module to check dependencies for
        install: If True, attempt to install missing dependencies
        show_system: If True, also show system information
    
    Returns:
        Boolean indicating whether all required dependencies are satisfied
    """
    success = True
    
    # First show system info if requested
    if show_system:
        sys_info = check_system_dependencies()
        print_system_info(sys_info)
    
    # Check module-specific dependencies if requested
    if module:
        print_colored(f"\n===== CHECKING DEPENDENCIES FOR MODULE '{module.upper()}' =====", "CYAN", bold=True)
        module_success, missing = check_module_dependencies(module)
        
        if module_success:
            print_colored(f"✓ All dependencies for module '{module}' are installed!", "GREEN")
        else:
            print_colored(f"Module '{module}' has dependency issues:", "YELLOW")
            for item in missing:
                if item['status'] == 'missing':
                    print_colored(f"  • Missing: {item['name']} {item['required'] or ''}", "YELLOW")
                elif item['status'] == 'incompatible':
                    print_colored(f"  • Incompatible: {item['name']}, required: {item['required']}, installed: {item['installed']}", "YELLOW")
            
            if install:
                print_colored("\nAttempting to install missing dependencies...", "CYAN")
                if install_dependencies(missing):
                    print_colored(f"✓ Successfully installed all dependencies for module '{module}'.", "GREEN")
                else:
                    print_colored(f"❌ Failed to install some dependencies for module '{module}'.", "RED")
                    success = False
            else:
                success = False
    else:
        # Check all dependencies
        print_colored("\n===== CHECKING DEPENDENCIES =====", "CYAN", bold=True)
        deps_success, missing = check_dependencies(required_only=core_only)
        
        print_dependency_status(missing, not core_only)
        
        if not deps_success:
            success = False
            
            if install:
                print_colored("\nAttempting to install missing dependencies...", "CYAN")
                core_issues = [item for item in missing if item.get('category') == 'core']
                optional_issues = [item for item in missing if item.get('category') != 'core']
                
                # Install core dependencies first
                if core_issues:
                    if install_dependencies(core_issues):
                        print_colored("✓ Successfully installed core dependencies.", "GREEN")
                    else:
                        print_colored("❌ Failed to install some core dependencies.", "RED")
                        success = False
                
                # Then install optional dependencies if requested
                if not core_only and optional_issues:
                    if install_dependencies(optional_issues):
                        print_colored("✓ Successfully installed optional dependencies.", "GREEN")
                    else:
                        print_colored("❌ Failed to install some optional dependencies.", "YELLOW")
    
    return success


def get_pip_command_for_missing(missing: List[Dict[str, Any]]) -> str:
    """Generate pip install command for missing packages."""
    missing_pkgs = []
    for item in missing:
        if item['status'] == 'missing':
            spec = item['name']
            if item['required']:
                spec += item['required']
            missing_pkgs.append(spec)
    
    if missing_pkgs:
        return f"pip install {' '.join(missing_pkgs)}"
    return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Verify dependencies for vAIn P2P AGI')
    parser.add_argument('--core-only', action='store_true', help='Check only core dependencies')
    parser.add_argument('--module', type=str, help='Check dependencies for a specific module')
    parser.add_argument('--install', action='store_true', help='Install missing dependencies')
    parser.add_argument('--no-system', action='store_true', help='Skip system dependency check')
    parser.add_argument('--generate-config', action='store_true', help='Generate dependencies.json config file')
    parser.add_argument('--generate-requirements', action='store_true', help='Generate requirements.txt file')
    
    args = parser.parse_args()
    
    if args.generate_config:
        if save_dependencies_config():
            print_colored("✓ Dependencies configuration saved to config/dependencies.json", "GREEN")
        else:
            print_colored("❌ Failed to save dependencies configuration", "RED")
        sys.exit(0)
    
    if args.generate_requirements:
        if generate_requirements_txt():
            print_colored("✓ Generated requirements.txt file", "GREEN")
        else:
            print_colored("❌ Failed to generate requirements.txt", "RED")
        sys.exit(0)
    
    # Run the main verification
    success = verify_dependencies(
        core_only=args.core_only,
        module=args.module,
        install=args.install,
        show_system=not args.no_system
    )
    
    sys.exit(0 if success else 1)
