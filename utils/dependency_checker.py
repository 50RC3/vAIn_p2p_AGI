import importlib
import logging
import sys
import subprocess
import pkg_resources
import json
import os
from typing import Dict, List, Tuple, Optional, Any, Set
from pathlib import Path

logger = logging.getLogger(__name__)

# Define core and optional dependencies with version requirements
CORE_DEPENDENCIES = {
    "numpy": ">=1.19.0",
    "torch": ">=1.9.0",
    "psutil": ">=5.8.0",
    "python-dotenv": ">=1.0.0"
}

OPTIONAL_DEPENDENCIES = {
    "visualization": {"matplotlib": ">=3.4.0", "tqdm": ">=4.61.0"},
    "networking": {"tabulate": ">=0.8.9", "asyncio": None, "websockets": ">=10.0"},
    "admin": {"tabulate": ">=0.8.9"},
    "security": {"cryptography": ">=41.0.0", "PyJWT": ">=2.8.0"},
    "storage": {"ipfshttpclient": ">=0.8.0", "pathlib": None},
    "metrics": {"prometheus-client": ">=0.17.0", "sentry-sdk": ">=1.25.0"},
    "ui": {"prompt_toolkit": ">=3.0.0"},
    "ml": {"scikit-learn": ">=1.2.0", "pandas": ">=2.0.0"},
    "web": {"fastapi": ">=0.95.0", "pydantic": ">=2.0.0", "uvicorn": ">=0.22.0"}
}

# Define module dependencies mapping with versions
MODULE_DEPENDENCIES = {
    "network": {"asyncio": None, "ssl": None, "json": None, "websockets": ">=10.0"},
    "training": {"torch": ">=1.9.0", "numpy": ">=1.19.0", "tqdm": ">=4.61.0"},
    "security": {"cryptography": ">=41.0.0", "PyJWT": ">=2.8.0", "passlib": ">=1.7.4"},
    "ai_core": {"torch": ">=1.9.0", "numpy": ">=1.19.0", "scikit-learn": ">=1.2.0"},
    "memory": {"torch": ">=1.9.0", "numpy": ">=1.19.0", "ipfshttpclient": ">=0.8.0"},
    "utils": {"psutil": ">=5.8.0", "numpy": ">=1.19.0"},
    "web": {"fastapi": ">=0.95.0", "pydantic": ">=2.0.0", "uvicorn": ">=0.22.0"}
}

def normalize_package_name(name: str) -> str:
    """Normalize package name for comparison."""
    return name.lower().replace('-', '_').replace('.', '_')

def get_installed_packages() -> Dict[str, str]:
    """Get a dictionary of all installed packages and their versions."""
    try:
        installed = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
        return installed
    except Exception as e:
        logger.error(f"Error getting installed packages: {str(e)}")
        return {}

def check_version_compatibility(installed_version: str, required_version: str) -> bool:
    """
    Check if installed version meets requirements.
    
    Args:
        installed_version: Current installed version
        required_version: Version requirement (e.g., ">=1.0.0")
        
    Returns:
        Boolean indicating compatibility
    """
    if not required_version:
        return True
    
    try:
        return pkg_resources.parse_version(installed_version) in pkg_resources.Requirement.parse(f"dummy{required_version}")
    except Exception as e:
        logger.warning(f"Version compatibility check error: {str(e)}")
        return False

def check_dependencies(required_only: bool = False) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Check if all required dependencies are installed.
    
    Args:
        required_only: If True, only check core dependencies
        
    Returns:
        Tuple of (success, missing_packages)
        where missing_packages is a list of dicts with 'name', 'required', 'installed', 'status'
    """
    installed_packages = get_installed_packages()
    missing_or_incompatible = []
    
    # Check core dependencies
    for package, version_req in CORE_DEPENDENCIES.items():
        normalized_name = normalize_package_name(package)
        if normalized_name in installed_packages:
            installed_version = installed_packages[normalized_name]
            is_compatible = check_version_compatibility(installed_version, version_req)
            if not is_compatible:
                missing_or_incompatible.append({
                    'name': package,
                    'required': version_req,
                    'installed': installed_version,
                    'status': 'incompatible',
                    'category': 'core'
                })
        else:
            missing_or_incompatible.append({
                'name': package,
                'required': version_req,
                'installed': None,
                'status': 'missing',
                'category': 'core'
            })
    
    # Check optional dependencies if requested
    if not required_only:
        for category, packages in OPTIONAL_DEPENDENCIES.items():
            for package, version_req in packages.items():
                normalized_name = normalize_package_name(package)
                # Skip standard library modules that don't need pip install
                if normalized_name in ('json', 'ssl', 'asyncio', 'pathlib'):
                    continue
                    
                if normalized_name in installed_packages:
                    installed_version = installed_packages[normalized_name]
                    is_compatible = check_version_compatibility(installed_version, version_req)
                    if not is_compatible:
                        missing_or_incompatible.append({
                            'name': package,
                            'required': version_req,
                            'installed': installed_version,
                            'status': 'incompatible',
                            'category': category
                        })
                else:
                    missing_or_incompatible.append({
                        'name': package,
                        'required': version_req,
                        'installed': None,
                        'status': 'missing',
                        'category': category
                    })
    
    return len(missing_or_incompatible) == 0, missing_or_incompatible

def check_module_dependencies(module: str) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Check dependencies for a specific module.
    
    Args:
        module: Module name
        
    Returns:
        Tuple of (success, missing_or_incompatible_packages)
    """
    if module not in MODULE_DEPENDENCIES:
        logger.warning(f"Unknown module: {module}")
        return False, [{'name': 'unknown_module', 'status': 'error', 'message': f"Unknown module: {module}"}]
    
    installed_packages = get_installed_packages()
    missing_or_incompatible = []
    
    for package, version_req in MODULE_DEPENDENCIES[module].items():
        normalized_name = normalize_package_name(package)
        # Skip standard library modules that don't need pip install
        if normalized_name in ('json', 'ssl', 'asyncio', 'pathlib'):
            continue
            
        if normalized_name in installed_packages:
            installed_version = installed_packages[normalized_name]
            is_compatible = check_version_compatibility(installed_version, version_req)
            if not is_compatible:
                missing_or_incompatible.append({
                    'name': package,
                    'required': version_req,
                    'installed': installed_version,
                    'status': 'incompatible',
                    'module': module
                })
        else:
            missing_or_incompatible.append({
                'name': package,
                'required': version_req,
                'installed': None,
                'status': 'missing',
                'module': module
            })
    
    return len(missing_or_incompatible) == 0, missing_or_incompatible

def install_dependencies(packages: List[Dict[str, Any]], pip_command: str = "pip") -> bool:
    """
    Install missing dependencies.
    
    Args:
        packages: List of package dictionaries with 'name' and 'required' keys
        pip_command: Command to use for pip (e.g., "pip", "pip3")
        
    Returns:
        Boolean indicating success
    """
    if not packages:
        return True
    
    success = True
    for package_info in packages:
        package = package_info['name']
        version_req = package_info.get('required')
        
        # Skip standard library modules
        if package in ('json', 'ssl', 'asyncio', 'pathlib'):
            continue
            
        install_spec = package
        if version_req:
            install_spec = f"{package}{version_req}"
        
        logger.info(f"Installing {install_spec}...")
        try:
            subprocess.check_call([pip_command, "install", install_spec])
            logger.info(f"Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install {package}: {str(e)}")
            success = False
    
    return success

def check_system_dependencies() -> Dict[str, Any]:
    """Check system dependencies and compatibility"""
    result = {
        "python_version": None,
        "torch_version": None,
        "cuda_available": False,
        "cuda_version": None,
        "node_js_version": None,
        "ipfs_available": False,
        "ipfs_version": None,
        "issues": []
    }
    
    try:
        # Check Python version
        result["python_version"] = sys.version
        
        # Check PyTorch and CUDA
        try:
            import torch
            result["torch_version"] = torch.__version__
            result["cuda_available"] = torch.cuda.is_available()
            
            if torch.cuda.is_available():
                result["cuda_version"] = torch.version.cuda
            else:
                result["issues"].append("CUDA not available, some modules may run slower")
        except ImportError:
            result["issues"].append("PyTorch not installed")
        
        # Check Node.js availability
        try:
            node_version = subprocess.check_output(["node", "--version"]).decode('utf-8').strip()
            result["node_js_version"] = node_version
        except (subprocess.CalledProcessError, FileNotFoundError):
            result["issues"].append("Node.js not found or not correctly installed")
        
        # Check IPFS availability
        try:
            ipfs_version = subprocess.check_output(["ipfs", "--version"]).decode('utf-8').strip()
            result["ipfs_available"] = True
            result["ipfs_version"] = ipfs_version
        except (subprocess.CalledProcessError, FileNotFoundError):
            result["issues"].append("IPFS not found or not correctly installed")
            
        return result
            
    except ImportError as e:
        result["issues"].append(f"Import error: {str(e)}")
        return result
        
    except Exception as e:
        result["issues"].append(f"Error checking dependencies: {str(e)}")
        return result

def load_dependencies_config() -> Dict[str, Any]:
    """Load dependencies configuration from file."""
    try:
        # Get the base directory from core.constants or use a fallback method
        try:
            from core.constants import BASE_DIR
            config_path = BASE_DIR / "config" / "dependencies.json"
        except ImportError:
            # Fallback: determine the project root directory based on this file's location
            current_dir = Path(__file__).resolve().parent
            base_dir = current_dir.parent
            config_path = base_dir / "config" / "dependencies.json"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load dependencies config: {e}")
    
    # If loading failed, return the hardcoded dependencies
    return {
        "core": CORE_DEPENDENCIES,
        "optional": OPTIONAL_DEPENDENCIES,
        "modules": MODULE_DEPENDENCIES
    }

def save_dependencies_config() -> bool:
    """Save current dependencies configuration to file."""
    try:
        # Get the base directory from core.constants or use a fallback method
        try:
            from core.constants import BASE_DIR, CONFIG_DIR
            config_dir = CONFIG_DIR
        except ImportError:
            # Fallback: determine the project root directory based on this file's location
            current_dir = Path(__file__).resolve().parent
            base_dir = current_dir.parent
            config_dir = base_dir / "config"
        
        # Ensure the config directory exists
        os.makedirs(config_dir, exist_ok=True)
        
        config_path = config_dir / "dependencies.json"
        
        config_data = {
            "core": CORE_DEPENDENCIES,
            "optional": OPTIONAL_DEPENDENCIES,
            "modules": MODULE_DEPENDENCIES
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        return True
    except Exception as e:
        logger.error(f"Failed to save dependencies config: {e}")
        return False

def generate_requirements_txt() -> bool:
    """Generate requirements.txt based on current dependency configuration."""
    try:
        # Get the base directory from core.constants or use a fallback method
        try:
            from core.constants import BASE_DIR
            base_dir = BASE_DIR
        except ImportError:
            # Fallback: determine the project root directory based on this file's location
            current_dir = Path(__file__).resolve().parent
            base_dir = current_dir.parent
        
        req_path = base_dir / "requirements.txt"
        
        # Collect all dependencies with their version requirements
        all_deps = {}
        
        # Add core dependencies
        for pkg, version in CORE_DEPENDENCIES.items():
            if version:
                all_deps[pkg] = version
            else:
                all_deps[pkg] = ""
        
        # Add optional dependencies
        for category, packages in OPTIONAL_DEPENDENCIES.items():
            for pkg, version in packages.items():
                # Skip standard library modules
                if pkg in ('json', 'ssl', 'asyncio', 'pathlib'):
                    continue
                
                if pkg not in all_deps and version:
                    all_deps[pkg] = version
                elif pkg not in all_deps:
                    all_deps[pkg] = ""
        
        # Write to requirements.txt
        with open(req_path, 'w') as f:
            f.write("# Auto-generated requirements.txt\n")
            f.write("# Core dependencies\n")
            
            # Write core dependencies first
            for pkg in sorted(CORE_DEPENDENCIES.keys()):
                if pkg in ('json', 'ssl', 'asyncio', 'pathlib'):
                    continue
                version = CORE_DEPENDENCIES[pkg] or ""
                f.write(f"{pkg}{version}\n")
            
            # Write optional dependencies grouped by category
            for category, packages in OPTIONAL_DEPENDENCIES.items():
                if packages:
                    f.write(f"\n# {category.capitalize()} dependencies\n")
                    for pkg in sorted(packages.keys()):
                        if pkg in ('json', 'ssl', 'asyncio', 'pathlib'):
                            continue
                        version = packages[pkg] or ""
                        f.write(f"{pkg}{version}\n")
        
        return True
    except Exception as e:
        logger.error(f"Failed to generate requirements.txt: {e}")
        return False

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Try to load config from file
    deps_config = load_dependencies_config()
    if deps_config:
        # Override default configurations if loaded successfully
        if "core" in deps_config:
            CORE_DEPENDENCIES = deps_config["core"]
        if "optional" in deps_config:
            OPTIONAL_DEPENDENCIES = deps_config["optional"]
        if "modules" in deps_config:
            MODULE_DEPENDENCIES = deps_config["modules"]
    
    import argparse
    parser = argparse.ArgumentParser(description='Dependency checker for vAIn P2P AGI')
    parser.add_argument('--core-only', action='store_true', help='Check only core dependencies')
    parser.add_argument('--module', type=str, help='Check dependencies for a specific module')
    parser.add_argument('--install', action='store_true', help='Install missing dependencies')
    parser.add_argument('--system', action='store_true', help='Check system dependencies')
    parser.add_argument('--generate-config', action='store_true', help='Generate dependencies.json config file')
    parser.add_argument('--generate-requirements', action='store_true', help='Generate requirements.txt file')
    
    args = parser.parse_args()
    
    if args.generate_config:
        if save_dependencies_config():
            logger.info("Dependencies configuration saved to config/dependencies.json")
        else:
            logger.error("Failed to save dependencies configuration")
        sys.exit(0)
    
    if args.generate_requirements:
        if generate_requirements_txt():
            logger.info("Generated requirements.txt file")
        else:
            logger.error("Failed to generate requirements.txt")
        sys.exit(0)
    
    if args.system:
        sys_deps = check_system_dependencies()
        
        print("\n===== System Dependencies =====")
        print(f"Python Version: {sys_deps['python_version']}")
        print(f"PyTorch Version: {sys_deps['torch_version']}")
        print(f"CUDA Available: {sys_deps['cuda_available']}")
        if sys_deps['cuda_available']:
            print(f"CUDA Version: {sys_deps['cuda_version']}")
        print(f"Node.js Version: {sys_deps['node_js_version']}")
        print(f"IPFS Available: {sys_deps['ipfs_available']}")
        if sys_deps['ipfs_available']:
            print(f"IPFS Version: {sys_deps['ipfs_version']}")
        
        if sys_deps['issues']:
            print("\nIssues:")
            for issue in sys_deps['issues']:
                print(f" - {issue}")
        
        sys.exit(0)
    
    if args.module:
        # Check specific module dependencies
        success, missing = check_module_dependencies(args.module)
        
        if success:
            logger.info(f"All dependencies for module '{args.module}' are installed!")
        else:
            logger.warning(f"Module '{args.module}' has dependency issues:")
            for item in missing:
                if item['status'] == 'missing':
                    logger.warning(f"  - Missing: {item['name']} {item['required'] or ''}")
                elif item['status'] == 'incompatible':
                    logger.warning(f"  - Incompatible: {item['name']}, required: {item['required']}, installed: {item['installed']}")
            
            if args.install:
                logger.info("Attempting to install missing dependencies...")
                if install_dependencies(missing):
                    logger.info(f"Successfully installed all dependencies for module '{args.module}'.")
                else:
                    logger.error(f"Failed to install some dependencies for module '{args.module}'.")
    else:
        # Check all dependencies
        success, missing = check_dependencies(required_only=args.core_only)
        
        if success:
            logger.info(f"All {'core ' if args.core_only else ''}dependencies are installed!")
        else:
            core_issues = [item for item in missing if item.get('category') == 'core']
            optional_issues = [item for item in missing if item.get('category') != 'core']
            
            if core_issues:
                logger.warning("Core dependency issues:")
                for item in core_issues:
                    if item['status'] == 'missing':
                        logger.warning(f"  - Missing: {item['name']} {item['required'] or ''}")
                    elif item['status'] == 'incompatible':
                        logger.warning(f"  - Incompatible: {item['name']}, required: {item['required']}, installed: {item['installed']}")
            
            if not args.core_only and optional_issues:
                logger.warning("Optional dependency issues:")
                by_category = {}
                for item in optional_issues:
                    category = item.get('category', 'unknown')
                    if category not in by_category:
                        by_category[category] = []
                    by_category[category].append(item)
                
                for category, items in by_category.items():
                    logger.warning(f"  Category: {category}")
                    for item in items:
                        if item['status'] == 'missing':
                            logger.warning(f"    - Missing: {item['name']} {item['required'] or ''}")
                        elif item['status'] == 'incompatible':
                            logger.warning(f"    - Incompatible: {item['name']}, required: {item['required']}, installed: {item['installed']}")
            
            # Ask if user wants to install dependencies
            if args.install:
                logger.info("Attempting to install missing dependencies...")
                # Install core dependencies first
                if core_issues:
                    if install_dependencies(core_issues):
                        logger.info("Successfully installed core dependencies.")
                    else:
                        logger.error("Failed to install some core dependencies.")
                
                # Then install optional dependencies if requested
                if not args.core_only and optional_issues:
                    if install_dependencies(optional_issues):
                        logger.info("Successfully installed optional dependencies.")
                    else:
                        logger.error("Failed to install some optional dependencies.")
            else:
                # Just show the pip command they could run
                missing_pkgs = []
                for item in missing:
                    if item['status'] == 'missing':
                        spec = item['name']
                        if item['required']:
                            spec += item['required']
                        missing_pkgs.append(spec)
                
                if missing_pkgs:
                    logger.info(f"You can install missing dependencies with: pip install {' '.join(missing_pkgs)}")
