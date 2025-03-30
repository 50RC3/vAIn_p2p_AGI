import importlib
import logging
import sys
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

# Define core and optional dependencies
CORE_DEPENDENCIES = [
    "numpy",
    "torch",
    "psutil"
]

OPTIONAL_DEPENDENCIES = {
    "visualization": ["matplotlib", "tqdm"],
    "networking": ["tabulate"],
    "admin": ["tabulate"],
}

def check_dependencies(required_only: bool = False) -> Tuple[bool, List[str]]:
    """
    Check if all required dependencies are installed.
    
    Args:
        required_only: If True, only check core dependencies
        
    Returns:
        Tuple of (success, missing_packages)
    """
    missing = []
    
    # Check core dependencies
    for package in CORE_DEPENDENCIES:
        try:
            importlib.import_module(package)
        except ImportError:
            missing.append(package)
    
    # Check optional dependencies if requested
    if not required_only:
        for category, packages in OPTIONAL_DEPENDENCIES.items():
            for package in packages:
                try:
                    importlib.import_module(package)
                except ImportError:
                    missing.append(f"{package} (optional - {category})")
    
    return len(missing) == 0, missing

def install_dependencies(packages: List[str], pip_command: str = "pip") -> bool:
    """
    Attempt to install missing dependencies
    
    Args:
        packages: List of package names to install
        pip_command: Command to use for pip (pip, pip3, etc.)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import subprocess
        for package in packages:
            if '(' in package:  # Handle optional dependencies format
                package = package.split('(')[0].strip()
            
            logger.info(f"Installing {package}...")
            subprocess.check_call([pip_command, "install", package])
        return True
    except Exception as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Check dependencies
    success, missing = check_dependencies()
    
    if success:
        logger.info("All dependencies are installed!")
    else:
        logger.warning(f"Missing dependencies: {', '.join(missing)}")
        
        # Ask if user wants to install
        if input("Do you want to install missing dependencies? (y/n): ").lower().startswith('y'):
            if install_dependencies(missing):
                logger.info("Successfully installed all dependencies.")
            else:
                logger.error("Failed to install some dependencies.")
                logger.info("Try installing them manually with: pip install tabulate")
