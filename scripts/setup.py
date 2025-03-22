import os
import logging
from typing import Dict
from tqdm import tqdm
from pathlib import Path
from core.interactive_utils import InteractiveSession, InteractiveConfig
from core.constants import InteractionLevel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_environment():
    """Initialize development environment interactively."""
    packages = {
        'python': [
            'torch>=1.9.0',
            'transformers>=4.11.0',
            'numpy>=1.19.5',
            'pandas>=1.3.0'
        ],
        'node': [
            '@openzeppelin/contracts',
            'ethers',
            'hardhat'
        ]
    }
    
    session = InteractiveSession(
        level=InteractionLevel.NORMAL,
        config=InteractiveConfig(timeout=600, safe_mode=True)
    )

    try:
        print("\nStarting Environment Setup")
        print("=" * 50)

        steps = ["Dependencies", "Database", "Contracts"]
        with tqdm(total=len(steps)) as pbar:
            for step in steps:
                try:
                    if step == "Dependencies":
                        install_dependencies(packages)
                    elif step == "Database":
                        setup_database()
                    else:
                        init_contracts()
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"{step} setup failed: {e}")
                    if input(f"\nRetry {step.lower()} setup? (Y/n): ").lower() != 'n':
                        pbar.n -= 1
                        continue
                    raise

    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
        raise
    finally:
        session.cleanup()

def install_dependencies(packages: Dict[str, list]):
    """Install dependencies with progress tracking."""
    print("\nInstalling Dependencies")
    print("=" * 50)

    for pkg_type, pkg_list in packages.items():
        print(f"\nInstalling {pkg_type} packages:")
        with tqdm(total=len(pkg_list)) as pbar:
            for package in pkg_list:
                try:
                    if pkg_type == 'python':
                        subprocess.run(['pip', 'install', package], check=True, capture_output=True)
                    else:
                        subprocess.run(['npm', 'install', package], check=True, capture_output=True)
                    pbar.update(1)
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to install {package}: {e}")
                    if not input("\nContinue with remaining packages? (Y/n): ").lower() == 'n':
                        continue
                    raise

def setup_database():
    """Initialize database with required tables."""
    from backend.src.models.init_db import init_database
    init_database()

def init_contracts():
    """Deploy smart contracts to local network."""
    subprocess.run(['npx', 'hardhat', 'compile'], check=True)
    subprocess.run(['bash', 'scripts/deploy.sh'], check=True)

if __name__ == "__main__":
    setup_environment()
