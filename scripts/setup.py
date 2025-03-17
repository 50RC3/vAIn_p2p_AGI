import os
import subprocess
from typing import Dict

def setup_environment():
    """Initialize development environment."""
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
    
    install_dependencies(packages)
    setup_database()
    init_contracts()

def install_dependencies(packages: Dict[str, list]):
    """Install Python and Node.js dependencies."""
    # Python dependencies
    subprocess.run(['pip', 'install', '-r', 'requirements.txt'], check=True)
    
    # Node.js dependencies
    subprocess.run(['npm', 'install'], check=True)

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
