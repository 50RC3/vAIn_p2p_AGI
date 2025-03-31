from setuptools import setup, find_packages
from pathlib import Path

here = Path(__file__).parent

# Read the version from the main __init__.py file
with open(here / "ai_core" / "__init__.py", encoding="utf-8") as f:
    for line in f:
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            version = line.split(delim)[1]
            break
    else:
        version = "0.1.0"  # Fallback version if not found

# Read description from README
with open(here / "README.md", encoding="utf-8") as f:
    long_description = f.read()

# Core dependencies
install_requires = [
    "numpy>=1.19.0",
    "torch>=1.7.0",
    "psutil>=5.8.0",
    "tqdm>=4.50.0",
    "asyncio>=3.4.3",
    "tabulate>=0.8.7"
]

# Optional dependencies
extras_require = {
    "visualization": ["matplotlib>=3.3.0"],
    "networking": ["kademlia>=2.2"],
    "security": ["cryptography>=35.0.0"],
    "full": ["matplotlib>=3.3.0", "kademlia>=2.2", "cryptography>=35.0.0", "prompt_toolkit>=3.0.0"]
}

setup(
    name="vain-p2p-agi",
    version=version,
    description="vAIn P2P AGI System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Mr.V",
    author_email="example@example.com",
    url="https://github.com/Mr.V/vAIn_p2p_AGI",
    packages=find_packages(exclude=["tests", "scripts"]),
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires=">=3.8.0",
    entry_points={
        "console_scripts": [
            "vain-start=start:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
