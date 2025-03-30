from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="vain_p2p_agi",
    version="0.2.1",
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.7',
    author="Mr.V",
    description="vAIn P2P AGI - Distributed intelligence framework",
)
