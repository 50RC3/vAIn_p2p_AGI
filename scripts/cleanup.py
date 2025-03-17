import subprocess
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_command(command, description):
    logger.info(f"Running {description}...")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"{description} failed:\n{result.stderr}")
        return False
    return True

def cleanup_code():
    # Python cleanup
    run_command("black .", "Python formatting")
    run_command("isort .", "Import sorting")
    run_command("mypy .", "Type checking")
    run_command("pylint **/*.py", "Python linting")
    
    # JavaScript cleanup
    run_command("npm run lint", "JavaScript linting")
    run_command("npm run format", "JavaScript formatting")
    
    # Solidity cleanup
    run_command("npm run solhint", "Solidity linting")

if __name__ == "__main__":
    cleanup_code()
