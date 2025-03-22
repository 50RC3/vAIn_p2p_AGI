import subprocess
import sys
import logging
from pathlib import Path
from tqdm import tqdm
from core.interactive_utils import InteractiveSession, InteractiveConfig
from core.constants import InteractionLevel

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
    """Run cleanup tasks with interactive monitoring."""
    session = InteractiveSession(
        level=InteractionLevel.NORMAL,
        config=InteractiveConfig(timeout=300, safe_mode=True)
    )

    try:
        print("\nStarting Code Cleanup")
        print("=" * 50)

        cleanup_tasks = {
            "Python": ["formatting", "import sorting", "type checking", "linting"],
            "JavaScript": ["linting", "formatting"],
            "Solidity": ["linting"]
        }

        for lang, tasks in cleanup_tasks.items():
            print(f"\n{lang} Cleanup:")
            with tqdm(total=len(tasks)) as pbar:
                for task in tasks:
                    try:
                        if run_command(get_command(lang, task), f"{lang} {task}"):
                            pbar.update(1)
                        elif input(f"\nRetry {lang} {task}? (Y/n): ").lower() != 'n':
                            pbar.n -= 1
                            continue
                        else:
                            logger.warning(f"Skipping {lang} {task}")
                    except Exception as e:
                        logger.error(f"{lang} {task} failed: {e}")
                        if not input("\nContinue with remaining tasks? (Y/n): ").lower() == 'n':
                            continue
                        raise

    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")
        raise
    finally:
        session.cleanup()

def get_command(lang: str, task: str) -> str:
    """Get command for cleanup task."""
    commands = {
        "Python": {
            "formatting": "black .",
            "import sorting": "isort .",
            "type checking": "mypy .",
            "linting": "pylint **/*.py"
        },
        "JavaScript": {
            "linting": "npm run lint",
            "formatting": "npm run format"
        },
        "Solidity": {
            "linting": "npm run solhint"
        }
    }
    return commands[lang][task]

if __name__ == "__main__":
    cleanup_code()
