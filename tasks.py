
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
from core.interactive_utils import InteractiveSession, InteractiveConfig
from core.constants import InteractionLevel, INTERACTION_TIMEOUTS

logger = logging.getLogger(__name__)

class TaskManager:
    def __init__(self):
        self.tasks = {}
        self.session = None
        self.interactive = True

    async def setup_session(self):
        self.session = InteractiveSession(
            level=InteractionLevel.NORMAL,
            config=InteractiveConfig(
                timeout=INTERACTION_TIMEOUTS["default"],
                persistent_state=True,
                safe_mode=True
            )
        )
        await self.session.__aenter__()

    async def cleanup(self):
        if self.session:
            await self.session.__aexit__(None, None, None)

    def register_task(self, name: str, func, description: str):
        self.tasks[name] = {
            "func": func,
            "description": description
        }

    async def run_task(self, name: str, **kwargs):
        if name not in self.tasks:
            logger.error(f"Task {name} not found")
            return False

        try:
            task = self.tasks[name]
            logger.info(f"Running task: {name}")
            logger.info(task["description"])
            
            if self.interactive:
                proceed = await self.session.get_confirmation(
                    f"\nRun task {name}? (y/n): ",
                    timeout=INTERACTION_TIMEOUTS["confirmation"]
                )
                if not proceed:
                    return False

            return await task["func"](**kwargs)

        except Exception as e:
            logger.error(f"Task {name} failed: {str(e)}")
            return False

async def main():
    manager = TaskManager()
    await manager.setup_session()

    try:
        # Register common tasks
        manager.register_task(
            "cleanup",
            lambda: run_command("python scripts/cleanup.py"),
            "Clean up code and remove temporary files"
        )
        manager.register_task(
            "test",
            lambda: run_command("pytest tests/"),
            "Run all tests"
        )
        manager.register_task(
            "deploy",
            lambda: run_command("bash scripts/deploy.sh"),
            "Deploy smart contracts"
        )

        # Get task name from command line
        import sys
        task_name = sys.argv[1] if len(sys.argv) > 1 else None

        if task_name:
            await manager.run_task(task_name)
        else:
            # List available tasks
            print("\nAvailable tasks:")
            for name, task in manager.tasks.items():
                print(f"- {name}: {task['description']}")

    finally:
        await manager.cleanup()

async def run_command(cmd: str) -> bool:
    """Run a shell command and return success status"""
    import subprocess
    try:
        subprocess.run(cmd, shell=True, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

if __name__ == "__main__":
    asyncio.run(main())
