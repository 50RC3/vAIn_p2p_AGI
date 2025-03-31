import asyncio
import logging
import json
import sys
import time
import psutil
from pathlib import Path
from typing import Dict, Callable, Any, Optional, List, Set, Union, Tuple
from dataclasses import dataclass, asdict
from core.interactive_utils import InteractiveSession, InteractiveConfig
from core.constants import InteractionLevel, INTERACTION_TIMEOUTS

# Configure structured logging for better analysis
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("task_manager.log")]
)
logger = logging.getLogger(__name__)

@dataclass
class TaskResult:
    """Data class to track task execution results."""
    success: bool
    execution_time: float
    task_name: str
    error_message: Optional[str] = None
    output: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    resource_usage: Optional[Dict[str, float]] = None

class TaskDependencyError(Exception):
    """Raised when task dependencies cannot be resolved."""
    pass

class Task:
    """Represents a task with its metadata and dependencies."""
    def __init__(self, 
                 name: str, 
                 func: Callable[..., Any], 
                 description: str,
                 dependencies: Optional[List[str]] = None,
                 timeout: Optional[float] = None) -> None:
        self.name = name
        self.func = func
        self.description = description
        self.dependencies = dependencies or []
        self.timeout = timeout
        self.metrics: Dict[str, float] = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "dependencies": self.dependencies,
            "metrics": self.metrics
        }

class TaskManager:
    def __init__(self) -> None:
        self.tasks: Dict[str, Task] = {}
        self.session: Optional[InteractiveSession] = None
        self.interactive: bool = True
        self.results: List[TaskResult] = []
        self.config_path: Path = Path("task_manager_config.json")
        self.load_config()

    def load_config(self) -> None:
        """Load task manager configuration from file."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    self.interactive = config.get("interactive", True)
                    logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config: {e}, using defaults")

    def save_config(self) -> None:
        """Save task manager configuration to file."""
        config = {
            "interactive": self.interactive,
            "last_run": time.time()
        }
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    async def setup_session(self) -> None:
        """Initialize an interactive session with proper error handling."""
        if not self.interactive:
            logger.info("Running in non-interactive mode")
            return

        try:
            self.session = InteractiveSession(
                session_id="task_manager",
                config=InteractiveConfig(
                    level=InteractionLevel.NORMAL,
                    timeout=INTERACTION_TIMEOUTS["default"],
                    persistent_state=True,
                    safe_mode=True
                )
            )
            if self.session:
                logger.info("Interactive session setup successfully")
        except ConnectionError as e:
            logger.error(f"Connection error setting up session: {e}")
            self.session = None
        except TimeoutError as e:
            logger.error(f"Timeout error setting up session: {e}")
            self.session = None
        except Exception as e:
            logger.error(f"Unexpected error setting up session: {e}")
            self.session = None

    async def cleanup(self) -> None:
        """Gracefully cleanup session resources and save results."""
        # Save execution metrics before cleanup
        self.save_execution_report()
        
        if self.session:
            try:
                await self.session.__aexit__(None, None, None)
                logger.info("Session closed successfully")
            except Exception as e:
                logger.error(f"Error during session cleanup: {e}")

    def save_execution_report(self) -> None:
        """Save execution results to a JSON report."""
        if not self.results:
            return
            
        try:
            report = {
                "timestamp": time.time(),
                "results": [asdict(result) for result in self.results],
                "success_rate": sum(1 for r in self.results if r.success) / len(self.results)
            }
            
            with open("task_execution_report.json", "w") as f:
                json.dump(report, f, indent=2)
                
            logger.info(f"Execution report saved: {len(self.results)} tasks, " 
                       f"{report['success_rate']*100:.1f}% success rate")
        except Exception as e:
            logger.error(f"Failed to save execution report: {e}")

    def register_task(self, 
                      name: str, 
                      func: Callable[..., Any], 
                      description: str,
                      dependencies: Optional[List[str]] = None,
                      timeout: Optional[float] = None) -> None:
        """Register a task to be executed later with optional dependencies."""
        if name in self.tasks:
            logger.warning(f"Task '{name}' already registered, overwriting")
            
        self.tasks[name] = Task(
            name=name,
            func=func,
            description=description,
            dependencies=dependencies,
            timeout=timeout
        )
        logger.debug(f"Registered task '{name}' with {len(dependencies or [])} dependencies")

    def get_task_execution_order(self) -> List[str]:
        """
        Determine task execution order based on dependencies.
        Raises TaskDependencyError if circular dependencies exist.
        """
        visited: Set[str] = set()
        temp_visited: Set[str] = set()
        order: List[str] = []
        
        def visit(task_name: str) -> None:
            if task_name in temp_visited:
                cycle_path = " -> ".join([task_name] + [t for t in temp_visited])
                raise TaskDependencyError(f"Circular dependency detected: {cycle_path}")
            
            if task_name not in visited and task_name in self.tasks:
                temp_visited.add(task_name)
                
                for dep in self.tasks[task_name].dependencies:
                    if dep not in self.tasks:
                        raise TaskDependencyError(f"Task '{task_name}' depends on unknown task '{dep}'")
                    visit(dep)
                    
                temp_visited.remove(task_name)
                visited.add(task_name)
                order.append(task_name)
        
        for task_name in self.tasks:
            if task_name not in visited:
                visit(task_name)
                
        return order
    
    def get_resource_usage(self) -> Dict[str, float]:
        """Get current system resource usage statistics."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_rss_mb": memory_info.rss / (1024 * 1024),
                "memory_vms_mb": memory_info.vms / (1024 * 1024),
                "num_threads": process.num_threads(),
                "disk_usage": psutil.disk_usage('/').percent
            }
        except Exception as e:
            logger.warning(f"Failed to get resource usage: {e}")
            return {}

    async def run_task(self, name: str, **kwargs: Any) -> Union[bool, TaskResult]:
        """Execute a registered task with detailed result tracking."""
        if name not in self.tasks:
            error_msg = f"Task '{name}' not found"
            logger.error(error_msg)
            result = TaskResult(success=False, execution_time=0, task_name=name, error_message=error_msg)
            self.results.append(result)
            return result if kwargs.get("detailed_result", False) else False

        task = self.tasks[name]
        logger.info(f"Running task: {name}")
        logger.info(task.description)
        
        # Check dependencies first
        for dep in task.dependencies:
            if dep not in self.tasks:
                error_msg = f"Dependency '{dep}' for task '{name}' not found"
                logger.error(error_msg)
                result = TaskResult(success=False, execution_time=0, task_name=name, error_message=error_msg)
                self.results.append(result)
                return result if kwargs.get("detailed_result", False) else False

        start_time = time.time()
        try:
            if self.interactive and self.session:
                try:
                    proceed = await self.session.get_confirmation(
                        f"Run task '{name}'? (y/n): ",
                        timeout=INTERACTION_TIMEOUTS.get("confirmation", 30)
                    )
                    if not proceed:
                        result = TaskResult(success=False, execution_time=0, task_name=name, 
                                          error_message="Task execution cancelled by user")
                        self.results.append(result)
                        return result if kwargs.get("detailed_result", False) else False
                except Exception as e:
                    logger.warning(f"Error in interactive confirmation: {e}, proceeding with task")

            # Capture resource usage before execution
            start_resource = self.get_resource_usage()
            
            # Execute with timeout if specified
            if task.timeout:
                try:
                    task_coro = task.func(**kwargs)
                    output = await asyncio.wait_for(task_coro, timeout=task.timeout)
                except asyncio.TimeoutError:
                    error_msg = f"Task '{name}' exceeded timeout of {task.timeout}s"
                    logger.error(error_msg)
                    result = TaskResult(success=False, execution_time=time.time() - start_time, 
                                      task_name=name, error_message=error_msg)
                    self.results.append(result)
                    return result if kwargs.get("detailed_result", False) else False
            else:
                output = await task.func(**kwargs)
                
            execution_time = time.time() - start_time
            
            # Capture resource usage after execution
            end_resource = self.get_resource_usage()
            resource_delta = {
                key: end_resource.get(key, 0) - start_resource.get(key, 0)
                for key in set(start_resource) | set(end_resource)
            }
            
            # Record metrics
            task.metrics["last_execution_time"] = execution_time
            if "avg_execution_time" not in task.metrics:
                task.metrics["avg_execution_time"] = execution_time
            else:
                # Calculate running average
                task.metrics["avg_execution_time"] = (
                    0.7 * task.metrics["avg_execution_time"] + 0.3 * execution_time
                )
            
            # Record success
            task.metrics["success_count"] = task.metrics.get("success_count", 0) + 1
            task.metrics["last_success_time"] = time.time()
            
            result = TaskResult(
                success=True if output is None else bool(output),
                execution_time=execution_time,
                task_name=name,
                output=str(output) if output is not None else None,
                metrics=task.metrics,
                resource_usage=resource_delta
            )
            self.results.append(result)
            
            logger.info(f"Task '{name}' completed in {execution_time:.2f}s")
            logger.debug(f"Resource usage for task '{name}': {resource_delta}")
            return result if kwargs.get("detailed_result", False) else bool(output)

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Task '{name}' failed: {error_msg}")
            
            # Record failure
            task.metrics["failure_count"] = task.metrics.get("failure_count", 0) + 1
            task.metrics["last_failure_time"] = time.time()
            
            # Include traceback for better debugging
            import traceback
            tb = traceback.format_exc()
            logger.debug(f"Exception traceback for task '{name}':\n{tb}")
            
            result = TaskResult(
                success=False,
                execution_time=execution_time,
                task_name=name,
                error_message=error_msg,
                metrics=task.metrics
            )
            self.results.append(result)
            return result if kwargs.get("detailed_result", False) else False

    async def run_tasks_in_sequence(self, task_names: List[str], **kwargs: Any) -> List[TaskResult]:
        """Run multiple tasks in sequence, respecting their dependencies."""
        if not task_names:
            # If no tasks specified, try to run all tasks in dependency order
            try:
                task_names = self.get_task_execution_order()
            except TaskDependencyError as e:
                logger.error(f"Failed to determine execution order: {e}")
                return []
        
        results = []
        for name in task_names:
            result = await self.run_task(name, detailed_result=True, **kwargs)
            if isinstance(result, TaskResult):
                results.append(result)
                if not result.success and kwargs.get("stop_on_failure", False):
                    logger.warning(f"Stopping task sequence after failure of '{name}'")
                    break
        return results
    
    async def run_tasks_in_parallel(self, task_names: List[str], **kwargs: Any) -> List[TaskResult]:
        """Run multiple tasks in parallel where possible, respecting dependencies."""
        if not task_names:
            return []
        
        # Build dependency graph
        dependent_on: Dict[str, Set[str]] = {}
        dependents: Dict[str, Set[str]] = {}
        
        # Initialize dependency tracking
        for name in task_names:
            if name not in self.tasks:
                continue
                
            dependent_on[name] = set()
            for dep in self.tasks[name].dependencies:
                if dep in task_names:
                    dependent_on[name].add(dep)
                    if dep not in dependents:
                        dependents[dep] = set()
                    dependents[dep].add(name)
        
        # Track completed tasks and their results
        completed: Set[str] = set()
        results: Dict[str, TaskResult] = {}
        
        # Find tasks with no dependencies
        queue = asyncio.Queue()
        for name in task_names:
            if name in dependent_on and len(dependent_on[name]) == 0:
                await queue.put(name)
        
        # Process tasks as their dependencies are satisfied
        running_tasks = set()
        max_concurrent = kwargs.get("max_concurrent", 5)  # Limit concurrent tasks
        stop_on_failure = kwargs.get("stop_on_failure", False)
        any_failure = False
        
        while len(completed) < len(task_names):
            # Start new tasks if possible
            while len(running_tasks) < max_concurrent and not queue.empty() and not any_failure:
                task_name = await queue.get()
                if task_name not in completed:
                    task = asyncio.create_task(self.run_task(task_name, detailed_result=True, **kwargs))
                    task.task_name = task_name  # Add task name for identification
                    running_tasks.add(task)
            
            if not running_tasks:
                # No more tasks to run or waiting on dependencies
                if any_failure and stop_on_failure:
                    logger.warning("Stopping parallel execution due to failure")
                    break
                if queue.empty() and len(completed) < len(task_names):
                    logger.error("Dependency deadlock detected")
                    break
            
            # Wait for any task to complete
            done, pending = await asyncio.wait(
                running_tasks, return_when=asyncio.FIRST_COMPLETED
            )
            
            # Process completed tasks
            for task in done:
                running_tasks.remove(task)
                task_name = task.task_name
                try:
                    result = task.result()
                    results[task_name] = result
                    completed.add(task_name)
                    
                    if not result.success:
                        any_failure = True
                        if stop_on_failure:
                            continue
                    
                    # Queue dependent tasks whose dependencies are now satisfied
                    if task_name in dependents:
                        for dependent in dependents[task_name]:
                            dependent_on[dependent].remove(task_name)
                            if len(dependent_on[dependent]) == 0:
                                await queue.put(dependent)
                except Exception as e:
                    logger.error(f"Error processing task result for {task_name}: {e}")
        
        # Return results in the order of the original task names
        ordered_results = []
        for name in task_names:
            if name in results:
                ordered_results.append(results[name])
        return ordered_results

async def run_command(cmd: str, capture_output: bool = True) -> Union[bool, str]:
    """Run a shell command asynchronously and return success status or output."""
    try:
        process = await asyncio.create_subprocess_shell(
            cmd, 
            stdout=asyncio.subprocess.PIPE if capture_output else None, 
            stderr=asyncio.subprocess.PIPE if capture_output else None
        )

        if capture_output:
            stdout, stderr = await process.communicate()
            
            stdout_text = stdout.decode().strip() if stdout else ""
            stderr_text = stderr.decode().strip() if stderr else ""
            
            if stdout_text:
                logger.info(f"Command output:\n{stdout_text}")
            if stderr_text:
                logger.error(f"Command error:\n{stderr_text}")
                
            if process.returncode == 0:
                return stdout_text  # Return actual output on success
            else:
                logger.error(f"Command failed with exit code {process.returncode}")
                return False
        else:
            await process.wait()
            return process.returncode == 0
    except asyncio.CancelledError:
        logger.warning(f"Command execution cancelled: {cmd}")
        raise
    except Exception as e:
        logger.error(f"Exception running command '{cmd}': {type(e).__name__}: {e}")
        return False

# Enhanced task functions
async def run_cleanup(remove_logs: bool = False) -> bool:
    """Execute cleanup task with configurable options."""
    cmd = "python scripts/cleanup.py"
    if remove_logs:
        cmd += " --remove-logs"
    return await run_command(cmd)

async def run_tests(path: str = "tests/", verbose: bool = False) -> bool:
    """Execute testing task with configurable test path and verbosity."""
    cmd = f"pytest {path}"
    if verbose:
        cmd += " -v"
    return await run_command(cmd)

async def run_deploy(env: str = "production", dry_run: bool = False) -> bool:
    """Execute deployment task with environment selection."""
    cmd = f"bash scripts/deploy.sh --env={env}"
    if dry_run:
        cmd += " --dry-run"
    return await run_command(cmd)

async def run_ml_training(model_name: str, epochs: int = 10) -> bool:
    """Execute ML model training task."""
    return await run_command(f"python scripts/train.py --model={model_name} --epochs={epochs}")

async def run_model_evaluation(model_path: str, test_data: str) -> Dict[str, float]:
    """Evaluate ML model performance on test data."""
    output = await run_command(
        f"python scripts/evaluate.py --model={model_path} --data={test_data} --json-output"
    )
    if output and isinstance(output, str):
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            logger.error("Failed to parse evaluation output as JSON")
    return {"accuracy": 0.0, "error": "Evaluation failed"}

async def run_ml_pipeline(cleanup_logs: bool = False, model_name: str = "default", 
                        epochs: int = 10, test_data: str = "test_data") -> bool:
    """Run the complete ML pipeline with proper error handling."""
    manager = TaskManager()
    
    # Register individual tasks with proper dependencies
    manager.register_task("pipeline_cleanup", 
                         lambda: run_cleanup(remove_logs=cleanup_logs),
                         "Pipeline cleanup")
    
    manager.register_task("pipeline_train", 
                         lambda: run_ml_training(model_name=model_name, epochs=epochs),
                         "Pipeline model training",
                         dependencies=["pipeline_cleanup"])
    
    manager.register_task("pipeline_evaluate", 
                         lambda: run_model_evaluation(model_path=model_name, test_data=test_data),
                         "Pipeline model evaluation",
                         dependencies=["pipeline_train"])
    
    manager.register_task("pipeline_deploy", 
                         lambda: run_deploy(dry_run=True),  # Using dry-run for safety
                         "Pipeline deployment",
                         dependencies=["pipeline_evaluate"])
    
    # Execute the sequence with dependency handling
    results = await manager.run_tasks_in_sequence(
        ["pipeline_cleanup", "pipeline_train", "pipeline_evaluate", "pipeline_deploy"],
        stop_on_failure=True
    )
    
    # Check if all tasks succeeded
    return all(result.success for result in results)

async def main() -> None:
    """Enhanced main entry point with more AI/ML focused tasks."""
    manager = TaskManager()
    await manager.setup_session()

    try:
        # Register basic tasks
        manager.register_task("cleanup", run_cleanup, "Clean up code and remove temporary files")
        manager.register_task("test", run_tests, "Run all tests")
        manager.register_task("deploy", run_deploy, "Deploy smart contracts", 
                             dependencies=["test"], timeout=300)
        
        # Register AI/ML specific tasks
        manager.register_task("train", run_ml_training, 
                             "Train ML model",
                             timeout=3600)  # 1 hour timeout
        
        manager.register_task("evaluate", run_model_evaluation, 
                             "Evaluate model performance",
                             dependencies=["train"], 
                             timeout=600)  # 10 min timeout
        
        # Register ML pipeline using proper async function instead of lambda
        manager.register_task("ml_pipeline", 
                             run_ml_pipeline,
                             "Run complete ML pipeline",
                             timeout=7200)  # 2 hour timeout

        # Parse arguments
        if len(sys.argv) > 1:
            task_name = sys.argv[1]
            args = {}
            
            # Parse additional arguments
            for arg in sys.argv[2:]:
                if '=' in arg:
                    key, value = arg.split('=', 1)
                    # Handle boolean values
                    if value.lower() in ('true', 'yes', '1'):
                        args[key] = True
                    elif value.lower() in ('false', 'no', '0'):
                        args[key] = False
                    # Handle numeric values
                    elif value.isdigit():
                        args[key] = int(value)
                    elif value.replace('.', '', 1).isdigit():
                        args[key] = float(value)
                    else:
                        args[key] = value
                else:
                    # Flag arguments
                    args[arg] = True
            
            # Run the task with parsed arguments
            result = await manager.run_task(task_name, **args)
            if isinstance(result, TaskResult):
                if result.success:
                    print(f"Task '{task_name}' completed successfully in {result.execution_time:.2f}s")
                    if result.resource_usage:
                        print(f"Resource usage: CPU: {result.resource_usage.get('cpu_percent', 0):.1f}%, "
                              f"Memory: {result.resource_usage.get('memory_percent', 0):.1f}%")
                    sys.exit(0)
                else:
                    print(f"Task '{task_name}' failed: {result.error_message}")
                    sys.exit(1)
            elif result:
                sys.exit(0)
            else:
                sys.exit(1)
        else:
            # List available tasks
            print("\nAvailable tasks:")
            for name, task in manager.tasks.items():
                deps = f" (depends on: {', '.join(task.dependencies)})" if task.dependencies else ""
                print(f"- {name}: {task.description}{deps}")

    finally:
        await manager.cleanup()

if __name__ == "__main__":
    asyncio.run(main())