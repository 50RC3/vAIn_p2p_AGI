import asyncio
import os
import sys
import json
import signal
import logging
import time
from typing import Dict, Optional, Callable, Set, List, Any, Awaitable, Union, TypeVar, cast
from datetime import datetime
from contextlib import contextmanager
from enum import Enum
from dataclasses import dataclass
from types import FrameType

try:
    import aiofiles
except ImportError:
    aiofiles = None  # type: ignore

from core.constants import (
    SESSION_SAVE_PATH,
    INTERACTION_TIMEOUTS,
)

# Default interaction timeout in seconds
INTERACTION_TIMEOUT = INTERACTION_TIMEOUTS.get('default', 30)

T = TypeVar('T')

logger = logging.getLogger(__name__)

class InteractionTimeout(Exception):
    """Exception raised when an interaction times out."""

class InteractionError(Exception):
    """Exception raised when an interaction encounters an error."""

class InteractionLevel(Enum):
    """Defines the level of interactivity for model operations"""
    NONE = 0     # No interaction
    LOW = 1      # Minimal interaction (errors only)
    NORMAL = 2   # Standard interaction
    HIGH = 3     # Verbose interaction

@dataclass
class InteractiveConfig:
    """Configuration for interactive sessions"""
    timeout: float = 30.0
    persistent_state: bool = True
    safe_mode: bool = True
    progress_tracking: bool = True
    memory_threshold: float = 0.9
    max_retries: int = 3
    cleanup_timeout: int = 30

class Session:
    """Manages interactive sessions for model operations"""
    
    def __init__(self, level: InteractionLevel = InteractionLevel.NORMAL,
                 config: Optional[InteractiveConfig] = None):
        self.level = level
        self.config = config or InteractiveConfig()
        self._state: Dict[str, Any] = {}
        self._active = False
        
    async def __aenter__(self) -> 'Session':
        """Async context manager entry"""
        self._active = True
        logger.info("Starting interactive session with level: %s", self.level)
        return self
        
    async def __aexit__(self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Optional[Any]) -> None:
        """Async context manager exit"""
        self._active = False
        if self.config.persistent_state:
            await self._save_state()
        logger.info("Interactive session ended")
        
    async def get_confirmation(self, message: str, timeout: Optional[float] = None) -> bool:
        """Get user confirmation for an action"""
        if not self._active:
            return True
            
        if self.level == InteractionLevel.NONE:
            return True
            
        try:
            # Implementation depends on your UI system
            # This is a simple console-based version
            print(f"\n{message} (y/n)")
            response = await asyncio.wait_for(
                self._get_input(),
                timeout or self.config.timeout
            )
            return response.lower().startswith('y')
        except asyncio.TimeoutError:
            logger.warning("Confirmation timed out, proceeding with default action")
            return True
            
    async def _save_progress(self, state: Optional[Dict[str, Any]] = None) -> None:
        """Save session progress"""
        if state:
            self._state.update(state)
        if self.config.persistent_state:
            await self._save_state()
            
    async def _save_state(self) -> None:
        """Save session state to persistent storage"""
        try:
            state_file = os.path.join(SESSION_SAVE_PATH, f"{id(self)}_state.json")
            if aiofiles:
                async with aiofiles.open(state_file, 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(self._state))
            else:
                with open(state_file, 'w', encoding='utf-8') as f:
                    json.dump(self._state, f)
            logger.debug("Session state saved to %s", state_file)
        except (IOError, OSError, PermissionError) as e:
            logger.warning("Failed to save session state due to file error: %s", e)
        except (TypeError, ValueError) as e:
            logger.warning("Failed to save session state due to serialization error: %s", e)
        
    @staticmethod
    async def _get_input() -> str:
        """Get asynchronous console input"""
        return await asyncio.get_event_loop().run_in_executor(None, input)

class InteractiveSession:
    """A session for interactive use with proper cleanup and resource monitoring."""

    def __init__(self, session_id: str, session_path: Optional[str] = None, config: Optional[InteractiveConfig] = None) -> None:
        self.session_id = session_id
        self.session_path = session_path or os.path.join(SESSION_SAVE_PATH, session_id)
        self.config = config or InteractiveConfig()
        
        # Create session directory if it doesn't exist
        os.makedirs(self.session_path, exist_ok=True)
        
        # Initialize session state
        self._run_mode = "interactive"
        
        # Save original handlers
        self._restore_session_func: Optional[Callable[[], None]] = None
        
        # Initialize data structures
        self._prev_handlers: Dict[int, Union[Callable[[int, Optional[FrameType]], Any], int, None]] = {}
        self._progress: Dict[str, Dict[str, Any]] = {}
        self._cleanup_hooks: List[Callable[[], None]] = []
        
        # Task tracking
        self._monitoring_task: Optional[asyncio.Task[Any]] = None
        self._shutdown_tasks: List[asyncio.Task[Any]] = []
        self._active_tasks: Set[asyncio.Task[Any]] = set()
        
        # Resource monitoring
        self._monitor_interval = 10  # seconds
        self._resource_stats: Dict[str, Any] = {}
        self._operation_history: List[Dict[str, Any]] = []
        self._active_operations: Set[str] = set()
    
    async def __aenter__(self) -> "InteractiveSession":
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Optional[type], 
                        exc_val: Optional[Exception],
                        exc_tb: Optional[Any]) -> None:
        await self.shutdown()

    async def initialize(self) -> None:
        """Initialize the interactive session."""
        self._setup_handlers()
        
        # Set up monitoring if progress tracking is enabled
        if hasattr(self, 'config') and getattr(self, 'config', None) and getattr(self.config, 'progress_tracking', False):
            self._start_resource_monitoring()
        
        # Attempt to restore previous session if available
        if self._restore_session_func is not None:
            try:
                self._restore_session_func()
                logger.info(f"Restored session {self.session_id}")
            except Exception as e:
                logger.warning("Failed to restore session: %s", e)
    
    def _setup_handlers(self) -> None:
        """Set up signal handlers for graceful termination."""
        try:
            # Save original handlers
            if hasattr(signal, 'SIGINT'):
                self._prev_handlers[signal.SIGINT] = signal.getsignal(signal.SIGINT)
                signal.signal(signal.SIGINT, self._handle_interrupt)
            
            if hasattr(signal, 'SIGTERM'):
                self._prev_handlers[signal.SIGTERM] = signal.getsignal(signal.SIGTERM)
                signal.signal(signal.SIGTERM, self._handle_termination)
                
            logger.debug("Signal handlers set up")
        except Exception as e:
            logger.warning(f"Failed to set up signal handlers: {e}")
    
    def _restore_handlers(self) -> None:
        """Restore original signal handlers."""
        try:
            # Restore original handlers
            for sig, handler in self._prev_handlers.items():
                signal.signal(sig, handler)
            logger.debug("Original signal handlers restored")
        except Exception as e:
            logger.warning(f"Failed to restore signal handlers: {e}")
    
    def _cleanup(self) -> None:
        """Clean up resources and run cleanup hooks."""
        try:
            # Run cleanup hooks
            for hook in self._cleanup_hooks:
                try:
                    hook()
                except Exception as e:
                    logger.warning(f"Cleanup hook failed: {e}")
            
            # Save session progress
            self._save_progress()
            
            # Restore original signal handlers
            self._restore_handlers()
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
    
    async def shutdown(self) -> None:
        """Shutdown the interactive session."""
        try:
            # Cancel monitoring task
            if self._monitoring_task and not self._monitoring_task.done():
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass
            
            # Run cleanup operations
            self._cleanup()
            
            # Cancel any remaining tasks
            for task in self._active_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for all shutdown tasks to complete
            if self._shutdown_tasks:
                await asyncio.gather(*self._shutdown_tasks, return_exceptions=True)
                
            logger.info("Session shutdown complete")
        except Exception as e:
            logger.warning(f"Error during shutdown: {e}")
    
    async def input(self, prompt: str, timeout: Optional[int] = None) -> str:
        """Get input from the user with optional timeout."""
        timeout = timeout or INTERACTION_TIMEOUT
        
        if not sys.stdin.isatty():
            # Non-interactive mode
            return input(prompt)
        
        try:
            loop = asyncio.get_event_loop()
            user_input = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: input(prompt)), 
                timeout
            )
            return user_input
        except asyncio.TimeoutError:
            # Handle timeout
            raise InteractionTimeout(f"Input timed out after {timeout} seconds")
        except (KeyboardInterrupt, EOFError, ValueError, IOError) as e:
            logger.warning(f"Input error: {e}")
            raise InteractionTimeout(f"Input error: {str(e)}")
    
    def _get_input_windows(self) -> str:
        """Windows-specific input handling."""
        import msvcrt
        
        result = []
        while True:
            if msvcrt.kbhit():
                char = msvcrt.getwche()
                if char == '\r':  # Enter key
                    print()
                    break
                result.append(char)
        return ''.join(result)
    
    def _get_input_unix(self) -> str:
        """Unix-specific input handling with timeout."""
        has_alarm = hasattr(signal, 'SIGALRM') and hasattr(signal, 'alarm')
        result = ""
        
        try:
            # Set alarm if available (Unix systems only)
            if has_alarm:
                signal.signal(getattr(signal, 'SIGALRM'), self._input_timeout_handler)
                if hasattr(signal, 'alarm'):
                    signal.alarm(int(INTERACTION_TIMEOUT))
            
            # Get input
            result = input()
            # Disable alarm if it was set
            if has_alarm and hasattr(signal, 'alarm'):
                try:
                    signal_module = signal  # Create local reference for type checking
                    if hasattr(signal_module, 'alarm'):
                        getattr(signal_module, 'alarm')(0)  # Use getattr to avoid static type checking issues
                except (AttributeError, ValueError):
                    pass
                
        except InteractionTimeout:
            return ""  # Empty string on timeout
        finally:
            # Ensure alarm is disabled if it was set
            # Ensure alarm is disabled if it was set
            if has_alarm and hasattr(signal, 'alarm'):
                try:
                    if hasattr(signal, 'alarm'):  # Double-check to satisfy type checker
                        signal.alarm(0)
                except (AttributeError, ValueError):
                    pass
        return result
        
        return result
    
    async def get_input(self, prompt: str = "", timeout: Optional[int] = None) -> str:
        """Get input with timeout handling."""
        timeout = timeout or INTERACTION_TIMEOUT
        
        print(prompt, end='', flush=True)
        
        try:
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, self._get_input_unix if os.name != 'nt' else self._get_input_windows),
                timeout
            )
            return result
        except asyncio.TimeoutError:
            self._restore_handlers()
            raise InteractionTimeout(f"Input timed out after {timeout} seconds")
        except (asyncio.CancelledError, OSError, IOError, ValueError) as e:
            self._restore_handlers()
            raise InteractionTimeout(f"Input error: {str(e)}") from e
    
    def _input_timeout_handler(self, signum: int, frame: Any) -> None:
        """Handle timeout for input operations."""
        raise InteractionTimeout(f"Input timed out after {INTERACTION_TIMEOUT} seconds")
    
    def _handle_interrupt(self, signum: int, frame: Any) -> None:
        """Handle interrupt signal (SIGINT)."""
        try:
            # Disable alarm if it was set to avoid alarm signal during cleanup
            if hasattr(signal, 'SIGALRM') and hasattr(signal, 'alarm'):
                try:
                    signal.alarm(0)
                except (AttributeError, ValueError):
                    pass
                    
            logger.info("Interrupt received, shutting down gracefully...")
            
            # Run cleanup immediately for synchronous context
            self._cleanup()
            
            print("\nSession interrupted. Exiting gracefully...")
            sys.exit(0)
        except Exception as e:
            logger.warning(f"Error handling interrupt: {e}")
            sys.exit(1)
    
    def _handle_termination(self, signum: int, frame: Any) -> None:
        """Handle termination signal (SIGTERM)."""
        try:
            logger.info("Termination signal received, shutting down...")
            
            # Run cleanup immediately for synchronous context
            self._cleanup()
            
            print("\nSession terminated. Exiting...")
            sys.exit(0)
        except Exception as e:
            logger.warning("Error handling termination: %s", e)
            sys.exit(1)
    
    async def prompt_yes_no(self, question: str, default: Optional[str] = None, timeout: Optional[int] = None) -> bool:
        """Prompt for a yes/no answer with timeout."""
        timeout = timeout or INTERACTION_TIMEOUT
        
        if default is not None:
            default = default.lower()
            if default not in ('y', 'n'):
                default = None
        
        choices = ' [Y/n]: ' if default == 'y' else ' [y/N]: ' if default == 'n' else ' [y/n]: '
        
        try:
            while True:
                response = await self.input(question + choices, timeout)
                response = response.lower().strip()
                
                if not response and default:
                    return default == 'y'
                
                if response in ('y', 'yes'):
                    return True
                if response in ('n', 'no'):
                    return False
                
                print("Please respond with 'y' or 'n'")
        except InteractionTimeout:
            if default is not None:
                logger.warning("Prompt timed out, using default: %s", default)
                return default == 'y'
            raise
    
    async def prompt_options(self, question: str, options: List[str], default: Optional[str] = None, timeout: Optional[int] = None) -> str:
        """Prompt for selecting from a list of options with timeout."""
        timeout = timeout or INTERACTION_TIMEOUT
        
        if default is not None and default not in options:
            default = None
        
        # Format options with numbers
        option_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])
        prompt_text = f"{question}\n{option_text}\nSelect an option (1-{len(options)})"
        
        if default is not None:
            default_num = options.index(default) + 1
            prompt_text += f" [default: {default_num}]: "
        else:
            prompt_text += ": "
        
        try:
            while True:
                response = await self.input(prompt_text, timeout)
                response = response.strip()
                
                if not response and default is not None:
                    return default
                
                try:
                    choice = int(response)
                    if 1 <= choice <= len(options):
                        return options[choice-1]
                except ValueError:
                    pass
                
                print(f"Please enter a number between 1 and {len(options)}")
        except InteractionTimeout:
            if default is not None:
                logger.warning("Prompt timed out, using default: %s", default)
                return default
            raise
    
    def _save_progress(self) -> None:
        """Save session progress to disk."""
        progress_file = os.path.join(self.session_path, "progress.json")
        try:
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(self._progress, f, indent=2)
            logger.debug("Session progress saved")
        except Exception as e:
            logger.warning("Failed to save session progress: %s", e)
    
    async def _save_to_disk(self, data: Dict[str, Any], filename: str) -> None:
        """Save data to disk asynchronously."""
        try:
            filepath = os.path.join(self.session_path, filename)
            if aiofiles:
                async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(data, indent=2))
            else:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            logger.debug("Data saved to %s", filename)
        except (IOError, OSError, PermissionError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to save data to {filename}: {e}")
    
    def register_cleanup_hook(self, hook: Callable[[], None]) -> None:
        """Register a function to be called during cleanup."""
        self._cleanup_hooks.append(hook)
    
    def update_progress(self, category: str, key: str, value: Any) -> None:
        """Update session progress for tracking."""
        if category not in self._progress:
            self._progress[category] = {}
        
        self._progress[category][key] = value
        self._save_progress()
    
    async def load_from_file(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load data from a file in the session directory."""
        filepath = os.path.join(self.session_path, filename)
        try:
            if not os.path.exists(filepath):
                return None
                
            if aiofiles:
                async with aiofiles.open(filepath, 'r', encoding='utf-8') as f:
                    content = await f.read()
            else:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
            return cast(Dict[str, Any], json.loads(content))
        except (IOError, OSError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load {filename}: {e}")
            return None
    
    def setup_signal_handler(self, signal_type: int, handler: Optional[Callable[[int, Any], None]] = None) -> Union[Callable[[int, Optional[FrameType]], Any], int, None]:
        """Set up a signal handler and return the previous one."""
        previous_handler = signal.getsignal(signal_type)
        
        if handler is not None:
            signal.signal(signal_type, handler)
            
        # Ensure alarm handling on Unix-like systems
        # Only set SIGALRM handler if it exists (Unix systems)
        if hasattr(signal, 'SIGALRM') and hasattr(signal, 'alarm'):
            try:
                signal.signal(getattr(signal, 'SIGALRM'), self._input_timeout_handler)
            except (AttributeError, ValueError):
                logger.debug("SIGALRM is not available on this platform")
            
        return previous_handler
    
    def add_task(self, task: asyncio.Task[Any]) -> None:
        """Track an active task."""
        self._active_tasks.add(task)
    
    def remove_task(self, task: asyncio.Task[Any]) -> None:
        """Remove a completed task from tracking."""
        if task in self._active_tasks:
            self._active_tasks.remove(task)
    
    def _start_resource_monitoring(self) -> None:
        """Start monitoring system resources."""
        async def _monitor_resources() -> None:
            try:
                while True:
                    # Record basic stats about memory, CPU, etc.
                    self._resource_stats = self._get_system_stats()
                    
                    # Save progress periodically
                    self._save_progress()
                    
                    # Check for critical resource levels
                    self._check_critical_levels()
                    
                    # Wait for next interval
                    await asyncio.sleep(self._monitor_interval)
            except Exception as e:
                logger.warning(f"Resource monitoring error: {e}")
        
        self._monitoring_task = asyncio.create_task(_monitor_resources())
    
    def _get_system_stats(self) -> Dict[str, Any]:
        """Get current system resource stats."""
        try:
            import psutil
            stats = {
                "timestamp": datetime.now().isoformat(),
                "memory_used": psutil.virtual_memory().percent,
                "cpu_usage": psutil.cpu_percent(),
                "disk_usage": psutil.disk_usage('/').percent
            }
            logger.debug(f"System stats: {stats}")
            return stats
        except (ImportError, MemoryError, OSError, RuntimeError) as e:
            logger.warning(f"Failed to get system stats: {e}")
            return {"error": str(e)}
    
    def _check_critical_levels(self) -> None:
        """Check if any resources are at critical levels."""
        if not self._resource_stats:
            return

        critical_thresholds = {
            "memory_used": 90,
            "cpu_usage": 95,
            "disk_usage": 95
        }

        for resource, value in self._resource_stats.items():
            if resource in critical_thresholds and value > critical_thresholds[resource]:
                logger.critical(
                    "Critical resource level - %s: %s%%",
                    resource, value
                )
    
    def _check_operations_status(self) -> Dict[str, str]:
        """Check the status of active operations."""
        try:
            status = {op: "running" for op in self._active_operations}
            
            # Update with completed operations from history
            for op in self._operation_history:
                if op["name"] in status and op["status"] == "completed":
                    status[op["name"]] = "completed"
                    
            logger.debug(f"Operations status: {status}")
            return status
        except Exception as e:
            logger.warning(f"Error checking operations: {e}")
            return {}
    
    async def track_operation(self, operation_name: str, timeout: Optional[int] = None, details: Optional[Dict[str, Any]] = None) -> bool:
        """Track an operation with timeout."""
        timeout = timeout or INTERACTION_TIMEOUT
        start_time = time.time()
        
        try:
            # Register operation as active
            self._active_operations.add(operation_name)
            
            # Record operation start
            operation_record = {
                "name": operation_name,
                "start_time": start_time,
                "details": details or {},
                "status": "running"
            }
            self._operation_history.append(operation_record)
            
            # Wait for timeout or until operation is manually marked as complete
            while time.time() - start_time < timeout:
                if operation_name not in self._active_operations:
                    # Operation was marked complete elsewhere
                    return True
                await asyncio.sleep(1)
                
            # Operation timed out
            logger.warning(f"Operation timed out: {operation_name}")
            return False
        except Exception as e:
            logger.warning(f"Error tracking operation: {e}")
            return False
        finally:
            # Always remove from active operations
            if operation_name in self._active_operations:
                self._active_operations.remove(operation_name)
                
                # Update operation history
                for op in self._operation_history:
                    if op["name"] == operation_name and op["status"] == "running":
                        op["end_time"] = time.time()
                        op["status"] = "completed" if operation_name not in self._active_operations else "failed"
                        op["duration"] = op["end_time"] - op["start_time"]
