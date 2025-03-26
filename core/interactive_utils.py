from __future__ import annotations
import signal
import asyncio
import platform
import logging
import json
import os
from pathlib import Path
from typing import Optional, Any, Callable, List, Dict
from functools import wraps
from dataclasses import dataclass, field
from core.constants import INTERACTION_TIMEOUTS, InteractionLevel
from tqdm import tqdm
import psutil
import aiofiles

logger = logging.getLogger(__name__)

@dataclass 
class InteractiveConfig:
    timeout: int = INTERACTION_TIMEOUTS["default"]
    max_retries: int = 3
    auto_save: bool = True
    progress_tracking: bool = True
    error_recovery: bool = True
    progress_file: str = "interactive_progress.json"
    backup_enabled: bool = True
    backup_interval: int = 300  # 5 minutes
    safe_mode: bool = True  # Enforces additional safety checks
    max_input_length: int = 1024
    cleanup_on_exit: bool = True
    persistent_state: bool = True
    recovery_enabled: bool = True
    error_retry_delay: float = 1.0
    max_cleanup_wait: int = 10
    input_encoding: str = 'utf-8'
    sanitize_patterns: List[str] = field(default_factory=lambda: [';', '--', '/*', '*/', '#!'])
    emergency_timeout: int = 30
    recovery_attempts: int = 3
    heartbeat_interval: int = 60
    memory_threshold: float = 0.9  # 90% memory usage threshold
    resource_monitoring: bool = True
    status_interval: int = 10
    session_persistence: bool = True
    feedback_verbosity: int = 2
    resource_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'cpu': 0.8,
        'memory': 0.8,
        'disk': 0.8
    })
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None

class InteractionTimeout(Exception):
    """Raised when an interactive operation times out"""
    pass

def with_timeout(timeout_sec: int):
    """Decorator to add timeout to interactive functions"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout_sec)
            except asyncio.TimeoutError:
                raise InteractionTimeout(f"Operation timed out after {timeout_sec}s")
        return wrapper
    return decorator

    def __init__(self, 
                level: InteractionLevel = InteractionLevel.NORMAL,
                config: Optional[InteractiveConfig] = None):
        self._last_input = None
        self._backup_task = None
        self._restore_session = self._restore_progress
        self.level = level
        self.config = config or InteractiveConfig()
        self._platform = platform.system()
        self._prev_handlers = {}
        self._interrupt_flag = False
        self._progress = {}
        self._cleanup_hooks = []
        self._progress_path = Path("./progress").joinpath(config.progress_file if config else "interactive_progress.json")
        self._progress_path.parent.mkdir(parents=True, exist_ok=True)
        self._shutdown_tasks = []
        self._active_tasks = set()
        self._recovery_mode = False
        self._memory_monitor = None
        self._heartbeat_task = None
        self._status_task = None
        self._resource_stats = {}
        self._operation_history = []
        self._active_operations = set()
        
    async def __aenter__(self):
        """Enhanced context manager entry with NORMAL mode features"""
        try:
            self._setup_handlers()
            
            if self.level == InteractionLevel.NORMAL:
                # Start monitoring tasks
                if self.config.resource_monitoring:
                    self._start_resource_monitoring()
                    
                if self.config.status_interval > 0:
                    self._status_task = asyncio.create_task(self._periodic_status())
                    
                # Restore session if enabled
                if self.config.session_persistence:
                    await self._restore_session()
                    
            return self
            
        except Exception as e:
            logger.error(f"Session initialization failed: {e}")
            await self.__aexit__(type(e), e, e.__traceback__)
            raise
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Improved context manager exit with resource cleanup"""
        try:
            # Cancel backup task if running
            if self._backup_task and not self._backup_task.done():
                self._backup_task.cancel()
                try:
                    await asyncio.wait_for(self._backup_task, timeout=self.config.max_cleanup_wait)
                except asyncio.TimeoutError:
                    logger.warning("Backup task cleanup timed out")
                except asyncio.CancelledError:
                    pass

            # Cancel monitoring tasks
            if self._memory_monitor:
                self._memory_monitor.cancel()

            # Save progress if needed
            if exc_type and self.config.persistent_state:
                await self._save_progress()

            # Run cleanup hooks with timeout
            for hook in self._cleanup_hooks:
                try:
                    if asyncio.iscoroutinefunction(hook):
                        await asyncio.wait_for(hook(), timeout=self.config.max_cleanup_wait)
                    else:
                        hook()
                except Exception as e:
                    logger.error(f"Cleanup hook failed: {e}")

            # Cancel all remaining tasks
            remaining = [t for t in self._active_tasks if not t.done()]
            if remaining:
                for task in remaining:
                    task.cancel()
                await asyncio.gather(*remaining, return_exceptions=True)
        finally:
            self._restore_handlers()
            if self.config.cleanup_on_exit:
                await self._cleanup()

    def _setup_handlers(self):
        def interrupt_handler(signum, frame):
            self._interrupt_flag = True
            logger.info("Received interrupt, initiating graceful shutdown")
            
        if self._platform != 'Windows':
            self._prev_handlers[signal.SIGINT] = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, interrupt_handler)

    async def get_confirmation(self, prompt: str, timeout: int = INTERACTION_TIMEOUTS["confirmation"]) -> bool:
        """Get user confirmation with timeout"""
        if self.level == InteractionLevel.NONE:
            return True
            
        try:
            response = await self._get_input(f"{prompt} (y/n): ", timeout)
            return response.lower().strip() == "y"
        except InteractionTimeout:
            return False

    @with_timeout(INTERACTION_TIMEOUTS["input"])
    async def get_input(self, prompt: str, validator: Optional[Callable] = None) -> str:
        """Get validated user input with timeout"""
        while True:
            try:
                value = await self._get_input(prompt)
                if validator is None or validator(value):
                    return value
                print("Invalid input, please try again")
            except InteractionTimeout:
                raise
    
    def _setup_timeout(self, timeout: int):
        """Setup cross-platform timeout handlers"""
        if hasattr(signal, 'SIGALRM'):
            self._prev_handlers[signal.SIGALRM] = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
        if self._platform != 'Windows':  # Unix-like systems
            self._prev_handlers[signal.SIGALRM] = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
        else:  # Windows fallback using asyncio
            if hasattr(signal, 'alarm'):
                signal.alarm(0)

    def _restore_handlers(self):
        """Restore original signal handlers"""
        if platform.system() != 'Windows':
            signal.alarm(0)
        for sig, handler in self._prev_handlers.items():
            signal.signal(sig, handler)
        self._prev_handlers.clear()

    async def _get_input(self, prompt: str, timeout: int = INTERACTION_TIMEOUTS["input"]) -> str:
        """Cross-platform input with timeout and safety checks"""
        if self.level == InteractionLevel.NONE:
            raise InteractionTimeout("Interactive mode disabled")
            
        try:
            self._setup_timeout(timeout)
            loop = asyncio.get_event_loop()
            
            # Input validation and sanitization
            result = await loop.run_in_executor(None, input, prompt)
            result = result.strip()
            
            if self.config.safe_mode:
                # Basic safety checks
                if len(result) > 1024:  # Prevent extremely long inputs
                    raise ValueError("Input too long")
                if any(char in result for char in '\x00\x0A\x0D'):  # Block control characters
                    raise ValueError("Invalid characters in input")
                    
            self._last_input = result
            self._restore_handlers()
            return result
            
        except Exception as e:
            self._restore_handlers()
            if isinstance(e, InteractionTimeout):
                raise
            raise InteractionTimeout(f"Input error: {str(e)}")

    async def _get_input_with_timeout(self, prompt: str, timeout: Optional[int] = None) -> str:
        """Get input with timeout and proper error handling"""
        timeout = timeout or self.config.timeout
        try:
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, input, prompt),
                timeout
            )
            return self._sanitize_input(result)
        except asyncio.TimeoutError:
            logger.warning(f"Input timeout after {timeout}s")
            raise InteractionTimeout(f"Input timeout after {timeout}s")
        except Exception as e:
            logger.error(f"Input error: {str(e)}")
            raise

    def _sanitize_input(self, value: str) -> str:
        """Sanitize user input for safety"""
        # Remove control characters and extra whitespace
        value = ''.join(char for char in value if ord(char) >= 32)
        value = ' '.join(value.split())
        
        # Enforce length limit
        return value[:self.config.max_input_length]

    def _validate_input(self, value: str) -> bool:
        """Validate input meets security requirements"""
        if not value:
            return False
        if len(value) > self.config.max_input_length:
            return False
        # Check for suspicious patterns
        suspicious_patterns = [';', '--', '/*', '*/', '#!/']
        return not any(pattern in value for pattern in suspicious_patterns)

    async def confirm_with_timeout(self, prompt: str, 
                                 timeout: int = 30,
                                 default: bool = False) -> bool:
        """Get confirmation with timeout and default fallback"""
        try:
            return await self.get_confirmation(prompt, timeout)
        except InteractionTimeout:
            logger.warning(f"Confirmation timeout, using default: {default}")
            return default

    async def get_validated_input(self, 
                                prompt: str,
                                validator: Optional[Callable[[str], bool]] = None,
                                error_msg: str = "Invalid input",
                                retries: Optional[int] = None,
                                timeout: Optional[int] = None) -> Optional[str]:
        retries = retries or self.config.max_retries
        timeout = timeout or self.config.timeout

        for attempt in range(retries):
            try:
                value = await self._get_input_with_timeout(prompt, timeout)
                
                # Safety checks
                if self.config.safe_mode:
                    if len(value) > self.config.max_input_length:
                        raise ValueError("Input exceeds maximum length")
                    if any(pattern in value for pattern in self.config.sanitize_patterns):
                        raise ValueError("Input contains invalid patterns")
                
                if validator is None or validator(value):
                    return value
                
                logger.warning(f"Validation failed: {error_msg}")
                if attempt < retries - 1:
                    await asyncio.sleep(self.config.error_retry_delay)
                    print(f"Please try again ({retries-attempt-1} attempts remaining)")
                    
            except (asyncio.TimeoutError, ValueError) as e:
                logger.warning(f"Input error on attempt {attempt+1}: {str(e)}")
                if attempt == retries - 1:
                    return None
                    
            except KeyboardInterrupt:
                if await self.get_confirmation("\nDo you want to cancel input? (y/n): "):
                    raise
                if attempt < retries - 1:
                    continue

        return None

    async def _periodic_backup(self):
        """Periodically backup progress"""
        while True:
            try:
                await asyncio.sleep(self.config.backup_interval)
                await self._save_progress()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Backup failed: {str(e)}")

    async def _save_progress(self):
        """Save current progress atomically"""
        if not self._progress:
            return

        try:
            temp_path = self._progress_path.with_suffix('.tmp')
            # Ensure directory exists
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to temporary file first
            async with aiofiles.open(temp_path, 'w') as f:
                data = {
                    'last_input': self._last_input,
                    'progress': self._progress,
                    'level': self.level.value,
                    'timestamp': asyncio.get_event_loop().time(),
                    'version': '1.0'  # For future compatibility
                }
                await f.write(json.dumps(data, indent=2))
            
            # Atomic rename
            temp_path.replace(self._progress_path)
            
        except Exception as e:
            logger.error(f"Failed to save progress: {str(e)}")
            # Attempt cleanup of temp file
            if temp_path.exists():
                temp_path.unlink()

    async def _restore_progress(self):
        """Restore previous progress if available"""
        try:
            if self._progress_path.exists():
                with open(self._progress_path) as f:
                    data = json.load(f)
                    self._progress = data.get('progress', {})
                    self._last_input = data.get('last_input')
                    logger.info("Restored previous progress")
        except Exception as e:
            logger.error(f"Failed to restore progress: {str(e)}")

    def _cleanup(self):
        """Clean up resources and handlers"""
        self._interrupt_flag = False
        if self._progress_path.exists() and not self.config.auto_save:
            try:
                self._progress_path.unlink()
            except Exception as e:
                logger.error(f"Failed to cleanup progress file: {str(e)}")
        self._progress.clear()

    def emergency_timeout(self, handler: Callable):
        """Set emergency timeout handler"""
        self._prev_handlers[signal.SIGALRM] = signal.signal(
            signal.SIGALRM, 
            lambda s, f: handler()
        )
        signal.alarm(300)  # 5 minute emergency timeout

    def register_cleanup(self, hook: Callable[[], None]):
        self._cleanup_hooks.append(hook)

    def _emergency_cleanup(self):
        """Emergency cleanup handler for unexpected termination"""
        if hasattr(self, '_progress_path') and self._progress_path.exists():
            try:
                self._progress_path.unlink()
            except Exception as e:
                logger.error(f"Emergency cleanup failed: {e}")

    def register_shutdown_task(self, task: Callable[[], None]):
        """Register a task to be run during shutdown"""
        self._shutdown_tasks.append(task)

    async def monitor_resources(self):
        """Monitor system resources during interactive session"""
        while True:
            try:
                metrics = await self.system_monitor.get_metrics()
                if metrics.memory_usage > self.config.memory_threshold:
                    logger.warning(f"High memory usage: {metrics.memory_usage}%")
                    await self._save_progress()
                    
                if self.system_monitor._check_critical_levels(metrics):
                    if not await self.get_confirmation(
                        "Critical resource usage detected. Continue?",
                        timeout=INTERACTION_TIMEOUTS["emergency"]
                    ):
                        await self.cleanup()
                        break
                        
                await asyncio.sleep(self.config.heartbeat_interval)
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                self.system_monitor.track_error(e, {"context": "resource_monitoring"})

    async def _periodic_status(self):
        """Periodic status updates for NORMAL mode"""
        while not self._interrupt_flag:
            try:
                if self._active_operations:
                    status = self._get_operations_status()
                    if self.config.progress_callback:
                        await self.config.progress_callback(status)
                    else:
                        logger.info(f"Active operations: {status}")
                        
                await asyncio.sleep(self.config.status_interval)
                
            except Exception as e:
                logger.error(f"Status update failed: {e}")

    async def confirm_with_feedback(self, prompt: str, 
                                  details: Optional[Dict] = None) -> bool:
        """Enhanced confirmation with detailed feedback for NORMAL mode"""
        if self.level != InteractionLevel.NORMAL:
            return await self.get_confirmation(prompt)
            
        try:
            if details and self.config.feedback_verbosity > 1:
                print("\nOperation Details:")
                for key, value in details.items():
                    print(f"  {key}: {value}")
                    
            return await self.get_confirmation(f"\n{prompt}")
            
        except Exception as e:
            logger.error(f"Confirmation dialog failed: {e}")
            return False
            
    def _start_resource_monitoring(self):
        """Start resource monitoring for NORMAL mode"""
        async def monitor_resources():
            while not self._interrupt_flag:
                try:
                    self._resource_stats = {
                        'cpu': psutil.cpu_percent(interval=1) / 100,
                        'memory': psutil.virtual_memory().percent / 100,
                        'disk': psutil.disk_usage('/').percent / 100
                    }
                    
                    # Check thresholds
                    for resource, value in self._resource_stats.items():
                        threshold = self.config.resource_thresholds.get(resource)
                        if threshold and value > threshold:
                            logger.warning(f"{resource.upper()} usage above threshold: {value:.1%}")
                            
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    logger.error(f"Resource monitoring failed: {e}")
                    
        self._active_tasks.add(asyncio.create_task(monitor_resources()))
