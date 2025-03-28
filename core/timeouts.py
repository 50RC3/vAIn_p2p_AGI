from typing import Any, Optional, Callable, TypeVar, Awaitable, Coroutine, Union, cast
import asyncio
import logging
import platform
import time
import signal
import functools

logger = logging.getLogger(__name__)

# Type for function return
T = TypeVar('T')

class CustomTimeoutError(Exception):
    """Raised when an operation times out."""
    pass

async def with_timeout(
    coro: Awaitable[T], 
    timeout: float, 
    timeout_message: str = "Operation timed out"
) -> T:
    """
    Execute a coroutine with a timeout.
    
    Args:
        coro: The coroutine to execute
        timeout: Timeout in seconds
        timeout_message: Message to include in the timeout exception
    
    Returns:
        The result of the coroutine
        
    Raises:
        TimeoutError: If the operation times out
    """
    try:
        return await asyncio.wait_for(coro, timeout)
    except asyncio.TimeoutError as exc:
        raise CustomTimeoutError(timeout_message) from exc

def timeout_after(seconds: float, error_message: str = "Function call timed out") -> Callable[[Callable[..., T]], Callable[..., Union[T, Coroutine[Any, Any, T]]]]:
    """
    Decorator for applying a timeout to a function.
    
    Args:
        seconds: Timeout duration in seconds
        error_message: Message to include in exception
    
    Usage:
        @timeout_after(5, "API call timed out")
        async def api_call():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., Union[T, Coroutine[Any, Any, T]]]:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> T:
                return await with_timeout(
                    func(*args, **kwargs), 
                    seconds, 
                    error_message
                )
            return async_wrapper
        else:
            # For non-async functions on Unix systems, we can use SIGALRM
            # On Windows, we'll use threading as a fallback
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> T:
                # Check if SIGALRM is available in the signal module
                if hasattr(signal, 'SIGALRM') and hasattr(signal, 'alarm'):
                    # Unix implementation using SIGALRM
                    def handler(signum: int, frame: Any) -> None:
                        raise CustomTimeoutError(error_message)
                    
                    # Set the alarm and previous handler
                    old_handler = signal.signal(getattr(signal, 'SIGALRM'), handler)
                    getattr(signal, 'alarm')(int(seconds))
                    
                    try:
                        result = func(*args, **kwargs)
                    finally:
                        # Cancel the alarm and restore the previous handler
                        getattr(signal, 'alarm')(0)
                        signal.signal(getattr(signal, 'SIGALRM'), old_handler)
                    return result
                else:
                    # Windows implementation using threading
                    import threading
                    from typing import List
                    func_result: List[T] = []
                    func_exception: List[Exception] = []
                    
                    def worker() -> None:
                        try:
                            func_result.append(func(*args, **kwargs))
                        except (ValueError, TypeError, RuntimeError, IOError) as e:
                            func_exception.append(e)
                    
                    thread = threading.Thread(target=worker)
                    thread.daemon = True
                    thread.start()
                    thread.join(seconds)
                    
                    if thread.is_alive():
                        raise CustomTimeoutError(error_message)
                    
                    if func_exception:
                        raise func_exception[0]
                    
                    if func_result:
                        return func_result[0]
                    
                    raise CustomTimeoutError("Function finished but returned no result")
            
            return sync_wrapper
    
    return decorator

# Constants for common timeouts
DEFAULT_TIMEOUTS = {
    "short": 5,     # For quick operations like simple API calls
    "medium": 30,   # For moderately complex operations
    "long": 120,    # For intensive computations
    "very_long": 300,  # For extremely intensive operations
    "infinite": None   # No timeout
}

# Platform-specific timeouts
if platform.system() == "Windows":
    # Windows often needs longer timeouts for some operations
    DEFAULT_TIMEOUTS.update({
        "process_spawn": 10,
        "network": 15,
    })
else:
    # Unix systems can be a bit faster
    DEFAULT_TIMEOUTS.update({
        "process_spawn": 5,
        "network": 10,
    })

def get_timeout(timeout_key: str, default: Optional[float] = 30) -> Optional[float]:
    """Get a timeout value by key from the default timeouts."""
    return DEFAULT_TIMEOUTS.get(timeout_key, default)
