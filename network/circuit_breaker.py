from enum import Enum
from datetime import datetime, timedelta
import threading
import logging
import asyncio
from core.interactive_utils import InteractiveSession, InteractiveConfig, InteractionLevel
from core.constants import INTERACTION_TIMEOUTS

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    def __init__(self, failure_threshold: int, reset_timeout: int, interactive: bool = True):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self._lock = threading.Lock()
        self.logger = logging.getLogger('CircuitBreaker')
        self.interactive = interactive
        self.session = None
        self._interrupt_requested = False
        self._cleanup_event = asyncio.Event()

    async def __aenter__(self):
        if self.interactive:
            self.session = InteractiveSession(
                level=InteractionLevel.NORMAL,
                config=InteractiveConfig(
                    timeout=INTERACTION_TIMEOUTS["emergency"],
                    persistent_state=True,
                    safe_mode=True
                )
            )
            await self.session.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.session:
            await self.session.__aexit__(exc_type, exc, tb)
            self.session = None
        await self._cleanup()

    async def allow_request_interactive(self) -> bool:
        """Interactive version of allow_request with monitoring"""
        try:
            with self._lock:
                if self.state == CircuitState.OPEN:
                    if await self._should_attempt_reset_interactive():
                        if self.interactive and self.session:
                            proceed = await self.session.confirm_with_timeout(
                                "\nCircuit is ready for reset attempt. Proceed?",
                                timeout=INTERACTION_TIMEOUTS["emergency"]
                            )
                            if not proceed:
                                return False
                        self.state = CircuitState.HALF_OPEN
                        self.logger.info("Circuit breaker entering half-open state")
                        return True
                    return False
                return True

        except Exception as e:
            self.logger.error(f"Error in allow_request_interactive: {str(e)}")
            raise

    async def record_failure_interactive(self):
        """Interactive version of record_failure with notifications"""
        try:
            with self._lock:
                self.failures += 1
                self.last_failure_time = datetime.now()

                if self.failures >= self.failure_threshold:
                    if self.interactive and self.session:
                        notify = await self.session.confirm_with_timeout(
                            f"\nFailure threshold ({self.failure_threshold}) reached. Open circuit?",
                            timeout=INTERACTION_TIMEOUTS["emergency"],
                            default=True
                        )
                        if not notify:
                            return

                    self.state = CircuitState.OPEN
                    self.logger.warning("Circuit breaker opened after reaching failure threshold")

        except Exception as e:
            self.logger.error(f"Error in record_failure_interactive: {str(e)}")
            raise

    async def _should_attempt_reset_interactive(self) -> bool:
        """Interactive version of reset check with timeout"""
        if not self.last_failure_time:
            return True

        time_since_failure = datetime.now() - self.last_failure_time
        should_reset = time_since_failure > timedelta(seconds=self.reset_timeout)

        if should_reset and self.interactive and self.session:
            self.logger.info(f"Circuit ready for reset after {time_since_failure.total_seconds():.1f}s")

        return should_reset

    async def _cleanup(self):
        """Cleanup resources"""
        try:
            self._cleanup_event.set()
            self.logger.info("Circuit breaker cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")

    def request_shutdown(self):
        """Request graceful shutdown"""
        self._interrupt_requested = True
        self.logger.info("Shutdown requested for circuit breaker")
