from enum import Enum, auto
from datetime import datetime, timedelta
import logging
import asyncio
from typing import Dict, List, Optional, NamedTuple
from statistics import mean, median
from dataclasses import dataclass, field
from core.interactive_utils import InteractiveSession, InteractiveConfig, InteractionLevel
from core.constants import INTERACTION_TIMEOUTS

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class FailureEvent:
    timestamp: datetime
    duration: float = 0.0
    recovery_time: float = 0.0

@dataclass
class CircuitMetrics:
    failure_events: List[FailureEvent] = field(default_factory=list)
    reset_attempts: int = 0
    successful_resets: int = 0
    last_reset_time: Optional[datetime] = None
    
    def add_failure(self, duration: float = 0.0):
        self.failure_events.append(FailureEvent(datetime.now(), duration))
    
    def record_reset_attempt(self, successful: bool):
        self.reset_attempts += 1
        if successful:
            self.successful_resets += 1
            self.last_reset_time = datetime.now()
            if self.failure_events:
                self.failure_events[-1].recovery_time = (
                    datetime.now() - self.failure_events[-1].timestamp
                ).total_seconds()

    def get_stats(self) -> Dict:
        if not self.failure_events:
            return {"failure_count": 0}
            
        return {
            "failure_count": len(self.failure_events),
            "avg_failure_duration": mean(e.duration for e in self.failure_events),
            "avg_recovery_time": mean(e.recovery_time for e in self.failure_events if e.recovery_time),
            "reset_success_rate": self.successful_resets / max(1, self.reset_attempts),
            "last_failure": self.failure_events[-1].timestamp.isoformat(),
            "last_reset": self.last_reset_time.isoformat() if self.last_reset_time else None
        }

class ResetStrategy(Enum):
    FIXED = auto()
    EXPONENTIAL = auto()
    INCREMENTAL = auto()

@dataclass
class ResetConfig:
    strategy: ResetStrategy = ResetStrategy.FIXED
    base_timeout: int = 60
    max_timeout: int = 3600
    multiplier: float = 2.0
    increment: int = 60

class CircuitBreaker:
    def __init__(self, failure_threshold: int, reset_timeout: int, 
                 interactive: bool = True, service_name: str = "default",
                 reset_config: Optional[ResetConfig] = None):
        self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self._lock = asyncio.Lock()  # Changed from threading.Lock
        self.logger = logging.getLogger('CircuitBreaker')
        self.interactive = interactive
        self.session = None
        self._interrupt_requested = False
        self._cleanup_event = asyncio.Event()
        self.reset_config = reset_config or ResetConfig()
        self.metrics = CircuitMetrics()
        self._current_timeout = reset_timeout

    async def __aenter__(self):
        if (self.interactive):
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
            async with self._lock:  # Changed from with to async with
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
            start_time = datetime.now()
            async with self._lock:  # Changed from with to async with
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
                
                failure_duration = (datetime.now() - start_time).total_seconds()
                self.metrics.add_failure(failure_duration)

        except Exception as e:
            self.logger.error(f"Error in record_failure_interactive: {str(e)}")
            raise

    async def _calculate_next_timeout(self) -> int:
        if self.reset_config.strategy == ResetStrategy.FIXED:
            return self.reset_timeout
        
        if self.reset_config.strategy == ResetStrategy.EXPONENTIAL:
            self._current_timeout = min(
                self._current_timeout * self.reset_config.multiplier,
                self.reset_config.max_timeout
            )
        else:  # INCREMENTAL
            self._current_timeout = min(
                self._current_timeout + self.reset_config.increment,
                self.reset_config.max_timeout
            )
        
        return self._current_timeout

    async def _should_attempt_reset_interactive(self) -> bool:
        """Interactive version of reset check with timeout"""
        if not self.last_failure_time:
            return True

        time_since_failure = datetime.now() - self.last_failure_time
        next_timeout = await self._calculate_next_timeout()
        should_reset = time_since_failure > timedelta(seconds=next_timeout)

        if should_reset and self.interactive and self.session:
            self.logger.info(
                f"Circuit ready for reset after {time_since_failure.total_seconds():.1f}s "
                f"(Strategy: {self.reset_config.strategy.name})"
            )
            self.metrics.record_reset_attempt(True)
        
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

class CircuitBreakerManager:
    def __init__(self, interactive: bool = True):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()
        self.interactive = interactive
        self.logger = logging.getLogger('CircuitBreakerManager')
        
    async def get_breaker(self, service_name: str, 
                         failure_threshold: int = 5, 
                         reset_timeout: int = 60) -> CircuitBreaker:
        """Get or create a circuit breaker for a service"""
        async with self._lock:
            if service_name not in self._breakers:
                self._breakers[service_name] = CircuitBreaker(
                    failure_threshold=failure_threshold,
                    reset_timeout=reset_timeout,
                    interactive=self.interactive,
                    service_name=service_name
                )
            return self._breakers[service_name]
            
    async def check_all_breakers(self) -> Dict[str, CircuitState]:
        """Get status of all circuit breakers"""
        return {name: breaker.state for name, breaker in self._breakers.items()}

    async def reset_all_breakers(self) -> None:
        """Reset all circuit breakers to closed state"""
        async with self._lock:
            for breaker in self._breakers.values():
                breaker.state = CircuitState.CLOSED
                breaker.failures = 0
                breaker.last_failure_time = None

    async def remove_breaker(self, service_name: str) -> None:
        """Remove a circuit breaker"""
        async with self._lock:
            if service_name in self._breakers:
                await self._breakers[service_name]._cleanup()
                del self._breakers[service_name]

    async def _cleanup(self) -> None:
        """Cleanup all circuit breakers"""
        async with self._lock:
            for breaker in self._breakers.values():
                await breaker._cleanup()
            self._breakers.clear()

    def request_shutdown(self) -> None:
        """Request shutdown for all circuit breakers"""
        for breaker in self._breakers.values():
            breaker.request_shutdown()

    async def get_metrics(self, service_name: str = None) -> Dict:
        """Get metrics for one or all circuit breakers"""
        if service_name and service_name in self._breakers:
            return {service_name: self._breakers[service_name].metrics.get_stats()}
            
        return {name: breaker.metrics.get_stats() 
                for name, breaker in self._breakers.items()}
