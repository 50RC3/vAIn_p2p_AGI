from typing import Dict, List, Callable, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
from core.interactive_utils import InteractiveSession, InteractionLevel, InteractiveConfig
from core.constants import INTERACTION_TIMEOUTS

logger = logging.getLogger(__name__)

@dataclass
class Alert:
    severity: str
    message: str
    timestamp: datetime
    metadata: Dict

class AlertSystem:
    def __init__(self, interactive: bool = True):
        self.handlers: Dict[str, List[Callable]] = {
            'critical': [],
            'warning': [],
            'info': []
        }
        self.interactive = interactive
        self.session: Optional[InteractiveSession] = None
        self._interrupt_requested = False
        self._alert_history: List[Alert] = []
        self._max_history = 1000

    def _rotate_history(self) -> None:
        """Maintain alert history size"""
        if len(self._alert_history) > self._max_history:
            self._alert_history = self._alert_history[-self._max_history:]

    async def trigger_alert_interactive(self, alert: Alert) -> bool:
        """Trigger alert with interactive confirmation and retry logic"""
        try:
            if not alert or not isinstance(alert, Alert):
                logger.error("Invalid alert object")
                return False

            # Store alert
            self._alert_history.append(alert)
            self._rotate_history()

            if not self.interactive:
                return await self._handle_alert(alert)

            self.session = InteractiveSession(
                level=InteractionLevel.NORMAL,
                config=InteractiveConfig(
                    timeout=INTERACTION_TIMEOUTS["emergency"],
                    persistent_state=True,
                    safe_mode=True
                )
            )

            async with self.session:
                return await self._handle_alert(alert)

        except Exception as e:
            logger.error(f"Failed to trigger alert: {str(e)}")
            return False
        finally:
            if self.session:
                await self.session.__aexit__(None, None, None)

    async def _handle_alert(self, alert: Alert) -> bool:
        """Handle alert with retries and timeouts"""
        if alert.severity == 'critical':
            if self.interactive and not await self.session.confirm_with_timeout(
                f"Critical alert: {alert.message}\nProceed with handlers?",
                timeout=15
            ):
                return False

        retries = 3
        for attempt in range(retries):
            try:
                handlers = self.handlers.get(alert.severity, [])
                if not handlers:
                    logger.warning(f"No handlers for severity: {alert.severity}")
                    return False

                for handler in handlers:
                    await self._handle_with_timeout(handler, alert)
                return True

            except Exception as e:
                if attempt == retries - 1:
                    logger.error(f"Alert handling failed after {retries} attempts: {str(e)}")
                    return False
                logger.warning(f"Retrying alert handling ({attempt + 1}/{retries})")
                continue

    async def _handle_with_timeout(self, handler: Callable, alert: Alert) -> None:
        """Execute handler with timeout"""
        if self.session:
            async with self.session.timeout(30):
                handler(alert)

    def add_handler(self, severity: str, handler: Callable):
        if severity not in self.handlers:
            raise ValueError(f"Invalid severity level: {severity}")
        self.handlers[severity].append(handler)

    def get_recent_alerts(self, severity: Optional[str] = None, limit: int = 100) -> List[Alert]:
        """Get recent alerts with optional filtering"""
        alerts = self._alert_history
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return alerts[-limit:]

    def cleanup(self) -> None:
        """Cleanup resources"""
        if hasattr(self, 'session') and self.session:
            self.session.cleanup()
