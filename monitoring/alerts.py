from typing import Dict, List, Callable
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Alert:
    severity: str
    message: str
    timestamp: datetime
    metadata: Dict

class AlertSystem:
    def __init__(self):
        self.handlers: Dict[str, List[Callable]] = {
            'critical': [],
            'warning': [],
            'info': []
        }
        
    def add_handler(self, severity: str, handler: Callable):
        self.handlers[severity].append(handler)
        
    def trigger_alert(self, alert: Alert):
        for handler in self.handlers[alert.severity]:
            handler(alert)
