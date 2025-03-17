import psutil
from typing import Dict
from dataclasses import dataclass
import time

@dataclass
class NodeHealth:
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, int]

class ResourceMonitor:
    def __init__(self, check_interval: int = 60):
        self.check_interval = check_interval
        self.last_check = 0
        
    def check_resources(self) -> NodeHealth:
        if time.time() - self.last_check < self.check_interval:
            return
            
        return NodeHealth(
            cpu_percent=psutil.cpu_percent(),
            memory_percent=psutil.virtual_memory().percent,
            disk_percent=psutil.disk_usage('/').percent,
            network_io=psutil.net_io_counters()._asdict()
        )
