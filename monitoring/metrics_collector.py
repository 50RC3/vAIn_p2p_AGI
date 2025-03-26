from typing import Dict, Any
import time
import logging
import asyncio
from dataclasses import dataclass, field
from collections import defaultdict
import json
from pathlib import Path
from cryptography.fernet import Fernet
import base64

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    network_io: Dict[str, float] = field(default_factory=dict)
    active_connections: int = 0
    error_count: int = 0
    last_updated: float = field(default_factory=time.time)

class MetricsCollector:
    def __init__(self, encryption_key: bytes, metrics_dir: str = "data/metrics"):
        self.metrics: Dict[str, Any] = defaultdict(dict)
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.fernet = Fernet(encryption_key)
        self._rotation_interval = 3600  # 1 hour
        self._last_rotation = time.time()
        
    async def record_metric(self, category: str, name: str, value: Any):
        timestamp = time.time()
        self.metrics[category][name] = {
            "value": value,
            "timestamp": timestamp
        }
        
        await self._check_rotation()
        await self._encrypt_and_save(category, name, value, timestamp)
    
    async def get_metrics(self, category: str = None) -> Dict[str, Any]:
        if category:
            return self.metrics.get(category, {})
        return self.metrics
    
    async def _encrypt_and_save(self, category: str, name: str, value: Any, timestamp: float):
        data = {
            "category": category,
            "name": name,
            "value": value,
            "timestamp": timestamp
        }
        encrypted = self.fernet.encrypt(json.dumps(data).encode())
        
        filepath = self.metrics_dir / f"{category}_{int(timestamp)}.enc"
        with open(filepath, "wb") as f:
            f.write(encrypted)
            
    async def _check_rotation(self):
        if time.time() - self._last_rotation >= self._rotation_interval:
            await self._rotate_metrics()
            
    async def _rotate_metrics(self):
        # Archive old metrics
        archive_dir = self.metrics_dir / "archive"
        archive_dir.mkdir(exist_ok=True)
        
        timestamp = int(time.time())
        for file in self.metrics_dir.glob("*.enc"):
            if file.stat().st_mtime < time.time() - self._rotation_interval:
                file.rename(archive_dir / f"{timestamp}_{file.name}")
                
        self._last_rotation = time.time()
