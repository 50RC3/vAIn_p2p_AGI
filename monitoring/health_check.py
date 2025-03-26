import logging
import time
import asyncio
from typing import Dict, Optional
from dataclasses import dataclass
from core.constants import INTERACTION_TIMEOUTS

logger = logging.getLogger(__name__)

@dataclass
class HealthStatus:
    healthy: bool = True
    last_check: float = 0
    error_count: int = 0
    message: str = ""

class HealthCheckManager:
    def __init__(self, check_interval: int = 60):
        self.check_interval = check_interval
        self.health_checks = {}
        self.status_cache = {}
        self._running = False
        self._lock = asyncio.Lock()
        
    async def register_check(self, name: str, check_fn, timeout: float = 5.0):
        """Register a new health check"""
        async with self._lock:
            self.health_checks[name] = {
                'function': check_fn,
                'timeout': timeout,
                'status': HealthStatus()
            }

    async def run_checks(self) -> Dict[str, HealthStatus]:
        """Run all registered health checks"""
        results = {}
        async with self._lock:
            for name, check in self.health_checks.items():
                try:
                    async with asyncio.timeout(check['timeout']):
                        healthy = await check['function']()
                        
                    status = check['status']
                    status.healthy = healthy
                    status.last_check = time.time()
                    
                    if not healthy:
                        status.error_count += 1
                        status.message = f"Check failed, attempts: {status.error_count}"
                    else:
                        status.error_count = 0
                        status.message = "Healthy"
                        
                    results[name] = status

                except asyncio.TimeoutError:
                    logger.error(f"Health check timeout: {name}")
                    results[name] = HealthStatus(
                        healthy=False,
                        last_check=time.time(),
                        message="Check timed out"
                    )
                except Exception as e:
                    logger.error(f"Health check error for {name}: {str(e)}")
                    results[name] = HealthStatus(
                        healthy=False,
                        last_check=time.time(),
                        message=f"Check error: {str(e)}"
                    )

        return results

    async def start_monitoring(self):
        """Start periodic health monitoring"""
        self._running = True
        while self._running:
            results = await self.run_checks()
            self.status_cache = results
            await asyncio.sleep(self.check_interval)

    async def stop(self):
        """Stop health monitoring"""
        self._running = False
