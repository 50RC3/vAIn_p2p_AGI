import os
import psutil
import asyncio
import logging
from typing import Callable, Optional

class MemoryMonitor:
    def __init__(self, threshold: float = 0.9, check_interval: int = 60):
        """
        Initialize memory monitor
        
        Args:
            threshold: Memory usage threshold (0-1) to trigger warnings
            check_interval: How often to check memory usage in seconds
        """
        self.threshold = threshold
        self.check_interval = check_interval
        self.logger = logging.getLogger('MemoryMonitor')
        self.callback: Optional[Callable] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._interrupt_requested = False
        
    async def start_monitoring(self, callback: Optional[Callable] = None):
        """Start memory monitoring loop"""
        self.callback = callback
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        
    async def _monitor_loop(self):
        """Main monitoring loop"""
        while not self._interrupt_requested:
            try:
                usage = psutil.Process(os.getpid()).memory_percent() / 100
                
                if usage > self.threshold:
                    self.logger.warning(f"High memory usage detected: {usage:.1%}")
                    if self.callback:
                        await self.callback(usage)
                        
            except Exception as e:
                self.logger.error(f"Memory monitoring error: {str(e)}")
                
            await asyncio.sleep(self.check_interval)
            
    def stop(self):
        """Stop memory monitoring"""
        self._interrupt_requested = True
        if self._monitor_task:
            self._monitor_task.cancel()
            self.logger.info("Memory monitoring stopped")
