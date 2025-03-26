import torch
import logging
import psutil
from dataclasses import dataclass
from typing import Optional, Dict, List
from queue import PriorityQueue
import threading
import time

logger = logging.getLogger(__name__)

@dataclass
class Task:
    priority: int
    data: Dict
    timestamp: float = time.time()

class TaskPriorityQueue:
    def __init__(self):
        self.queue = PriorityQueue()
        self.lock = threading.Lock()
        
    def push(self, priority: int, data: Dict):
        with self.lock:
            self.queue.put(Task(priority, data))
            
    def pop(self) -> Optional[Task]:
        with self.lock:
            if not self.queue.empty():
                return self.queue.get()
            return None

class ResourceMonitor:
    def __init__(self, threshold: float = 0.85, check_interval: int = 30):
        self.threshold = threshold
        self.check_interval = check_interval
        self.running = False
        self._monitor_thread = None
        
    def start(self):
        if not self.running:
            self.running = True
            self._monitor_thread = threading.Thread(target=self._monitor_resources)
            self._monitor_thread.daemon = True
            self._monitor_thread.start()
            
    def stop(self):
        self.running = False
        if self._monitor_thread:
            self._monitor_thread.join()
            
    def _monitor_resources(self):
        while self.running:
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            if cpu_percent > self.threshold * 100 or memory_percent > self.threshold * 100:
                logger.warning(f"Resource usage high - CPU: {cpu_percent}%, Memory: {memory_percent}%")
                
            time.sleep(self.check_interval)

class LocalTrainer:
    def __init__(self):
        self.batch_size = self._calculate_optimal_batch()
        self.priority_queue = TaskPriorityQueue()
        self.resource_monitor = ResourceMonitor(
            threshold=0.85,
            check_interval=30
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resource_monitor.start()
        
    def _calculate_optimal_batch(self) -> int:
        """Calculate optimal batch size based on available resources"""
        try:
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.get_device_properties(0).total_memory
                return min(512, max(32, gpu_mem // (1024 * 1024 * 128)))  # 128MB per batch
            else:
                system_mem = psutil.virtual_memory().total
                return min(256, max(16, system_mem // (1024 * 1024 * 1024)))  # 1GB per batch
        except Exception as e:
            logger.warning(f"Error calculating batch size: {e}, using default")
            return 32

    def __del__(self):
        """Cleanup on deletion"""
        if hasattr(self, 'resource_monitor'):
            self.resource_monitor.stop()
