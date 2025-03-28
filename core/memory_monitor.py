import os
import gc
import logging
import asyncio
import tracemalloc
import time
from typing import Dict, Any, Optional, List, Tuple
import psutil

logger = logging.getLogger(__name__)

class MemoryMonitor:
    """
    Utility class for monitoring and managing memory usage.
    
    This class provides functionality to:
    - Track memory usage over time
    - Detect memory leaks
    - Perform garbage collection when needed
    - Provide memory statistics
    """
    
    def __init__(self, 
                 warning_threshold: float = 80.0,
                 critical_threshold: float = 90.0,
                 gc_threshold: float = 85.0,
                 enable_tracemalloc: bool = False,
                 check_interval: float = 30.0):
        """
        Initialize the memory monitor.
        
        Args:
            warning_threshold: Percentage of memory usage that triggers warnings
            critical_threshold: Percentage of memory usage considered critical
            gc_threshold: Percentage of memory usage that triggers garbage collection
            enable_tracemalloc: Whether to enable detailed memory tracking with tracemalloc
            check_interval: How often to check memory usage (seconds)
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.gc_threshold = gc_threshold
        self.check_interval = check_interval
        self.history = []
        self.max_history_items = 100
        self._monitor_task = None
        self.running = False
        self.enable_tracemalloc = enable_tracemalloc
        self.tracemalloc_started = False
        
        # Platform-specific setup
        self.process = psutil.Process(os.getpid())
        
        # Try to enable garbage collection optimization
        gc.enable()
        
        # Enable tracemalloc if requested
        if enable_tracemalloc:
            self._start_tracemalloc()
    
    def _start_tracemalloc(self):
        """Start detailed memory allocation tracking."""
        if not self.tracemalloc_started:
            tracemalloc.start()
            self.tracemalloc_started = True
            logger.debug("Tracemalloc tracking enabled")
    
    def _stop_tracemalloc(self):
        """Stop tracemalloc to free up resources."""
        if self.tracemalloc_started:
            tracemalloc.stop()
            self.tracemalloc_started = False
            logger.debug("Tracemalloc tracking disabled")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get current memory usage statistics.
        
        Returns:
            Dict with memory usage information
        """
        try:
            # Get system-wide memory info
            system_memory = psutil.virtual_memory()
            
            # Get process-specific memory info
            process_memory = self.process.memory_info()
            
            # Calculate memory usage percentage for this process
            process_percent = (process_memory.rss / system_memory.total) * 100
            
            return {
                "system": {
                    "total": system_memory.total,
                    "available": system_memory.available,
                    "used": system_memory.used,
                    "percent": system_memory.percent,
                },
                "process": {
                    "rss": process_memory.rss,  # Resident Set Size
                    "vms": process_memory.vms,  # Virtual Memory Size
                    "percent": process_percent,
                },
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return {
                "error": str(e),
                "timestamp": time.time()
            }
    
    def get_memory_snapshot(self) -> Optional[List[Tuple]]:
        """
        Get a detailed memory snapshot if tracemalloc is enabled.
        
        Returns:
            List of memory allocation records or None if tracemalloc is disabled
        """
        if not self.tracemalloc_started:
            return None
            
        try:
            snapshot = tracemalloc.take_snapshot()
            return snapshot.statistics('lineno')
        except Exception as e:
            logger.error(f"Error taking memory snapshot: {e}")
            return None
    
    def force_garbage_collection(self):
        """Force a garbage collection cycle."""
        try:
            # Get memory before collection
            before = self.get_memory_usage()
            
            # Perform garbage collection
            collected = gc.collect()
            
            # Get memory after collection
            after = self.get_memory_usage()
            
            # Calculate memory freed
            if 'error' not in before and 'error' not in after:
                freed = before['process']['rss'] - after['process']['rss']
                
                logger.info(f"Garbage collection freed {freed / (1024 * 1024):.2f} MB, "
                          f"collected {collected} objects")
                
                return {
                    'collected_objects': collected,
                    'memory_freed': freed,
                    'before': before,
                    'after': after
                }
        except Exception as e:
            logger.error(f"Error during garbage collection: {e}")
    
    async def start_monitoring(self):
        """Start the memory monitoring background task."""
        if self.running:
            return
            
        self.running = True
        self._monitor_task = asyncio.create_task(self._monitor_memory())
        logger.info("Memory monitoring started")
    
    async def stop_monitoring(self):
        """Stop the memory monitoring background task."""
        self.running = False
        
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            
        if self.tracemalloc_started:
            self._stop_tracemalloc()
            
        logger.info("Memory monitoring stopped")
    
    async def _monitor_memory(self):
        """Background task that periodically checks memory usage."""
        try:
            while self.running:
                # Get current memory usage
                memory_info = self.get_memory_usage()
                
                # Store in history
                self.history.append(memory_info)
                
                # Trim history if needed
                if len(self.history) > self.max_history_items:
                    self.history = self.history[-self.max_history_items:]
                
                # Check thresholds
                if 'error' not in memory_info:
                    system_percent = memory_info['system']['percent']
                    process_percent = memory_info['process']['percent']
                    
                    # Check if we should trigger garbage collection
                    if system_percent > self.gc_threshold:
                        logger.warning(f"Memory usage high ({system_percent:.1f}%), triggering garbage collection")
                        self.force_garbage_collection()
                    
                    # Check warning threshold
                    if system_percent > self.warning_threshold:
                        logger.warning(f"High memory usage: System {system_percent:.1f}%, Process {process_percent:.1f}%")
                    
                    # Check critical threshold
                    if system_percent > self.critical_threshold:
                        logger.error(f"CRITICAL: Memory usage at {system_percent:.1f}%")
                        # Take memory snapshot if tracemalloc is enabled
                        if self.tracemalloc_started:
                            snapshot = self.get_memory_snapshot()
                            if snapshot:
                                top_stats = snapshot[:10]
                                logger.error("Top memory allocations:")
                                for stat in top_stats:
                                    logger.error(f"{stat}")
                
                # Sleep for the check interval
                await asyncio.sleep(self.check_interval)
        
        except asyncio.CancelledError:
            logger.debug("Memory monitoring task cancelled")
        except Exception as e:
            logger.error(f"Error in memory monitoring: {e}")
    
    async def get_memory_trend(self, minutes: int = 5) -> Dict[str, Any]:
        """
        Analyze memory usage trend over the specified time period.
        
        Args:
            minutes: Number of minutes to analyze
            
        Returns:
            Dict with trend analysis information
        """
        if not self.history:
            return {"error": "No history available"}
        
        # Filter history by time range
        now = time.time()
        time_threshold = now - (minutes * 60)
        relevant_history = [entry for entry in self.history 
                           if entry.get('timestamp', 0) >= time_threshold]
        
        if not relevant_history:
            return {"error": "No data available for the specified time range"}
        
        try:
            # Calculate start and end values
            start_entry = relevant_history[0]
            end_entry = relevant_history[-1]
            
            # Calculate deltas
            if 'error' not in start_entry and 'error' not in end_entry:
                system_delta = end_entry['system']['percent'] - start_entry['system']['percent']
                process_delta = end_entry['process']['percent'] - start_entry['process']['percent']
                
                # Calculate rate of change (percentage points per minute)
                elapsed_minutes = (end_entry['timestamp'] - start_entry['timestamp']) / 60
                if elapsed_minutes > 0:
                    system_rate = system_delta / elapsed_minutes
                    process_rate = process_delta / elapsed_minutes
                else:
                    system_rate = 0
                    process_rate = 0
                
                return {
                    "start_time": start_entry['timestamp'],
                    "end_time": end_entry['timestamp'],
                    "elapsed_minutes": elapsed_minutes,
                    "system_delta": system_delta,
                    "process_delta": process_delta,
                    "system_rate": system_rate,  # percentage points per minute
                    "process_rate": process_rate,
                    "current_system_percent": end_entry['system']['percent'],
                    "current_process_percent": end_entry['process']['percent'],
                    "trend": "increasing" if system_rate > 0.5 else 
                            "decreasing" if system_rate < -0.5 else "stable"
                }
        
        except Exception as e:
            logger.error(f"Error analyzing memory trend: {e}")
            return {"error": f"Error analyzing trend: {str(e)}"}
