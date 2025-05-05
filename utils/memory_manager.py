import gc
import psutil
import threading
import time
import weakref
from typing import Dict, Set, Any, Optional, Callable, List
import logging
import tracemalloc
import asyncio
from dataclasses import dataclass, field

from utils.unified_logger import get_logger

logger = get_logger("memory_manager")

@dataclass
class MemoryStats:
    """Memory statistics"""
    total_system_mb: float = 0.0
    available_system_mb: float = 0.0
    used_system_mb: float = 0.0
    system_percent: float = 0.0
    process_mb: float = 0.0
    process_percent: float = 0.0
    tracked_objects: int = 0
    largest_objects: List[Dict[str, Any]] = field(default_factory=list)

class MemoryManager:
    """
    Advanced memory management system for controlling and optimizing memory usage
    in the vAIn P2P AGI system.
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls) -> "MemoryManager":
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = MemoryManager()
        return cls._instance
        
    def __init__(self):
        """Initialize memory manager"""
        self._monitoring = False
        self._monitor_thread = None
        self._tracked_objects: Dict[int, weakref.ref] = {}
        self._critical_threshold = 85.0  # percent
        self._warning_threshold = 75.0  # percent
        self._monitoring_interval = 60  # seconds
        self._last_stats: Optional[MemoryStats] = None
        self._callbacks: Dict[str, List[Callable]] = {
            "warning": [],
            "critical": [],
            "normal": []
        }
        self._current_state = "normal"
        self._enable_tracemalloc = False
        self._history: List[MemoryStats] = []
        self._max_history_size = 100
        self._process = psutil.Process()
        self._lock = threading.RLock()
        
    def start_monitoring(self, interval: int = 60, trace_memory: bool = False) -> None:
        """Start memory monitoring"""
        if self._monitoring:
            logger.warning("Memory monitoring already active")
            return
            
        self._monitoring_interval = interval
        self._enable_tracemalloc = trace_memory
        self._monitoring = True
        
        if trace_memory and not tracemalloc.is_tracing():
            tracemalloc.start()
            logger.info("Trace memory allocation tracking started")
        
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            name="memory-monitor",
            daemon=True
        )
        self._monitor_thread.start()
        logger.info(f"Memory monitoring started (interval: {interval}s)")
        
    def stop_monitoring(self) -> None:
        """Stop memory monitoring"""
        if not self._monitoring:
            return
            
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None
            
        if tracemalloc.is_tracing():
            tracemalloc.stop()
            
        logger.info("Memory monitoring stopped")
        
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self._monitoring:
            try:
                stats = self.get_memory_stats()
                
                # Check for threshold crossings
                new_state = self._get_state(stats.system_percent)
                if new_state != self._current_state:
                    self._handle_state_change(new_state, stats)
                    
                self._current_state = new_state
                
                # Store in history
                with self._lock:
                    self._history.append(stats)
                    if len(self._history) > self._max_history_size:
                        self._history = self._history[-self._max_history_size:]
                    
                time.sleep(self._monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}", exc_info=True)
                time.sleep(10)  # Sleep briefly before retry
                
    def _handle_state_change(self, new_state: str, stats: MemoryStats) -> None:
        """Handle memory state changes and trigger callbacks"""
        logger.info(f"Memory state changed from {self._current_state} to {new_state}")
        
        # Execute callbacks for the new state
        for callback in self._callbacks.get(new_state, []):
            try:
                callback(stats)
            except Exception as e:
                logger.error(f"Error in memory {new_state} callback: {e}", exc_info=True)
                
        # If transitioning to critical, run garbage collection
        if new_state == "critical":
            logger.warning("Memory usage critical, running garbage collection")
            self.force_garbage_collection()
                
    def _get_state(self, memory_percent: float) -> str:
        """Determine memory state based on thresholds"""
        if memory_percent >= self._critical_threshold:
            return "critical"
        elif memory_percent >= self._warning_threshold:
            return "warning"
        else:
            return "normal"
            
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        try:
            # System memory
            system_memory = psutil.virtual_memory()
            
            # Process memory
            process_memory = self._process.memory_info()
            process_mb = process_memory.rss / (1024 * 1024)  # Convert to MB
            
            # Create stats object
            stats = MemoryStats(
                total_system_mb=system_memory.total / (1024 * 1024),
                available_system_mb=system_memory.available / (1024 * 1024),
                used_system_mb=(system_memory.total - system_memory.available) / (1024 * 1024),
                system_percent=system_memory.percent,
                process_mb=process_mb,
                process_percent=(process_mb / (system_memory.total / (1024 * 1024))) * 100,
                tracked_objects=len(self._tracked_objects),
            )
            
            # Add tracemalloc data if enabled
            if self._enable_tracemalloc and tracemalloc.is_tracing():
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics('lineno')
                
                # Add top memory consumers
                stats.largest_objects = [
                    {
                        "file": str(stat.traceback.filename),
                        "line": stat.traceback.lineno,
                        "size_mb": stat.size / (1024 * 1024)
                    }
                    for stat in top_stats[:10]  # Top 10 memory consumers
                ]
                
            self._last_stats = stats
            return stats
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}", exc_info=True)
            return MemoryStats()
    
    def track_object(self, obj: Any, description: str = "") -> int:
        """Track object for memory management"""
        obj_id = id(obj)
        with self._lock:
            # Store weak reference to avoid keeping object alive
            self._tracked_objects[obj_id] = (weakref.ref(obj), description)
        return obj_id
        
    def untrack_object(self, obj_id: int) -> None:
        """Stop tracking an object"""
        with self._lock:
            if obj_id in self._tracked_objects:
                del self._tracked_objects[obj_id]
                
    def get_tracked_objects(self) -> List[Dict[str, Any]]:
        """Get information about tracked objects"""
        result = []
        dead_refs = []
        
        with self._lock:
            for obj_id, (obj_ref, description) in self._tracked_objects.items():
                # Get the actual object from the weak reference
                obj = obj_ref()
                if obj is None:
                    # Reference is dead, mark for removal
                    dead_refs.append(obj_id)
                    continue
                    
                try:
                    # Try to estimate memory usage
                    memory_usage = 0
                    if hasattr(obj, "__sizeof__"):
                        memory_usage = obj.__sizeof__()
                    
                    result.append({
                        "id": obj_id,
                        "type": type(obj).__name__,
                        "description": description,
                        "estimated_size": memory_usage,
                    })
                except Exception:
                    result.append({
                        "id": obj_id,
                        "type": "unknown",
                        "description": description,
                        "error": "failed to get object info"
                    })
            
            # Clean up dead references
            for obj_id in dead_refs:
                del self._tracked_objects[obj_id]
                
        return result
        
    def force_garbage_collection(self) -> Dict[str, Any]:
        """Force garbage collection and return stats"""
        start = time.time()
        
        # Run GC with full generational cleanup
        collected = gc.collect(generation=2)
        
        end = time.time()
        
        stats = {
            "collected_objects": collected,
            "time_seconds": end - start,
            "uncollectable_objects": len(gc.garbage),
        }
        
        logger.info(f"Garbage collection: collected {collected} objects in {end-start:.3f}s")
        return stats
        
    def register_callback(self, state: str, callback: Callable) -> None:
        """Register callback for memory state changes"""
        if state in self._callbacks:
            self._callbacks[state].append(callback)
        else:
            logger.warning(f"Invalid memory state: {state}")
            
    def set_thresholds(self, warning: float, critical: float) -> None:
        """Set memory usage thresholds"""
        if warning >= critical:
            logger.error("Warning threshold must be less than critical threshold")
            return
            
        self._warning_threshold = warning
        self._critical_threshold = critical
        logger.info(f"Memory thresholds set: warning={warning}%, critical={critical}%")
        
    def get_memory_history(self) -> List[MemoryStats]:
        """Get memory usage history"""
        with self._lock:
            return list(self._history)  # Return a copy
            
    async def run_memory_evaluation(self) -> Dict[str, Any]:
        """
        Run comprehensive memory evaluation and return improvement recommendations
        """
        try:
            # Force garbage collection first
            gc_stats = self.force_garbage_collection()
            
            # Get current memory stats
            stats = self.get_memory_stats()
            
            # Get tracked objects
            tracked_objects = self.get_tracked_objects()
            
            # Get largest memory consumers if tracemalloc is enabled
            largest_consumers = []
            if self._enable_tracemalloc and tracemalloc.is_tracing():
                snapshot = tracemalloc.take_snapshot()
                largest_consumers = [
                    {
                        "file": str(stat.traceback.filename),
                        "line": stat.traceback.lineno,
                        "size_mb": stat.size / (1024 * 1024),
                        "count": stat.count
                    }
                    for stat in snapshot.statistics('lineno')[:20]  # Top 20
                ]
            
            # Generate recommendations
            recommendations = []
            
            # Check overall memory usage
            if stats.system_percent > 80:
                recommendations.append({
                    "priority": "high",
                    "issue": "High system memory usage",
                    "recommendation": "Consider scaling infrastructure or optimizing memory-intensive components"
                })
                
            # Check for memory leaks based on history
            with self._lock:
                if len(self._history) >= 10:
                    first_10_avg = sum(s.process_mb for s in self._history[:10]) / 10
                    last_10_avg = sum(s.process_mb for s in self._history[-10:]) / 10
                    if last_10_avg > first_10_avg * 1.2:  # 20% increase
                        recommendations.append({
                            "priority": "high",
                            "issue": "Potential memory leak detected",
                            "recommendation": "Investigate tracked objects for resource leaks"
                        })
            
            # Check for large tracked objects
            large_tracked = [obj for obj in tracked_objects 
                            if obj.get("estimated_size", 0) > 10 * 1024 * 1024]  # > 10MB
            if large_tracked:
                recommendations.append({
                    "priority": "medium",
                    "issue": f"Large tracked objects ({len(large_tracked)} objects > 10MB)",
                    "recommendation": "Consider implementing object pooling or reducing object size"
                })
                
            return {
                "timestamp": time.time(),
                "memory_stats": {
                    "system_percent": stats.system_percent,
                    "process_mb": stats.process_mb,
                    "available_mb": stats.available_system_mb
                },
                "gc_stats": gc_stats,
                "tracked_objects_count": len(tracked_objects),
                "largest_consumers": largest_consumers,
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Error in memory evaluation: {e}", exc_info=True)
            return {
                "timestamp": time.time(),
                "error": str(e),
                "recommendations": [{
                    "priority": "high",
                    "issue": "Memory evaluation failed",
                    "recommendation": "Check logs for memory manager errors"
                }]
            }

# Global memory manager instance
memory_manager = MemoryManager.get_instance()