import asyncio
from typing import Dict, Any, Optional, List, Set
from android.hardware import SensorManager
from android.hardware.Sensor import TYPE_ALL
from android.os import BatteryManager, Context
import logging
import time

from training.compression import AdaptiveCompression
from core.cache import CacheManager
from core.system_coordinator import get_coordinator

logger = logging.getLogger(__name__)

class SensorDataCollector:
    def __init__(self):
        self.active_sensors: Set[str] = set()
        self.data_buffers: Dict[str, List[float]] = {}
        
    async def collect_sensor_data(self, sensor_type: str) -> Dict:
        """Collect and preprocess mobile sensor data"""
        raw_data = await self._read_sensor(sensor_type)
        return self._preprocess_sensor_data(raw_data)
        
    async def _read_sensor(self, sensor_type: str) -> List[float]:
        """Read raw data from sensor"""
        if sensor_type not in self.data_buffers:
            return []
        return self.data_buffers.get(sensor_type, [])
        
    def _preprocess_sensor_data(self, data: List[float]) -> Dict[str, float]:
        """Preprocess sensor data for transmission"""
        if not data:
            return {}
            
        return {
            'mean': sum(data) / len(data),
            'max': max(data),
            'min': min(data),
            'timestamp': time.time()
        }

class MobileNode:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.sensor_manager = None
        self.sensor_collector = SensorDataCollector()
        self._interrupt_requested = False
        
    async def initialize(self):
        """Initialize mobile node and sensors"""
        try:
            self.sensor_manager = SensorManager.getSystemService(SensorManager.SENSOR_SERVICE)
            await self._setup_sensors()
            return True
        except Exception as e:
            logger.error(f"Failed to initialize mobile node: {e}")
            return False
            
    async def _setup_sensors(self):
        """Setup available sensors"""
        sensor_types = [
            TYPE_ALL.ACCELEROMETER,
            TYPE_ALL.GYROSCOPE,
            TYPE_ALL.LIGHT,
            TYPE_ALL.PROXIMITY
        ]
        
        for sensor_type in sensor_types:
            sensor = self.sensor_manager.getDefaultSensor(sensor_type)
            if sensor:
                self.sensor_collector.active_sensors.add(sensor_type)
                self.sensor_collector.data_buffers[sensor_type] = []

    async def get_sensor_data(self) -> Dict[str, Any]:
        """Get current sensor data readings"""
        return {
            sensor_type: await self.sensor_collector.collect_sensor_data(sensor_type)
            for sensor_type in self.sensor_collector.active_sensors
        }

    async def start_sensor_stream(self, sensor_type: str) -> bool:
        """Start streaming data from a specific sensor"""
        if sensor_type not in self.sensor_collector.active_sensors:
            return False
            
        try:
            stream_id = f"{self.node_id}_{sensor_type}"
            self.sensor_collector.data_buffers[sensor_type] = asyncio.Queue(maxsize=100)
            self._start_sensor_polling(sensor_type)
            return True
        except Exception as e:
            logger.error(f"Failed to start sensor stream: {e}")
            return False
            
    async def _get_battery_level(self) -> float:
        """Get current battery level"""
        try:
            battery_manager = self.context.getSystemService(Context.BATTERY_SERVICE)
            return battery_manager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY) / 100.0
        except Exception as e:
            logger.warning(f"Failed to get battery level: {e}")
            return 1.0

    async def get_network_quality(self) -> float:
        """Get network quality score (0-1)"""
        if not hasattr(self, 'network_monitor'):
            from .network_quality import NetworkQualityMonitor
            self.network_monitor = NetworkQualityMonitor()
        return await self.network_monitor.get_quality()

    def _start_sensor_polling(self, sensor_type: str):
        """Start polling sensor data"""
        pass

class ResourceMonitor:
    """Monitor available resources on the mobile device."""
    
    def __init__(self):
        self.memory_monitor = None
        self.cpu_threshold = 0.8
        self.memory_threshold = 0.7
        self.battery_threshold = 0.2
        self.last_check = 0
        self.check_interval = 60  # seconds
        self.logger = logging.getLogger('ResourceMonitor')
        self.callbacks = {}
        self.system_coordinator = get_coordinator()
        
    async def initialize(self, context=None):
        """Initialize resource monitoring."""
        self.context = context
        
        # Set up memory monitoring
        try:
            from ..memory_monitor import MemoryMonitor
            self.memory_monitor = MemoryMonitor(threshold=self.memory_threshold, check_interval=30)
            await self.memory_monitor.start_monitoring(callback=self._on_high_memory_usage)
            self.logger.info("Memory monitoring started")
            
            # Register with system coordinator
            if self.system_coordinator:
                self.system_coordinator.register_component("mobile_resource_monitor", self)
                self.system_coordinator.register_event_listener("critical_memory", self._handle_critical_memory)
                self.system_coordinator.register_event_listener("critical_cpu", self._handle_critical_cpu)
                
            return True
        except ImportError:
            self.logger.warning("MemoryMonitor not available, resource monitoring will be limited")
            return False
            
    async def _on_high_memory_usage(self, usage):
        """Handle high memory usage alerts."""
        self.logger.warning(f"High memory usage detected: {usage:.1%}")
        if "memory_warning" in self.callbacks:
            await self.callbacks["memory_warning"](usage)
            
        # Send event to system coordinator
        if self.system_coordinator:
            self.system_coordinator.dispatch_event("high_memory", {"usage": usage, "source": "mobile"})
    
    async def _handle_critical_memory(self, data):
        """Handle critical memory events from system coordinator."""
        self.logger.warning(f"Critical memory event received: {data}")
        await self._reduce_resource_usage("memory")
    
    async def _handle_critical_cpu(self, data):
        """Handle critical CPU events from system coordinator."""
        self.logger.warning(f"Critical CPU event received: {data}")
        await self._reduce_resource_usage("cpu")
    
    async def _reduce_resource_usage(self, resource_type):
        """Reduce resource usage based on resource type."""
        if resource_type == "memory":
            # Clear caches
            if hasattr(self, "cache_manager"):
                await self.cache_manager.clear()
                
            # Stop non-essential operations
            if "resource_action" in self.callbacks:
                await self.callbacks["resource_action"]({
                    "action": "reduce",
                    "resource": resource_type
                })
            
    async def get_resource_state(self):
        """Get current resource state."""
        try:
            import psutil
            
            # Only check every check_interval seconds
            current_time = time.time()
            if current_time - self.last_check < self.check_interval:
                return self._last_state
                
            self.last_check = current_time
            
            mem = psutil.virtual_memory()
            cpu = psutil.cpu_percent() / 100
            
            self._last_state = {
                "memory_available": mem.available,
                "memory_percent": mem.percent / 100,
                "cpu_percent": cpu,
                "battery": await self._get_battery_level(),
                "timestamp": current_time
            }
            
            return self._last_state
        except Exception as e:
            self.logger.error(f"Failed to get resource state: {e}")
            return {
                "memory_percent": 0,
                "cpu_percent": 0,
                "battery": 1.0,
                "error": str(e)
            }
            
    def register_callback(self, event_type, callback):
        """Register callback for resource events."""
        self.callbacks[event_type] = callback
        
    async def cleanup(self):
        """Clean up resources."""
        if self.memory_monitor:
            self.memory_monitor.stop()
            
        # Unregister from system coordinator
        if self.system_coordinator and hasattr(self.system_coordinator, "components"):
            if "mobile_resource_monitor" in self.system_coordinator.components:
                self.system_coordinator.components.pop("mobile_resource_monitor")

class EdgeNode:
    """Edge node implementation with adaptive compression and caching."""
    
    def __init__(self):
        self.compression = AdaptiveCompression(
            base_rate=0.1,
            min_rate=0.01, 
            max_rate=0.3,
            quality_threshold=0.85
        )
        self.cache_manager = CacheManager({
            'memory': {'size': '2GB', 'ttl': 3600},
            'disk': {'size': '20GB', 'ttl': 86400}
        })
        self.resource_monitor = ResourceMonitor()
        
    async def process_data(self, data: bytes) -> Optional[bytes]:
        """Process data with compression and caching."""
        try:
            # Check cache first
            cache_key = self.cache_manager.get_key(data)
            cached = await self.cache_manager.get(cache_key)
            if cached:
                return cached
                
            # Apply compression based on network conditions
            compressed = await self.compression.compress(data)
            
            # Cache the result
            await self.cache_manager.set(cache_key, compressed)
            
            return compressed
            
        except Exception as e:
            logger.error(f"Data processing failed: {e}")
            return None
            
    async def cleanup(self):
        """Cleanup resources."""
        await self.cache_manager.cleanup()
        await self.compression.cleanup()
