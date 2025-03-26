import asyncio
from typing import Dict, Any, Optional, List, Set
from android.hardware import SensorManager
from android.hardware.Sensor import TYPE_ALL
from android.os import BatteryManager, Context
import logging
import time

from training.compression import AdaptiveCompression
from core.cache import CacheManager

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
