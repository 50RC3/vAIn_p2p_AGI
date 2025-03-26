from typing import Dict, Any, Optional, Tuple
import torch
import psutil
import asyncio
import logging
from ..edge_computing import EdgeComputingService

logger = logging.getLogger(__name__)

class MobileEdgeService:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.edge_service = EdgeComputingService()
        self.resource_monitor = MobileResourceMonitor()
        self.task_queue = asyncio.Queue()
        
    async def register_as_edge_node(self):
        """Register mobile device as edge node with capabilities"""
        capabilities = await self.resource_monitor.get_capabilities()
        await self.edge_service.register_edge_node(
            self.node_id,
            capabilities
        )
        
    async def process_task(self, task_type: str, data: Any) -> Optional[Any]:
        """Process edge computing task with mobile optimizations"""
        if not self.resource_monitor.can_process_task():
            return None
            
        if task_type == "preprocessing":
            return await self._preprocess_mobile(data)
        elif task_type == "model_aggregation":
            return await self._aggregate_mobile(data)
            
        return None

    async def _preprocess_mobile(self, data: torch.Tensor) -> torch.Tensor:
        """Mobile-optimized preprocessing"""
        try:
            if len(data.shape) > 2:
                data = data.squeeze()
            return self.mobile_compressor.compress(data)
        except Exception as e:
            logger.error(f"Mobile preprocessing failed: {e}")
            return data

    async def offload_training(self, model: torch.nn.Module, 
                             data: torch.Tensor) -> Tuple[torch.nn.Module, float]:
        """Offload training to edge node"""
        if not await self._verify_edge_availability():
            raise RuntimeError("No edge nodes available")
            
        compressed_data = await self._preprocess_mobile(data)
        result = await self.edge_service.offload_task(
            "training",
            {
                "model_state": model.state_dict(),
                "data": compressed_data,
                "device_type": "mobile"
            }
        )
        return self._update_model(model, result)

    async def _verify_edge_availability(self) -> bool:
        """Verify if edge nodes are available for offloading"""
        try:
            nodes = await self.edge_service.get_available_nodes()
            return len(nodes) > 0
        except Exception as e:
            logger.error(f"Failed to verify edge availability: {e}")
            return False

    def _update_model(self, model: torch.nn.Module, result: Dict) -> Tuple[torch.nn.Module, float]:
        """Update model with results from edge training"""
        if 'model_state' in result:
            model.load_state_dict(result['model_state'])
        return model, result.get('loss', float('inf'))

class MobileEdgeManager:
    def __init__(self):
        self.mobile_nodes = {}
        self.sensor_streams = {}
        self.compression_rate = 0.1  # 10x compression for mobile
        
    async def register_mobile_node(self, node_id: str, capabilities: Dict):
        """Register mobile device with capabilities and sensors"""
        self.mobile_nodes[node_id] = {
            'battery_level': capabilities.get('battery', 100),
            'available_sensors': capabilities.get('sensors', []),
            'network_type': capabilities.get('network', '4G'),
            'compute_capacity': self._calculate_compute_score(capabilities)
        }
        
    def _calculate_compute_score(self, capabilities: Dict) -> float:
        """Calculate node's compute capacity score"""
        cpu_score = capabilities.get('cpu_cores', 1) * 0.4
        memory_score = (capabilities.get('memory', 0) / (1024 * 1024 * 1024)) * 0.3
        battery_score = capabilities.get('battery', 0) * 0.3
        return cpu_score + memory_score + battery_score

class MobileResourceMonitor:
    def __init__(self):
        self.battery_threshold = 0.2  # 20% battery minimum
        self.network_monitor = NetworkQualityMonitor()
        
    async def get_capabilities(self) -> Dict[str, Any]:
        """Get mobile device capabilities"""
        capabilities = {
            'cpu_cores': psutil.cpu_count(),
            'memory': psutil.virtual_memory().total,
            'battery': await self._get_battery_level(),
            'device_type': 'mobile'
        }
        capabilities.update({
            'network': await self.network_monitor.get_network_type(),
            'sensors': await self._get_available_sensors()
        })
        return capabilities
