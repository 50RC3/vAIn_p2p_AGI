from typing import Optional, Dict, List
from dataclasses import dataclass
import torch
from .interface import ChatbotInterface
from .rl_trainer import RLTrainer, RLConfig

@dataclass
class MobileConfig:
    min_batch_size: int = 8
    max_batch_size: int = 32
    battery_threshold: float = 0.2
    network_threshold: float = 0.5
    enable_power_saving: bool = True

class MobileTrainer:
    def __init__(self, config: MobileConfig):
        self.batch_size = self._calculate_optimal_batch()
        self.power_aware = True
        self.network_monitor = NetworkQualityMonitor()
        self.config = config
        self.training_stats = {'local': 0, 'offloaded': 0}
        
    async def train_on_device(self, model: torch.nn.Module, data: torch.Tensor):
        """Execute training with battery/network awareness"""
        if self._should_offload():
            return await self.offload_to_edge(model, data)
        return await self._local_training(model, data)

    def _calculate_optimal_batch(self) -> int:
        """Calculate optimal batch size based on device capabilities"""
        available_memory = psutil.virtual_memory().available
        return min(max(self.config.min_batch_size,
                      available_memory // (1024 * 1024 * 10)),  # 10MB per sample estimate
                  self.config.max_batch_size)

    async def _should_offload(self) -> bool:
        """Determine if training should be offloaded"""
        battery_level = await self._get_battery_level()
        network_quality = await self.network_monitor.get_quality()
        return (battery_level < self.config.battery_threshold and 
                network_quality > self.config.network_threshold)

class MobileChatInterface(ChatbotInterface):
    def __init__(self, model: torch.nn.Module, storage, 
                 max_history: int = 100):  # Reduced history for mobile
        super().__init__(model, storage, max_history)
        self.mobile_mode = True
        self.compression_enabled = True
        
    async def process_message(self, message: str) -> Dict:
        """Process message with mobile optimizations"""
        try:
            # Use compressed model if available
            if hasattr(self.model, 'mobile_forward'):
                response = await self._mobile_inference(message)
            else:
                response = await super().process_message(message)
                
            # Optimize response for mobile
            if self.compression_enabled:
                response = self._compress_response(response)
                
            return response
            
        except Exception as e:
            logger.error(f"Mobile interface error: {e}")
            raise

    async def _mobile_inference(self, message: str) -> Dict:
        """Optimized inference for mobile devices"""
        compressed_input = self._compress_input(message)
        response = await self.model.mobile_forward(compressed_input)
        
        # Add sensor context if available
        if hasattr(self, 'sensor_context'):
            response['sensor_data'] = await self._get_sensor_context()
        
        return response
        
    def _compress_input(self, message: str) -> torch.Tensor:
        """Compress input for mobile transmission"""
        tensor = super()._preprocess_message(message)
        return self.mobile_compressor.compress(tensor)
