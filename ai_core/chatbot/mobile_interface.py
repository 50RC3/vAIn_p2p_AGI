from typing import Optional, Dict, List
from dataclasses import dataclass
import torch
import logging
import asyncio
from .interface import ChatbotInterface, LearningConfig
from .rl_trainer import RLTrainer, RLConfig

logger = logging.getLogger(__name__)

@dataclass
class MobileConfig:
    min_batch_size: int = 8
    max_batch_size: int = 32
    battery_threshold: float = 0.2
    network_threshold: float = 0.5
    enable_power_saving: bool = True
    # Learning settings for mobile
    enable_learning: bool = False
    learning_offline_only: bool = True
    compressed_learning: bool = True
    max_buffer_size: int = 100

class NetworkQualityMonitor:
    """Monitor network quality for mobile devices"""
    
    def __init__(self):
        self.last_quality = 0.5  # Default to medium quality
        
    async def get_quality(self) -> float:
        """Get current network quality (0.0-1.0)"""
        try:
            # In a real implementation, this would check actual network quality
            # For demo purposes, we'll just return the last value
            return self.last_quality
        except Exception as e:
            logger.error(f"Error checking network quality: {e}")
            return 0.0

class MobileTrainer:
    def __init__(self, config: MobileConfig):
        self.batch_size = self._calculate_optimal_batch()
        self.power_aware = True
        self.network_monitor = NetworkQualityMonitor()
        self.config = config
        self.training_stats = {'local': 0, 'offloaded': 0}
        self.learning_buffer = []
        
    async def train_on_device(self, model: torch.nn.Module, data: torch.Tensor):
        """Execute training with battery/network awareness"""
        if await self._should_offload():
            return await self.offload_to_edge(model, data)
        return await self._local_training(model, data)
        
    async def offload_to_edge(self, model: torch.nn.Module, data: torch.Tensor):
        """Offload training to edge server when battery/network conditions require"""
        try:
            logger.info("Offloading training to edge server")
            # Compress model and data for efficient transfer
            compressed_model = self._compress_model_for_transfer(model)
            compressed_data = self._compress_data(data)
            
            # In a real implementation, this would make a network request
            # For demo purposes, we'll simulate it with a delay
            await asyncio.sleep(0.5)
            
            self.training_stats['offloaded'] += 1
            return {'status': 'offloaded', 'model_updated': True}
        except Exception as e:
            logger.error(f"Error offloading training: {e}")
            raise

    async def _local_training(self, model: torch.nn.Module, data: torch.Tensor):
        """Execute training locally on the device with resource constraints"""
        try:
            logger.info("Training locally on mobile device")
            
            # Set model to training mode
            model.train()
            
            # Train with smaller batch size and fewer iterations for mobile
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # Split data into mini-batches to avoid memory issues
            for i in range(0, data.size(0), self.batch_size):
                batch = data[i:i+self.batch_size]
                optimizer.zero_grad()
                
                # Forward pass
                output = model(batch)
                
                # Calculate loss (this would depend on your specific task)
                loss = torch.nn.functional.mse_loss(output, torch.zeros_like(output))
                
                # Backward pass with reduced precision to save battery
                with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    loss.backward()
                
                # Update weights
                optimizer.step()
            
            self.training_stats['local'] += 1
            return {'status': 'local', 'model_updated': True}
        except Exception as e:
            logger.error(f"Error in local training: {e}")
            raise

    def _calculate_optimal_batch(self) -> int:
        """Calculate optimal batch size based on device capabilities"""
        try:
            import psutil
            available_memory = psutil.virtual_memory().available
            return min(max(self.config.min_batch_size,
                        available_memory // (1024 * 1024 * 10)),  # 10MB per sample estimate
                    self.config.max_batch_size)
        except ImportError:
            # If psutil is not available, use minimum batch size
            return self.config.min_batch_size

    async def _should_offload(self) -> bool:
        """Determine if training should be offloaded"""
        try:
            battery_level = await self._get_battery_level()
            network_quality = await self.network_monitor.get_quality()
            return (battery_level < self.config.battery_threshold and 
                    network_quality > self.config.network_threshold)
        except Exception as e:
            logger.error(f"Error determining offload status: {e}")
            return False
    
    async def _get_battery_level(self) -> float:
        """Get current battery level (0.0-1.0)"""
        try:
            # In a real implementation, this would check actual battery level
            # For demo purposes, we'll just return a fixed value
            return 0.5  # 50% battery
        except Exception as e:
            logger.error(f"Error checking battery level: {e}")
            return 1.0  # Assume full battery on error (safer)

    def _compress_model_for_transfer(self, model: torch.nn.Module):
        """Compress model for efficient transfer to edge server"""
        # This is a placeholder for real compression logic
        return model
        
    def _compress_data(self, data: torch.Tensor):
        """Compress data for efficient transfer"""
        # This is a placeholder for real compression logic
        return data
        
    async def process_learning_buffer(self):
        """Process learning data stored in buffer"""
        if not self.learning_buffer:
            return
            
        # In mobile settings, only process when conditions are good
        if self.config.learning_offline_only and await self._should_offload():
            return
            
        try:
            # Process learning data in batches to save resources
            batch_size = min(10, len(self.learning_buffer))
            data = self.learning_buffer[:batch_size]
            
            # Process the data (in a real implementation this would use the learning modules)
            
            # Remove processed data from buffer
            self.learning_buffer = self.learning_buffer[batch_size:]
            
        except Exception as e:
            logger.error(f"Error processing learning buffer: {e}")


class MobileChatInterface(ChatbotInterface):
    def __init__(self, model: torch.nn.Module, storage, 
                 max_history: int = 100,  # Reduced history for mobile
                 mobile_config: Optional[MobileConfig] = None,
                 learning_config: Optional[LearningConfig] = None):
        
        # Configure learning for mobile environment
        if learning_config:
            # Adjust learning parameters for mobile
            learning_config.batch_size = min(learning_config.batch_size, 4)
            learning_config.save_interval = learning_config.save_interval * 2  # Save less frequently
        
        super().__init__(model, storage, max_history, learning_config=learning_config)
        self.mobile_mode = True
        self.compression_enabled = True
        self.mobile_config = mobile_config or MobileConfig()
        self.mobile_trainer = MobileTrainer(self.mobile_config)
        
        # Configure learning for mobile
        self.learning_buffer = []
        
        # Initialize mobile compressor if not exists
        if not hasattr(self, 'mobile_compressor'):
            self.mobile_compressor = MobileCompressor()
            
    async def process_message(self, message: str, 
                            context: Optional[Dict[str, Any]] = None) -> ChatResponse:
        """Process message with mobile optimizations"""
        try:
            # Process learning in mobile-friendly way
            if self.learning_enabled and self.mobile_config.enable_learning:
                await self._buffer_for_learning(message)
            
            # Use compressed model if available
            if hasattr(self.model, 'mobile_forward'):
                response = await self._mobile_inference(message, context)
            else:
                response = await super().process_message(message, context)
                
            # Optimize response for mobile
            if self.compression_enabled:
                response = self._compress_response(response)
                
            # Process learning buffer in background
            if self.learning_buffer:
                asyncio.create_task(self.mobile_trainer.process_learning_buffer())
                
            return response
            
        except Exception as e:
            logger.error(f"Mobile interface error: {e}")
            raise

    async def _buffer_for_learning(self, message: str):
        """Store message for learning without immediate processing"""
        if len(message.split()) < self.learning_config.min_sample_length:
            return
            
        # Add to buffer for later processing
        self.learning_buffer.append(message)
        
        # Trim buffer if it gets too large
        if len(self.learning_buffer) > self.mobile_config.max_buffer_size:
            self.learning_buffer = self.learning_buffer[-self.mobile_config.max_buffer_size:]

    async def _mobile_inference(self, message: str, context: Optional[Dict[str, Any]] = None) -> ChatResponse:
        """Optimized inference for mobile devices"""
        try:
            # Start timing for latency measurement
            start_time = time.time()
            
            # Compress input to reduce memory usage
            compressed_input = self._compress_input(message)
            
            # Forward pass with mobile-optimized model
            output = await self.model.mobile_forward(compressed_input)
            
            # Generate response
            if isinstance(output, str):
                response_text = output
            else:
                response_text = self._postprocess_output(output)
                
            # Calculate latency
            latency = time.time() - start_time
            
            # Try to get model version
            try:
                model_version = await self.storage.get_model_version()
            except:
                model_version = "mobile"
                
            # Add sensor context if available
            sensor_data = {}
            if hasattr(self, 'sensor_context'):
                sensor_data = await self._get_sensor_context()
            
            # Create chat response
            response = ChatResponse(
                text=response_text,
                confidence=0.8,  # Default confidence for mobile
                model_version=model_version,
                latency=latency
            )
            
            if sensor_data:
                if not hasattr(response, 'metadata'):
                    response.metadata = {}
                response.metadata['sensor_data'] = sensor_data
                
            return response
            
        except Exception as e:
            logger.error(f"Mobile inference error: {e}")
            # Return error response with minimal info
            return ChatResponse(
                text="Sorry, I encountered an issue.",
                confidence=0.0,
                model_version="error",
                latency=0.0,
                error=str(e)
            )
        
    def _compress_input(self, message: str) -> torch.Tensor:
        """Compress input for mobile transmission"""
        tensor = self._preprocess_message(message)
        return self.mobile_compressor.compress(tensor)
    
    def _compress_response(self, response: ChatResponse) -> ChatResponse:
        """Optimize response for mobile delivery"""
        # Create a copy to avoid modifying the original
        mobile_response = ChatResponse(
            text=response.text,
            confidence=response.confidence,
            model_version=response.model_version,
            latency=response.latency,
            error=response.error
        )
        
        # Truncate text if it's too long
        if len(mobile_response.text) > 500:
            mobile_response.text = mobile_response.text[:497] + "..."
            
        return mobile_response


# Helper class for mobile compression
class MobileCompressor:
    """Compress tensors for efficient mobile usage"""
    
    def __init__(self, compression_rate: float = 0.5):
        self.compression_rate = compression_rate
        
    def compress(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compress input tensor"""
        # For demonstration - in reality would use proper compression
        # Here we just reduce precision
        return tensor.to(torch.float16)
        
    def decompress(self, tensor: torch.Tensor) -> torch.Tensor:
        """Decompress tensor"""
        # Convert back to original precision
        return tensor.to(torch.float32)
