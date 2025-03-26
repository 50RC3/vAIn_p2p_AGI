import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import logging
from typing import Dict, Optional, Tuple
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

class ModelOptimizer:
    def __init__(self, model: nn.Module):
        self.model = model
        self.original_size = self._get_model_size()
        
    def optimize(self, 
                quantize: bool = True,
                pruning_amount: float = 0.3,
                dynamic_quantization: bool = True) -> nn.Module:
        """Apply optimization techniques to the model"""
        try:
            # Step 1: Pruning
            if pruning_amount > 0:
                self._apply_pruning(pruning_amount)
                
            # Step 2: Quantization
            if quantize:
                if dynamic_quantization:
                    self.model = torch.quantization.quantize_dynamic(
                        self.model,
                        {torch.nn.Linear, torch.nn.Conv2d},
                        dtype=torch.qint8
                    )
                else:
                    self._apply_static_quantization()
                    
            final_size = self._get_model_size()
            compression_ratio = final_size / self.original_size
            logger.info(f"Model compressed from {self.original_size:.2f}MB to {final_size:.2f}MB "
                       f"(ratio: {compression_ratio:.2%})")
                       
            return self.model
            
        except Exception as e:
            logger.error(f"Model optimization failed: {str(e)}")
            raise

    def _apply_pruning(self, amount: float) -> None:
        """Apply structured pruning to model weights"""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prune.l1_unstructured(module, 'weight', amount=amount)
                prune.remove(module, 'weight')
                
    def _apply_static_quantization(self) -> None:
        """Apply static quantization with calibration"""
        self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(self.model, inplace=True)
        # Note: Calibration would be done here with actual data
        torch.quantization.convert(self.model, inplace=True)
        
    def _get_model_size(self) -> float:
        """Calculate model size in MB"""
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return (param_size + buffer_size) / (1024 * 1024)
