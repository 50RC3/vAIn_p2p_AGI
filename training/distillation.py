import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Optional, Tuple
from tqdm import tqdm
from .model_optimizer import ModelOptimizer

logger = logging.getLogger(__name__)

class DistillationTrainer:
    def __init__(self, 
                 teacher_model: nn.Module,
                 student_model: nn.Module,
                 temperature: float = 2.0,
                 alpha: float = 0.5):
        """
        Args:
            teacher_model: The large production model
            student_model: The smaller model to be trained
            temperature: Temperature for softening probability distributions
            alpha: Weight for balancing soft/hard targets (0-1)
        """
        self.teacher = teacher_model.eval()
        self.student = student_model.train()
        self.temperature = temperature
        self.alpha = alpha
        self.optimizer = torch.optim.Adam(self.student.parameters())
        
    def train_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Single distillation training step"""
        # Get soft targets from teacher
        with torch.no_grad():
            teacher_logits = self.teacher(inputs)
            soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
            
        # Train student
        student_logits = self.student(inputs)
        student_probs = F.softmax(student_logits / self.temperature, dim=1)
        
        # Compute losses
        distillation_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            soft_targets,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        student_loss = F.cross_entropy(student_logits, targets)
        loss = self.alpha * distillation_loss + (1 - self.alpha) * student_loss
        
        # Update student
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': loss.item(),
            'distillation_loss': distillation_loss.item(),
            'student_loss': student_loss.item()
        }

    def compress_student(self, optimize_for_mobile: bool = True) -> nn.Module:
        """Compress and optimize student model for mobile deployment"""
        self.student.eval()
        
        # Apply comprehensive optimization
        optimizer = ModelOptimizer(self.student)
        optimized_student = optimizer.optimize(
            quantize=True,
            pruning_amount=0.3,
            dynamic_quantization=not optimize_for_mobile
        )
        
        if optimize_for_mobile:
            # Export for mobile deployment
            scripted_model = torch.jit.script(optimized_student)
            return scripted_model
            
        return optimized_student

    def export_for_mobile(self, path: str) -> None:
        """Export optimized model for mobile deployment"""
        try:
            mobile_model = self.compress_student(optimize_for_mobile=True)
            mobile_model.save(path)
            logger.info(f"Mobile-optimized model saved to {path}")
        except Exception as e:
            logger.error(f"Mobile export failed: {str(e)}")
            raise
