import torch
import torch.nn as nn
import torch.optim as optim
import copy
import logging
import psutil
from typing import Tuple, Optional, Dict, List
from contextlib import contextmanager
import time

logger = logging.getLogger(__name__)

class MetaReptile:
    """Meta-Reptile implementation for DNC-based models with production safeguards."""
    def __init__(self, 
                 model: nn.Module,
                 inner_lr: float = 0.01,
                 meta_lr: float = 0.001,
                 num_inner_steps: int = 5,
                 memory_threshold: float = 90.0):
        if not isinstance(model, nn.Module):
            raise TypeError("Model must be a PyTorch Module")
        if not all(p > 0 for p in [inner_lr, meta_lr]):
            raise ValueError("Learning rates must be positive")
        if num_inner_steps < 1:
            raise ValueError("num_inner_steps must be positive")
            
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.num_inner_steps = num_inner_steps
        self.memory_threshold = memory_threshold
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=self.meta_lr)
        self.task_loss = nn.CrossEntropyLoss()
        self._interrupt_requested = False
        
        # Add cognitive evolution tracking
        self.cognitive_stats = {
            'adaptation_speed': [],
            'learning_efficiency': [],
            'memory_retention': []
        }

    @contextmanager
    def _resource_check(self):
        """Context manager for monitoring system resources"""
        try:
            memory_usage = psutil.Process().memory_percent()
            if memory_usage > self.memory_threshold:
                logger.warning(f"High memory usage: {memory_usage:.1f}%")
            yield
        except Exception as e:
            logger.error(f"Resource monitoring failed: {str(e)}")
            yield

    def _validate_inputs(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Validate input tensors"""
        if not torch.is_tensor(x) or not torch.is_tensor(y):
            raise TypeError("Inputs must be PyTorch tensors")
        if x.size(0) != y.size(0):
            raise ValueError("Batch sizes must match")
        if torch.isnan(x).any() or torch.isnan(y).any():
            raise ValueError("Inputs contain NaN values")

    def inner_update(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[nn.Module, float]:
        """Perform inner loop updates on a task with error handling."""
        try:
            self._validate_inputs(x, y)
            temp_model = copy.deepcopy(self.model)
            inner_optimizer = optim.SGD(temp_model.parameters(), lr=self.inner_lr)
            
            final_loss = 0.0
            with self._resource_check():
                for step in range(self.num_inner_steps):
                    if self._interrupt_requested:
                        logger.info("Inner update interrupted")
                        break

                    inner_optimizer.zero_grad()
                    try:
                        transformer_out, read_vector = temp_model(x)
                        loss = self.task_loss(transformer_out[-1], y)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(temp_model.parameters(), max_norm=1.0)
                        inner_optimizer.step()
                        final_loss = loss.item()
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            logger.error("GPU OOM in inner update")
                            if hasattr(torch.cuda, 'empty_cache'):
                                torch.cuda.empty_cache()
                        raise

            return temp_model, final_loss
        except Exception as e:
            logger.error(f"Inner update failed: {str(e)}")
            raise

    def meta_update(self, temp_model: nn.Module) -> None:
        """Update meta-parameters using Reptile update rule with validation."""
        try:
            if self._interrupt_requested:
                logger.info("Meta update interrupted")
                return

            with self._resource_check():
                for meta_param, temp_param in zip(self.model.parameters(), temp_model.parameters()):
                    if meta_param.grad is None:
                        meta_param.grad = torch.zeros_like(meta_param.data)
                    
                    grad_update = (temp_param.data - meta_param.data) / self.meta_lr
                    if torch.isnan(grad_update).any():
                        raise ValueError("NaN detected in meta gradients")
                        
                    meta_param.grad.data.add_(grad_update)

        except Exception as e:
            logger.error(f"Meta update failed: {str(e)}")
            raise

    def adapt_to_task(self, support_x: torch.Tensor, support_y: torch.Tensor) -> float:
        """Adapt model to new task using support set with full error handling."""
        try:
            self._validate_inputs(support_x, support_y)
            self.meta_optimizer.zero_grad()
            
            temp_model, loss = self.inner_update(support_x, support_y)
            self.meta_update(temp_model)
            
            # Validate gradients before optimization step
            for param in self.model.parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    raise ValueError("NaN detected in model gradients")
            
            self.meta_optimizer.step()
            return loss

        except Exception as e:
            logger.error(f"Task adaptation failed: {str(e)}")
            raise

    def evaluate_task(self, query_x: torch.Tensor, query_y: torch.Tensor) -> Tuple[float, float]:
        """Evaluate model on query set after adaptation with error handling."""
        try:
            self._validate_inputs(query_x, query_y)
            with torch.no_grad(), self._resource_check():
                transformer_out, _ = self.model(query_x)
                loss = self.task_loss(transformer_out[-1], query_y)
                pred = transformer_out[-1].argmax(dim=1)
                accuracy = (pred == query_y).float().mean()
            return loss.item(), accuracy.item()
            
        except Exception as e:
            logger.error(f"Task evaluation failed: {str(e)}")
            raise

    def request_interrupt(self) -> None:
        """Request graceful interruption of training"""
        self._interrupt_requested = True
        logger.info("Interrupt requested for MetaReptile")
        
    async def evolve_cognitive_abilities(self, 
                                       support_set: List[torch.Tensor],
                                       steps: int) -> Dict:
        """Evolve and enhance cognitive capabilities"""
        try:
            stats = {}
            
            # Measure initial capabilities
            initial_perf = await self._evaluate_cognitive_metrics(support_set)
            
            # Evolutionary adaptation
            for step in range(steps):
                # Adapt and measure improvements
                adapted_state = await self.adapt_interactive(support_set, 1)
                if adapted_state is None:
                    break
                    
                current_perf = await self._evaluate_cognitive_metrics(support_set)
                
                # Track cognitive evolution
                for metric in self.cognitive_stats:
                    improvement = current_perf[metric] - initial_perf[metric]
                    self.cognitive_stats[metric].append(improvement)
                    
                stats[f'step_{step}'] = current_perf
                
            return stats
            
        except Exception as e:
            logger.error(f"Cognitive evolution failed: {str(e)}")
            raise
            
    async def _evaluate_cognitive_metrics(self, 
                                        test_set: List[torch.Tensor]) -> Dict:
        """Evaluate current cognitive capabilities"""
        try:
            metrics = {}
            
            # Test adaptation speed
            start_time = time.time()
            adapted = await self.adapt_interactive(test_set, 1)
            metrics['adaptation_speed'] = time.time() - start_time
            
            # Test learning efficiency
            metrics['learning_efficiency'] = self._measure_learning_rate(test_set)
            
            # Test memory retention
            metrics['memory_retention'] = self._test_memory_retention(test_set)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Metrics evaluation failed: {str(e)}")
            raise
