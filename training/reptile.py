import torch
import logging
from typing import List, Dict, Optional
from models.reptile_meta.reptile_model import ReptileModel
from core.interactive_utils import InteractiveSession, InteractiveConfig, InteractionLevel
from core.constants import INTERACTION_TIMEOUTS
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ReptileTrainer:
    def __init__(self, model: ReptileModel, 
                 inner_lr: float = 0.01, 
                 outer_lr: float = 0.001,
                 interactive: bool = True):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.interactive = interactive
        self.session = None
        self._interrupt_requested = False

    async def adapt_to_task_interactive(self, support_set: List[torch.Tensor], 
                                      query_set: List[torch.Tensor], 
                                      num_inner_steps: int = 5) -> bool:
        """Interactive version of adapt_to_task with progress tracking and validation"""
        try:
            if self.interactive:
                self.session = InteractiveSession(
                    level=InteractionLevel.NORMAL,
                    config=InteractiveConfig(
                        timeout=INTERACTION_TIMEOUTS["batch"],
                        persistent_state=True,
                        safe_mode=True
                    )
                )

            async with self.session:
                # Store original parameters with validation
                orig_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                
                # Progress tracking
                if self.interactive:
                    print("\nAdapting to task...")
                    print("=" * 50)
                    
                # Get adapted state with interactive monitoring
                adapted_state = await self.model.adapt_interactive(support_set, num_inner_steps)
                if adapted_state is None:
                    return False
                
                if self._interrupt_requested:
                    logger.info("Task adaptation interrupted by user")
                    return False

                # Compute and validate meta gradients
                try:
                    meta_grads = {k: orig_state[k] - adapted_state[k] 
                                for k in orig_state.keys()}
                except Exception as e:
                    logger.error(f"Meta gradient computation failed: {str(e)}")
                    return False

                # Update model parameters with safety checks
                for k, v in self.model.state_dict().items():
                    if torch.isnan(meta_grads[k]).any():
                        logger.error(f"NaN detected in meta gradients for {k}")
                        return False
                    v.data.add_(meta_grads[k], alpha=-self.outer_lr)

                return True

        except Exception as e:
            logger.error(f"Interactive task adaptation failed: {str(e)}")
            return False
        finally:
            if self.session:
                await self.session.__aexit__(None, None, None)

    def adapt_to_task(self, support_set: List[torch.Tensor], 
                      query_set: List[torch.Tensor], 
                      num_inner_steps: int = 5):
        """Non-interactive adaptation with production safety measures"""
        if self.interactive:
            import asyncio
            return asyncio.run(self.adapt_to_task_interactive(
                support_set, query_set, num_inner_steps))

        # Validate inputs
        if not support_set or not query_set:
            raise ValueError("Support and query sets cannot be empty")
        if num_inner_steps < 1:
            raise ValueError("num_inner_steps must be positive")
            
        try:
            # Store original parameters with validation
            orig_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            if not orig_state:
                raise RuntimeError("Failed to clone model state")

            # Adapt to task with timeout
            adapted_state = self.model.adapt(support_set, num_inner_steps)
            if adapted_state is None:
                logger.error("Model adaptation failed")
                return False

            # Compute and validate meta gradients
            try:
                meta_grads = {k: orig_state[k] - adapted_state[k] 
                            for k in orig_state.keys()}
                            
                # Validate meta gradients
                for k, grad in meta_grads.items():
                    if torch.isnan(grad).any():
                        logger.error(f"NaN detected in meta gradients for {k}")
                        return False
                    if torch.isinf(grad).any():
                        logger.error(f"Inf detected in meta gradients for {k}")
                        return False
                        
            except Exception as e:
                logger.error(f"Meta gradient computation failed: {str(e)}")
                return False

            # Update model parameters with safety checks
            for k, v in self.model.state_dict().items():
                if k not in meta_grads:
                    logger.error(f"Missing meta gradient for parameter {k}")
                    continue
                v.data.add_(meta_grads[k], alpha=-self.outer_lr)

            return True

        except Exception as e:
            logger.error(f"Task adaptation failed: {str(e)}")
            return False

    def request_interrupt(self):
        """Request graceful interruption of training"""
        self._interrupt_requested = True
        try:
            if self.model:
                self.model.request_interrupt()
            logger.info("Interrupt requested successfully")
        except Exception as e:
            logger.error(f"Failed to request interrupt: {str(e)}")
