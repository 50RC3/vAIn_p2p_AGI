import torch
import torch.nn as nn
from typing import Dict, List, Optional
import logging
from tqdm import tqdm
import psutil
from core.interactive_utils import InteractiveSession, InteractiveConfig, InteractionLevel
from core.constants import INTERACTION_TIMEOUTS

logger = logging.getLogger(__name__)

class ReptileModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.inner_lr = config.inner_learning_rate
        self.feature_extractor = nn.Sequential(
            nn.Linear(784, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        self.interactive = config.interactive_mode
        self.session = None
        self._interrupt_requested = False
        self.progress_file = "reptile_progress.json"

    def adapt(self, support_set: List[torch.Tensor], num_steps: int) -> Dict[str, torch.Tensor]:
        """Legacy non-interactive adaptation method"""
        return self.adapt_interactive(support_set, num_steps)

    async def adapt_interactive(self, support_set: List[torch.Tensor], num_steps: int) -> Optional[Dict[str, torch.Tensor]]:
        """Interactive model adaptation with progress tracking and error handling"""
        if not self.interactive:
            return self.adapt(support_set, num_steps)

        try:
            self.session = InteractiveSession(
                level=InteractionLevel.NORMAL,
                config=InteractiveConfig(
                    timeout=INTERACTION_TIMEOUTS["batch"],
                    persistent_state=True,
                    safe_mode=True
                )
            )

            async with self.session:
                # Restore previous progress if available
                saved_progress = await self._load_progress()
                if saved_progress:
                    logger.info("Restoring from previous adaptation state")
                    self.load_state_dict(saved_progress["model_state"])
                    start_step = saved_progress["current_step"]
                else:
                    start_step = 0

                adapted_state = {k: v.clone() for k, v in self.state_dict().items()}
                optimizer = torch.optim.SGD(self.parameters(), lr=self.inner_lr)

                try:
                    with tqdm(total=num_steps, initial=start_step, desc="Adaptation Progress") as pbar:
                        for step in range(start_step, num_steps):
                            if self._interrupt_requested:
                                logger.info("Adaptation interrupted by user")
                                break

                            if not await self._check_resources():
                                break

                            for x, y in support_set:
                                try:
                                    optimizer.zero_grad()
                                    loss = self.compute_loss(x, y)
                                    loss.backward()
                                    optimizer.step()

                                    # Update progress
                                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})

                                except RuntimeError as e:
                                    if "out of memory" in str(e):
                                        logger.error("GPU OOM, attempting recovery")
                                        if hasattr(torch.cuda, 'empty_cache'):
                                            torch.cuda.empty_cache()
                                        await self._save_progress(adapted_state, step)
                                        raise

                            pbar.update(1)
                            
                            # Periodic progress save
                            if step % 10 == 0:
                                await self._save_progress(adapted_state, step)

                    return adapted_state

                except Exception as e:
                    logger.error(f"Adaptation error: {str(e)}")
                    await self._save_progress(adapted_state, step)
                    raise

        except Exception as e:
            logger.error(f"Interactive adaptation failed: {str(e)}")
            raise
        finally:
            if self.session:
                await self.session.__aexit__(None, None, None)

    async def _check_resources(self) -> bool:
        """Monitor system resources"""
        try:
            memory_usage = psutil.Process().memory_percent()
            if memory_usage > 90:
                msg = f"High memory usage detected: {memory_usage:.1f}%"
                if self.interactive and self.session:
                    proceed = await self.session.get_confirmation(
                        f"{msg}. Continue?",
                        timeout=INTERACTION_TIMEOUTS["emergency"]
                    )
                    if not proceed:
                        logger.warning("Adaptation stopped due to high memory usage")
                        return False
                else:
                    logger.warning(msg)
            return True
        except Exception as e:
            logger.error(f"Resource check failed: {str(e)}")
            return True  # Continue on monitoring error

    async def _save_progress(self, adapted_state: Dict[str, torch.Tensor], current_step: int) -> None:
        """Save adaptation progress"""
        try:
            progress = {
                "model_state": adapted_state,
                "current_step": current_step
            }
            torch.save(progress, self.progress_file)
        except Exception as e:
            logger.error(f"Failed to save progress: {str(e)}")

    async def _load_progress(self) -> Optional[Dict]:
        """Load saved adaptation progress"""
        try:
            if not torch.load(self.progress_file):
                return None
            return torch.load(self.progress_file)
        except:
            return None
