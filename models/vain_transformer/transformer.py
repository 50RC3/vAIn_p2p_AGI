import torch
from torch import nn
import logging
from typing import Dict, Any, Optional, Union

from core.interactive_utils import Session, InteractionLevel, InteractiveConfig
from core.constants import INTERACTION_TIMEOUTS

logger = logging.getLogger(__name__)

class VAInTransformer(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 num_heads: int,
                 num_layers: int,
                 memory_threshold: float = 0.9) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.session: Optional[Session] = None
        self.interactive = False
        self._session_registered = False
        self._interrupt_requested = False
        self.memory_threshold = memory_threshold
        self.encoder = nn.Linear(input_dim, hidden_dim)  # Placeholder for actual encoder
    async def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.session is None:
            self.session = await self._register_session()
        
        try:
            output = await self._generate_response(x)
            if self.session:
                await self.session.save_progress()
            return output
        except Exception as e:
            logger.error("Forward pass failed: %s", str(e))
            raise
    async def _register_session(self) -> Session:
        if not self._session_registered:
            self.session = Session(
                level=InteractionLevel.NORMAL,
                config=InteractiveConfig(
                    timeout=INTERACTION_TIMEOUTS["batch"],
                    persistent_state=True,
                    safe_mode=True,
                    progress_tracking=True,
                    memory_threshold=self.memory_threshold
                )
            )
            await self.session.__aenter__()
            self._session_registered = True
        return self.session
    async def forward_interactive(self, x) -> torch.Tensor:
        """Interactive forward pass with resource monitoring"""
        if not self.interactive:
            return self.forward(x)

        try:
            session = await self._register_session()
            async with session:
                await self._check_resources()
                if self.session:
                    await self.session.save_progress({"stage": "starting_forward"})

                if self._interrupt_requested:
                    if self.session and await self.session.get_confirmation(
                        "Continue forward pass?",
                        timeout=INTERACTION_TIMEOUTS["emergency"]
                    ):
                        self._interrupt_requested = False
                    else:
                        raise RuntimeError("Forward pass interrupted by user")

                output = self.forward(x)
                if self.session:
                    await self.session.save_progress({"stage": "completed", "success": True})
                return output

        except Exception as e:
            logger.error("Interactive forward failed: %s", str(e))
            if self.session:
                await self.session.save_progress({"stage": "error", "error": str(e)})
            raise
    async def _check_resources(self) -> bool:
        """Check system resources"""
        try:
            import psutil
            mem = psutil.virtual_memory()
            if mem.percent > self.memory_threshold * 100:
                if self.session:
                    return await self.session.get_confirmation(
                        f"High memory usage ({mem.percent}%). Continue?",
                        timeout=INTERACTION_TIMEOUTS["emergency"]
                    )
            return True
        except Exception as e:
            logger.error(f"Resource check failed: {str(e)}")
            return True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
            logger.error(f"Resource check failed: {str(e)}")
            return True

    def forward(self, x):
        return self.encoder(x)
