import torch
from typing import Dict, Any, Optional
from models import ModelOutput
from core.interactive_utils import InteractiveSession

class ModelInterface:
    def __init__(self, model: torch.nn.Module, interactive: bool = True):
        self.model = model
        self.interactive = interactive
        self.session = None
        self._cached_outputs = {}
        
    async def forward(self, x: torch.Tensor) -> ModelOutput:
        if hasattr(self.model, 'forward_interactive') and self.interactive:
            output = await self.model.forward_interactive(x)
        else:
            output = self.model(x)
            
        # Standardize output format
        if isinstance(output, tuple):
            main_output, *extras = output
            return ModelOutput(
                output=main_output,
                attention=extras[0] if len(extras) > 0 else None,
                memory_state=extras[1] if len(extras) > 1 else None
            )
        return ModelOutput(output=output)

    async def adapt(self, support_set: Dict[str, torch.Tensor]) -> bool:
        if hasattr(self.model, 'adapt_interactive'):
            return await self.model.adapt_interactive(support_set)
        elif hasattr(self.model, 'adapt'):
            return self.model.adapt(support_set)
        return False

    async def share_state(self, source_id: str, state: Dict[str, torch.Tensor]) -> bool:
        """Share model state with other models"""
        try:
            for key, tensor in state.items():
                shared_key = f"{source_id}_{key}"
                self._cached_outputs[shared_key] = tensor
            return True
        except Exception as e:
            logger.error(f"State sharing failed: {e}")
            return False

    async def get_shared_state(self, source_id: str) -> Optional[Dict[str, torch.Tensor]]:
        """Retrieve shared state from other models"""
        try:
            state = {}
            prefix = f"{source_id}_"
            for key in self._cached_outputs:
                if key.startswith(prefix):
                    state[key[len(prefix):]] = self._cached_outputs[key]
            return state
        except Exception as e:
            logger.error(f"State retrieval failed: {e}")
            return None

    def cleanup(self):
        self._cached_outputs.clear()
        if hasattr(self.model, 'cleanup'):
            self.model.cleanup()
