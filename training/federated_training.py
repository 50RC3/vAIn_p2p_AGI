import torch
import logging
from typing import List, Optional
from tqdm import tqdm
from core.constants import ModelStatus
from core.interactive_utils import InteractiveSession
from .federated_client import FederatedClient
from .aggregation import aggregate_models

logger = logging.getLogger(__name__)

class FederatedTraining:
    def __init__(self, model, data_loader, config):
        self.model = model
        self.data_loader = data_loader
        self.config = config
        self.clients = []
        self.session = InteractiveSession(level=config.interaction_level)
        self.status = ModelStatus.PENDING
        self._validate_config()

    def _validate_config(self):
        """Validate training configuration"""
        if not self.config.num_rounds > 0:
            raise ValueError("num_rounds must be positive")
        if not 0 < self.config.client_fraction <= 1:
            raise ValueError("client_fraction must be between 0 and 1")

    def _select_clients(self) -> List[FederatedClient]:
        """Select active clients based on hardware and reputation"""
        try:
            available = [c for c in self.clients if c.is_available()]
            client_count = max(1, int(len(available) * self.config.client_fraction))
            return sorted(available, 
                        key=lambda c: (c.get_reliability_score(), c.get_hardware_score()),
                        reverse=True)[:client_count]
        except Exception as e:
            logger.error(f"Client selection failed: {e}")
            raise

    async def train(self) -> Optional[torch.nn.Module]:
        """Execute federated training with monitoring and error handling"""
        try:
            self.status = ModelStatus.TRAINING
            progress = tqdm(range(self.config.num_rounds), desc="Training rounds")

            for round in progress:
                if await self.session.should_continue():
                    # Select and validate clients
                    active_clients = self._select_clients()
                    if len(active_clients) == 0:
                        logger.warning("No clients available, skipping round")
                        continue

                    # Train on each client with timeout
                    client_models = []
                    for client in active_clients:
                        try:
                            model = await self.session.run_with_timeout(
                                client.train(),
                                timeout=self.config.training_timeout
                            )
                            if self._validate_model(model):
                                client_models.append(model)
                        except Exception as e:
                            logger.error(f"Client training failed: {e}")
                            continue

                    # Aggregate validated models
                    if client_models:
                        self.model = aggregate_models(client_models)
                        await self._save_checkpoint(round)
                    
                    # Update progress metrics
                    progress.set_postfix({
                        'active_clients': len(active_clients),
                        'success_rate': len(client_models) / len(active_clients)
                    })

            self.status = ModelStatus.VALIDATED
            return self.model

        except Exception as e:
            self.status = ModelStatus.REJECTED
            logger.error(f"Training failed: {e}")
            raise
        finally:
            await self.session.cleanup()

    def _validate_model(self, model: torch.nn.Module) -> bool:
        """Validate model updates for Byzantine behavior"""
        try:
            # Check for NaN values
            for param in model.parameters():
                if torch.isnan(param).any():
                    return False
                    
            # Check update magnitude
            magnitude = torch.norm(
                torch.cat([p.view(-1) for p in model.parameters()])
            )
            if magnitude > self.config.max_update_norm:
                return False
                
            return True
        except Exception:
            return False

    async def _save_checkpoint(self, round: int):
        """Save training checkpoint with error handling"""
        try:
            if self.config.checkpointing_enabled:
                state = {
                    'round': round,
                    'model': self.model.state_dict(),
                    'status': self.status
                }
                torch.save(state, f"checkpoint_round_{round}.pt")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
