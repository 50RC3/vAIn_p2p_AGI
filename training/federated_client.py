import torch
from torch.optim import Adam
import torch.nn.functional as F
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class FederatedClientError(Exception):
    pass

class FederatedClient:
    def __init__(self, model, data_loader, config):
        try:
            self.model = model
            self.data_loader = data_loader
            self.config = config
            self.optimizer = Adam(model.parameters(), lr=config.learning_rate)
            self.criterion = F.cross_entropy  # Define loss function
            logger.info("Initialized FederatedClient")
        except Exception as e:
            logger.error(f"Failed to initialize FederatedClient: {str(e)}")
            raise FederatedClientError(f"Initialization failed: {str(e)}")

    def train(self) -> Optional[Dict]:
        try:
            self.model.train()
            total_loss = 0
            for epoch in range(self.config.num_epochs):
                epoch_loss = 0
                for batch_idx, (data, target) in enumerate(self.data_loader):
                    try:
                        loss = self._train_batch(data, target)
                        epoch_loss += loss
                    except Exception as e:
                        logger.error(f"Batch training failed: {str(e)}")
                        continue
                
                avg_epoch_loss = epoch_loss / len(self.data_loader)
                logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}, Loss: {avg_epoch_loss:.4f}")
                total_loss += avg_epoch_loss
            
            return self.model.state_dict()
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return None

    def _train_batch(self, data, target):
        self.optimizer.zero_grad()
        output = self.model(data)
        if isinstance(output, tuple):
            output = output[0]
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()
