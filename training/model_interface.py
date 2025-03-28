import torch
import torch.nn as nn
import logging
import asyncio
import os
import psutil
from typing import Dict, Any, Optional, List, Union

logger = logging.getLogger(__name__)

class ModelInterface:
    """Interface for model training and inference operations"""
    
    def __init__(self, model_path: Optional[str] = None, config: Dict[str, Any] = None):
        self.model = None
        self.model_path = model_path
        self.config = config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training = False
        self.initialized = False
        
    async def initialize(self):
        """Initialize the model asynchronously"""
        if self.initialized:
            return
            
        try:
            logger.info(f"Initializing model from {self.model_path}")
            # Model initialization logic here
            if self.model_path and os.path.exists(self.model_path):
                self.model = torch.load(self.model_path, map_location=self.device)
            else:
                # Create a default model if path doesn't exist
                self.model = self._create_default_model()
                
            self.initialized = True
            logger.info("Model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            self.initialized = False
            
    def _create_default_model(self) -> nn.Module:
        """Create a default model if none exists"""
        # Simple placeholder model - replace with actual model architecture
        model = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )
        return model.to(self.device)
        
    async def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run prediction with the model"""
        if not self.initialized:
            await self.initialize()
            
        try:
            # Ensure inputs are on the correct device
            inputs = inputs.to(self.device)
            
            # Run prediction
            with torch.no_grad():
                output = self.model(inputs)
                
            return output
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
            
    async def train(self, training_data, validation_data=None, epochs=10):
        """Train the model with the provided data"""
        if not self.initialized:
            await self.initialize()
            
        try:
            self.training = True
            logger.info(f"Starting training for {epochs} epochs")
            
            # Simple training loop - replace with actual training logic
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            for epoch in range(epochs):
                # Training logic here
                inputs, targets = training_data
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
                
                # Check if we should stop training
                if self._should_stop_training():
                    logger.info("Training stopped early")
                    break
                    
            self.training = False
            logger.info("Training completed")
            
            # Save the trained model
            if self.model_path:
                torch.save(self.model, self.model_path)
                logger.info(f"Model saved to {self.model_path}")
                
            return {"status": "success", "epochs_completed": epoch + 1}
        except Exception as e:
            self.training = False
            logger.error(f"Training error: {str(e)}")
            raise RuntimeError(f"Training failed: {str(e)}")
            
    def _should_stop_training(self) -> bool:
        """Check if training should be stopped early"""
        # Add logic for early stopping here
        # Example: Check system resources
        if psutil.virtual_memory().percent > 95:
            logger.warning("Memory usage too high, stopping training")
            return True
            
        return False
        
    def stop_training(self):
        """Signal to stop the training process"""
        if self.training:
            logger.info("Requested to stop training")
            self.training = False
