import os
import json
import logging
import torch
import asyncio
import random
import torch.nn as nn
import time
from typing import Dict, Optional, Any, List, Tuple

logger = logging.getLogger(__name__)

class SelfSupervisedLearning:
    """Self-supervised learning for text using masked language modeling"""
    
    def __init__(self, model_name: str = "bert-base-uncased", 
                 learning_rate: float = 2e-5, 
                 mask_probability: float = 0.15,
                 save_interval: int = 1000,
                 model_path: str = "./models/self_supervised"):
        """Initialize self-supervised learning module
        
        Args:
            model_name: Name of the pretrained model to use
            learning_rate: Learning rate for fine-tuning
            mask_probability: Probability of masking a token
            save_interval: How often to save model (in examples)
            model_path: Path to save the model
        """
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.mask_probability = mask_probability
        self.save_interval = save_interval
        self.model_path = model_path
        self.examples_processed = 0
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.training_lock = asyncio.Lock()
        self.recent_losses = []  # Track recent losses for adaptive learning
        self.last_save_time = time.time()
        self.adaptive_lr = True  # Enable adaptive learning rate based on loss trends
        self.min_lr = learning_rate / 10  # Minimum learning rate
        self.max_lr = learning_rate * 2   # Maximum learning rate
        
        # Try to initialize model and tokenizer
        success = self._try_init_model()
        
        # Try to load saved model if initialization was successful
        if success:
            asyncio.create_task(self._try_load_saved_model())
    
    async def _try_load_saved_model(self):
        """Attempt to load a previously saved model"""
        try:
            if os.path.exists(self.model_path):
                await self.load_model()
        except Exception as e:
            logger.warning(f"Could not load saved model: {e}")
    
    def _try_init_model(self) -> bool:
        """Try to initialize model and tokenizer with proper error handling"""
        # First try to use the transformers library
        try:
            # Try to import transformers
            from transformers import AutoModelForMaskedLM, AutoTokenizer
            
            # Initialize tokenizer and model
            logger.info(f"Loading model {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Try to detect and use GPU if available for better performance
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Add gradient checkpointing for memory efficiency
            try:
                self.model = AutoModelForMaskedLM.from_pretrained(
                    self.model_name,
                    gradient_checkpointing=True
                ).to(device)
            except:
                # Fallback without gradient checkpointing if not supported
                self.model = AutoModelForMaskedLM.from_pretrained(self.model_name).to(device)
            
            # Initialize optimizer with weight decay
            from torch.optim import AdamW
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() 
                              if not any(nd in n for nd in no_decay)],
                    "weight_decay": 0.01,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() 
                              if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
            
            # Create model directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            logger.info(f"Successfully initialized self-supervised learning with {self.model_name} on {device}")
            return True
            
        except ImportError:
            logger.warning("Could not import transformers library - using dummy implementation")
            self._init_dummy_model()
            return True
        except Exception as e:
            logger.error(f"Error initializing self-supervised learning: {e}")
            self._init_dummy_model()
            return False
    
    def _init_dummy_model(self):
        """Initialize a dummy model for when transformers is not available"""
        # Simple embedding layer as a dummy model
        self.model = nn.Embedding(10000, 768)  # 10k vocab, 768 embedding dim
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        logger.info("Using dummy model for self-supervised learning")
    
    async def training_step(self, text: str) -> Optional[float]:
        """Perform a training step with the given text"""
        if self.model is None:
            return None
            
        # Skip very short texts
        if len(text.strip()) < 3:
            return None
            
        async with self.training_lock:
            try:
                # Real transformer case
                if self.tokenizer is not None:
                    try:
                        # Use the actual tokenizer
                        inputs = self.tokenizer(text, return_tensors="pt", 
                                               truncation=True, max_length=512)
                        
                        # Move to same device as model
                        device = next(self.model.parameters()).device
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        
                        # Create masked version
                        masked_inputs = inputs.copy()
                        input_ids = masked_inputs["input_ids"].clone()
                        labels = input_ids.clone()
                        
                        # Apply random masking
                        rand = torch.rand(input_ids.shape, device=device)
                        mask_indices = (rand < self.mask_probability) & (input_ids != self.tokenizer.pad_token_id)
                        
                        # Replace with mask tokens
                        input_ids[mask_indices] = self.tokenizer.mask_token_id
                        masked_inputs["input_ids"] = input_ids
                        
                        # Set labels - will be ignored for unmasked positions
                        labels[~mask_indices] = -100
                        
                        # Adjust learning rate based on recent losses if enabled
                        if self.adaptive_lr and len(self.recent_losses) >= 10:
                            self._adjust_learning_rate()
                        
                        # Forward pass
                        outputs = self.model(**masked_inputs, labels=labels)
                        loss = outputs.loss
                        
                        # Optimization step
                        self.optimizer.zero_grad()
                        loss.backward()
                        
                        # Apply gradient clipping to prevent exploding gradients
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        
                        self.optimizer.step()
                        
                        # Track loss
                        loss_value = loss.item()
                        self.recent_losses.append(loss_value)
                        if len(self.recent_losses) > 100:
                            self.recent_losses.pop(0)
                            
                        self.examples_processed += 1
                        
                        # Check if it's time to save
                        if self.examples_processed % self.save_interval == 0:
                            asyncio.create_task(self.save_model())
                            
                        return loss_value
                        
                    except Exception as transformer_e:
                        logger.warning(f"Error using transformer implementation: {transformer_e}")
                        # Fall back to dummy implementation
                        return await self._dummy_training_step(text)
                else:
                    # Dummy implementation
                    return await self._dummy_training_step(text)
                    
            except Exception as e:
                logger.error(f"Error in self-supervised training step: {e}")
                return None
    
    async def _dummy_training_step(self, text: str) -> Optional[float]:
        """Fallback dummy implementation when transformers are not available"""
        try:
            # Tokenize input text
            tokens = [ord(c) % 10000 for c in text[:512]]  # Simple dummy tokenization
            
            # Apply masking
            masked_tokens, labels = self._mask_tokens(tokens)
            
            # In real implementation, convert to tensors and forward pass
            inputs = torch.tensor([masked_tokens])
            labels = torch.tensor([labels])
            
            # Forward pass and calculate loss
            if hasattr(self.model, 'forward'):
                # Dummy forward pass
                outputs = self.model(inputs)
                loss = torch.mean((outputs - labels.float()) ** 2)  # Simple MSE loss
                
                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                self.examples_processed += 1
                
                # Return loss value
                return loss.item()
            else:
                # Can't perform training with this model
                return None
        except Exception as e:
            logger.error(f"Error in dummy training step: {e}")
            return None
    
    def _mask_tokens(self, tokens: List[int]) -> Tuple[List[int], List[int]]:
        """Apply random masking to tokens for MLM training
        
        Returns:
            Tuple of (masked_tokens, labels) where labels are original tokens
        """
        masked_tokens = tokens.copy()
        labels = [-100] * len(tokens)  # Default label is -100 (ignored in loss)
        
        for i in range(len(tokens)):
            if random.random() < self.mask_probability:
                # Save original token as label
                labels[i] = tokens[i]
                
                # Apply masking 80% of the time
                if random.random() < 0.8:
                    masked_tokens[i] = 1  # [MASK] token (using 1 as placeholder)
                else:
                    # Replace with random token 10% of time
                    if random.random() < 0.5:
                        masked_tokens[i] = random.randint(0, 9999)
                    # Keep original token 10% of time (no change needed)
        
        return masked_tokens, labels
    
    def _adjust_learning_rate(self):
        """Adjust learning rate based on recent loss trends"""
        if len(self.recent_losses) < 10:
            return
            
        # Calculate trend using simple linear regression
        x = list(range(len(self.recent_losses)))
        y = self.recent_losses
        n = len(x)
        
        # Calculate slope of trend line
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return
            
        slope = numerator / denominator
        
        # If loss is increasing (positive slope), decrease learning rate
        if slope > 0.01:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'] * 0.8, self.min_lr)
            logger.debug(f"Decreased learning rate to {self.optimizer.param_groups[0]['lr']:.6f}")
        
        # If loss is decreasing rapidly, can slightly increase learning rate
        elif slope < -0.05:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = min(param_group['lr'] * 1.1, self.max_lr)
            logger.debug(f"Increased learning rate to {self.optimizer.param_groups[0]['lr']:.6f}")
    
    async def save_model(self) -> bool:
        """Save the model to disk with additional metadata"""
        try:
            if not os.path.exists(os.path.dirname(self.model_path)):
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Calculate average recent loss
            avg_loss = sum(self.recent_losses) / max(1, len(self.recent_losses))
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr'] if self.optimizer.param_groups else self.learning_rate
            
            # Save more comprehensive state
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'examples_processed': self.examples_processed,
                'avg_loss': avg_loss,
                'timestamp': time.time(),
                'model_name': self.model_name,
                'learning_rate': current_lr,
                'recent_losses': self.recent_losses[-10:] if self.recent_losses else []
            }, self.model_path)
            
            self.last_save_time = time.time()
            logger.info(f"Model saved to {self.model_path} after {self.examples_processed} examples (avg loss: {avg_loss:.4f})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    async def load_model(self) -> bool:
        """Load the model from disk"""
        try:
            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.examples_processed = checkpoint.get('examples_processed', 0)
                
                # Restore recent losses if available
                if 'recent_losses' in checkpoint:
                    self.recent_losses = checkpoint['recent_losses']
                
                logger.info(f"Model loaded from {self.model_path}, processed {self.examples_processed} examples")
                return True
            else:
                logger.warning(f"Model file not found at {self.model_path}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    async def get_embedding(self, text: str) -> torch.Tensor:
        """Get text embedding from the model"""
        # Handle empty text
        if not text or len(text.strip()) == 0:
            return torch.zeros(768)  # Return zero embedding for empty text
            
        try:
            # Use transformer tokenizer if available
            if self.tokenizer is not None:
                try:
                    # Tokenize the text
                    inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
                    
                    # Move to correct device
                    device = next(self.model.parameters()).device
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    # Get model's hidden states - use more efficient approach with reduced memory usage
                    with torch.no_grad():
                        outputs = self.model(**inputs, output_hidden_states=True)
                        
                        # Use the last 4 hidden layers for better embeddings by averaging them
                        last_layers = outputs.hidden_states[-4:]
                        
                        # Stack and average across layers
                        token_embeddings = torch.stack(last_layers, dim=0).mean(dim=0)
                        
                        # Pool the token embeddings to get a single embedding for the text
                        # Use attention-weighted pooling
                        attention_mask = inputs['attention_mask']
                        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                        
                        # Get sentence embedding by attention-weighted sum
                        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                        
                        return sum_embeddings / sum_mask
                
                except Exception as e:
                    logger.error(f"Error getting transformer embedding: {e}")
                    return torch.randn(768)  # Return random embedding on error
            
            # Fallback to simple embedding
            if hasattr(self.model, 'get_input_embeddings'):
                # Use proper model's embedding layer
                tokenized = [ord(c) % 10000 for c in text[:128]]  # Simple tokenization
                inputs = torch.tensor([tokenized])
                with torch.no_grad():
                    # Get embeddings from first layer
                    embeddings = self.model.get_input_embeddings()(inputs)
                    # Average the token embeddings to get text embedding
                    text_embedding = torch.mean(embeddings, dim=1)
                return text_embedding.squeeze(0)
            else:
                # Fallback to random embedding
                return torch.randn(768)  # Standard embedding dimension
                
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return torch.randn(768)  # Return random embedding on error
    
    def get_embedder(self) -> nn.Module:
        """Return the embedding component of the model"""
        if hasattr(self.model, 'get_input_embeddings'):
            return self.model.get_input_embeddings()
        else:
            # Return a simple embedder
            return nn.Embedding(10000, 768)
