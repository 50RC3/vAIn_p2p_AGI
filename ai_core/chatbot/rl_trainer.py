"""
Reinforcement Learning Trainer for the chatbot interface.
Implements online learning based on user feedback.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import asyncio
import time
import random
import numpy as np
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class RLConfig:
    """Configuration for reinforcement learning"""
    learning_rate: float = 0.001
    batch_size: int = 32
    gamma: float = 0.99
    update_interval: int = 1000
    memory_size: int = 10000
    min_samples_to_train: int = 64
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay: int = 10000
    training_frequency: int = 60  # seconds
    prioritized_replay: bool = True
    alpha: float = 0.6
    beta: float = 0.4
    
    def __post_init__(self):
        """Initialize device after construction"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclass
class TrainerState:
    """State tracking for RL trainer"""
    running: bool = True
    last_update_time: float = 0
    update_count: int = 0
    callbacks: Dict[str, Callable[[Any], Any]] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    total_experiences: int = 0
    total_updates: int = 0
    steps_since_last_update: int = 0

class ReplayBuffer:
    """Memory buffer for experience replay"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        
    def __len__(self) -> int:
        return len(self.buffer)
        
    def add(self, state: Any, action: Any, reward: float, next_state: Any, done: bool) -> None:
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample random batch from buffer"""
        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough samples in buffer ({len(self.buffer)}/{batch_size})")
            
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        # Sample batch_size items from buffer
        indices = random.sample(range(len(self.buffer)), batch_size)
        for idx in indices:
            state, action, reward, next_state, done = self.buffer[idx]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            
        return (states, actions, rewards, next_states, dones)


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized replay buffer for experience replay with importance sampling"""
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0=uniform sampling, higher=more prioritization)
            beta: Importance sampling exponent (0=no correction, 1=full correction)
            beta_increment: Amount to increase beta each sampling
        """
        super().__init__(capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = 1e-6  # Small constant to avoid zero priority
        self.max_priority = 1.0
    
    def add(self, state: Any, action: Any, reward: float, next_state: Any, done: bool) -> None:
        """Add experience with max priority to buffer"""
        super().add(state, action, reward, next_state, done)
        self.priorities.append(self.max_priority)
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample batch based on priorities"""
        if len(self.buffer) < batch_size:
            # Fallback to uniform sampling if buffer is too small
            return super().sample(batch_size)
            
        # Increase beta for importance sampling
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs = probs / np.sum(probs)
        
        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights = weights / np.max(weights)  # Normalize
        
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for idx in indices:
            state, action, reward, next_state, done = self.buffer[idx]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            
        return (states, actions, rewards, next_states, dones, indices, weights)
    
    def update_priorities(self, indices: List[int], new_priorities: List[float]) -> None:
        """Update priorities for experiences at the specified indices"""
        for idx, priority in zip(indices, new_priorities):
            if 0 <= idx < len(self.priorities):
                # Add small constant to avoid zero priority
                self.priorities[idx] = priority + self.epsilon
                self.max_priority = max(self.max_priority, self.priorities[idx])


class RLTrainer:
    """Reinforcement learning trainer for response quality improvement"""
    
    def __init__(self, 
                 quality_model: nn.Module, 
                 config: RLConfig, 
                 device: Optional[torch.device] = None,
                 embedder: Optional[nn.Module] = None):
        """
        Initialize the RL trainer with quality model and config
        
        Args:
            quality_model: Neural network for quality prediction
            config: Configuration for reinforcement learning
            device: Device to run calculations on (defaults to config.device)
            embedder: Optional model to create embeddings
        """
        self.quality_model = quality_model
        self.config = config
        self.device = device or config.device
        self.grad_step = 0
        self.optimizer = optim.Adam(
            self.quality_model.parameters(), 
            lr=self.config.learning_rate
        )
        self.running = True
        self.lock = asyncio.Lock()
        
        # Move model to device
        self.quality_model = self.quality_model.to(self.device)
        
        # Initialize embedder if provided
        self.embedder = None
        if embedder is not None:
            self.embedder = embedder.to(self.device)
            
        # Initialize replay buffer
        if self.config.prioritized_replay:
            self.memory = PrioritizedReplayBuffer(
                capacity=self.config.memory_size,
                alpha=self.config.alpha,
                beta=self.config.beta
            )
        else:
            self.memory = ReplayBuffer(capacity=self.config.memory_size)
        
        # Training state
        self.state = TrainerState()
        self._training_stats = {
            "total_interactions": 0,
            "successful_updates": 0
        }
        self._latest_loss = None
        self._last_update_time = time.time()
        
        logger.info(f"RL trainer initialized with {self.device} device")
        
    def register_callback(self, event_name: str, callback: Callable[[Any], Any]) -> None:
        """
        Register callback for specific events.
        
        Args:
            event_name: Name of event to register for
            callback: Callback function
        """
        self.state.callbacks[event_name] = callback
            
    async def add_experience(self, 
                          input_embedding: torch.Tensor, 
                          response_embedding: torch.Tensor, 
                          feedback_score: float) -> None:
        """
        Add experience to replay buffer.
        
        Args:
            input_embedding: Embedding of input text
            response_embedding: Embedding of response text
            feedback_score: User feedback score normalized between 0 and 1
        """
        if not self.running:
            return
            
        # Create combined state representation
        state = torch.cat([input_embedding, response_embedding], dim=-1)
        
        # Simulate next state as a slight variation
        next_state = state + torch.randn_like(state) * 0.01
        
        # Convert feedback to reward (-1 to 1 range)
        reward = (feedback_score - 0.5) * 2.0
        
        # Add to replay buffer
        self.memory.add(state, None, reward, next_state, False)
        self.state.total_experiences += 1
        self.state.steps_since_last_update += 1
        
        # Log the addition
        logger.debug(f"Added experience with reward {reward:.3f}, total: {self.state.total_experiences}")
        
        # Trigger training if enough time has passed
        if (time.time() - self.state.last_update_time > self.config.training_frequency and 
            len(self.memory) > self.config.min_samples_to_train):
            await self._update()
            
    async def _update(self) -> None:
        """Update quality model using samples from replay buffer"""
        if not self.running or len(self.memory) < self.config.batch_size:
            return
            
        # Track update time
        self.state.last_update_time = time.time()
        self.state.update_count += 1
        
        # Sample from replay buffer
        if self.config.prioritized_replay:
            states, _, rewards, next_states, _, indices, weights = self.memory.sample(self.config.batch_size)
            weight_tensor = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
        else:
            states, _, rewards, next_states, _ = self.memory.sample(self.config.batch_size)
            weight_tensor = None
        
        # Convert to tensors
        state_tensor = torch.stack([s for s in states if isinstance(s, torch.Tensor)])
        reward_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        
        # Predict quality scores
        predicted_quality = self.quality_model(state_tensor)
        
        # Calculate loss (MSE to target rewards)
        if weight_tensor is not None:
            # Weighted MSE loss for prioritized replay
            squared_error = (predicted_quality - reward_tensor) ** 2
            loss = (squared_error * weight_tensor).mean()
            
            # Update priorities based on TD error
            with torch.no_grad():
                td_errors = torch.abs(predicted_quality - reward_tensor).detach().cpu().numpy()
                self.memory.update_priorities(indices, td_errors.flatten())
        else:
            loss = F.mse_loss(predicted_quality, reward_tensor)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(self.quality_model.parameters(), 1.0)
        self.optimizer.step()
        
        # Update metrics
        self.state.metrics["loss"] = loss.item()
        self.state.metrics["mean_reward"] = reward_tensor.mean().item()
        self.state.metrics["buffer_size"] = len(self.memory)
        self.state.total_updates += 1
        self.state.steps_since_last_update = 0
        
        # Log the update
        logger.info(f"RL update #{self.state.total_updates}: loss={loss.item():.4f}, mean_reward={self.state.metrics['mean_reward']:.4f}")
        
        # Call update callback if registered
        if "update_completed" in self.state.callbacks:
            callback_data = {
                "loss": loss.item(),
                "mean_reward": self.state.metrics["mean_reward"],
                "update_count": self.state.total_updates,
                "buffer_size": len(self.memory)
            }
            await self._call_callback("update_completed", callback_data)
            
    async def _call_callback(self, event_name: str, data: Any) -> None:
        """Safely call a registered callback"""
        if event_name not in self.state.callbacks:
            return
            
        try:
            callback = self.state.callbacks[event_name]
            if asyncio.iscoroutinefunction(callback):
                await callback(data)
            else:
                callback(data)
        except Exception as e:
            logger.error(f"Error in RL callback {event_name}: {e}")
            
    def predict_quality(self, state_embedding: torch.Tensor) -> float:
        """
        Predict quality score for a given state embedding
        
        Args:
            state_embedding: Combined state embedding
            
        Returns:
            float: Predicted quality score 
        """
        with torch.no_grad():
            score = self.quality_model(state_embedding).item()
            return score
            
    async def shutdown(self) -> None:
        """Shutdown the trainer"""
        self.running = False
        # Wait for any pending operations
        async with self.lock:
            pass
