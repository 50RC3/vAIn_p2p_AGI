import random
import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)

class ReplayBuffer:
    """Simple replay buffer for reinforcement learning"""
    
    def __init__(self, buffer_size: int = 10000):
        """Initialize a ReplayBuffer
        
        Args:
            buffer_size: Maximum size of buffer
        """
        self.buffer = deque(maxlen=buffer_size)
        
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of experiences from memory."""
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
            
        samples = random.sample(self.buffer, batch_size)
        
        states = torch.cat([s[0] for s in samples]) if torch.is_tensor(samples[0][0]) else np.array([s[0] for s in samples])
        actions = torch.cat([s[1] for s in samples]) if torch.is_tensor(samples[0][1]) else np.array([s[1] for s in samples])
        rewards = torch.tensor([s[2] for s in samples]).float().unsqueeze(1) if torch.is_tensor(samples[0][2]) else np.array([s[2] for s in samples])
        next_states = torch.cat([s[3] for s in samples]) if torch.is_tensor(samples[0][3]) else np.array([s[3] for s in samples])
        dones = torch.tensor([s[4] for s in samples]).float().unsqueeze(1) if torch.is_tensor(samples[0][4]) else np.array([s[4] for s in samples])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        return len(self.buffer)


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Replay Buffer implementation for reinforcement learning"""
    
    def __init__(self, buffer_size: int = 10000, alpha: float = 0.6, beta: float = 0.4):
        """Initialize a prioritized replay buffer
        
        Args:
            buffer_size: Maximum size of buffer
            alpha: How much prioritization to use (0 = no prioritization, 1 = full prioritization)
            beta: Importance sampling correction factor (0 = no correction, 1 = full correction)
        """
        super().__init__(buffer_size)
        self.alpha = alpha
        self.beta = beta
        self.priorities = deque(maxlen=buffer_size)
        self.eps = 1e-6  # small constant to ensure non-zero priority
        
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory with max priority."""
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
        
        # New experiences get max priority
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.priorities.append(max_priority)
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of experiences from memory using priorities."""
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
            
        # Convert priorities to probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** -self.beta
        weights /= weights.max()  # normalize
        weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
        
        # Convert samples to tensors
        states = torch.cat([s[0] for s in samples]) if torch.is_tensor(samples[0][0]) else np.array([s[0] for s in samples])
        actions = torch.cat([s[1] for s in samples]) if torch.is_tensor(samples[0][1]) else np.array([s[1] for s in samples])
        rewards = torch.tensor([s[2] for s in samples]).float().unsqueeze(1) if torch.is_tensor(samples[0][2]) else np.array([s[2] for s in samples])
        next_states = torch.cat([s[3] for s in samples]) if torch.is_tensor(samples[0][3]) else np.array([s[3] for s in samples])
        dones = torch.tensor([s[4] for s in samples]).float().unsqueeze(1) if torch.is_tensor(samples[0][4]) else np.array([s[4] for s in samples])
        
        return states, actions, rewards, next_states, dones, weights, indices
    
    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        """Update priorities of sampled transitions."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.eps