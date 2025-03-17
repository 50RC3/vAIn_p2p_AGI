import torch
import torch.nn as nn
from typing import List, Tuple, Dict
from dataclasses import dataclass

@dataclass
class RLConfig:
    gamma: float = 0.99
    lr: float = 0.001
    batch_size: int = 32
    update_interval: int = 100
    memory_size: int = 1000

class RLTrainer:
    def __init__(self, model: nn.Module, config: RLConfig):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        self.memory = []
        self.max_memory_size = config.memory_size
        
    def store_interaction(self, state: torch.Tensor, action: torch.Tensor, 
                         reward: float, next_state: torch.Tensor):
        """Stores and processes model interactions for learning"""
        self.memory.append((state, action, reward, next_state))
        self.clear_old_samples()
        if len(self.memory) >= self.config.batch_size:
            self._update_policy()
            
    def _update_policy(self):
        batch = self._sample_batch()
        states, actions, rewards, next_states = zip(*batch)
        
        # Compute TD error and update
        current_values = self.model(torch.stack(states))
        next_values = self.model(torch.stack(next_states))
        td_target = torch.tensor(rewards) + self.config.gamma * next_values.max(1)[0]
        td_error = td_target - current_values.gather(1, torch.tensor(actions))
        
        # Update model
        loss = td_error.pow(2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.latest_loss = loss.item()
        
    def _sample_batch(self) -> List[Tuple]:
        """Sample random batch from memory"""
        indices = torch.randperm(len(self.memory))[:self.config.batch_size]
        return [self.memory[i] for i in indices]
        
    def clear_old_samples(self):
        """Remove old samples if memory exceeds max size"""
        if len(self.memory) > self.max_memory_size:
            self.memory = self.memory[-self.max_memory_size:]
            
    def get_training_stats(self) -> Dict[str, float]:
        """Return current training statistics"""
        return {
            'memory_size': len(self.memory),
            'avg_reward': sum(r for _, _, r, _ in self.memory) / len(self.memory),
            'latest_loss': self.latest_loss if hasattr(self, 'latest_loss') else None
        }
