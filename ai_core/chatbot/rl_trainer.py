import torch
import torch.nn as nn
import logging
import asyncio
from typing import List, Tuple, Dict, Optional, Callable, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class RLConfig:
    gamma: float = 0.99
    lr: float = 0.001
    batch_size: int = 32
    update_interval: int = 100
    memory_size: int = 1000
    min_samples_to_train: int = 100
    max_batch_retries: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    priority_alpha: float = 0.6  # Priority sampling parameter
    priority_beta: float = 0.4   # Importance sampling parameter
    grad_accum_steps: int = 4    # Number of steps for gradient accumulation

@dataclass
class TrainerState:
    is_training: bool = False
    total_steps: int = 0
    callbacks: Dict[str, Callable] = field(default_factory=dict)

class RLTrainer:
    def __init__(self, model: nn.Module, config: RLConfig):
        """Initialize RL trainer with model and config"""
        self.model = model.to(config.device)
        self.config = config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        self.memory: List[Tuple] = []
        self.max_memory_size = config.memory_size
        self.training_lock = asyncio.Lock()
        self._latest_loss: Optional[float] = None
        self._training_stats = {"total_interactions": 0, "successful_updates": 0}
        self.priorities = torch.ones(config.memory_size)
        self.grad_step = 0
        self.scaler = torch.cuda.amp.GradScaler()
        self.state = TrainerState()
        
    def register_callback(self, event: str, callback: Callable):
        """Register callback for trainer events"""
        self.state.callbacks[event] = callback

    async def _notify_callback(self, event: str, data: Any):
        """Notify registered callback"""
        if event in self.state.callbacks:
            await self.state.callbacks[event](data)
        
    async def store_interaction(self, state: torch.Tensor, action: torch.Tensor, 
                              reward: float, next_state: torch.Tensor) -> None:
        """Stores interaction and triggers async policy update if needed"""
        try:
            # Validate inputs
            if not all(isinstance(t, torch.Tensor) for t in [state, action, next_state]):
                raise ValueError("State and action must be torch tensors")
            if not isinstance(reward, (int, float)):
                raise ValueError("Reward must be numeric")

            # Move tensors to device and store
            interaction = (
                state.to(self.config.device),
                action.to(self.config.device),
                reward,
                next_state.to(self.config.device)
            )
            self.memory.append(interaction)
            self._training_stats["total_interactions"] += 1
            self.priorities[len(self.memory)-1] = 1.0  # New experiences get max priority
            
            # Clear old samples if needed
            await self.clear_old_samples()
            
            # Update policy if enough samples
            if len(self.memory) >= self.config.min_samples_to_train:
                async with self.training_lock:
                    await self._update_policy()
                    
            await self._notify_callback('interaction_stored', {
                'state': state,
                'action': action,
                'reward': reward
            })
                    
        except Exception as e:
            logger.error(f"Error storing interaction: {str(e)}")
            raise

    async def _update_policy(self) -> None:
        """Update policy using sampled batch with retry logic"""
        for attempt in range(self.config.max_batch_retries):
            try:
                batch, indices, weights = await self._sample_batch()
                
                with torch.cuda.amp.autocast():
                    states, actions, rewards, next_states = zip(*batch)
                    
                    # Stack and move tensors, ensuring proper shapes
                    states_t = torch.stack(states).squeeze(1)  # Remove extra dim
                    next_states_t = torch.stack(next_states).squeeze(1)
                    actions_t = torch.stack(actions).to(self.config.device)
                    rewards_t = torch.tensor(rewards, device=self.config.device).float()

                    # Compute TD error with proper dimensionality
                    with torch.no_grad():
                        next_values = self.model(next_states_t).detach()
                    current_values = self.model(states_t)
                    
                    td_target = rewards_t + self.config.gamma * next_values.max(1)[0]
                    td_error = td_target.unsqueeze(1) - current_values.gather(1, actions_t.unsqueeze(1))
                    
                    # Update model with gradient clipping
                    loss = (td_error.pow(2) * weights.unsqueeze(1)).mean()
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.config.grad_accum_steps

                self.scaler.scale(loss).backward()
                
                # Update priorities
                self.priorities[indices] = td_error.abs().detach().cpu() ** self.config.priority_alpha
                
                self.grad_step += 1
                if self.grad_step % self.config.grad_accum_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                
                self._latest_loss = loss.item()
                self._training_stats["successful_updates"] += 1
                return

            except Exception as e:
                if attempt == self.config.max_batch_retries - 1:
                    logger.error(f"Policy update failed after {attempt+1} attempts: {str(e)}")
                    raise
                logger.warning(f"Policy update attempt {attempt+1} failed: {str(e)}")
                await asyncio.sleep(0.1 * (attempt + 1))

    async def _sample_batch(self) -> Tuple[List[Tuple], torch.Tensor, torch.Tensor]:
        """Sample random batch from memory with validation"""
        if len(self.memory) < self.config.batch_size:
            raise ValueError(f"Not enough samples in memory: {len(self.memory)}")
        probs = self.priorities[:len(self.memory)] / self.priorities[:len(self.memory)].sum()
        indices = torch.multinomial(probs, self.config.batch_size)
        weights = (len(self.memory) * probs[indices]) ** -self.config.priority_beta
        weights = weights / weights.max()
        return [self.memory[i] for i in indices], indices, weights.to(self.config.device)
        
    async def clear_old_samples(self) -> None:
        """Remove old samples if memory exceeds max size"""
        if len(self.memory) > self.max_memory_size:
            self.memory = self.memory[-self.max_memory_size:]
            logger.debug(f"Cleared old samples. New memory size: {len(self.memory)}")

    def get_training_stats(self) -> Dict[str, float]:
        """Return comprehensive training statistics"""
        stats = {
            'memory_size': len(self.memory),
            'device': self.config.device,
            'latest_loss': self._latest_loss,
            **self._training_stats
        }
        
        if self.memory:
            stats['avg_reward'] = sum(r for _, _, r, _ in self.memory) / len(self.memory)
        
        return stats

    def save_state(self, path: str) -> None:
        """Save trainer state for recovery"""
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'memory': self.memory,
            'stats': self._training_stats,
        }, path)

    def load_state(self, path: str) -> None:
        """Load trainer state"""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.memory = checkpoint['memory']
        self._training_stats = checkpoint['stats']
