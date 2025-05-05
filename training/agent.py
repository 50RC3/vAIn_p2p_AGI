"""
Agent implementation for the multi-agent system
"""
from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn
import logging
import asyncio
from config.agent_config import AgentConfig

logger = logging.getLogger(__name__)

class Agent:
    """An intelligent agent in the multi-agent system"""
    
    def __init__(self, config: AgentConfig):
        """
        Initialize an agent with the specified configuration.
        
        Args:
            config: The agent configuration
        """
        self.config = config
        self.model = self._create_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.knowledge_base: Dict[str, Any] = {}
        self.cognitive_state: Dict[str, float] = {
            "reasoning": 0.1,
            "memory": 0.1,
            "learning": 0.1,
            "adaptation": 0.1
        }
        self.experience: List[Dict[str, Any]] = []
        
        logger.info("Agent initialized")
        
    def _create_model(self) -> nn.Module:
        """
        Create the neural network model for this agent.
        
        Returns:
            nn.Module: The neural network model
        """
        # Create a simple model for demonstration
        input_size = self.config.input_dim
        hidden_size = self.config.hidden_dim
        output_size = self.config.output_dim
        
        model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
        return model
    
    def local_update(self, x: torch.Tensor, y: torch.Tensor, steps: int) -> float:
        """
        Perform a local model update.
        
        Args:
            x: Input data tensor
            y: Target data tensor
            steps: Number of optimization steps
            
        Returns:
            float: The final loss value
        """
        final_loss = 0.0
        
        for _ in range(steps):
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = nn.functional.mse_loss(output, y)
            loss.backward()
            self.optimizer.step()
            final_loss = loss.item()
            
            # Record this experience
            self.experience.append({
                "loss": final_loss,
                "improvement": 1.0 if len(self.experience) == 0 else 
                              self.experience[-1]["loss"] - final_loss
            })
            
            # Limit experience memory
            if len(self.experience) > 1000:
                self.experience = self.experience[-1000:]
            
        return final_loss
    
    async def extract_knowledge(self) -> Dict[str, Any]:
        """
        Extract knowledge from the agent's experience and model.
        
        Returns:
            Dict[str, Any]: Knowledge extracted from the agent
        """
        # Simulate knowledge extraction process
        await asyncio.sleep(0.01)  # Simulate processing time
        
        knowledge = {
            "model_weights_mean": {name: float(param.mean()) 
                                for name, param in self.model.named_parameters()},
            "learning_progress": self._calculate_learning_progress(),
            "cognitive_state": self.cognitive_state.copy()
        }
        
        # Add to knowledge base
        for key, value in knowledge.items():
            if key not in self.knowledge_base:
                self.knowledge_base[key] = []
            self.knowledge_base[key].append(value)
            
        # Only share the latest knowledge
        return {k: v[-1] if isinstance(v, list) and v else v 
                for k, v in self.knowledge_base.items()}
    
    async def integrate_knowledge(self, global_knowledge: Dict[str, Any]) -> None:
        """
        Integrate global knowledge into this agent's knowledge base.
        
        Args:
            global_knowledge: The global knowledge to integrate
        """
        # Simulate knowledge integration process
        await asyncio.sleep(0.01)  # Simulate processing time
        
        # Only integrate new knowledge
        for key, value in global_knowledge.items():
            if key not in self.knowledge_base:
                self.knowledge_base[key] = value
                
            # Special handling for cognitive state - learn from the global average
            if key == "cognitive_state" and isinstance(value, dict):
                for cog_key, cog_value in value.items():
                    if cog_key in self.cognitive_state:
                        # Blend agent's cognitive state with global knowledge
                        self.cognitive_state[cog_key] = 0.7 * self.cognitive_state[cog_key] + 0.3 * cog_value
    
    async def evolve_cognition(self, global_knowledge: Dict[str, Any]) -> Dict[str, float]:
        """
        Evolve the agent's cognitive abilities based on experience and global knowledge.
        
        Args:
            global_knowledge: The global knowledge to use for evolution
            
        Returns:
            Dict[str, float]: Improvements in cognitive abilities
        """
        previous_state = self.cognitive_state.copy()
        
        # Simulate neural plasticity and cognitive development
        await asyncio.sleep(0.02)  # Simulate deeper processing
        
        # Calculate learning improvement from experience
        if self.experience:
            recent_exp = self.experience[-20:]  # Last 20 experiences
            avg_improvement = sum(exp.get("improvement", 0) for exp in recent_exp) / len(recent_exp)
            
            # Update cognitive state based on experience
            self.cognitive_state["learning"] += max(0, min(0.01, avg_improvement * 0.001))
            self.cognitive_state["memory"] += 0.005  # Memory improves with experience
            
        # Update reasoning based on model complexity
        param_count = sum(p.numel() for p in self.model.parameters())
        self.cognitive_state["reasoning"] += 0.001 * (param_count / 10000)
        
        # Update adaptation based on global knowledge integration
        self.cognitive_state["adaptation"] += 0.003
        
        # Calculate and return improvements
        improvements = {k: self.cognitive_state[k] - previous_state[k] 
                      for k in self.cognitive_state}
        
        return improvements
    
    def _calculate_learning_progress(self) -> float:
        """
        Calculate the agent's learning progress based on experience.
        
        Returns:
            float: Learning progress metric
        """
        if not self.experience:
            return 0.0
            
        if len(self.experience) < 5:
            return 0.0
            
        # Compare first and last losses in experience window
        first_losses = self.experience[:5]
        last_losses = self.experience[-5:]
        
        avg_first = sum(exp["loss"] for exp in first_losses) / len(first_losses)
        avg_last = sum(exp["loss"] for exp in last_losses) / len(last_losses)
        
        if avg_first == 0:
            return 0.0
        
        # Learning progress as a percentage improvement
        progress = (avg_first - avg_last) / avg_first
        return max(0, min(1, progress))  # Clamp between 0 and 1
