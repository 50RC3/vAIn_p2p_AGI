from typing import List, Optional, Dict, Any, Tuple
import logging
import torch
from tqdm import tqdm
from .agent import Agent
from .federated import FederatedLearningManager
from config.agent_config import AgentConfig
from core.interactive_utils import InteractiveSession
from .evolution_tracker import EvolutionTracker
from .evolution import EvolutionTracker

logger = logging.getLogger(__name__)

class MultiAgentSystem:
    def __init__(self, config: AgentConfig, interactive_session: Optional[InteractiveSession] = None):
        if not isinstance(config, AgentConfig):
            raise ValueError("Invalid agent configuration provided")
        
        self.config = config
        self.interactive = interactive_session
        self.agents: List[Agent] = []
        
        # Initialize agents with validation
        for i in range(config.num_agents):
            try:
                agent = Agent(config)
                self.agents.append(agent)
            except Exception as e:
                logger.error("Failed to initialize agent {}: {}", i, str(e))
                raise
                
        logger.info("Successfully initialized {} agents", len(self.agents))
        
        # Add AGI coordination
        self.global_knowledge: Dict[str, Any] = {}
        self.collective_intelligence: float = 0.0
        self.evolution_tracker = EvolutionTracker()

    async def _evolve_cognitive_abilities(self) -> Dict[str, float]:
        """
        Evolve cognitive abilities across all agents.
        
        Returns:
            Dict[str, float]: Metrics of cognitive improvements
        """
        improvements = {
            "reasoning": 0.0,
            "memory": 0.0,
            "learning": 0.0,
            "adaptation": 0.0
        }
        
        # Apply cognitive evolution to each agent
        for agent in self.agents:
            if hasattr(agent, 'evolve_cognition'):
                agent_improvements = await agent.evolve_cognition(self.global_knowledge)
                for key, value in agent_improvements.items():
                    if key in improvements:
                        improvements[key] += value / len(self.agents)
                        
        # Track these improvements
        self.evolution_tracker.track_evolution_step({
            "cognitive_improvements": improvements,
            "collective_intelligence": self.collective_intelligence
        })
        
        return improvements

    def federated_update(self, federated_learning: FederatedLearningManager) -> bool:
        """Perform federated update with validation and error handling"""
        try:
            # Validate all agents have models
            if not all(hasattr(agent, 'model') for agent in self.agents):
                raise ValueError("Some agents missing models")

            # Collect and validate local models
            local_models = []
            for i, agent in enumerate(self.agents):
                try:
                    model_state = agent.model.state_dict()
                    # Basic validation of model state
                    if not model_state:
                        raise ValueError(f"Empty model state from agent {i}")
                    local_models.append(model_state)
                except Exception as e:
                    logger.error("Error getting model state from agent {}: {}", i, str(e))
                    if self.config.strict_mode:
                        raise
                    continue

            if not local_models:
                raise ValueError("No valid local models collected")

            # Aggregate models with timeout protection
            aggregated_state = federated_learning.aggregate_models(local_models)

            # Update all agents
            update_failed = False
            for i, agent in enumerate(self.agents):
                try:
                    agent.model.load_state_dict(aggregated_state)
                except Exception as e:
                    logger.error("Failed to update agent {}: {}", i, str(e))
                    update_failed = True
                    if self.config.strict_mode:
                        raise

            return not update_failed

        except Exception as e:
            logger.error("Federated update failed: {}", str(e))
            return False

    def train(self, federated_learning: FederatedLearningManager, 
              clients_data: List[Any], rounds: int, meta_steps: int) -> Dict[str, Any]:
        """Train the multi-agent system with progress tracking and validation"""
        if not self.agents or not clients_data:
            raise ValueError("Agents or client data not properly initialized")
        if len(self.agents) != len(clients_data):
            raise ValueError("Number of agents must match client data sets")

        training_stats: Dict[str, Any] = {
            'round_losses': [],
            'failed_updates': 0,
            'completed_rounds': 0
        }

        try:
            # Main training loop with progress bar
            with tqdm(total=rounds, desc="Training Progress") as pbar:
                for round_num in range(rounds):
                    round_losses = []
                    
                    # Train each agent
                    for agent_idx, (agent, client_data) in enumerate(zip(self.agents, clients_data)):
                        try:
                            batch_losses = []
                            for x, y in client_data:
                                # Validate input data
                                if not torch.is_tensor(x) or not torch.is_tensor(y):
                                    raise ValueError(f"Invalid input tensors from client {agent_idx}")
                                    
                                loss = agent.local_update(x, y, meta_steps)
                                batch_losses.append(loss)
                                
                            # Calculate average batch loss safely
                            if batch_losses:
                                avg_batch_loss = sum(batch_losses) / len(batch_losses)
                                round_losses.append(avg_batch_loss)
                            
                        except Exception as e:
                            logger.error("Error training agent {} in round {}: {}", agent_idx, round_num, str(e))
                            if self.config.strict_mode:
                                raise
                            continue

                    # Perform federated update
                    if not self.federated_update(federated_learning):
                        training_stats['failed_updates'] += 1
                        if self.config.strict_mode:
                            raise RuntimeError(f"Federated update failed in round {round_num}")

                    # Update statistics
                    if round_losses:
                        avg_round_loss = sum(round_losses) / len(round_losses)
                        training_stats['round_losses'].append(float(avg_round_loss))
                        logger.info("Round {} complete, Average Loss: {:.4f}", round_num+1, avg_round_loss)
                    
                    training_stats['completed_rounds'] = round_num + 1
                    pbar.update(1)

        except Exception as e:
            logger.error("Training interrupted: {}", str(e))
            if self.config.strict_mode:
                raise
        
        return training_stats
    
    async def coordinate_global_learning(self, 
                                      federated_learning: FederatedLearningManager,
                                      rounds: int) -> Dict[str, Any]:
        """Coordinate global AGI evolution"""
        try:
            evolution_stats = []
            
            for round_num in range(rounds):
                # Synchronize agent knowledge
                await self._share_global_knowledge()
                
                # Evolve cognitive abilities
                cognitive_improvements = await self._evolve_cognitive_abilities()
                
                # Update collective intelligence
                self.collective_intelligence = await federated_learning.update_global_intelligence(
                    [agent.model for agent in self.agents]
                )
                
                # Track evolution
                stats = {
                    'round': round_num,
                    'collective_intelligence': self.collective_intelligence,
                    'cognitive_improvements': cognitive_improvements,
                    'global_knowledge': len(self.global_knowledge)
                }
                evolution_stats.append(stats)
                
                logger.info("Global AGI Evolution - Round {}", round_num)
                logger.info("Collective Intelligence: {:.4f}", self.collective_intelligence)
                
            # Return the final state
            return {
                "evolution_rounds": rounds,
                "final_intelligence": self.collective_intelligence,
                "evolution_history": evolution_stats
            }
            
        except Exception as e:
            logger.error("Global coordination failed: {}", str(e))
            raise
            
    async def _share_global_knowledge(self) -> None:
        """Share and integrate knowledge across agents"""
        try:
            for agent in self.agents:
                # Extract agent's unique knowledge
                new_knowledge = await agent.extract_knowledge()
                
                # Integrate into global knowledge
                self.global_knowledge.update(new_knowledge)
                
                # Share global knowledge back
                await agent.integrate_knowledge(self.global_knowledge)
                
        except Exception as e:
            logger.error("Knowledge sharing failed: {}", str(e))
            raise
