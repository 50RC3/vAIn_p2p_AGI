from typing import List, Optional
import logging
import torch
from tqdm import tqdm
from .agent import Agent
from .federated import FederatedLearning
from config.agent_config import AgentConfig
from core.interactive_utils import InteractiveSession
from core.constants import InteractionLevel

logger = logging.getLogger(__name__)

class MultiAgentSystem:
    def __init__(self, config: AgentConfig, interactive_session: Optional[InteractiveSession] = None):
        if not isinstance(config, AgentConfig):
            raise ValueError("Invalid agent configuration provided")
        
        self.config = config
        self.interactive = interactive_session
        self.agents = []
        
        # Initialize agents with validation
        for i in range(config.num_agents):
            try:
                agent = Agent(config)
                self.agents.append(agent)
            except Exception as e:
                logger.error(f"Failed to initialize agent {i}: {str(e)}")
                raise
                
        logger.info(f"Successfully initialized {len(self.agents)} agents")

    def federated_update(self, federated_learning: FederatedLearning) -> bool:
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
                    logger.error(f"Error getting model state from agent {i}: {str(e)}")
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
                    logger.error(f"Failed to update agent {i}: {str(e)}")
                    update_failed = True
                    if self.config.strict_mode:
                        raise

            return not update_failed

        except Exception as e:
            logger.error(f"Federated update failed: {str(e)}")
            return False

    def train(self, federated_learning: FederatedLearning, 
              clients_data: List, rounds: int, meta_steps: int) -> dict:
        """Train the multi-agent system with progress tracking and validation"""
        if not self.agents or not clients_data:
            raise ValueError("Agents or client data not properly initialized")
        if len(self.agents) != len(clients_data):
            raise ValueError("Number of agents must match client data sets")

        training_stats = {
            'round_losses': [],
            'failed_updates': 0,
            'completed_rounds': 0
        }

        try:
            # Main training loop with progress bar
            with tqdm(total=rounds, desc="Training Progress") as pbar:
                for round in range(rounds):
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
                                
                            avg_batch_loss = sum(batch_losses) / len(batch_losses)
                            round_losses.append(avg_batch_loss)
                            
                        except Exception as e:
                            logger.error(f"Error training agent {agent_idx} in round {round}: {str(e)}")
                            if self.config.strict_mode:
                                raise
                            continue

                    # Perform federated update
                    if not self.federated_update(federated_learning):
                        training_stats['failed_updates'] += 1
                        if self.config.strict_mode:
                            raise RuntimeError(f"Federated update failed in round {round}")

                    # Update statistics
                    if round_losses:
                        avg_round_loss = sum(round_losses) / len(round_losses)
                        training_stats['round_losses'].append(avg_round_loss)
                        logger.info(f'Round {round+1} complete, Average Loss: {avg_round_loss:.4f}')
                    
                    training_stats['completed_rounds'] = round + 1
                    pbar.update(1)

        except Exception as e:
            logger.error(f"Training interrupted: {str(e)}")
            if self.config.strict_mode:
                raise
        
        return training_stats
