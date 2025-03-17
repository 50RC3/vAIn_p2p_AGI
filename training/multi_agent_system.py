from typing import List
from .agent import Agent
from .federated import FederatedLearning
from config.agent_config import AgentConfig

class MultiAgentSystem:
    def __init__(self, config: AgentConfig):
        self.agents = [Agent(config) for _ in range(config.num_agents)]
        
    def federated_update(self, federated_learning: FederatedLearning):
        local_models = [agent.model.state_dict() for agent in self.agents]
        aggregated_state = federated_learning.aggregate_models(local_models)
        
        # Update all agents with aggregated model
        for agent in self.agents:
            agent.model.load_state_dict(aggregated_state)
            
    def train(self, federated_learning: FederatedLearning, 
              clients_data: List, rounds: int, meta_steps: int):
        for round in range(rounds):
            round_losses = []
            for agent, client_data in zip(self.agents, clients_data):
                batch_losses = []
                for x, y in client_data:
                    loss = agent.local_update(x, y, meta_steps)
                    batch_losses.append(loss)
                round_losses.append(sum(batch_losses) / len(batch_losses))
                    
            self.federated_update(federated_learning)
            avg_loss = sum(round_losses) / len(round_losses)
            print(f'Federated Round {round+1} complete, Average Loss: {avg_loss:.4f}')
