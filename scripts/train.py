import torch
from config.agent_config import AgentConfig
from models.multi_agent_system import MultiAgentSystem
from training.federated import FederatedLearning

def main():
    params = {
        'hidden_size': 20,
        'memory_size': 128,
        'memory_vector_dim': 40,
        'nhead': 2,
        'num_layers': 2
    }
    input_size = 10
    num_agents = 5
    rounds = 10
    local_epochs = 5
    meta_steps = 5

    # Create synthetic training data
    clients_data = [[(torch.randn(10, 5, input_size), 
                     torch.randn(10, 5, params['hidden_size'])) 
                    for _ in range(3)] for _ in range(num_agents)]

    # Initialize config
    config = AgentConfig.from_dict(params, num_agents, input_size)
    
    # Setup training components
    federated_learning = FederatedLearning(
        model=None,
        clients_data=clients_data,
        num_rounds=rounds,
        local_epochs=local_epochs,
        lr=0.01
    )
    
    multi_agent_system = MultiAgentSystem(config)

    # Start training
    multi_agent_system.train(federated_learning, clients_data, rounds, meta_steps)

if __name__ == '__main__':
    main()
