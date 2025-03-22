import torch
import logging
from tqdm import tqdm
from core.constants import InteractionLevel
from core.interactive_utils import InteractiveSession, InteractiveConfig
from config.agent_config import AgentConfig
from models.multi_agent_system import MultiAgentSystem
from training.federated import FederatedLearning

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize interactive session
    session = InteractiveSession(
        level=InteractionLevel.NORMAL,
        config=InteractiveConfig(
            timeout=300,  # 5 minute timeout
            persistent_state=True,
            safe_mode=True
        )
    )
    
    try:
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
        
        # Interactive configuration
        if input("\nReview configuration before starting? (y/N): ").lower() == 'y':
            config.update_interactive()

        print("\nInitializing Training Session")
        print("=" * 50)
        
        # Setup training components
        federated_learning = FederatedLearning(
            model=None,
            clients_data=clients_data,
            num_rounds=rounds,
            local_epochs=local_epochs,
            lr=0.01
        )
        
        multi_agent_system = MultiAgentSystem(config)

        # Setup training with progress tracking
        with tqdm(total=rounds, desc="Training Rounds") as pbar:
            try:
                multi_agent_system.train(
                    federated_learning, 
                    clients_data, 
                    rounds, 
                    meta_steps,
                    progress_callback=lambda x: pbar.update(1)
                )
            except KeyboardInterrupt:
                logger.info("Training interrupted by user")
                if input("\nSave current progress? (Y/n): ").lower() != 'n':
                    multi_agent_system.save_checkpoint()
            except Exception as e:
                logger.error(f"Training error: {str(e)}")
                raise

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise
    finally:
        session.cleanup()

if __name__ == '__main__':
    main()
