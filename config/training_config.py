from dataclasses import dataclass
import os

@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float
    num_epochs: int
    hidden_size: int
    num_layers: int
    num_rounds: int
    min_clients: int
    clients_per_round: int
    
    # DNC parameters
    input_size: int = int(os.getenv('INPUT_SIZE', 256))
    memory_size: int = int(os.getenv('MEMORY_SIZE', 128))
    memory_vector_dim: int = int(os.getenv('MEMORY_VECTOR_DIM', 64))
    num_heads: int = int(os.getenv('NUM_HEADS', 4))

    @classmethod
    def from_env(cls):
        from dotenv import load_dotenv
        load_dotenv()
        
        return cls(
            batch_size=int(os.getenv('BATCH_SIZE', 32)),
            learning_rate=float(os.getenv('LEARNING_RATE', 0.001)),
            num_epochs=int(os.getenv('NUM_EPOCHS', 100)),
            hidden_size=int(os.getenv('HIDDEN_SIZE', 256)),
            num_layers=int(os.getenv('NUM_LAYERS', 4)),
            num_rounds=int(os.getenv('NUM_ROUNDS', 50)),
            min_clients=int(os.getenv('MIN_CLIENTS', 3)),
            clients_per_round=int(os.getenv('CLIENTS_PER_ROUND', 10)),
            input_size=int(os.getenv('INPUT_SIZE', 256)),
            memory_size=int(os.getenv('MEMORY_SIZE', 128)),
            memory_vector_dim=int(os.getenv('MEMORY_VECTOR_DIM', 64)),
            num_heads=int(os.getenv('NUM_HEADS', 4))
        )
