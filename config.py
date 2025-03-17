import os
from dotenv import load_dotenv

class Config:
    def __init__(self):
        self.load_env()
        
        # Node identification
        self.node_id = os.getenv('NODE_ID', 'default_node_id')
        
        # Network configuration
        self.network = {
            'udp': {
                'port': int(os.getenv('UDP_PORT', 8468))
            },
            'dht': {
                'bootstrap_nodes': os.getenv('DHT_BOOTSTRAP_NODES', '').split(',')
            },
            'secret_key': os.getenv('NETWORK_SECRET', 'default_secret')
        }
        
        # Training parameters
        self.batch_size = int(os.getenv('BATCH_SIZE', 32))
        self.learning_rate = float(os.getenv('LEARNING_RATE', 0.001))
        self.num_epochs = 100
        
        # Model parameters
        self.hidden_size = 256
        self.num_layers = 4
        
        # Federated learning parameters
        self.num_rounds = 50
        self.min_clients = 3
        self.clients_per_round = 10
        
        # Chatbot parameters
        self.chatbot = {
            'max_context': int(os.getenv('CHATBOT_MAX_CONTEXT', 1024)),
            'response_temp': float(os.getenv('CHATBOT_RESPONSE_TEMP', 0.7)),
            'top_p': float(os.getenv('CHATBOT_TOP_P', 0.9)),
            'max_tokens': int(os.getenv('CHATBOT_MAX_TOKENS', 256))
        }
        
        # RL parameters
        self.rl = {
            'gamma': float(os.getenv('RL_GAMMA', 0.99)),
            'learning_rate': float(os.getenv('RL_LEARNING_RATE', 0.001)),
            'batch_size': int(os.getenv('RL_BATCH_SIZE', 32)),
            'update_interval': int(os.getenv('RL_UPDATE_INTERVAL', 100)),
            'memory_size': int(os.getenv('RL_MEMORY_SIZE', 10000))
        }
        
    def load_env(self):
        """Load environment variables from .env file"""
        load_dotenv()
        
    def validate_config(self):
        """Validate configuration parameters"""
        assert self.chatbot['max_context'] > 0, "Invalid max context length"
        assert 0 < self.chatbot['response_temp'] <= 1, "Invalid response temperature"
        assert 0 < self.chatbot['top_p'] <= 1, "Invalid top_p value"
        assert self.rl['gamma'] > 0 and self.rl['gamma'] <= 1, "Invalid gamma value"

# Chatbot Configuration
CHATBOT_CONFIG = {
    'max_context_length': 1024,
    'response_temp': 0.7,
    'top_p': 0.9,
    'max_tokens': 256
}

# Reinforcement Learning
RL_CONFIG = {
    'gamma': 0.99,
    'learning_rate': 0.001,
    'batch_size': 32,
    'update_interval': 100,
    'memory_size': 10000,
    'min_samples': 1000
}
