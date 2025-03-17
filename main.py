import os
from config import Config
from models.simple_nn import SimpleNN
from training.federated_training import FederatedTraining
from data.data_loader import DataLoader
from network.p2p_network import P2PNetwork
import logging
from utils.logger_init import init_logger
import signal
import sys

def setup_signal_handlers(network: P2PNetwork):
    def signal_handler(sig, frame):
        logging.info("Shutting down vAIn node...")
        network.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def main():
    # Initialize logger first
    logger = init_logger(
        name="vAIn",
        config_path="config/logging.json",
        log_dir="logs"
    )
    
    try:
        # Initialize config with environment variables
        config = Config()
        
        # Initialize P2P network with new config structure
        network = P2PNetwork(config.node_id, config.network)
        setup_signal_handlers(network)
        
        logger.info(f"Starting vAIn node {config.node_id}...")
        network.start()
    except Exception as e:
        logger.error(f"Error running vAIn node: {e}")
        if 'network' in locals():
            network.stop()
        sys.exit(1)

if __name__ == "__main__":
    main()
