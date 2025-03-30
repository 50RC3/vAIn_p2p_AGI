import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Any, Union
from web3 import Web3
from dotenv import load_dotenv
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

HAS_WEB3 = True

@dataclass 
class BlockchainConfig:
    private_key: str
    infura_project_id: str 
    network: str
    gas_limit: int = 6000000
    gas_price_gwei: int = 50
    max_gas_price_gwei: int = 150
    retry_attempts: int = 3
    retry_delay: int = 5
    _web3: Optional[Web3] = None
    development_mode: bool = False
    
    network_configs: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'mainnet': {
            'chain_id': 1,
            'min_gas_price': 15,
            'block_time': 15,
            'confirmations': 5
        },
        'polygon': {
            'chain_id': 137, 
            'min_gas_price': 30,
            'block_time': 2,
            'confirmations': 15
        },
        'development': {
            'chain_id': 1337,
            'min_gas_price': 1,
            'block_time': 2,
            'confirmations': 1
        }
    })

    def __post_init__(self) -> None:
        self.validate_config()

    def validate_config(self) -> None:
        """Validate blockchain configuration"""
        # In development mode, we allow empty keys
        if self.network == 'development':
            self.development_mode = True
            # For development, provide a warning but continue
            if not self.private_key or len(self.private_key) != 64:
                logger.warning("Using development mode with invalid/missing private key")
                return
        # For production networks, enforce strict validation
        else:
            if not self.private_key or len(self.private_key) != 64:
                raise ValueError("Invalid private key format")
            
            if not self.infura_project_id:
                raise ValueError("Infura project ID required for non-development networks")
            
        # Common validations for all modes
        if self.network not in self.network_configs:
            raise ValueError(f"Unsupported network: {self.network}")
        
        if self.gas_price_gwei <= 0:
            raise ValueError("Gas price must be positive")

        if self.gas_limit < 21000:
            raise ValueError("Gas limit must be at least 21000")

    @property
    def web3(self) -> Web3:
        """Get or create Web3 instance with automatic retry"""
        if not self._web3:
            self._web3 = self._create_web3()
        return self._web3

    def _create_web3(self) -> Web3:
        for attempt in range(self.retry_attempts):
            try:
                if self.network == 'development':
                    w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:8545'))
                else:
                    w3 = Web3(Web3.HTTPProvider(
                        f"https://{self.network}.infura.io/v3/{self.infura_project_id}"
                    ))
                if w3.is_connected():
                    return w3
            except (ConnectionError, TimeoutError, ValueError) as e:
                logger.warning("Connection attempt %s failed: %s", attempt + 1, e)
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise ConnectionError("Failed to establish Web3 connection") from e
        raise ConnectionError("Failed to establish Web3 connection after all retry attempts")

    def estimate_gas_price(self) -> int:
        """Estimate optimal gas price with safety bounds"""
        try:
            network_config = self.network_configs[self.network]
            base_fee = self.web3.eth.gas_price
            suggested_gwei = Web3.from_wei(base_fee, 'gwei')
            
            # Apply network-specific minimum
            min_gwei = network_config['min_gas_price']
            max_gwei = self.max_gas_price_gwei
            
            # Ensure gas price is within bounds
            final_gwei = max(min_gwei, min(int(float(suggested_gwei) * 1.1), max_gwei))
            logger.info(f"Estimated gas price: {final_gwei} gwei")
            return int(final_gwei)
            
        except Exception as e:
            logger.error(f"Gas price estimation failed: {e}")
            return self.gas_price_gwei

    def update_interactive(self) -> None:
        """Interactive configuration update"""
        try:
            print("\nCurrent Configuration:")
            print(f"Network: {self.network}")
            print(f"Gas Price (gwei): {self.gas_price_gwei}")
            print(f"Gas Limit: {self.gas_limit}")
            
            if input("\nUpdate configuration? (y/n): ").lower() != 'y':
                return

            # Load contract addresses
            addresses_path = Path(__file__).parent.parent / 'contracts' / 'addresses.json'
            with open(addresses_path) as f:
                addresses = json.load(f)

            networks = list(addresses.keys())
            print("\nAvailable networks:", ", ".join(networks))
            
            while True:
                new_network = input(f"Enter network [{self.network}]: ").lower() or self.network
                if new_network in networks:
                    self.network = new_network
                    break
                print(f"Invalid network. Must be one of: {networks}")

            while True:
                try:
                    gas_price = input(f"Enter gas price in gwei [{self.gas_price_gwei}]: ")
                    self.gas_price_gwei = int(gas_price) if gas_price else self.gas_price_gwei
                    if self.gas_price_gwei <= 0:
                        print("Gas price must be positive")
                        continue
                    break
                except ValueError:
                    print("Gas price must be a number")

            while True:
                try:
                    gas_limit = input(f"Enter gas limit [{self.gas_limit}]: ")
                    self.gas_limit = int(gas_limit) if gas_limit else self.gas_limit
                    if self.gas_limit < 21000:
                        print("Gas limit must be at least 21000")
                        continue
                    break
                except ValueError:
                    print("Gas limit must be a number")

            self.validate_config()
            logger.info("Blockchain configuration updated successfully")

        except KeyboardInterrupt:
            print("\nConfiguration update cancelled")
            return
        except Exception as e:
            logger.error(f"Configuration update failed: {str(e)}")
            raise

    @classmethod
    def from_env(cls) -> 'BlockchainConfig':
        load_dotenv()
        
        network = os.getenv('NETWORK', 'development')
        
        config = cls(
            private_key=os.getenv('PRIVATE_KEY', ''),
            infura_project_id=os.getenv('INFURA_PROJECT_ID', ''),
            network=network
        )
        
        try:
            if network != 'development' and config.web3.is_connected():
                logger.info("Successfully connected to %s", config.network)
        except Exception as e:
            if network != 'development':
                logger.warning("Connection test failed: %s", str(e))
            else:
                logger.info("Running in development mode without blockchain connection")
            
        return config
