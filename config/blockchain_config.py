
from dataclasses import dataclass

@dataclass
class BlockchainConfig:
    private_key: str
    infura_project_id: str
    network: str
    gas_limit: int = 6000000
    gas_price_gwei: int = 50

    @classmethod
    def from_env(cls):
        from dotenv import load_dotenv
        import os
        load_dotenv()
        
        return cls(
            private_key=os.getenv('PRIVATE_KEY', ''),
            infura_project_id=os.getenv('INFURA_PROJECT_ID', ''),
            network=os.getenv('NETWORK', 'development')
        )
