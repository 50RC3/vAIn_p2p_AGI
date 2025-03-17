from dataclasses import dataclass

@dataclass
class NetworkConfig:
    node_env: str
    port: int
    database_url: str

    @classmethod
    def from_env(cls):
        from dotenv import load_dotenv
        import os
        load_dotenv()
        
        return cls(
            node_env=os.getenv('NODE_ENV', 'development'),
            port=int(os.getenv('PORT', 3000)),
            database_url=os.getenv('DATABASE_URL', '')
        )
