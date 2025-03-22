from dataclasses import dataclass
from typing import Optional
import os
import json
from pathlib import Path
import socket
import ssl
import logging
try:
    from dotenv import load_dotenv
except ImportError:
    print("python-dotenv not installed. Run: pip install python-dotenv")
    raise

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class NetworkConfig:
    node_env: str
    port: int
    database_url: str
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None
    max_connections: int = 100
    timeout: int = 30

    @classmethod
    def from_env(cls, database_url: str):
        return cls(
            node_env=os.getenv('NODE_ENV', 'production'),
            port=int(os.getenv('PORT', 443)),
            database_url=database_url,
            ssl_cert_path=os.getenv('SSL_CERT_PATH'),
            ssl_key_path=os.getenv('SSL_KEY_PATH'),
            max_connections=int(os.getenv('MAX_CONNECTIONS', 100)),
            timeout=int(os.getenv('TIMEOUT', 30))
        )

    def validate(self) -> None:
        """Validate configuration"""
        if self.node_env not in ['development', 'staging', 'production']:
            raise ValueError(f"Invalid environment: {self.node_env}")
            
        if not (1024 <= self.port <= 65535):
            raise ValueError(f"Invalid port: {self.port}")
                raise ValueError(f"Invalid SSL certificate: {str(e)}") from e
        if self.ssl_cert_path:
            cert_path = Path(self.ssl_cert_path)
            if not cert_path.exists():
                raise ValueError(f"SSL certificate not found: {self.ssl_cert_path}")
            try:
                ssl.SSLContext().load_cert_chain(cert_path)
            except ssl.SSLError as e:
                raise ValueError(f"Invalid SSL certificate: {str(e)}")

    @property
    def is_ssl_enabled(self) -> bool:
        return bool(self.ssl_cert_path and self.ssl_key_path)

    def update_interactive(self) -> None:
        """Interactive configuration update with validation"""
        try:
            print("\nCurrent Network Configuration:")
            self._display_current_config()
            
            if input("\nUpdate configuration? (y/n): ").lower() != 'y':
                return

            valid_envs = ['development', 'staging', 'production'] 
            while True:
                env = input(f"Enter environment {valid_envs} [{self.node_env}]: ") or self.node_env
                if env in valid_envs:
                    self.node_env = env
                    break
                print(f"Invalid environment. Must be one of: {valid_envs}")

            while True:
                try:
                    port = input(f"Enter port (1024-65535) [{self.port}]: ")
                    port = int(port) if port else self.port
                    
                    if not (1024 <= port <= 65535):
                        print("Port must be between 1024 and 65535")
                        continue

                    # Test if port is available
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    if sock.connect_ex(('localhost', port)) == 0:
                        print(f"Warning: Port {port} is already in use")
                        if input("Continue anyway? (y/n): ").lower() != 'y':
                            continue
                    sock.close()
                    self.port = port
                    break
                except ValueError:
                    print("Port must be a number")

            # SSL Configuration
            if input("Configure SSL? (y/n): ").lower() == 'y':
                while True:
                    cert_path = input("Enter SSL certificate path: ").strip()
                    if not cert_path:
                        break
                        
                    cert_path = Path(cert_path)
                    if not cert_path.exists():
                        print("Certificate file not found")
                        continue
                        
                    try:
                        ssl.SSLContext().load_cert_chain(cert_path)
                        self.ssl_cert_path = str(cert_path)
                        break
                    except ssl.SSLError:
                        print("Invalid SSL certificate")

            self.validate()
            self._save_config()
            logger.info("Network configuration updated successfully")
            
        except KeyboardInterrupt:
            print("\nConfiguration update cancelled")
            return
        except Exception as e:
            logger.error(f"Configuration update failed: {str(e)}")
            raise

    def _display_current_config(self) -> None:
        """Display current configuration settings"""
        print(f"Environment: {self.node_env}")
        print(f"Port: {self.port}")
        print(f"Max Connections: {self.max_connections}")
        print(f"SSL Enabled: {self.is_ssl_enabled}")
        print(f"Timeout: {self.timeout}s")

    def _save_config(self) -> None:
        """Save configuration to file"""
        config_path = Path('config/network.json')
        config_path.parent.mkdir(exist_ok=True)
        
        with config_path.open('w', encoding='utf-8') as f:
            json.dump({
                'node_env': self.node_env,
                'port': self.port,
                'max_connections': self.max_connections,
                'ssl_cert_path': self.ssl_cert_path,
                'ssl_key_path': self.ssl_key_path,
                'timeout': self.timeout
            }, f, indent=2)
