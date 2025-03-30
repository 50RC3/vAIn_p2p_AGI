import os
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass
import socket
import ssl
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Optional dotenv support
try:
    from dotenv import load_dotenv
    load_dotenv()
    HAVE_DOTENV = True
except ImportError:
    logger.warning("python-dotenv not installed. Using environment variables directly.")
    HAVE_DOTENV = False

@dataclass
class NetworkConfig:
    node_env: str
    port: int
    database_url: str
    cert_path: Optional[Path] = None
    key_path: Optional[Path] = None
    node_id: Optional[str] = None
    peer_timeout: int = 30
    max_connections: int = 10

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def from_env(cls) -> "NetworkConfig":
        """Create configuration from environment with fallbacks"""
        # Get env vars with fallbacks
        try:
            cert_path_str = os.environ.get('SSL_CERT_PATH')
            key_path_str = os.environ.get('SSL_KEY_PATH')
            database_url = os.environ.get('DATABASE_URL', 'sqlite:///db.sqlite3')
            
            return cls(
                node_env=os.environ.get('NODE_ENV', 'development'),
                port=int(os.environ.get('PORT', '3000')), 
                database_url=database_url,
                cert_path=Path(cert_path_str) if cert_path_str else None,
                key_path=Path(key_path_str) if key_path_str else None,
                max_connections=int(os.environ.get('MAX_CONNECTIONS', '100')),
                peer_timeout=int(os.environ.get('TIMEOUT', '30'))
            )
        except ValueError as e:
            raise ValueError(f"Invalid environment variable: {str(e)}")

    def validate(self) -> None:
        """Validate network configuration"""
        valid_envs = ['development', 'testing', 'production']
        if self.node_env not in valid_envs:
            raise ValueError(f"Invalid node_env: {self.node_env}")

        if not isinstance(self.port, int) or self.port < 1024 or self.port > 65535:
            raise ValueError(f"Invalid port number: {self.port}")

        if not self.database_url:
            raise ValueError("Database URL is required")

        # Validate SSL paths if provided
        if self.cert_path:
            # Convert to Path if it's not already
            self.cert_path = Path(self.cert_path) if not isinstance(self.cert_path, Path) else self.cert_path
            if not self.cert_path.exists():
                raise ValueError(f"Certificate file not found: {self.cert_path}")

        if self.key_path:
            # Convert to Path if it's not already
            self.key_path = Path(self.key_path) if not isinstance(self.key_path, Path) else self.key_path
            if not self.key_path.exists():
                raise ValueError(f"Key file not found: {self.key_path}")

    @property
    def is_ssl_enabled(self) -> bool:
        return bool(self.cert_path and self.key_path)

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
                    port_input = input(f"Enter port (1024-65535) [{self.port}]: ")
                    port = int(port_input) if port_input else self.port
                    
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
                    cert_path_str = input("Enter SSL certificate path: ").strip()
                    if not cert_path_str:
                        break
                        
                    cert_path = Path(cert_path_str)
                    if not cert_path.exists():
                        print("Certificate file not found")
                        continue
                        
                    try:
                        ssl.SSLContext().load_cert_chain(cert_path)
                        self.cert_path = cert_path
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
    def _save_config(self) -> None:
        """Save configuration to file"""
        import json
        config_path = Path('config/network.json')
        config_path.parent.mkdir(exist_ok=True)
        
        with config_path.open('w', encoding='utf-8') as f:
            json.dump({
                'node_env': self.node_env,
                'port': self.port,
                'max_connections': self.max_connections,
                'cert_path': str(self.cert_path) if self.cert_path else None,
                'key_path': str(self.key_path) if self.key_path else None,
                'timeout': self.peer_timeout
            }, f, indent=2)
