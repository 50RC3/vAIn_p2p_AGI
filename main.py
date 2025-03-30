import os
import sys
import logging
import argparse
import base64
import asyncio
import random
from pathlib import Path
from datetime import datetime

# Configure logging with custom config if available
try:
    from logging_config import configure_logging
    logger = configure_logging(level=logging.INFO)
except ImportError:
    # Fallback to basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger(__name__)

# Check dependencies before importing other modules
try:
    from utils.dependency_checker import check_dependencies, install_dependencies

    # Check only required dependencies for faster startup
    success, missing = check_dependencies(required_only=True)
    if not success:
        logger.warning(f"Missing core dependencies: {', '.join(missing)}")
        logger.info("Attempting to install missing dependencies...")
        
        if not install_dependencies(missing):
            logger.error("Failed to install required dependencies. Exiting.")
            sys.exit(1)
except ImportError:
    logger.warning("Dependency checker not found, skipping dependency verification")

# Try loading configuration
try:
    from config import get_config
    config = get_config(interactive=True)
except ImportError as e:
    logger.warning(f"Could not import config module: {e}")
    # Fallback to standalone Config class if needed
    try:
        # If we can't import from the module, load a minimal config
        logger.info("Using minimal configuration instead")
        config = None
    except Exception as e:
        logger.error(f"Failed to create config: {e}")
        config = None

# Enhanced import error handling for key components
def import_core_components():
    """Import core components with proper error handling and fallbacks"""
    components = {}
    
    # Try to import the UI components
    try:
        from ai_core.ui import run_cli
        components['run_cli'] = run_cli
    except ImportError as e:
        logger.error(f"Could not import UI components: {e}")
        components['run_cli'] = None
    
    # Try to import the model storage
    try:
        from ai_core.model_storage import ModelStorage
        components['ModelStorage'] = ModelStorage
    except ImportError as e:
        logger.error(f"Could not import model storage: {e}")
        components['ModelStorage'] = None
    
    # Import SimpleNN with fallback, avoiding the import through __init__.py
    try:
        # Direct import to bypass potential circular imports
        from models.simple_nn import SimpleNN  # Import directly from module
        components['SimpleNN'] = SimpleNN
        logger.info("Successfully imported SimpleNN directly from models.simple_nn")
    except ImportError as e:
        logger.error(f"Could not import model classes: {e}")
        # Define a minimal fallback SimpleNN class if import fails
        import torch.nn as nn
        class FallbackSimpleNN(nn.Module):
            def __init__(self, input_size=512, output_size=512, hidden_size=256):
                super().__init__()
                self.linear = nn.Linear(input_size, output_size)
            
            def forward(self, x):
                return self.linear(x)
        
        components['SimpleNN'] = FallbackSimpleNN
        logger.warning("Using fallback SimpleNN implementation")
    except Exception as e:
        logger.error(f"Unexpected error importing SimpleNN: {e}")
        components['SimpleNN'] = None
        
    return components

# Import components with fallbacks
core_components = import_core_components()

async def setup_interface():
    """Set up and initialize the user interface."""
    logger.info("Initializing user interface...")
    
    try:
        # Initialize spaCy if available
        try:
            from ai_core.nlp.utils import load_spacy_model
            logger.info("Loading spaCy model...")
            await load_spacy_model()
            logger.info("spaCy model loaded successfully")
        except ImportError:
            logger.warning("spaCy not available. NLP features will be limited.")
        except Exception as e:
            logger.warning(f"Error loading spaCy: {e}")
            
        # Create a simple model for demonstration purposes
        # In a real system, you would load your trained model here
        SimpleNN = core_components['SimpleNN']
        if SimpleNN is None:
            raise ImportError("SimpleNN model class is not available")
                
        model = SimpleNN(input_size=512, output_size=512, hidden_size=256)
        
        # Check if model was instantiated correctly
        if model is None:
            raise RuntimeError("Failed to instantiate SimpleNN model")
            
        # Create storage instance with defensive fallback
        try:
            from ai_core.model_storage import ModelStorage
            storage = ModelStorage(storage_dir="./model_storage")
            
            # Verify storage has required methods
            if not hasattr(storage, 'get_model_version'):
                logger.warning("ModelStorage missing get_model_version method, using mock implementation")
                
                # Create simple async method implementation
                async def get_model_version():
                    return "0.1.0"
                
                # Add method to storage instance
                storage.get_model_version = get_model_version
                
            if not hasattr(storage, 'store_feedback'):
                logger.warning("ModelStorage missing store_feedback method, using mock implementation")
                async def store_feedback(feedback):
                    return None
                storage.store_feedback = store_feedback
                
            if not hasattr(storage, 'persist_feedback'):
                logger.warning("ModelStorage missing persist_feedback method, using mock implementation")
                async def persist_feedback(session_id, feedback):
                    return None
                storage.persist_feedback = persist_feedback
                
        except ImportError:
            logger.warning("ModelStorage not available, using mock implementation")
            
            # Create a minimal mock storage
            class MockStorage:
                async def get_model_version(self):
                    return "0.1.0"
                
                async def store_feedback(self, feedback):
                    return None
                    
                async def persist_feedback(self, session_id, feedback):
                    return None
                                
            storage = MockStorage()
        
        # Create LearningConfig using setup_learning_config
        learning_config = setup_learning_config()
        
        # Initialize the chatbot interface with defensive error handling
        from ai_core.chatbot.interface import ChatbotInterface
        interface = ChatbotInterface(
            model=model,
            storage=storage,
            learning_config=learning_config
        )
        
        # Verify that critical components are available - use more direct check with proper error handling
        try:
            if not hasattr(interface, 'process_message'):
                logger.error("ChatbotInterface missing process_message method")
                raise AttributeError("Interface missing required methods")
                
            # Additionally check that it's callable
            if not callable(getattr(interface, 'process_message', None)):
                logger.error("process_message is not a callable method")
                raise AttributeError("process_message is not properly implemented")
                
        except Exception as e:
            logger.error(f"Interface validation failed: {e}")
            raise
        
        # Start a new session
        session_id = await interface.start_session()
        print(f"Started new chat session: {session_id}")
        print("Type 'exit', 'quit', or Ctrl+C to end the session.")
        print("Type 'clear' to clear the current session.")
        print("--------------------------------------------------")
        
        # For demonstration purposes, clear the session to show the interface is working
        await interface.clear_session()
        
        # Return the interface for interactive usage
        return interface
        
    except Exception as e:
        logger.error(f"Error setting up interface: {e}")
        return None  # Return None instead of raising to allow for graceful fallback

def setup_learning_config():
    """Set up and return the learning configuration."""
    from ai_core.chatbot.interface import LearningConfig
    return LearningConfig(
        # Self-supervised config
        enable_self_supervised=True,
        self_supervised_model="distilbert-base-uncased",
        mask_probability=0.15,
        
        # Unsupervised config
        enable_unsupervised=True,
        unsupervised_clusters=10,
        
        # RL config - fixed constructor
        enable_reinforcement=True,
        rl_learning_rate=1e-4,
        discount_factor=0.95,
        rl_update_interval=5,
        
        # Added options for stability
        adaptive_learning=True,
        cross_learning=True,
        memory_size=10000,
        embedding_cache_size=1000
    )

async def setup_monitoring_dashboard():
    """Set up and initialize the network monitoring dashboard."""
    try:
        # First try importing from network.monitoring
        try:
            from network.monitoring import ResourceMonitor, NetworkMonitor, get_resource_metrics
            from network.connection_diagnostics import ConnectionDiagnostics
        except ImportError as e:
            # Fallback to creating mock versions
            logger.warning(f"Could not import monitoring components: {e}")
            
            # Add missing import
            from dataclasses import dataclass, field
            from typing import Dict, List, Any
            
            @dataclass
            class ResourceMetrics:
                cpu_usage: float = 0.0
                memory_usage: float = 0.0
                disk_usage: float = 0.0
                
            @dataclass
            class NetworkHealthMetrics:
                peer_count: int = 0
                connection_success_rate: float = 0.0
                avg_latency: float = 0.0
                bandwidth_usage: float = 0.0
                overall_health: float = 0.0
                
            @dataclass
            class DiagnosticResult:
                dns_resolution: Dict[str, Any] = field(default_factory=dict)
                recommendations: List[str] = field(default_factory=list)
            
            class MockNetworkMonitor:
                def __init__(self, metrics_dir="./metrics"):
                    self.metrics_dir = metrics_dir
                    Path(metrics_dir).mkdir(exist_ok=True, parents=True)
                    
                async def start_monitoring(self):
                    logger.info("Started mock network monitoring")
                    
                async def check_network_health(self) -> NetworkHealthMetrics:
                    return NetworkHealthMetrics()
                    
                def request_shutdown(self):
                    logger.info("Mock network monitor shutdown requested")
                    
            class MockResourceMonitor:
                def __init__(self, check_interval=30, interactive=True):
                    self.check_interval = check_interval
                    self.interactive = interactive
                    
                async def check_resources_interactive(self):
                    return ResourceMetrics()
                    
            class MockConnectionDiagnostics:
                async def diagnose_connection(self, host, port):
                    return DiagnosticResult(
                        dns_resolution={"success": True},
                        recommendations=["No issues detected"]
                    )
            
            def mock_get_resource_metrics() -> ResourceMetrics:
                import psutil
                return ResourceMetrics(
                    cpu_usage=psutil.cpu_percent(),
                    memory_usage=psutil.virtual_memory().percent,
                    disk_usage=psutil.disk_usage('/').percent
                )
            
            # Create mocks
            ResourceMonitor = MockResourceMonitor
            NetworkMonitor = MockNetworkMonitor
            ConnectionDiagnostics = MockConnectionDiagnostics
            get_resource_metrics = mock_get_resource_metrics

        # Initialize monitoring components
        diagnostics = ConnectionDiagnostics()
        resource_monitor = ResourceMonitor(check_interval=30, interactive=True)
        network_monitor = NetworkMonitor(metrics_dir="./metrics")
        
        # Start background monitoring
        await network_monitor.start_monitoring()
        
        return {
            'diagnostics': diagnostics,
            'resource_monitor': resource_monitor,
            'network_monitor': network_monitor,
            'get_resource_metrics': get_resource_metrics
        }
    except Exception as e:
        logger.error(f"Error setting up monitoring dashboard: {e}")
        return None

async def show_quick_status(monitoring):
    """Show a quick summary of network status."""
    if not monitoring:
        print("Monitoring components not available.")
        return
        
    try:
        # Import psutil directly if needed
        import psutil
        
        # Get resource metrics with fallback
        if 'get_resource_metrics' in monitoring:
            metrics = monitoring['get_resource_metrics']()
        else:
            # Fallback to direct psutil call
            class SimpleMetrics:
                def __init__(self):
                    self.cpu_usage = psutil.cpu_percent()
                    self.memory_usage = psutil.virtual_memory().percent
                    self.disk_usage = psutil.disk_usage('/').percent
            metrics = SimpleMetrics()
        
        # Check if we're connected to the network
        connected = False
        peer_count = 0
        if 'network_monitor' in monitoring:
            try:
                health = await monitoring['network_monitor'].check_network_health()
                connected = health.peer_count > 0
                peer_count = health.peer_count
            except Exception as e:
                logger.error(f"Error checking network health: {e}")
        
        # Print status
        print("\n==== SYSTEM STATUS ====")
        print(f"Network Connected: {'✓' if connected else '✗'}")
        print(f"Peers: {peer_count}")
        print(f"CPU Usage: {metrics.cpu_usage:.1f}%")
        print(f"Memory Usage: {metrics.memory_usage:.1f}%")
        print(f"Disk Usage: {metrics.disk_usage:.1f}%")
        print("=====================")
        
    except Exception as e:
        print(f"Error getting status: {e}")

async def show_monitoring_dashboard(monitoring):
    """Show the full network monitoring dashboard."""
    if not monitoring:
        print("Monitoring components not available.")
        return
        
    try:
        # Clear screen for better visibility
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("\n====== NETWORK MONITORING DASHBOARD ======")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("==========================================")
        
        # Get resource metrics
        metrics = monitoring['get_resource_metrics']()
        print("\n== SYSTEM RESOURCES ==")
        print(f"CPU: {metrics.cpu_usage:.1f}% | Memory: {metrics.memory_usage:.1f}% | Disk: {metrics.disk_usage:.1f}%")
        
        # Get network health if available
        if 'network_monitor' in monitoring:
            health = await monitoring['network_monitor'].check_network_health()
            
            # Connection status
            conn_status = "CONNECTED" if health.peer_count > 0 else "DISCONNECTED"
            status_color = "\033[92m" if health.peer_count > 0 else "\033[91m"  # Green or Red
            reset_color = "\033[0m"
            
            print(f"\n== NETWORK STATUS: {status_color}{conn_status}{reset_color} ==")
            print(f"Peers: {health.peer_count}")
            print(f"Connection Success Rate: {health.connection_success_rate:.1%}")
            print(f"Average Latency: {health.avg_latency:.2f} ms")
            print(f"Bandwidth Usage: {health.bandwidth_usage:.2f} MB/s")
            print(f"Overall Health: {health.overall_health:.1%}")
            
            # If active peers exist, show connection quality
            if health.peer_count > 0:
                print("\n== CONNECTION QUALITY ==")
                quality = "Excellent" if health.avg_latency < 100 else \
                          "Good" if health.avg_latency < 200 else \
                          "Fair" if health.avg_latency < 500 else "Poor"
                print(f"Quality: {quality}")
                
        # Connection diagnostics if available
        if 'diagnostics' in monitoring:
            print("\n== DIAGNOSTICS ==")
            result = await monitoring['diagnostics'].diagnose_connection("google.com", 80)
            print(f"Internet Connectivity: {'✓' if result.dns_resolution.get('success', False) else '✗'}")
            if not result.dns_resolution.get('success', False):
                print(f"  - DNS Issue: {result.dns_resolution.get('error', 'Unknown')}")
                
            # Show recommendations if any
            if result.recommendations:
                print("\n== RECOMMENDATIONS ==")
                for i, rec in enumerate(result.recommendations, 1):
                    print(f"{i}. {rec}")
        
        # Resource monitor details if available
        if 'resource_monitor' in monitoring:
            resource_health = await monitoring['resource_monitor'].check_resources_interactive()
            if resource_health:
                print("\n== RESOURCE HEALTH ==")
                for key, value in resource_health.__dict__.items():
                    if isinstance(value, dict) or key == 'load_metrics':
                        continue
                    print(f"{key}: {value}")
                    
        print("\n==========================================")
        print("Press Enter to return to main interface...")
        input()
        
        # Clear screen when returning
        os.system('cls' if os.name == 'nt' else 'clear')
        
    except Exception as e:
        print(f"Error displaying monitoring dashboard: {e}")
        print("\nPress Enter to continue...")
        input()

async def main():
    """Main entry point for the application."""
    interface = None
    monitoring = None
    try:
        # Setup and get the interface
        interface = await setup_interface()
        
        # Setup network monitoring
        monitoring = await setup_monitoring_dashboard()
        
        if interface is None:
            logger.error("Failed to set up interface. Exiting.")
            return
        
        # Parse arguments
        parser = argparse.ArgumentParser(description="vAIn P2P AGI System")
        parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
        parser.add_argument("--monitor", action="store_true", help="Show network monitoring dashboard")
        args = parser.parse_args()
        
        # If monitor flag is set, show the monitoring dashboard first
        if args.monitor and monitoring:
            await show_monitoring_dashboard(monitoring)
        
        while True:
            try:
                # Show command prompt with options
                print("\nOptions: [m]onitor | [c]hat | [s]tatus | [q]uit")
                user_input = input("> ")
                
                if user_input.lower() in ["exit", "quit", "q"]:
                    break
                    
                if user_input.lower() in ["monitor", "m"] and monitoring:
                    await show_monitoring_dashboard(monitoring)
                    continue
                    
                if user_input.lower() in ["status", "s"] and monitoring:
                    await show_quick_status(monitoring)
                    continue
                    
                if user_input.lower() == "clear":
                    await interface.clear_session()
                    session_id = await interface.start_session()
                    print(f"Started new session: {session_id}")
                    continue
                
                # Skip empty inputs instead of processing them
                if not user_input.strip():
                    continue
                
                # Process message and get response
                response = await interface.process_message(user_input)
                
                # Print response
                print(f"Bot: {response.text}")
                
                # Ask for rating if appropriate
                if random.random() < 0.2:  # 20% chance to ask for rating
                    rating = input("Rate this response (0-1, or skip with Enter): ")
                    if rating.strip() and rating.replace('.', '', 1).isdigit():
                        rating_float = float(rating)
                        if 0 <= rating_float <= 1:
                            await interface.store_feedback(response.id, rating_float)
                            print(f"Feedback recorded: {rating_float}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)
    finally:
        # Properly clean up interface resources
        if interface and hasattr(interface, 'cleanup'):
            await interface.cleanup()
            
        # Clean up monitoring resources
        if monitoring:
            if 'network_monitor' in monitoring:
                monitoring['network_monitor'].request_shutdown()

if __name__ == "__main__":
    # Add signal handlers for more graceful shutdown
    if os.name != 'nt':  # Not on Windows
        import signal
        signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Exiting.")
        sys.exit(0)

