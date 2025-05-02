#!/usr/bin/env python3
"""
Main application entry point for vAIn_p2p_AGI project.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import logging
import argparse
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("vAIn_p2p_AGI")

# Import project modules
try:
    from config.agent_config import AgentConfig
    from models.multi_agent_system import MultiAgentSystem
    from core.interactive_utils import InteractiveSession, InteractiveConfig
    from core.constants import InteractionLevel
    from ui.interface_manager import UserInterfaceManager
    from ui.terminal_ui import TerminalUI
except ImportError as e:
    logger.error("Failed to import required modules: %s", e)
    logger.error("Make sure you're running this script from the project root directory.")
    sys.exit(1)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run vAIn_p2p_AGI application")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to custom configuration file")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Enable interactive mode")
    parser.add_argument("--verbose", "-v", action="count", default=0,
                        help="Increase verbosity level")
    return parser.parse_args()

def setup_environment(args):
    """Setup environment based on arguments."""
    # Set appropriate log level
    if args.verbose == 1:
        logger.setLevel(logging.INFO)
    elif args.verbose >= 2:
        logger.setLevel(logging.DEBUG)

    # Configure paths
    project_root = Path(__file__).parent.absolute()
    os.environ['PROJECT_ROOT'] = str(project_root)

    # Return config info
    return {
        "interactive": args.interactive,
        "debug": args.debug,
        "config_path": args.config
    }

def initialize_user_interface(config):
    """Initialize the appropriate user interface based on configuration."""
    logger.info("Initializing user interface...")

    ui_type = os.environ.get('UI_TYPE', 'terminal')

    # Create the UI manager
    ui_manager = UserInterfaceManager()

    # Initialize the appropriate UI based on configuration
    if ui_type.lower() == 'web':
        try:
            from ui.web_ui import WebUI
            # Ensure web UI can handle Windows paths if needed
            if os.name == 'nt':  # Check if running on Windows
                os.environ['WEBUI_WINDOWS_MODE'] = 'true'
            ui = WebUI(debug=config["debug"])
            logger.info("Web UI initialized on http://localhost:8080")
        except ImportError:
            logger.warning("Web UI dependencies not found. Falling back to terminal UI.")
            ui = TerminalUI(interactive=config["interactive"])
    elif ui_type.lower() == 'gui':
        try:
            from ui.gui import GraphicalUI
            ui = GraphicalUI(theme="dark" if os.environ.get('DARK_MODE') else "light")
            logger.info("Graphical UI initialized")
        except ImportError:
            logger.warning("GUI dependencies not found. Falling back to terminal UI.")
            ui = TerminalUI(interactive=config["interactive"])
    else:
        # Default to terminal UI
        ui = TerminalUI(interactive=config["interactive"])
        logger.info("Terminal UI initialized")

    # Register the UI with the manager
    ui_manager.register_interface(ui)

    return ui_manager

def main():
    """Main entry point for the application."""
    args = parse_args()
    config = setup_environment(args)
    
    # Initialize interactive session
    # Use appropriate InteractionLevel values based on what's available in the enum
    interaction_level = InteractionLevel.ADVANCED if config["interactive"] else InteractionLevel.NORMAL
    session = InteractiveSession(
        level=interaction_level,
        config=InteractiveConfig(
            timeout=300,  # 5 minute timeout
            persistent_state=True,
            safe_mode=not config["debug"]
        )
    )
    
    try:
        logger.info("Initializing vAIn_p2p_AGI application...")
        
        # Load configuration
        if config["config_path"]:
            agent_config = AgentConfig.load(config["config_path"])
        else:
            # Use default configuration
            params = {
                'hidden_size': 64,
                'memory_size': 128,
                'memory_vector_dim': 40,
                'nhead': 4,
                'num_layers': 2
            }
            agent_config = AgentConfig.from_dict(params, num_agents=5, input_size=10)
            
            if config["interactive"]:
                print("\nDefault configuration loaded.")
                if input("Update configuration interactively? (y/N): ").lower() == 'y':
                    agent_config.update_interactive()
        
        # Initialize the multi-agent system
        logger.info("Creating multi-agent system...")
        system = MultiAgentSystem(agent_config)
        
        # Initialize the user interface
        ui_manager = initialize_user_interface(config)
        
        # Connect the system to the UI
        ui_manager.connect_system(system)
        
        # Start the system
        logger.info("Starting application...")
        system.initialize()  # Assuming 'initialize' is the correct method
        
        # Start the UI
        ui_manager.start()
        
        # Main application loop now handled by UI manager
        ui_manager.run_event_loop()
            
    except (ValueError, IOError, ImportError) as e:
        # Catch common expected exceptions
        logger.error("Application error: %s", str(e))
        if config["debug"]:
            import traceback
            traceback.print_exc()
        return 1
    except KeyboardInterrupt:
        logger.info("Application terminated by user.")
        return 0
    finally:
        # Clean up resources
        logger.info("Cleaning up resources...")
        if 'ui_manager' in locals():
            ui_manager.shutdown()
        session._cleanup()
        
    logger.info("Application terminated successfully.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
