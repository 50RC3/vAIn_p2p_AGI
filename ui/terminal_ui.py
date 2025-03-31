"""
Terminal-based user interface.
"""
import sys
import logging
import threading
import time
from typing import Dict, Any, Optional

from ui.base_ui import BaseUI

logger = logging.getLogger(__name__)

class TerminalUI(BaseUI):
    """
    Terminal-based user interface for interacting with the system.
    Provides a simple command-line interface for basic interactions.
    """
    
    def __init__(self, interactive=False):
        super().__init__(name="Terminal")
        self.interactive = interactive
        self.running = False
        self.commands = {
            "help": self._cmd_help,
            "status": self._cmd_status,
            "exit": self._cmd_exit,
            "version": self._cmd_version,
        }
    
    def start(self):
        """Initialize and start the terminal UI."""
        self.running = True
        
        # Print welcome message
        print("\n" + "=" * 50)
        print("vAIn_p2p_AGI Terminal Interface")
        print("=" * 50)
        print("Type 'help' for available commands or 'exit' to quit.\n")
    
    def shutdown(self):
        """Clean up and shut down the terminal UI."""
        self.running = False
        print("\nTerminal interface shutdown.")
    
    def run_event_loop(self):
        """Run the terminal UI event loop."""
        if not self.interactive:
            # In non-interactive mode, just keep alive and wait for signals
            try:
                while self.running:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                self.running = False
                return
        
        # Interactive command loop
        while self.running:
            try:
                cmd = input("> ").strip()
                if not cmd:
                    continue
                    
                parts = cmd.split()
                cmd_name = parts[0].lower()
                args = parts[1:]
                
                if cmd_name in self.commands:
                    self.commands[cmd_name](*args)
                else:
                    print(f"Unknown command: {cmd_name}. Type 'help' for available commands.")
            
            except KeyboardInterrupt:
                print("\nUse 'exit' to quit the application.")
            except Exception as e:
                logger.error(f"Error processing command: {e}")
    
    def display_message(self, message, level="info"):
        """Display a message in the terminal."""
        prefix_map = {
            "debug": "DEBUG",
            "info": "INFO",
            "warning": "WARNING",
            "error": "ERROR",
        }
        
        prefix = prefix_map.get(level.lower(), "INFO")
        print(f"[{prefix}] {message}")
        
        # Also log to file
        super().display_message(message, level)
    
    # Command handlers
    def _cmd_help(self, *args):
        """Display help information."""
        print("\nAvailable commands:")
        print("  help      - Show this help message")
        print("  status    - Show system status")
        print("  version   - Show version information")
        print("  exit      - Exit the application\n")
    
    def _cmd_status(self, *args):
        """Show system status."""
        if self.system:
            print("\nSystem Status:")
            print(f"  Active: {self.system.is_active()}")
            print(f"  Agents: {self.system.get_agent_count()}")
            # Add more status information as needed
        else:
            print("\nSystem not connected.")
    
    def _cmd_exit(self, *args):
        """Exit the application."""
        self.running = False
        print("\nExiting application...")
    
    def _cmd_version(self, *args):
        """Show version information."""
        print("\nvAIn_p2p_AGI v0.1.0")
        print("© 2023 vAIn Project Contributors")
