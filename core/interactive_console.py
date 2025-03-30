import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
import os
import sys

from .constants import InteractionLevel
from .interactive_utils import InteractiveSession, InteractiveConfig
from .menu_system import MenuSystem, MenuItem
from network.admin_commands import AdminCommands
from security.firewall_rules import FirewallManager
from security.auth_manager import AuthManager
from network.monitoring import ResourceMonitor

logger = logging.getLogger(__name__)

class InteractiveConsole:
    """Central interactive console that provides access to all system functionality."""
    
    def __init__(self, interaction_level: InteractionLevel = InteractionLevel.NORMAL):
        self.interaction_level = interaction_level
        self.session: Optional[InteractiveSession] = None
        self.menu_system = MenuSystem("vAIn System Console")
        self.running = False
        self._setup_menus()
        
    def _setup_menus(self) -> None:
        """Set up the menu structure for the interactive console."""
        
        # Main menu
        self.menu_system.add_menu_item(MenuItem(
            "network", "Network Management", self._network_menu
        ))
        self.menu_system.add_menu_item(MenuItem(
            "security", "Security Controls", self._security_menu
        ))
        self.menu_system.add_menu_item(MenuItem(
            "model", "AI Model Management", self._model_menu
        ))
        self.menu_system.add_menu_item(MenuItem(
            "system", "System Monitoring & Configuration", self._system_menu
        ))
        self.menu_system.add_menu_item(MenuItem(
            "help", "Help & Documentation", self._show_help
        ))
        self.menu_system.add_menu_item(MenuItem(
            "exit", "Exit Console", None
        ))
        
    async def start(self) -> None:
        """Start the interactive console."""
        try:
            self.running = True
            config = InteractiveConfig(
                timeout=300,
                persistent_state=True,
                safe_mode=True,
                recovery_enabled=True,
                max_cleanup_wait=30
            )
            
            self.session = InteractiveSession(
                session_id="main_console",
                config=config
            )
            
            async with self.session:
                logger.info("Interactive console started")
                print("\nWelcome to vAIn System Console")
                print("=" * 50)
                
                while self.running:
                    await self.menu_system.display(self.session)
                    command = await self.session.get_input("\nEnter command: ")
                    
                    if command.lower() == "exit":
                        await self._confirm_exit()
                        continue
                    
                    await self.menu_system.handle_command(command, self.session)
        except KeyboardInterrupt:
            print("\nConsole interrupted. Shutting down...")
        except Exception as e:
            logger.error(f"Console error: {str(e)}")
        finally:
            self.running = False
            print("\nExiting console. Goodbye!")
            
    async def _confirm_exit(self) -> None:
        """Confirm exit from the console."""
        if await self.session.prompt_yes_no("Are you sure you want to exit?"):
            self.running = False
            
    async def _network_menu(self, session: InteractiveSession) -> None:
        """Handle network management menu."""
        admin = AdminCommands()
        
        network_menu = MenuSystem("Network Management")
        network_menu.add_menu_item(MenuItem("peers", "Show Connected Peers", 
                                           lambda s: admin._handle_show_command("peers", [])))
        network_menu.add_menu_item(MenuItem("ban", "Ban a Peer", self._ban_peer))
        network_menu.add_menu_item(MenuItem("unban", "Unban a Peer", self._unban_peer))
        network_menu.add_menu_item(MenuItem("reputation", "Show Reputation Table", 
                                           lambda s: admin._handle_show_command("reputation", [])))
        network_menu.add_menu_item(MenuItem("consensus", "Show Consensus Status", 
                                           lambda s: admin._handle_show_command("consensus", [])))
        network_menu.add_menu_item(MenuItem("back", "Back to Main Menu", None))
        
        await network_menu.display(session)
        while True:
            command = await session.get_input("\nNetwork command: ")
            if command.lower() == "back":
                break
            await network_menu.handle_command(command, session)
            
    async def _ban_peer(self, session: InteractiveSession) -> None:
        """Ban a peer interactively."""
        admin = AdminCommands()
        peer_id = await session.get_input("Enter peer ID to ban: ")
        reason = await session.get_input("Enter reason for ban (optional): ")
        if not reason:
            reason = "Manual ban from console"
        await admin._ban_peer(peer_id, reason)
        
    async def _unban_peer(self, session: InteractiveSession) -> None:
        """Unban a peer interactively."""
        admin = AdminCommands()
        peer_id = await session.get_input("Enter peer ID to unban: ")
        await admin.network.unban_peer(peer_id)
        print(f"Peer {peer_id} has been unbanned.")
            
    async def _security_menu(self, session: InteractiveSession) -> None:
        """Handle security controls menu."""
        security_menu = MenuSystem("Security Controls")
        security_menu.add_menu_item(MenuItem("firewall", "Firewall Management", self._firewall_menu))
        security_menu.add_menu_item(MenuItem("auth", "Authentication Management", self._auth_menu))
        security_menu.add_menu_item(MenuItem("back", "Back to Main Menu", None))
        
        await security_menu.display(session)
        while True:
            command = await session.get_input("\nSecurity command: ")
            if command.lower() == "back":
                break
            await security_menu.handle_command(command, session)
            
    async def _firewall_menu(self, session: InteractiveSession) -> None:
        """Handle firewall management menu."""
        firewall = FirewallManager()
        
        firewall_menu = MenuSystem("Firewall Management")
        firewall_menu.add_menu_item(MenuItem("show", "Show Current Rules", self._show_firewall_rules))
        firewall_menu.add_menu_item(MenuItem("add", "Add New Rule", self._add_firewall_rule))
        firewall_menu.add_menu_item(MenuItem("back", "Back to Security Menu", None))
        
        await firewall_menu.display(session)
        while True:
            command = await session.get_input("\nFirewall command: ")
            if command.lower() == "back":
                break
            await firewall_menu.handle_command(command, session)
            
    async def _show_firewall_rules(self, session: InteractiveSession) -> None:
        """Show current firewall rules."""
        firewall = FirewallManager()
        print("\nCurrent Firewall Rules:")
        print("-" * 80)
        print(f"{'Protocol':<10} {'Port':<8} {'Direction':<12} {'Action':<8} {'Priority':<10}")
        print("-" * 80)
        for rule in firewall.rules:
            print(f"{rule.protocol:<10} {rule.port:<8} {rule.direction:<12} {rule.action:<8} {rule.priority:<10}")
            
    async def _add_firewall_rule(self, session: InteractiveSession) -> None:
        """Add a new firewall rule interactively."""
        from security.firewall_rules import FirewallRule
        
        firewall = FirewallManager()
        
        print("\nAdd New Firewall Rule:")
        protocol = await session.prompt_options(
            "Select protocol:", ["TCP", "UDP"], "TCP")
        
        port = None
        while not port:
            try:
                port_str = await session.get_input("Enter port number: ")
                port = int(port_str)
                if port < 1 or port > 65535:
                    print("Port must be between 1 and 65535")
                    port = None
            except ValueError:
                print("Port must be a number")
                
        direction = await session.prompt_options(
            "Select direction:", ["INBOUND", "OUTBOUND", "BOTH"], "BOTH")
        
        action = await session.prompt_options(
            "Select action:", ["ALLOW", "DENY"], "ALLOW")
        
        priority = None
        while not priority:
            try:
                priority_str = await session.get_input("Enter priority (1-100, higher is more important): ")
                priority = int(priority_str)
                if priority < 1 or priority > 100:
                    print("Priority must be between 1 and 100")
                    priority = None
            except ValueError:
                print("Priority must be a number")
                
        rule = FirewallRule(protocol, port, direction, action, priority)
        success = await firewall.add_rule_interactive(rule)
        if success:
            print("\nFirewall rule added successfully!")
        else:
            print("\nFailed to add firewall rule.")
            
    async def _auth_menu(self, session: InteractiveSession) -> None:
        """Handle authentication management menu."""
        auth_manager = AuthManager()
        
        auth_menu = MenuSystem("Authentication Management")
        auth_menu.add_menu_item(MenuItem("rotate", "Rotate Authentication Keys", self._rotate_auth_keys))
        auth_menu.add_menu_item(MenuItem("stats", "Show Authentication Statistics", self._show_auth_stats))
        auth_menu.add_menu_item(MenuItem("back", "Back to Security Menu", None))
        
        await auth_menu.display(session)
        while True:
            command = await session.get_input("\nAuthentication command: ")
            if command.lower() == "back":
                break
            await auth_menu.handle_command(command, session)
            
    async def _rotate_auth_keys(self, session: InteractiveSession) -> None:
        """Rotate authentication keys interactively."""
        auth_manager = AuthManager()
        success = await auth_manager.rotate_keys_interactive()
        if success:
            print("\nAuthentication keys rotated successfully!")
        else:
            print("\nFailed to rotate authentication keys.")
            
    async def _show_auth_stats(self, session: InteractiveSession) -> None:
        """Show authentication statistics."""
        auth_manager = AuthManager()
        print("\nAuthentication Statistics:")
        print("-" * 50)
        for key, value in auth_manager._stats.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
            
    async def _model_menu(self, session: InteractiveSession) -> None:
        """Handle AI model management menu."""
        model_menu = MenuSystem("AI Model Management")
        model_menu.add_menu_item(MenuItem("version", "Version Control", self._version_control))
        model_menu.add_menu_item(MenuItem("status", "Model Status", self._model_status))
        model_menu.add_menu_item(MenuItem("back", "Back to Main Menu", None))
        
        await model_menu.display(session)
        while True:
            command = await session.get_input("\nModel management command: ")
            if command.lower() == "back":
                break
            await model_menu.handle_command(command, session)
    
    async def _version_control(self, session: InteractiveSession) -> None:
        """Handle model version control."""
        from core.version_control import ModelVersionControl
        
        version_control = ModelVersionControl()
        
        version_menu = MenuSystem("Version Control")
        version_menu.add_menu_item(MenuItem("list", "List Versions", self._list_versions))
        version_menu.add_menu_item(MenuItem("save", "Save New Version", self._save_version))
        version_menu.add_menu_item(MenuItem("load", "Load Version", self._load_version))
        version_menu.add_menu_item(MenuItem("back", "Back to Model Menu", None))
        
        await version_menu.display(session)
        while True:
            command = await session.get_input("\nVersion control command: ")
            if command.lower() == "back":
                break
            await version_menu.handle_command(command, session)
    
    async def _list_versions(self, session: InteractiveSession) -> None:
        """List available model versions."""
        from core.version_control import ModelVersionControl
        
        version_control = ModelVersionControl()
        versions = version_control.version_history
        
        print("\nAvailable Model Versions:")
        print("-" * 100)
        print(f"{'Version Hash':<40} {'Timestamp':<25} {'Author':<20} {'Description'}")
        print("-" * 100)
        
        for hash, version in versions.items():
            print(f"{hash[:8]:<40} {version.timestamp:<25} {version.metadata.get('author', 'Unknown'):<20} {version.metadata.get('description', 'No description')}")
    
    async def _save_version(self, session: InteractiveSession) -> None:
        """Save a new model version."""
        from core.version_control import ModelVersionControl
        
        version_control = ModelVersionControl()
        # Use the existing interactive save functionality
        version_control.interactive_save()
    
    async def _load_version(self, session: InteractiveSession) -> None:
        """Load a model version."""
        from core.version_control import ModelVersionControl
        
        version_control = ModelVersionControl()
        
        # First list available versions
        await self._list_versions(session)
        
        # Ask for version hash
        version_hash = await session.get_input("\nEnter version hash to load (or partial hash): ")
        if not version_hash:
            print("Operation cancelled")
            return
            
        # Find matching version
        matching_versions = [
            hash for hash in version_control.version_history.keys() 
            if hash.startswith(version_hash)
        ]
        
        if not matching_versions:
            print(f"No versions found matching '{version_hash}'")
            return
            
        if len(matching_versions) > 1:
            print(f"Multiple matching versions found. Please be more specific:")
            for hash in matching_versions:
                version = version_control.version_history[hash]
                print(f"{hash[:8]} - {version.metadata.get('description', 'No description')}")
            return
            
        # Load the version
        try:
            version_control.load_version(matching_versions[0])
            print(f"\nSuccessfully loaded version: {matching_versions[0]}")
        except Exception as e:
            print(f"\nError loading version: {str(e)}")
    
    async def _model_status(self, session: InteractiveSession) -> None:
        """Show model status."""
        # This would need to be connected to your model tracking system
        print("\nModel Status:")
        print("-" * 50)
        print("This functionality requires integration with your model tracking system.")
        print("Please implement based on your specific model management approach.")
            
    async def _system_menu(self, session: InteractiveSession) -> None:
        """Handle system monitoring and configuration menu."""
        system_menu = MenuSystem("System Monitoring & Configuration")
        system_menu.add_menu_item(MenuItem("resources", "Resource Monitoring", self._resource_monitoring))
        system_menu.add_menu_item(MenuItem("config", "System Configuration", self._system_config))
        system_menu.add_menu_item(MenuItem("back", "Back to Main Menu", None))
        
        await system_menu.display(session)
        while True:
            command = await session.get_input("\nSystem command: ")
            if command.lower() == "back":
                break
            await system_menu.handle_command(command, session)
    
    async def _resource_monitoring(self, session: InteractiveSession) -> None:
        """Handle resource monitoring."""
        monitor = ResourceMonitor()
        
        health = await monitor.check_resources_interactive()
        if not health:
            print("Failed to retrieve resource metrics")
            return
            
        print("\nSystem Resource Report:")
        print("-" * 50)
        print(f"CPU Usage: {health.cpu_percent:.1f}%")
        print(f"Memory Usage: {health.memory_percent:.1f}%")
        print(f"Disk Usage: {health.disk_percent:.1f}%")
        print(f"Network IO: {health.network_io_read/1024/1024:.2f} MB read, {health.network_io_write/1024/1024:.2f} MB written")
        print(f"Active Connections: {health.active_connections}")
        
        if health.warnings:
            print("\nWarnings:")
            for warning in health.warnings:
                print(f"- {warning}")
    
    async def _system_config(self, session: InteractiveSession) -> None:
        """Handle system configuration."""
        from config import Config
        
        config = Config()
        await config.update_interactive()
    
    async def _show_help(self, session: InteractiveSession) -> None:
        """Show help and documentation."""
        print("\nHelp & Documentation")
        print("=" * 50)
        print("The vAIn System Console provides a central interface to all system functionality.")
        print("\nMain Areas:")
        print("- network: Manage peer connections, bans, and network status")
        print("- security: Control firewall rules and authentication settings")
        print("- model: Manage AI model versions and training")
        print("- system: Monitor resources and configure system settings")
        print("\nNavigation:")
        print("- Type the command name to access a feature")
        print("- Use 'back' to return to the previous menu")
        print("- Use 'exit' to quit the console")
        print("\nFor detailed documentation, see: docs/user_guide.md")
