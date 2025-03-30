import asyncio
from typing import List, Dict, Optional, Callable, Any
import sys

class MenuItem:
    """Represents an item in a menu."""
    
    def __init__(self, command: str, description: str, 
                 handler: Optional[Callable] = None):
        self.command = command.lower()
        self.description = description
        self.handler = handler

class MenuSystem:
    """A system for creating and managing interactive menus."""
    
    def __init__(self, title: str):
        self.title = title
        self.items: List[MenuItem] = []
        
    def add_menu_item(self, item: MenuItem) -> None:
        """Add a menu item to the menu."""
        self.items.append(item)
        
    async def display(self, session) -> None:
        """Display the menu to the user."""
        print(f"\n{self.title}")
        print("=" * len(self.title))
        
        for i, item in enumerate(self.items, 1):
            print(f"{i}. {item.description} ['{item.command}']")
    
    async def handle_command(self, command: str, session) -> bool:
        """Handle a command input by the user."""
        command = command.lower()
        
        # Check if command is a number
        try:
            index = int(command) - 1
            if 0 <= index < len(self.items):
                command = self.items[index].command
            else:
                print(f"Invalid menu option: {command}")
                return False
        except ValueError:
            # Not a number, treat as text command
            pass
        
        # Find the matching command
        for item in self.items:
            if item.command == command:
                if item.handler:
                    await item.handler(session)
                return True
                
        print(f"Unknown command: {command}")
        print("Type a command name or number from the menu.")
        return False
