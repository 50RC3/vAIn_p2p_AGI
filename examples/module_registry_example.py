"""
Example demonstrating the use of the Module Registry system.
"""

import asyncio
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import the necessary components
from core.module_registry import ModuleRegistry
from core.resource_management import ResourceManager

# Example module classes
class BaseModule:
    """Base module class that all modules should inherit from."""
    
    async def initialize(self) -> None:
        """Initialize the module."""
        print(f"[{self.__class__.__name__}] Initializing...")
        
    async def shutdown(self) -> None:
        """
        Shut down the module.

        This method performs necessary cleanup and resource deallocation
        to ensure a graceful shutdown of the module.
        """
        # Log the shutdown process for the module
        print(f"[{self.__class__.__name__}] Shutting down...")

        # Perform additional shutdown procedures here
        # For example, closing database connections, flushing caches, etc.
        
    async def heartbeat(self) -> bool:
        """Return True if the module is healthy."""
        return True

class DataStorageModule(BaseModule):
    """Example data storage module."""
    
    async def initialize(self) -> None:
        await super().initialize()
        print("[DataStorage] Setting up data connections...")
        # In a real module, we would set up database connections here
        
    async def store_data(self, key: str, value: Any) -> bool:
        """Store data in the storage."""
        print(f"[DataStorage] Storing data: {key} = {value}")
        return True

class NLPModule(BaseModule):
    """Example Natural Language Processing module."""
    
    async def initialize(self) -> None:
        await super().initialize()
        print("[NLP] Loading language models...")
        # In a real module, we would load language models here
        
    async def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text using NLP techniques."""
        print(f"[NLP] Analyzing text: {text}")
        return {"sentiment": "positive", "entities": ["example"], "length": len(text)}

class ReasoningModule(BaseModule):
    """Example reasoning module that depends on NLP."""
    
    def __init__(self) -> None:
        self.nlp = None
        
    async def initialize(self) -> None:
        await super().initialize()
        print("[Reasoning] Setting up reasoning frameworks...")
        # In a real module, we would set up reasoning frameworks here
        
    async def reason(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform reasoning on input data."""
        print(f"[Reasoning] Reasoning about: {input_data}")
        return {"conclusion": "example conclusion", "confidence": 0.95}

async def setup_registry() -> tuple[ModuleRegistry, ResourceManager]:
    """Set up and initialize the module registry with example modules."""
    
    # Create resource manager
    resource_manager = ResourceManager()
    resource_manager.start_monitoring()
    
    # Create module registry
    registry = ModuleRegistry(
        config_path="examples/config/modules.json",
        resource_manager=resource_manager
    )
    
    # Register modules
    registry.register_module("storage", DataStorageModule, dependencies=[])
    registry.register_module("nlp", NLPModule, dependencies=["storage"])
    registry.register_module("reasoning", ReasoningModule, dependencies=["nlp"])
    
    # Register callback to demonstrate event handling
    def on_module_status_change(data: Dict[str, str]) -> None:
        module_id = data["module_id"]
        old_status = data["old_status"]
        new_status = data["new_status"]
        print(f"Status change callback: {module_id} changed from {old_status} to {new_status}")
    
    registry.callback_manager.register_callback("module_status_change", on_module_status_change)
    
    # Initialize registry (which will initialize all modules)
    await registry.initialize()  # type: ignore
    
    return registry, resource_manager

async def example_workflow(registry: ModuleRegistry) -> None:
    """Run an example workflow using the registered modules."""
    
    # Get module instances
    storage = registry.get_module("storage")
    nlp = registry.get_module("nlp")
    reasoning = registry.get_module("reasoning")
    
    # Example workflow
    print("\nRunning example workflow:")
    
    # Store some input text
    await storage.store_data("input_text", "This is an example of module registry in action.")
    
    # Analyze text with NLP
    analysis = await nlp.analyze_text("This is an example of module registry in action.")
    
    # Use reasoning on the analysis
    result = await reasoning.reason(analysis)
    
    print(f"\nWorkflow result: {result}\n")

async def main() -> None:
    """
    Main entry point for the module registry example.
    This async function demonstrates the core functionality of the module registry:
    - Setting up a registry and resource manager
    - Running an example workflow 
    - Updating module status
    - Handling registry shutdown
    Returns:
        None
    Raises:
        Exception: Propagates any exceptions that occur during execution
    """
    try:
        print("Starting Module Registry Example...")
        
        registry, resource_manager = await setup_registry()
        
        # Run an example workflow
        await example_workflow(registry)
        
        # Update status of a module
        registry.update_module_status("nlp", "suspended")
        
        # Wait to see status change callback
        await asyncio.sleep(1)
        
        # Clean shutdown
        await registry.shutdown()  # type: ignore
        resource_manager.stop_monitoring()
        
        print("Example completed successfully.")
        
    except Exception as e:
        print(f"Error in example: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())