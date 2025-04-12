"""
Example of how to use the component integration system.
"""
import asyncio
import logging
import os
import sys
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import integration components
from core.component_integration import ComponentIntegration, ComponentFactory
from core.integration_helper import IntegrationHelper
from core.interactive_config import InteractiveConfig

async def main():
    """
    Example of using component integration.
    """
    # Create configuration for integration
    config = {
        'interactive_mode': True,
        'resource_monitoring': True,
        'status_check_interval': 60,
        'memory_threshold': 0.9,
        'safe_mode': True,
        'model': {
            'path': os.path.join('models', 'default_model.pt'),
            'batch_size': 16,
            'learning_rate': 0.001,
            'use_gpu': torch.cuda.is_available()
        },
        'use_ui_manager': True
    }
    
    # Set up the complete system
    try:
        # Create the system with all components
        system = await IntegrationHelper.setup_complete_system(config)
        
        integration = system["integration"]
        model_interface = system["model_interface"]
        ui_components = system["ui_components"]
        ui_manager = system["ui_manager"]
        
        # Start UI components
        await IntegrationHelper.start_ui_components(ui_components)
        
        # Example: Run a training operation
        print("\nRunning example training operation...")
        
        # Create test data
        test_inputs = torch.randn(100, 10)
        test_targets = torch.randn(100, 10)
        test_data = (test_inputs, test_targets)
        
        # Create validation data
        val_inputs = torch.randn(20, 10)
        val_targets = torch.randn(20, 10)
        val_data = (val_inputs, val_targets)
        
        # Train the model
        try:
            result = await model_interface.train(test_data, val_data, epochs=3)
            print(f"Training result: {result}")
        except Exception as e:
            print(f"Training error: {e}")
        
        # Example: Run a prediction
        print("\nRunning example prediction...")
        test_input = torch.randn(1, 10)
        try:
            output = await model_interface.predict(test_input)
            print(f"Model prediction shape: {output.shape}")
            print(f"Sample output: {output[0][:3]}...")
        except Exception as e:
            print(f"Prediction error: {e}")
        
        # Get system status
        print("\nGetting system status...")
        status = await integration.get_status()
        print(f"System initialized: {status['initialized']}")
        print(f"Interactive mode: {status['interactive_mode']}")
        print(f"Resource monitoring: {status['resource_monitoring']}")
        print(f"Number of components: {len(status['components'])}")
        
        if 'resources' in status:
            print("\nResource usage:")
            print(f"CPU: {status['resources'].get('cpu_usage', 'N/A')}%")
            print(f"Memory: {status['resources'].get('memory_usage', 'N/A')}%")
            print(f"Disk: {status['resources'].get('disk_usage', 'N/A')}%")
            if 'gpu_usage' in status['resources']:
                print(f"GPU: {status['resources'].get('gpu_usage')}%")
        
        # Run CLI if it was created
        if "cli" in ui_components:
            cli = ui_components["cli"]
            # If running in non-interactive example mode, don't start the CLI
            # Otherwise, uncomment the next line to start the CLI
            # await cli.start()
            
        print("\nExample completed. Would normally run until CLI is closed.")
        print("Cleaning up resources...")
        
    except Exception as e:
        logger.error(f"Error in integration example: {e}", exc_info=True)
    finally:
        # Clean up components
        if 'system' in locals():
            await IntegrationHelper.cleanup_system(
                system["integration"],
                {"ui_manager": system.get("ui_manager")}
            )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExample interrupted by user")
    except Exception as e:
        print(f"Fatal error: {e}")
