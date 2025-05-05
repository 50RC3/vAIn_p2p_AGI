#!/usr/bin/env python3
"""
Script that imports and runs the code_style_fixer utility from another location.
"""
import os
import sys
import importlib.util
import logging
import types
from typing import Optional

def load_module_from_path(
    module_path: str,
    module_name: str
) -> Optional[types.ModuleType]:
    """
    Load a Python module from a file path.

    Args:
        module_path (str): Path to the Python file
        module_name (str): Name to give the imported module

    Returns:
        module: The loaded module or None if loading failed
    """
    if not module_path:
        logging.error("Module path is not provided.")
        return None
    if not module_name:
        logging.error("Module name is not provided.")
        return None

    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)

        # If the specification is not found, log an error and return None
        if not spec:
            logging.error("Could not load the module loader from %s", module_path)
            return None

        # If the module loader is not found, log an error and return None
        if not spec.loader:
            logging.error("Could not load the module loader from %s", module_path)
            return None

        # Create a module object from the specification
        module = importlib.util.module_from_spec(spec)
        if not module:
            logging.error("Could not create the module object from %s", module_path)
            return None

        # Execute the module loader to load the module
        # This is the part that actually runs the code
        spec.loader.exec_module(module)

        return module
    except AttributeError as e:
        # If an AttributeError is raised, log an error and return None
        logging.error(
            "Could not load the module or loader from %s: %s", 
            module_path, e
        )
        return None
    except FileNotFoundError as e:
        # If a FileNotFoundError is raised, log an error and return None
        logging.error(
            "Could not find the module %s at %s: %s", 
            module_name, module_path, e
        )
        return None
    except ImportError as e:
        # If an ImportError is raised, log an error and return None
        logging.error(
            "Could not import the module %s from %s: %s", 
            module_name, module_path, e
        )
        return None
    except (ValueError, TypeError) as e:
        # If any other specific exception is raised, log and return None
        logging.error(
            "Error loading module %s from %s: %s", 
            module_name, module_path, e
        )
        return None

def main() -> int:
    # Configure logging to print out all information
    # about the loading process
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Path to the Python file that contains the code
    # to be executed
    original_script_path = new_func()

    # Load the specified Python file as a module
    # and get a reference to the loaded module
    code_style_fixer = load_module_from_path(original_script_path, "code_style_fixer")
    if not code_style_fixer:
        # If the module couldn't be loaded for some reason,
        # return an error code
        return 1

    # Run the main function from the imported module
    # to execute the code
    return int(code_style_fixer.main())


def new_func() -> str:
    # Dynamically construct the path using environment variables or relative paths
    base_dir = os.environ.get("VAI_BETA_BASE_DIR", r"C:\Users\Mr.V\Documents\GitHub\vAIn_Beta\beta")
    original_script_path = os.path.join(base_dir, "utils", "code_style_fixer.py")
    return original_script_path

if __name__ == "__main__":
    sys.exit(main())
