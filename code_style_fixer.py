import sys
import os
import importlib.util

# Path to the original script
original_script_path = r"C:\Users\Mr.V\Documents\GitHub\vAIn_Beta\beta\utils\code_style_fixer.py"

# Import the original script as a module
spec = importlib.util.spec_from_file_location("code_style_fixer", original_script_path)
code_style_fixer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(code_style_fixer)

# Run the main function
if __name__ == "__main__":
    sys.exit(code_style_fixer.main())
