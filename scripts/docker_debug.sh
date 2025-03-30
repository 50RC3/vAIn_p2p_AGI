#!/bin/bash

# Docker debugging helper script for vAIn_p2p_AGI
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

# Ensure we're in the project root
cd "$PROJECT_ROOT" || { echo "Failed to navigate to project root"; exit 1; }

# Execute the Python debugging module
python -m scripts.module_debug "$@"
