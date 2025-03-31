#!/bin/bash
# Install dependencies and set up configuration

# Create necessary directories
mkdir -p config/backups logs/metrics

echo "Creating default configurations..."
python -c "from tools.config_manager import ConfigManager; ConfigManager().create_default_configs()"

# Install root project dependencies
npm install

# Install backend dependencies
cd backend
npm install

echo "Installation complete! You can now run 'npm run dev' to start the development server."
echo "Use 'va config' to manage system configurations."
