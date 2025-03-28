#!/bin/bash
# Install root project dependencies
npm install

# Install backend dependencies
cd backend
npm install

echo "Installation complete! You can now run 'npm run dev' to start the development server."
