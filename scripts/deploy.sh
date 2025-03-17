#!/bin/bash

echo "Deploying vAIn contracts..."

# Deploy core contracts
npx hardhat run scripts/deploy.js --network $NETWORK

# Verify contracts on Etherscan
npx hardhat verify --network $NETWORK $TOKEN_ADDRESS
