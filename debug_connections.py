#!/usr/bin/env python
"""
Connection debugger for vAIn P2P network
Run this script to diagnose connection issues
"""

import asyncio
import argparse
import logging
from utils.network_diagnostics import NetworkDiagnostics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("connection_debugger")

async def debug_vscode_connection(port=52762):
    """Debug VSCode debugpy connection issues"""
    logger.info(f"Checking debugpy connection on port {port}")
    
    # Check if the port is available
    port_available = await NetworkDiagnostics.check_port_availability("127.0.0.1", port)
    if port_available:
        logger.warning(f"Port {port} is available, which means nothing is listening on it!")
        logger.warning("This explains the ConnectionRefusedError - the debugpy server isn't running")
        logger.info("Possible solutions:")
        logger.info("1. Ensure VSCode is properly configured for debugging")
        logger.info("2. Try restarting VSCode")
        logger.info("3. Check if any firewall is blocking the connection")
    else:
        logger.info(f"Port {port} is in use, something is listening")
        
        # Try to connect to the port
        connection_success, message = await NetworkDiagnostics.check_connection("127.0.0.1", port)
        logger.info(f"Connection test result: {message}")

async def debug_p2p_connections(host="localhost", port=8468):
    """Debug P2P network connections"""
    logger.info(f"Running diagnostics for P2P connection to {host}:{port}")
    results = await NetworkDiagnostics.run_diagnostics(host, port)
    
    # Provide recommendations based on results
    if not results.get("connection", {}).get("success", False):
        logger.info("\nRecommendations:")
        
        if not results.get("dns_resolution", {}).get("success", True):
            logger.info("- DNS resolution failed: Check if the hostname is correct")
        
        if not results.get("ping", {}).get("success", False):
            logger.info("- Host is not responding to ping: Check if the host is online")
            logger.info("- Check your network connection and firewall settings")
        
        if host == "localhost" or host == "127.0.0.1":
            if not results.get("port_available", False):
                logger.info("- Port is in use but connection failed: Another application might be using this port")
            else:
                logger.info("- Port is available: No application is listening on this port")
                logger.info("- Make sure the P2P service is running on this port")
        else:
            logger.info("- Check if the remote service is running and accepting connections")
            logger.info("- Verify that the port is correctly configured and open in firewalls")

async def main():
    """Main entry point for the connection debugger"""
    parser = argparse.ArgumentParser(description="vAIn P2P Network Connection Debugger")
    parser.add_argument("--vscode", action="store_true", help="Debug VSCode debugpy connection")
    parser.add_argument("--port", type=int, default=52762, help="Port to debug (default: 52762 for VSCode)")
    parser.add_argument("--host", type=str, default="localhost", help="Host to debug (default: localhost)")
    parser.add_argument("--p2p-port", type=int, default=8468, help="P2P port to debug (default: 8468)")
    
    args = parser.parse_args()
    
    logger.info("Starting vAIn P2P Network Connection Debugger")
    logger.info("=" * 60)
    
    if args.vscode:
        await debug_vscode_connection(args.port)
    else:
        await debug_p2p_connections(args.host, args.p2p_port)
    
    logger.info("=" * 60)
    logger.info("Connection debugging completed")

if __name__ == "__main__":
    asyncio.run(main())
