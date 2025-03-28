import socket
import asyncio
import logging
import time
import subprocess
from typing import Dict, List, Tuple, Optional
import platform

logger = logging.getLogger(__name__)

class NetworkDiagnostics:
    """Utility class for diagnosing network issues"""
    
    @staticmethod
    async def check_port_availability(host: str, port: int) -> bool:
        """Check if a port is available for binding"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            sock.bind((host, port))
            sock.close()
            return True
        except OSError:
            return False
            
    @staticmethod
    async def check_connection(host: str, port: int, timeout: float = 2.0) -> Tuple[bool, str]:
        """Check if a connection can be established to host:port"""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=timeout
            )
            writer.close()
            await writer.wait_closed()
            return True, "Connection successful"
        except asyncio.TimeoutError:
            return False, "Connection timed out"
        except ConnectionRefusedError:
            return False, "Connection refused (port not listening)"
        except socket.gaierror:
            return False, "DNS resolution error"
        except OSError as e:
            return False, f"Network error: {str(e)}"
    
    @staticmethod
    async def test_local_ports(start_port: int = 8000, count: int = 10) -> Dict[int, bool]:
        """Test a range of local ports for availability"""
        results = {}
        for port in range(start_port, start_port + count):
            results[port] = await NetworkDiagnostics.check_port_availability("127.0.0.1", port)
        return results
    
    @staticmethod
    async def ping_host(host: str) -> Tuple[bool, str]:
        """Ping a host to check basic connectivity"""
        param = "-n" if platform.system().lower() == "windows" else "-c"
        command = ["ping", param, "1", host]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return True, "Ping successful"
            else:
                return False, f"Ping failed: {result.stderr}"
        except subprocess.TimeoutExpired:
            return False, "Ping timed out"
        except Exception as e:
            return False, f"Ping error: {str(e)}"

    @staticmethod
    async def diagnose_connection_issue(host: str, port: int) -> Dict:
        """Run a full diagnostic on connection issues to host:port"""
        results = {}
        
        # Check DNS resolution
        try:
            ip = socket.gethostbyname(host)
            results["dns_resolution"] = {"success": True, "ip": ip}
        except socket.gaierror:
            results["dns_resolution"] = {"success": False, "error": "Could not resolve hostname"}
        
        # Ping test
        ping_success, ping_message = await NetworkDiagnostics.ping_host(host)
        results["ping"] = {"success": ping_success, "message": ping_message}
        
        # Port check
        connection_success, conn_message = await NetworkDiagnostics.check_connection(host, port)
        results["connection"] = {"success": connection_success, "message": conn_message}
        
        # Check for local port conflicts
        if host == "127.0.0.1" or host == "localhost":
            results["port_available"] = await NetworkDiagnostics.check_port_availability(host, port)
        
        return results

    @staticmethod
    async def run_diagnostics(target_host: str, target_port: int):
        """Run and log diagnostics for a connection issue"""
        logger.info(f"Running network diagnostics for {target_host}:{target_port}")
        
        results = await NetworkDiagnostics.diagnose_connection_issue(target_host, target_port)
        
        logger.info("Diagnostic results:")
        for key, value in results.items():
            logger.info(f"  {key}: {value}")
            
        if not results.get("connection", {}).get("success", False):
            logger.error(f"Connection to {target_host}:{target_port} failed: {results.get('connection', {}).get('message')}")
            
        return results
