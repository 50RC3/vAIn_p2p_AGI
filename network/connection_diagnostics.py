import asyncio
import logging
import socket
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class DiagnosticResult:
    """Result of network diagnostics"""
    dns_resolution: Dict[str, Any] = field(default_factory=dict)
    tcp_connection: Dict[str, Any] = field(default_factory=dict)
    latency: Optional[float] = None
    recommendations: List[str] = field(default_factory=list)

class ConnectionDiagnostics:
    """Diagnostics for network connections"""
    
    async def diagnose_connection(self, host: str, port: int) -> DiagnosticResult:
        """Diagnose connection to a host:port"""
        result = DiagnosticResult()
        
        # Check DNS resolution
        try:
            ip = await self._resolve_dns(host)
            result.dns_resolution = {
                "success": True,
                "ip": ip
            }
        except Exception as e:
            logger.error(f"DNS resolution failed: {e}")
            result.dns_resolution = {
                "success": False,
                "error": str(e)
            }
            result.recommendations.append(f"Check DNS settings. Could not resolve {host}.")
            return result
            
        # Check TCP connection
        try:
            latency = await self._check_tcp_connection(ip, port)
            result.tcp_connection = {
                "success": True,
                "latency": latency
            }
            result.latency = latency
            
            if latency > 500:
                result.recommendations.append(f"High latency ({latency:.1f}ms) detected. Consider optimizing network.")
        except Exception as e:
            logger.error(f"TCP connection failed: {e}")
            result.tcp_connection = {
                "success": False,
                "error": str(e)
            }
            result.recommendations.append(f"Could not connect to {host}:{port}. Check firewall settings.")
            
        return result
        
    async def _resolve_dns(self, host: str) -> str:
        """Resolve DNS asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, socket.gethostbyname, host)
        
    async def _check_tcp_connection(self, host: str, port: int) -> float:
        """Check TCP connection and measure latency"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            reader, writer = await asyncio.open_connection(host, port)
            latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            
            writer.close()
            await writer.wait_closed()
            
            return latency_ms
        except Exception:
            # If connection fails, return a high latency value as indicator
            return 9999.0
        
    async def diagnose_p2p_network(self) -> Dict[str, Any]:
        """Diagnose P2P network (placeholder for actual implementation)"""
        return {
            "status": "offline",
            "reason": "P2P network diagnostics not implemented yet"
        }
