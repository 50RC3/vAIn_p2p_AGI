"""
Debug utilities for network monitoring and diagnostics.
Provides components for interactive network monitoring, debugging and diagnostics.
"""

import asyncio
import logging
import time
import psutil
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from core.constants import InteractionLevel
from core.interactive_utils import InteractiveSession, InteractiveConfig
# Fix the circular import by importing DebugManager from utils.debug_utils instead of from itself
from utils.debug_utils import DebugManager

logger = logging.getLogger(__name__)

@dataclass
class DebugMetrics:
    """Debug metrics for network monitoring"""
    timestamp: float = field(default_factory=time.time)
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_io: Dict[str, float] = field(default_factory=dict)
    error_count: int = 0
    latency: float = 0.0
    connection_count: int = 0
    peer_count: int = 0


@dataclass
class DebugSession:
    """Debug session information"""
    session_id: str
    start_time: float = field(default_factory=time.time)
    metrics_history: List[DebugMetrics] = field(default_factory=list)
    active: bool = True
    interactive: bool = True


class DebugMonitor:
    """Monitor network for debugging purposes"""
    
    def __init__(self, interactive: bool = True, log_dir: Optional[str] = None):
        self.interactive = interactive
        self.log_dir = Path(log_dir) if log_dir else Path("logs/debug")
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.session: Optional[InteractiveSession] = None
        self.metrics_history: List[DebugMetrics] = []
        self.active_sessions: Dict[str, DebugSession] = {}
        self.is_monitoring = False
        self._monitor_task = None
        self.interaction_level = InteractionLevel.NORMAL
        self.debug_manager = DebugManager(str(self.log_dir))

    async def start_monitoring(self, session: Optional[InteractiveSession] = None) -> None:
        """Start monitoring network activity"""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.session = session
        
        # Create monitoring task
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Debug network monitoring started")
        
        if self.interactive and self.session:
            await self.session.log_info("Debug network monitoring active")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop"""
        try:
            while self.is_monitoring:
                # Collect metrics
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Trim history if needed
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                    
                # Log stats periodically
                if len(self.metrics_history) % 10 == 0:
                    self._log_stats(metrics)
                
                # Save metrics
                await self._save_progress(metrics)
                    
                # Wait for next check
                await asyncio.sleep(5)
                
        except asyncio.CancelledError:
            logger.debug("Debug monitoring task cancelled")
        except Exception as e:
            logger.error(f"Error in debug monitoring: {e}")
            self.debug_manager.track_error(e)

    def _collect_metrics(self) -> DebugMetrics:
        """Collect current system metrics"""
        metrics = DebugMetrics()
        
        try:
            # System resources
            metrics.cpu_usage = psutil.cpu_percent()
            metrics.memory_usage = psutil.virtual_memory().percent
            metrics.disk_usage = psutil.disk_usage('/').percent
            
            # Network IO
            net_io = psutil.net_io_counters()
            metrics.network_io = {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv
            }
            
            # Network connections
            connections = psutil.net_connections()
            metrics.connection_count = len([c for c in connections if c.status == 'ESTABLISHED'])
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            metrics.error_count += 1
            self.debug_manager.track_error(e)
            
        return metrics
    
    def _log_stats(self, metrics: DebugMetrics) -> None:
        """Log current stats to console"""
        logger.debug(
            f"Debug stats: CPU {metrics.cpu_usage:.1f}%, "
            f"Memory {metrics.memory_usage:.1f}%, "
            f"Network {len(metrics.network_io)} connections"
        )
    
    async def _save_progress(self, metrics: DebugMetrics) -> None:
        """Save debug progress to session"""
        if not self.session:
            return
            
        try:
            await self.session.save_progress({
                'debug_metrics': {
                    'timestamp': metrics.timestamp,
                    'cpu_usage': metrics.cpu_usage,
                    'memory_usage': metrics.memory_usage,
                    'network_io': metrics.network_io,
                    'connection_count': metrics.connection_count
                }
            })
        except Exception as e:
            logger.error(f"Error saving debug progress: {e}")
            self.debug_manager.track_error(e)
    
    async def check_network_health(self) -> Dict[str, Any]:
        """Check current network health"""
        # Get latest metrics if available
        if self.metrics_history:
            metrics = self.metrics_history[-1]
        else:
            metrics = self._collect_metrics()
            
        health = {
            'timestamp': time.time(),
            'peer_count': metrics.peer_count,
            'connection_success_rate': 1.0 - (metrics.error_count / max(1, len(self.metrics_history))),
            'avg_latency': metrics.latency,
            'bandwidth_usage': sum(metrics.network_io.values()) / 1024 / 1024 if metrics.network_io else 0,
            'overall_health': 1.0 - (metrics.cpu_usage / 100)  # Simple metric based on CPU usage
        }
        
        return health
    
    async def diagnose_connection(self, host: str, port: int) -> Dict[str, Any]:
        """Diagnose connection issues to host:port"""
        result = {
            'dns_resolution': {'success': True},
            'recommendations': []
        }
        
        # We're in debug mode, so add some diagnostic info
        result['recommendations'].append("This is a debug connection diagnostic")
        result['recommendations'].append(f"Testing connection to {host}:{port}")
        result['recommendations'].append("For detailed diagnostics, use NetworkDiagnostics")
        
        return result
    
    def stop_monitoring(self) -> None:
        """Stop network monitoring"""
        self.is_monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
        logger.info("Debug network monitoring stopped")


def create_debug_monitor(interactive: bool = True) -> DebugMonitor:
    """Create and return a debug monitor instance"""
    return DebugMonitor(interactive=interactive)


# Export the components that would be imported by main.py
__all__ = [
    'DebugMetrics',
    'DebugSession',
    'DebugMonitor',
    'create_debug_monitor'
]
