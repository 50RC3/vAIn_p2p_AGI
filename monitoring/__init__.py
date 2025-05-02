"""
Monitoring package for vAIn P2P AGI system.
Provides automated log monitoring, system monitoring, and recovery capabilities.
"""

import os
import logging
import asyncio
from typing import Dict, Any, Optional, List

# Import monitoring components
from .log_monitor import LogMonitor, LogAlert, setup_log_monitor

logger = logging.getLogger(__name__)

# Global monitor instance
_log_monitor = None

async def initialize_monitoring(config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Initialize the monitoring system with the given configuration.
    
    Args:
        config: Configuration dictionary for the monitoring system
        
    Returns:
        bool: True if initialization was successful
    """
    global _log_monitor
    
    try:
        # Create log directories
        os.makedirs("logs", exist_ok=True)
        
        # Start log monitor
        _log_monitor = await setup_log_monitor(config)
        
        # Register with system coordinator if available
        try:
            from ai_core.system_coordinator import SystemCoordinator
            coordinator = SystemCoordinator.get_instance()
            if coordinator:
                coordinator.register_monitor("log_monitor", _log_monitor)
                logger.info("Registered log monitor with system coordinator")
        except (ImportError, AttributeError):
            logger.debug("System coordinator not available for monitor registration")
        
        logger.info("Monitoring system initialized")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize monitoring: {e}")
        return False

async def stop_monitoring() -> None:
    """Stop all monitoring components"""
    global _log_monitor
    
    if _log_monitor:
        await _log_monitor.stop()
        _log_monitor = None
        logger.info("Monitoring system stopped")

def get_log_monitor() -> Optional[LogMonitor]:
    """Get the current log monitor instance"""
    return _log_monitor

def register_alert_handler(handler):
    """Register a handler for log alerts"""
    global _log_monitor
    if _log_monitor:
        _log_monitor.register_callback(handler)
        return True
    return False

def get_recent_alerts(level=None, resolved=None, limit=100):
    """Get recent alerts from the log monitor"""
    global _log_monitor
    if _log_monitor:
        return _log_monitor.get_alerts(level, resolved, limit)
    return []

def get_monitoring_stats():
    """Get monitoring statistics"""
    global _log_monitor
    if _log_monitor:
        return _log_monitor.get_stats()
    return {"status": "not_running"}

__all__ = [
    'LogMonitor',
    'LogAlert',
    'initialize_monitoring',
    'stop_monitoring',
    'get_log_monitor',
    'register_alert_handler',
    'get_recent_alerts',
    'get_monitoring_stats'
]