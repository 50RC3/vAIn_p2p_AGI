"""
Enhanced log monitoring system for vAIn P2P AGI.
Provides proactive monitoring, early warning system, and automated recovery.
"""

import os
import re
import time
import logging
import asyncio
import traceback
import hashlib
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
import concurrent.futures

# Import existing monitoring utilities
try:
    from utils.resource_monitor import ResourceMonitor
except ImportError:
    ResourceMonitor = None

try:
    from utils.debug_utils import DebugManager, debug_manager
except ImportError:
    debug_manager = None

logger = logging.getLogger(__name__)

@dataclass
class LogAlert:
    """Alert generated from log monitoring"""
    level: str
    message: str
    source_file: str
    line_number: int
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)
    occurrences: int = 1
    resolved: bool = False
    resolution_action: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for serialization"""
        return {
            "level": self.level,
            "message": self.message,
            "source_file": self.source_file,
            "line_number": self.line_number,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "context": self.context,
            "occurrences": self.occurrences,
            "resolved": self.resolved,
            "resolution_action": self.resolution_action
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LogAlert':
        """Create alert from dictionary"""
        alert = cls(
            level=data["level"],
            message=data["message"],
            source_file=data["source_file"],
            line_number=data["line_number"],
            timestamp=data["timestamp"],
            context=data["context"],
            occurrences=data["occurrences"]
        )
        if "resolved" in data:
            alert.resolved = data["resolved"]
        if "resolution_action" in data:
            alert.resolution_action = data["resolution_action"]
        return alert
        
    def get_hash(self) -> str:
        """Get a hash representation of this alert for deduplication"""
        key = f"{self.level}:{self.source_file}:{self.message}"
        return hashlib.md5(key.encode()).hexdigest()


class EnhancedLogMonitor:
    """
    Enhanced automated log monitoring system with proactive detection
    and automated recovery capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize enhanced log monitor with configuration"""
        self.config = config or {
            "log_dir": "logs",
            "check_interval": 15,  # seconds
            "alert_history_size": 2000,
            "pattern_file": "config/log_patterns.json",
            "notification_webhooks": [],
            "max_lines_per_check": 10000,  # Prevent excessive CPU usage
            "alert_deduplication_window": 300,  # seconds
            "enable_automatic_recovery": True,
            "recovery_scripts_dir": "monitoring/recovery_scripts",
            "alert_priority_levels": {
                "critical": 1,
                "error": 2,
                "warning": 3,
                "info": 4
            },
            "alert_retention_days": 30,
            "max_concurrent_file_checks": 10
        }
        
        # Create log directory if it doesn't exist
        self.log_dir = Path(self.config["log_dir"])
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # Create recovery scripts directory if enabled and doesn't exist
        if self.config["enable_automatic_recovery"]:
            recovery_dir = Path(self.config["recovery_scripts_dir"])
            recovery_dir.mkdir(exist_ok=True, parents=True)
        
        # Alert storage
        self.alerts: List[LogAlert] = []
        self.recent_alert_hashes: Dict[str, float] = {}
        
        # File tracking
        self.file_positions: Dict[str, int] = {}
        self.monitored_files: Set[str] = set()
        
        # Watch patterns
        self.patterns = self._load_patterns()
        
        # Monitoring state
        self.running = False
        self.monitor_task = None
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config["max_concurrent_file_checks"]
        )
        
        # Callbacks
        self.alert_callbacks: List[Callable[[LogAlert], None]] = []
        
        # Extended tracking
        self.alert_stats: Dict[str, Dict[str, Any]] = {
            "total_by_level": {"critical": 0, "error": 0, "warning": 0, "info": 0},
            "resolved_count": 0,
            "auto_resolved_count": 0,
            "recurring_issues": {}
        }

        # Resource monitor connection
        self.resource_monitor = ResourceMonitor() if ResourceMonitor else None
        
        # Debug integration
        self.debug_manager = debug_manager

    def _load_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load monitoring patterns from configuration file"""
        pattern_file = Path(self.config["pattern_file"])
        
        # Default patterns if file doesn't exist
        default_patterns = {
            "error": [
                {
                    "pattern": r"ERROR.*?Exception: (.*)",
                    "level": "error",
                    "action": "alert",
                    "recovery": "restart_component"
                },
                {
                    "pattern": r"CRITICAL: (.*)",
                    "level": "critical",
                    "action": "alert,notify",
                    "recovery": "escalate"
                }
            ],
            "resource": [
                {
                    "pattern": r"(Out of memory|MemoryError)",
                    "level": "critical",
                    "action": "alert,notify",
                    "recovery": "free_memory"
                },
                {
                    "pattern": r"(Disk full|No space left on device)",
                    "level": "critical", 
                    "action": "alert,notify",
                    "recovery": "clean_temp_files"
                }
            ],
            "security": [
                {
                    "pattern": r"(Authentication failure|Unauthorized access|Forbidden)",
                    "level": "critical",
                    "action": "alert,notify",
                    "recovery": "lock_account"
                }
            ],
            "performance": [
                {
                    "pattern": r"(Response time exceeded|Timeout occurred|Performance degraded)",
                    "level": "warning",
                    "action": "alert",
                    "recovery": "optimize_resources"
                }
            ],
            "network": [
                {
                    "pattern": r"(Connection refused|Failed to connect|Network timeout)",
                    "level": "warning",
                    "action": "alert",
                    "recovery": "retry_connection"
                },
                {
                    "pattern": r"(Socket error|Connection reset|Broken pipe)",
                    "level": "error",
                    "action": "alert",
                    "recovery": "reset_network"
                }
            ],
            "system": [
                {
                    "pattern": r"(System shutdown|Initialization failure)",
                    "level": "critical",
                    "action": "alert,notify,recover",
                    "recovery": "restart_system"
                }
            ],
            "ai_core": [
                {
                    "pattern": r"(Model loading failed|Inference error|Training diverged)",
                    "level": "error",
                    "action": "alert",
                    "recovery": "reload_model"
                }
            ],
            "database": [
                {
                    "pattern": r"(Database connection failed|Query timeout|Deadlock detected)",
                    "level": "error",
                    "action": "alert",
                    "recovery": "reconnect_db"
                }
            ]
        }
        
        # Create config directory and default pattern file if they don't exist
        if not pattern_file.exists():
            pattern_file.parent.mkdir(exist_ok=True, parents=True)
            with open(pattern_file, 'w') as f:
                json.dump(default_patterns, f, indent=2)
            logger.info(f"Created default log patterns file at {pattern_file}")
            return default_patterns
        
        # Load patterns from file
        try:
            with open(pattern_file, 'r') as f:
                patterns = json.load(f)
            logger.info(f"Loaded {sum(len(p) for p in patterns.values())} log monitoring patterns")
            return patterns
        except Exception as e:
            logger.error(f"Error loading log patterns: {e}, using defaults")
            return default_patterns
    
    async def start(self) -> None:
        """Start enhanced log monitoring"""
        if self.running:
            logger.warning("Log monitor already running")
            return
            
        logger.info("Starting enhanced automated log monitoring")
        self.running = True
        
        # Load existing alerts if available
        self._load_alert_history()
        
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        
        # Discover log files
        self._discover_log_files()
    
    async def stop(self) -> None:
        """Stop log monitoring"""
        if not self.running:
            return
            
        logger.info("Stopping enhanced automated log monitoring")
        self.running = False
        
        if self.monitor_task and not self.monitor_task.done():
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
                
        # Save alert history
        self._save_alert_history()
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
    
    def register_alert_callback(self, callback: Callable[[LogAlert], None]) -> None:
        """Register callback for log alerts"""
        self.alert_callbacks.append(callback)
    
    def _discover_log_files(self) -> None:
        """Discover log files to monitor"""
        if not self.log_dir.exists():
            logger.warning(f"Log directory {self.log_dir} does not exist")
            return
            
        # Scan all log files recursively
        for file_path in self.log_dir.glob("**/*.log"):
            self.monitored_files.add(str(file_path))
            
            # Initialize file position if not already tracked
            if str(file_path) not in self.file_positions:
                try:
                    self.file_positions[str(file_path)] = os.path.getsize(file_path)
                except (FileNotFoundError, PermissionError) as e:
                    logger.warning(f"Cannot access log file {file_path}: {e}")
        
        # Also check for new log files in app-specific directories
        common_log_dirs = ["backend/logs", "ai_core/logs", "network/logs", "system/logs"]
        for log_dir in common_log_dirs:
            dir_path = Path(log_dir)
            if dir_path.exists():
                for file_path in dir_path.glob("*.log"):
                    self.monitored_files.add(str(file_path))
                    if str(file_path) not in self.file_positions:
                        try:
                            self.file_positions[str(file_path)] = os.path.getsize(file_path)
                        except (FileNotFoundError, PermissionError) as e:
                            logger.warning(f"Cannot access log file {file_path}: {e}")
        
        logger.info(f"Monitoring {len(self.monitored_files)} log files")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        try:
            while self.running:
                start_time = time.time()
                try:
                    # Discover new log files periodically
                    self._discover_log_files()
                    
                    # Process log files concurrently
                    file_paths = list(self.monitored_files)
                    futures = []
                    
                    for file_path in file_paths:
                        future = self.thread_pool.submit(
                            self._read_log_file, 
                            file_path
                        )
                        futures.append((file_path, future))
                    
                    # Process results as they complete
                    for file_path, future in futures:
                        try:
                            new_lines = future.result()
                            if new_lines:
                                await self._process_log_lines(file_path, new_lines)
                        except Exception as e:
                            logger.error(f"Error processing {file_path}: {e}")
                    
                    # Clean up old alerts from deduplication window
                    self._clean_recent_alerts()
                    
                    # Save alert history periodically
                    if (int(time.time()) % 600) < self.config["check_interval"]:  # Every ~10 minutes
                        self._save_alert_history()
                    
                except Exception as e:
                    logger.error(f"Error in log monitoring loop: {e}")
                    logger.debug(traceback.format_exc())
                
                # Calculate sleep time to maintain consistent check interval
                elapsed = time.time() - start_time
                sleep_time = max(0.1, self.config["check_interval"] - elapsed)
                
                # Wait for next check interval
                await asyncio.sleep(sleep_time)
        
        except asyncio.CancelledError:
            logger.info("Log monitoring task cancelled")
    
    def _read_log_file(self, file_path: str) -> List[Tuple[int, str]]:
        """Read new lines from log file (runs in thread pool)"""
        try:
            if not os.path.exists(file_path):
                logger.warning(f"Log file no longer exists: {file_path}")
                self.monitored_files.discard(file_path)
                return []
                
            # Get current size
            current_size = os.path.getsize(file_path)
            last_position = self.file_positions.get(file_path, 0)
            
            # File was truncated or is new
            if current_size < last_position:
                last_position = 0
            
            if current_size == last_position:
                # No new content
                return []
                
            new_lines = []
            line_number = 0
            max_lines = self.config["max_lines_per_check"]
            
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                # Skip to last position
                f.seek(last_position)
                
                # Read new lines
                for i, line in enumerate(f):
                    if i >= max_lines:
                        logger.warning(f"Reached maximum lines ({max_lines}) for {file_path}")
                        break
                    
                    line = line.rstrip('\r\n')
                    if line:  # Skip empty lines
                        line_number = last_position + f.tell()
                        new_lines.append((line_number, line))
                
                # Update file position
                self.file_positions[file_path] = f.tell()
            
            return new_lines
            
        except Exception as e:
            logger.error(f"Error reading log file {file_path}: {e}")
            return []
    
    async def _process_log_lines(self, file_path: str, lines: List[Tuple[int, str]]) -> None:
        """Process new log lines and detect issues"""
        for line_number, line in lines:
            # Check against each pattern category
            for category, patterns in self.patterns.items():
                for pattern_def in patterns:
                    pattern = pattern_def["pattern"]
                    match = re.search(pattern, line)
                    
                    if match:
                        # Extract captured message if available
                        message = match.group(1) if match.lastindex else line
                        
                        # Create alert
                        alert = LogAlert(
                            level=pattern_def["level"],
                            message=message,
                            source_file=file_path,
                            line_number=line_number,
                            context={
                                "category": category,
                                "full_line": line,
                                "pattern": pattern
                            }
                        )
                        
                        # Check for duplicates
                        alert_hash = alert.get_hash()
                        current_time = time.time()
                        
                        if alert_hash in self.recent_alert_hashes:
                            # Update occurrence count in existing alert
                            for existing_alert in self.alerts:
                                if existing_alert.get_hash() == alert_hash and not existing_alert.resolved:
                                    existing_alert.occurrences += 1
                                    self.recent_alert_hashes[alert_hash] = current_time
                                    break
                        else:
                            # Add new alert
                            self.alerts.append(alert)
                            self.recent_alert_hashes[alert_hash] = current_time
                            
                            # Update stats
                            self.alert_stats["total_by_level"][alert.level] = (
                                self.alert_stats["total_by_level"].get(alert.level, 0) + 1
                            )
                            
                            # Track recurring issues
                            issue_key = f"{category}:{re.sub(r'\d+', 'X', message)}"
                            if issue_key in self.alert_stats["recurring_issues"]:
                                self.alert_stats["recurring_issues"][issue_key] += 1
                            else:
                                self.alert_stats["recurring_issues"][issue_key] = 1
                            
                            # Notify callbacks
                            for callback in self.alert_callbacks:
                                try:
                                    callback(alert)
                                except Exception as cb_err:
                                    logger.error(f"Error in alert callback: {cb_err}")
                            
                            # Process actions
                            await self._process_alert_actions(alert, pattern_def)
                            
                            # Trim alert history if needed
                            if len(self.alerts) > self.config["alert_history_size"]:
                                oldest_unresolved_idx = next(
                                    (i for i, a in enumerate(self.alerts) if not a.resolved),
                                    0
                                )
                                self.alerts = self.alerts[oldest_unresolved_idx:]
    
    async def _process_alert_actions(self, alert: LogAlert, pattern_def: Dict[str, Any]) -> None:
        """Process actions defined for an alert"""
        actions = pattern_def.get("action", "").split(",")
        
        for action in actions:
            action = action.strip()
            
            if action == "notify":
                self._send_notification(alert)
            
            elif action == "recover" and self.config["enable_automatic_recovery"]:
                recovery_action = pattern_def.get("recovery")
                if recovery_action:
                    success = await self._attempt_recovery(alert, recovery_action)
                    if success:
                        alert.resolved = True
                        alert.resolution_action = recovery_action
                        self.alert_stats["auto_resolved_count"] += 1
    
    def _send_notification(self, alert: LogAlert) -> None:
        """Send notification for critical alert"""
        if not self.config["notification_webhooks"]:
            return
            
        # Format notification message
        message = (
            f"ALERT: {alert.level.upper()}\n"
            f"Message: {alert.message}\n"
            f"Source: {alert.source_file}:{alert.line_number}\n"
            f"Time: {datetime.fromtimestamp(alert.timestamp).strftime('%Y-%m-%d %H:%M:%S')}"
        )

        # Non-blocking webhook notification
        asyncio.create_task(self._send_webhook_notification(message, alert))

    async def _send_webhook_notification(self, message: str, alert: LogAlert) -> None:
        """Send webhook notification (non-blocking)"""
        # This would use a proper HTTP client in production code
        for webhook_url in self.config["notification_webhooks"]:
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        webhook_url, 
                        json={"message": message, "alert": alert.to_dict()}
                    ) as response:
                        if response.status >= 400:
                            logger.warning(
                                f"Failed to send webhook notification: {response.status}"
                            )
            except ImportError:
                logger.warning("aiohttp package not available for webhook notifications")
                return
            except Exception as e:
                logger.error(f"Error sending webhook notification: {e}")

    async def _attempt_recovery(self, alert: LogAlert, recovery_action: str) -> bool:
        """Attempt to recover from an issue automatically"""
        logger.info(f"Attempting recovery action '{recovery_action}' for alert: {alert.message}")

        try:
            # Check for custom recovery script
            script_path = Path(self.config["recovery_scripts_dir"]) / f"{recovery_action}.py"
            if script_path.exists():
                # Execute custom recovery script
                return await self._run_recovery_script(script_path, alert)
            
            # Built-in recovery actions
            if recovery_action == "restart_component":
                return await self._recover_restart_component(alert)
            elif recovery_action == "free_memory":
                return self._recover_free_memory()
            elif recovery_action == "clean_temp_files":
                return self._recover_clean_temp_files()
            elif recovery_action == "retry_connection":
                return await self._recover_retry_connection(alert)
            elif recovery_action == "reset_network":
                return await self._recover_reset_network()
            elif recovery_action == "reload_model":
                return await self._recover_reload_model(alert)
            elif recovery_action == "reconnect_db":
                return await self._recover_reconnect_db(alert)
            else:
                logger.warning(f"Unknown recovery action: {recovery_action}")
                return False

        except Exception as e:
            logger.error(f"Recovery action '{recovery_action}' failed: {e}")
            return False

    async def _run_recovery_script(self, script_path: Path, alert: LogAlert) -> bool:
        """Run a custom recovery script"""
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("recovery_module", script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if hasattr(module, "recover"):
                if asyncio.iscoroutinefunction(module.recover):
                    return await module.recover(alert, logger)
                else:
                    return module.recover(alert, logger)
            else:
                logger.warning(f"Recovery script missing 'recover' function: {script_path}")
                return False

        except Exception as e:
            logger.error(f"Error running recovery script: {e}")
            return False

    async def _recover_restart_component(self, alert: LogAlert) -> bool:
        """Restart a failing component based on log file"""
        component_name = None

        # Try to identify component from file path
        file_path = Path(alert.source_file)
        if "ai_core" in str(file_path):
            component_name = "ai_core"
        elif "network" in str(file_path):
            component_name = "network"
        elif "blockchain" in str(file_path):
            component_name = "blockchain"
        elif "backend" in str(file_path):
            component_name = "backend"

        if not component_name:
            logger.warning("Could not identify component to restart")
            return False

        # Request component restart via system coordinator if available
        try:
            # This would normally use an actual API to the system coordinator
            from ai_core.system_coordinator import SystemCoordinator
            coordinator = SystemCoordinator.get_instance()
            if coordinator:
                success = await coordinator.restart_component(component_name)
                logger.info(f"Component restart {'succeeded' if success else 'failed'}: {component_name}")
                return success
        except ImportError:
            logger.warning("System coordinator not available for component restart")
            return False
        except Exception as e:
            logger.error(f"Error restarting component: {e}")
            return False

        return False

    def _recover_free_memory(self) -> bool:
        """Attempt to free memory by cleaning caches and requesting GC"""
        try:
            # Run garbage collection
            import gc
            gc.collect()

            # Clear python object caches when possible
            for module_name in sys.modules.copy():
                if module_name.startswith('vAIn_'):
                    module = sys.modules[module_name]
                    if hasattr(module, 'clear_cache'):
                        try:
                            module.clear_cache()
                        except Exception:
                            pass

            # On Windows, attempt to clear system file cache (requires admin)
            if sys.platform == "win32" and os.geteuid() == 0:
                try:
                    os.system("PowerShell -Command \"Write-VolumeCache C:\"")
                except Exception:
                    pass
            
            # On Linux, attempt to clear page cache (requires sudo)
            elif sys.platform.startswith("linux") and os.geteuid() == 0:
                try:
                    os.system("sync && echo 1 > /proc/sys/vm/drop_caches")
                except Exception:
                    pass
            
            logger.info("Memory cleanup procedure executed")
            return True
            
        except Exception as e:
            logger.error("Memory cleanup failed: %s", e)
            return False
    
    def _recover_clean_temp_files(self) -> bool:
        """Clean temporary files to free disk space"""
        try:
            temp_dirs = ["temp", "tmp", "logs/archive", "logs/old", ".cache"]
            
            for temp_dir in temp_dirs:
                if os.path.exists(temp_dir) and os.path.isdir(temp_dir):
                    # Find files older than 7 days
                    now = time.time()
                    for root, _, files in os.walk(temp_dir):
                        for f in files:
                            filepath = os.path.join(root, f)
                            try:
                                if os.path.isfile(filepath):
                                    if now - os.path.getmtime(filepath) > 7 * 86400:
                                        os.remove(filepath)
                            except Exception as e:
                                logger.debug(f"Could not remove temp file: {e}")
            
            # Archive old log files
            logs_dir = Path(self.config["log_dir"])
            if logs_dir.exists():
                archive_dir = logs_dir / "archive"
                archive_dir.mkdir(exist_ok=True)
                
                for log_file in logs_dir.glob("*.log"):
                    try:
                        # If file is older than 14 days, archive it
                        if time.time() - os.path.getmtime(log_file) > 14 * 86400:
                            # Compress and move to archive
                            import gzip
                            archive_path = archive_dir / f"{log_file.name}.{int(time.time())}.gz"
                            with open(log_file, 'rb') as f_in:
                                with gzip.open(archive_path, 'wb') as f_out:
                                    f_out.write(f_in.read())
                            os.remove(log_file)
                    except Exception as e:
                        logger.debug(f"Could not archive log file: {e}")
            
            logger.info("Temporary file cleanup executed")
            return True
            
        except Exception as e:
            logger.error(f"Temporary file cleanup failed: {e}")
            return False
    
    async def _recover_retry_connection(self, alert: LogAlert) -> bool:
        """Retry failed network connection"""
        # Extract host/endpoint from alert if possible
        host = None
        
        # Try to extract connection details from alert
        if "Connection refused" in alert.message or "Failed to connect" in alert.message:
            match = re.search(r"(?:to|with)\s+([a-zA-Z0-9.-]+(?::\d+)?)", alert.message)
            if match:
                host = match.group(1)
        
        if not host:
            return False
        
        try:
            # Simple connection retry with timeout
            import socket
            
            # Parse host and port
            if ":" in host:
                hostname, port = host.split(":")
                port = int(port)
            else:
                hostname = host
                port = 80  # Default port
            
            # Try to connect with timeout
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            result = sock.connect_ex((hostname, port))
            sock.close()
            
            # Update connection registry if available
            try:
                from network.connections import ConnectionRegistry
                registry = ConnectionRegistry.get_instance()
                if registry and result == 0:
                    await registry.refresh_connection(hostname, port)
            except ImportError:
                pass
            
            return result == 0
            
        except Exception as e:
            logger.error(f"Connection retry failed: {e}")
            return False
    
    async def _recover_reset_network(self) -> bool:
        """Reset network connections"""
        try:
            # Check for network module and call reset
            try:
                from network.service import NetworkService
                service = NetworkService.get_instance()
                if service:
                    await service.reset_connections()
                    logger.info("Network connections reset")
                    return True
            except ImportError:
                pass
            
            # Fallback: reset through system coordinator
            try:
                from ai_core.system_coordinator import SystemCoordinator
                coordinator = SystemCoordinator.get_instance()
                if coordinator:
                    success = await coordinator.reset_subsystem("network")
                    logger.info(f"Network reset {'succeeded' if success else 'failed'}")
                    return success
            except ImportError:
                pass
                
            return False
            
        except Exception as e:
            logger.error(f"Network reset failed: {e}")
            return False
    
    async def _recover_reload_model(self, alert: LogAlert) -> bool:
        """Reload AI model that failed"""
        model_name = None
        
        # Try to extract model name from alert
        match = re.search(r"model\s+['\"]([\w\-_.]+)['\"]", alert.message, re.IGNORECASE)
        if match:
            model_name = match.group(1)
        
        if not model_name:
            match = re.search(r"([\w\-_.]+) model", alert.message, re.IGNORECASE)
            if match:
                model_name = match.group(1)
        
        try:
            # Attempt to reload model via AI core
            try:
                from ai_core.model_manager import ModelManager
                manager = ModelManager.get_instance()
                if manager and model_name:
                    success = await manager.reload_model(model_name)
                    logger.info(f"Model reload {'succeeded' if success else 'failed'}: {model_name}")
                    return success
                elif manager:  # Reload default model
                    success = await manager.reload_default_model()
                    logger.info(f"Default model reload {'succeeded' if success else 'failed'}")
                    return success
            except ImportError:
                pass
                
            return False
            
        except (ConnectionError, TimeoutError, RuntimeError, AttributeError) as e:
            logger.error(f"Model reload failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during model reload: {e}")
            logger.debug(traceback.format_exc())
            return False
    
    async def _recover_reconnect_db(self, alert: LogAlert) -> bool:
        """Reconnect to database"""
        try:
            # Attempt to reconnect database
            try:
                from database.connection import DatabaseManager
                db_manager = DatabaseManager.get_instance()
                if db_manager:
                    success = await db_manager.reconnect()
                    logger.info(f"Database reconnection {'succeeded' if success else 'failed'}")
                    return success
            except ImportError:
                pass
                
            return False
            
        except Exception as e:
            logger.error(f"Database reconnection failed: {e}")
            return False
    
    def _clean_recent_alerts(self) -> None:
        """Clean up old alerts from deduplication window"""
        current_time = time.time()
        dedup_window = self.config["alert_deduplication_window"]
        
        # Remove alerts older than deduplication window
        self.recent_alert_hashes = {
            alert_hash: timestamp
            for alert_hash, timestamp in self.recent_alert_hashes.items()
            if current_time - timestamp < dedup_window
        }
    
    def _load_alert_history(self) -> None:
        """Load alert history from file"""
        history_file = self.log_dir / "alert_history.json"
        
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    
                # Load alerts
                self.alerts = [LogAlert.from_dict(alert_data) for alert_data in data["alerts"]]
                
                # Load stats
                if "stats" in data:
                    self.alert_stats = data["stats"]
                    
                logger.info(f"Loaded {len(self.alerts)} alerts from history")
                
            except Exception as e:
                logger.error(f"Error loading alert history: {e}")
    
    def _save_alert_history(self) -> None:
        """Save alert history to file"""
        history_file = self.log_dir / "alert_history.json"
        
        try:
            # Prepare data
            data = {
                "alerts": [alert.to_dict() for alert in self.alerts],
                "stats": self.alert_stats,
                "timestamp": time.time()
            }
            
            # Save to file
            with open(history_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.debug(f"Saved {len(self.alerts)} alerts to history")
            
        except Exception as e:
            logger.error(f"Error saving alert history: {e}")
    
    def get_alerts(self, 
                  level: Optional[str] = None, 
                  resolved: Optional[bool] = None,
                  limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent alerts with optional filtering"""
        filtered_alerts = self.alerts
        
        if level:
            filtered_alerts = [a for a in filtered_alerts if a.level == level]
            
        if resolved is not None:
            filtered_alerts = [a for a in filtered_alerts if a.resolved == resolved]
            
        # Sort by timestamp (newest first)
        filtered_alerts.sort(key=lambda a: a.timestamp, reverse=True)
        
        # Limit results
        filtered_alerts = filtered_alerts[:limit]
        
        return [alert.to_dict() for alert in filtered_alerts]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get alert statistics"""
        return {
            "total_alerts": len(self.alerts),
            "active_alerts": len([a for a in self.alerts if not a.resolved]),
            "alert_counts_by_level": self.alert_stats["total_by_level"],
            "resolved_count": self.alert_stats["resolved_count"],
            "auto_resolved_count": self.alert_stats["auto_resolved_count"],
            "top_recurring_issues": dict(
                sorted(
                    self.alert_stats["recurring_issues"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            )
        }
    
    def resolve_alert(self, alert_id: str, resolution_note: str) -> bool:
        """Manually resolve an alert"""
        for alert in self.alerts:
            if alert.get_hash() == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolution_action = resolution_note
                self.alert_stats["resolved_count"] += 1
                return True
                
        return False
        
    async def analyze_patterns(self) -> Dict[str, Any]:
        """Analyze alert patterns to find correlations and root causes"""
        # This would be more sophisticated in a real implementation
        if len(self.alerts) < 10:
            return {"message": "Not enough alerts to analyze patterns"}
        
        # Get recent alerts
        recent_alerts = sorted(
            [a for a in self.alerts if time.time() - a.timestamp < 86400],
            key=lambda x: x.timestamp
        )
        
        # Simple time-based correlation
        correlated_alerts = []
        time_windows = {}
        
        # Group alerts by 5-minute windows
        for alert in recent_alerts:
            window = int(alert.timestamp / 300)
            if window not in time_windows:
                time_windows[window] = []
            time_windows[window].append(alert)
        
        # Find windows with multiple alerts
        for window, alerts in time_windows.items():
            if len(alerts) >= 3:  # At least 3 alerts in a 5-minute window
                # Check if different components involved
                components = set()
                for alert in alerts:
                    file_path = alert.source_file
                    component = file_path.split(os.sep)[0] if os.sep in file_path else "unknown"
                    components.add(component)
                
                if len(components) >= 2:  # Multiple components involved
                    correlated_alerts.append({
                        "window": datetime.fromtimestamp(window * 300).isoformat(),
                        "alert_count": len(alerts),
                        "components": list(components),
                        "first_alert": alerts[0].message
                    })
        
        return {
            "correlations_found": len(correlated_alerts),
            "correlated_events": correlated_alerts,
            "total_alerts_analyzed": len(recent_alerts)
        }


# Helper function to create and configure the log monitor
def setup_log_monitor(config: Optional[Dict[str, Any]] = None) -> EnhancedLogMonitor:
    """Create and configure an enhanced log monitor instance"""
    monitor = EnhancedLogMonitor(config)
    return monitor


# Create startup script for enhanced log monitoring
async def main():
    """Run the enhanced log monitor as a standalone service"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Log Monitoring Service')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--log-dir', type=str, help='Directory for logs')
    parser.add_argument('--interval', type=int, help='Check interval in seconds')
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/log_monitor.log")
        ]
    )
    
    # Load config from file if specified
    config = None
    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
    
    # Override config with command line arguments
    if not config:
        config = {}
    
    if args.log_dir:
        config["log_dir"] = args.log_dir
    
    if args.interval:
        config["check_interval"] = args.interval
    
    # Create and start monitor
    monitor = EnhancedLogMonitor(config)
    await monitor.start()
    
    try:
        # Run indefinitely
        while True:
            await asyncio.sleep(3600)  # Check every hour if still running
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutting down log monitor")
    finally:
        await monitor.stop()

if __name__ == "__main__":
    asyncio.run(main())