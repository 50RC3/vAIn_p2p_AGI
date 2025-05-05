"""
Enhanced log monitoring system for vAIn P2P AGI.
Provides real-time log analysis, issue detection and automated recovery capabilities.
"""

import os
import re
import time
import json
import logging
import asyncio
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
import concurrent.futures
import psutil

logger = logging.getLogger(__name__)

@dataclass
class LogAlert:
    """Alert generated from log monitoring"""
    level: str
    message: str
    source_file: str
    timestamp: float
    line_number: int = 0
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


class LogMonitor:
    """
    Enhanced automated log monitoring system with proactive detection
    and automated recovery capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize log monitor with configuration"""
        self.config = config or {
            "log_dir": "logs",
            "check_interval": 15,  # seconds
            "pattern_file": "config/log_patterns.json",
            "notification_webhooks": [],
            "max_lines_per_check": 10000,
            "enable_automatic_recovery": True,
            "recovery_scripts_dir": "monitoring/recovery_scripts",
            "alert_priority_levels": {
                "critical": 1,
                "error": 2,
                "warning": 3,
                "info": 4
            }
        }
        
        # Create log directory if it doesn't exist
        self.log_dir = Path(self.config["log_dir"])
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # Create recovery scripts directory if enabled
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
            max_workers=10
        )
        
        # Callbacks for integration
        self.alert_callbacks: List[Callable[[LogAlert], None]] = []
        
        # Stats tracking
        self.stats = {
            "total_alerts": 0,
            "critical": 0,
            "error": 0,
            "warning": 0,
            "info": 0,
            "auto_resolved": 0,
            "start_time": time.time()
        }
    
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
            "ai_core": [
                {
                    "pattern": r"(Model loading failed|Inference error|Training diverged)",
                    "level": "error",
                    "action": "alert",
                    "recovery": "reload_model"
                }
            ]
        }
        
        # Create pattern file with defaults if it doesn't exist
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
        """Start log monitoring"""
        if self.running:
            logger.warning("Log monitor already running")
            return
            
        logger.info("Starting automated log monitoring")
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
            
        logger.info("Stopping automated log monitoring")
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
        self.thread_pool.shutdown(wait=False)
    
    def register_callback(self, callback: Callable[[LogAlert], None]) -> None:
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
        
        # Add main app log from root directory
        main_log_file = Path("vain_p2p.log")
        if main_log_file.exists():
            self.monitored_files.add(str(main_log_file))
            if str(main_log_file) not in self.file_positions:
                try:
                    self.file_positions[str(main_log_file)] = os.path.getsize(main_log_file)
                except Exception:
                    pass
        
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
                        message = match.group(1) if match.groups() else line
                        
                        # Create alert
                        alert = LogAlert(
                            level=pattern_def["level"],
                            message=message,
                            source_file=file_path,
                            line_number=line_number,
                            timestamp=time.time(),
                            context={
                                "category": category,
                                "full_line": line,
                                "pattern": pattern
                            }
                        )
                        
                        # Add to alerts list
                        self.alerts.append(alert)
                        
                        # Update stats
                        self.stats["total_alerts"] += 1
                        self.stats[alert.level] = self.stats.get(alert.level, 0) + 1
                            
                        # Notify callbacks
                        for callback in self.alert_callbacks:
                            try:
                                callback(alert)
                            except Exception as cb_err:
                                logger.error(f"Error in alert callback: {cb_err}")
                            
                        # Process actions
                        await self._process_alert_actions(alert, pattern_def)
                        
                        # Trim alert history if needed
                        if len(self.alerts) > 1000:
                            self.alerts = self.alerts[-1000:]
    
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
                        self.stats["auto_resolved"] += 1
    
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
        
        # Async webhook notification (non-blocking)
        asyncio.create_task(self._send_webhook_notification(message, alert))
    
    async def _send_webhook_notification(self, message: str, alert: LogAlert) -> None:
        """Send webhook notification (non-blocking)"""
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
            
            # Handle built-in recovery actions
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
        """Restart a failing component based on log file.
        
        This method attempts to identify which component is failing based on
        the log file path and then uses the system coordinator to restart
        that specific component.
        
        Args:
            alert: The alert that triggered this recovery action
            
        Returns:
            bool: True if the component was successfully restarted
        """
        component_name = None
        
        # Try to identify component from file path
        file_path = Path(alert.source_file)
        component_parts = str(file_path).split(os.sep)
        
        # Look for common component names in path
        for part in component_parts:
            if part in ["ai_core", "network", "memory", "models", "training"]:
                component_name = part
                break
        
        if not component_name:
            logger.warning("Could not identify component to restart")
            return False
        
        try:
            # Try to import SystemCoordinator for component restart
            from ai_core.system_coordinator import SystemCoordinator
            
            coordinator = None
            try:
                # Try to get instance if it's a singleton
                if hasattr(SystemCoordinator, 'get_instance'):
                    coordinator = SystemCoordinator.get_instance()
                else:
                    # Otherwise create a new instance
                    coordinator = SystemCoordinator()
            except Exception:
                logger.error("Failed to get SystemCoordinator instance")
                return False
                
            if coordinator:
                try:
                    # Attempt restart using system coordinator
                    success = await coordinator.restart_component(component_name)
                    logger.info(f"Component restart {'succeeded' if success else 'failed'}: {component_name}")
                    return success
                except Exception as e:
                    logger.error(f"Error restarting component: {e}")
                    return False
            
        except ImportError:
            logger.warning("SystemCoordinator not available, cannot restart component")
            return False
            
        return False
    
    def _recover_free_memory(self) -> bool:
        """Attempt to free memory by cleaning caches and requesting GC"""
        try:
            import gc
            logger.info("Running memory cleanup procedure")
            
            # Run garbage collection
            collected = gc.collect()
            logger.info(f"Garbage collection freed {collected} objects")
            
            # Try to reduce memory usage in common modules
            try:
                import torch
                if hasattr(torch, 'cuda') and hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                    logger.info("Cleared PyTorch CUDA cache")
            except ImportError:
                pass
            
            return True
        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")
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
        try:
            # Extract host/endpoint from alert if possible
            host = None
            
            # Try to extract connection details from alert
            if "Connection refused" in alert.message or "Failed to connect" in alert.message:
                match = re.search(r"(?:to|with)\s+([a-zA-Z0-9.-]+(?::\d+)?)", alert.message)
                if match:
                    host = match.group(1)
            try:
                from ai_core.system_coordinator import SystemCoordinator
                coordinator = None
                if hasattr(SystemCoordinator, 'get_instance'):
                    coordinator = SystemCoordinator.get_instance()
                else:
                    coordinator = SystemCoordinator()
                if coordinator:
                    success = await coordinator.restart_component("network")
                    logger.info(f"Network component restart: {'success' if success else 'failed'}")
                    return success
                coordinator = SystemCoordinator.get_instance()
                if coordinator:
                    success = await coordinator.restart_component("network")
                    logger.info(f"Network component restart: {'success' if success else 'failed'}")
                    return success
            except ImportError:
                logger.warning("SystemCoordinator not available for network retry")
            
            logger.info(f"Attempted connection retry to {host}")
            return True
        except Exception as e:
            logger.error(f"Connection retry failed: {e}")
            return False

    async def _recover_reset_network(self) -> bool:
        """Reset network components after failure"""
        try:
            # Try to reset network through system coordinator
            try:
                from ai_core.system_coordinator import SystemCoordinator
                coordinator = None
                if hasattr(SystemCoordinator, 'get_instance'):
                    coordinator = SystemCoordinator.get_instance()
                else:
                    coordinator = SystemCoordinator()
                if coordinator:
                    success = await coordinator.restart_component("network")
                    logger.info(f"Network reset: {'success' if success else 'failed'}")
                    return success
            except ImportError:
                logger.warning("SystemCoordinator not available for network reset")
            
            logger.info("Attempted network connection reset")
            return True
            
        except Exception as e:
            logger.error(f"Network reset failed: {e}")
            return False
    
    async def _recover_reload_model(self, alert: LogAlert) -> bool:
        """Reload AI model after failure"""
        try:
            # Try to find model name in error message
            model_name = None
            match = re.search(r"model\s+['\"](.*?)['\"]", alert.message, re.IGNORECASE)
            if match:
                model_name = match.group(1)
            
            # Try to reload model via system coordinator
            try:
                from ai_core.system_coordinator import SystemCoordinator
                coordinator = SystemCoordinator.get_instance()
                if coordinator:
                    # Use specific model name if available, otherwise restart ai_core
                    if model_name:
                        logger.info(f"Attempting to reload model: {model_name}")
                    success = await coordinator.restart_component("ai_core")
                    logger.info(f"AI core restart: {'success' if success else 'failed'}")
                    return success
            except ImportError:
                logger.warning("SystemCoordinator not available for model reload")
            
            return False
            
        except Exception as e:
            logger.error(f"Model reload failed: {e}")
            return False
    
    def _load_alert_history(self) -> None:
            """Load alert history from file"""
            history_file = self.log_dir / "alert_history.json"
            
            if history_file.exists():
                try:
                    with open(history_file, 'r') as f:
                        data = json.load(f)
                    for alert_data in data["alerts"]:
                        alert = LogAlert(
                            level=alert_data.get("level", "info"),
                            message=alert_data.get("message", ""),
                            source_file=alert_data.get("source_file", "unknown"),
                            line_number=alert_data.get("line_number", 0),
                            timestamp=alert_data.get("timestamp", time.time()),
                            context=alert_data.get("context", {}),
                            occurrences=alert_data.get("occurrences", 1),
                            resolved=alert_data.get("resolved", False),
                            resolution_action=alert_data.get("resolution_action")
                        )
                        self.alerts.append(alert)
                
                # Load stats
                if "stats" in data:
                    self.stats = data["stats"]
                    self.stats["start_time"] = time.time()
                    
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
                "stats": self.stats
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
        uptime = time.time() - self.stats["start_time"]
        return {
            "total_alerts": self.stats["total_alerts"],
            "by_level": {
                "critical": self.stats.get("critical", 0),
                "error": self.stats.get("error", 0),
                "warning": self.stats.get("warning", 0),
                "info": self.stats.get("info", 0)
            },
            "auto_resolved": self.stats.get("auto_resolved", 0),
            "uptime_seconds": uptime,
            "uptime_formatted": str(timedelta(seconds=int(uptime)))
        }
    
    def resolve_alert(self, alert_id: str, resolution_note: str) -> bool:
        """Manually resolve an alert by ID"""
        try:
            alert_id = int(alert_id)
            if 0 <= alert_id < len(self.alerts):
                self.alerts[alert_id].resolved = True
                self.alerts[alert_id].resolution_action = resolution_note
                return True
        except (ValueError, IndexError):
            pass
            
        return False

    def process_system_metrics(self):
        """Process system metrics and detect anomalies"""
        try:
            # Process CPU, memory, disk metrics
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Check for resource issues
            if cpu_percent > 95:
                self.add_alert("high_cpu", f"CPU usage at {cpu_percent}%", level="warning")
                
            if memory.percent > 90:
                self.add_alert("high_memory", f"Memory usage at {memory.percent}%", level="warning")
                
            if disk.percent > 95:
                self.add_alert("low_disk", f"Disk space at {disk.percent}% used", level="warning")
                
        except Exception as e:
            logger.error("Error processing system metrics: %s", e)


# Helper function to create and start log monitor
async def setup_log_monitor(config=None):
    """Create and start a log monitor instance"""
    monitor = LogMonitor(config)
    await monitor.start()
    return monitor
