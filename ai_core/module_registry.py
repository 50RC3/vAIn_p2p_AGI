# Refactored module_registry.py with separation of concerns

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Set, Type, Callable, Union, TypedDict
import json
import os
import inspect
import threading

from .resource_management import ResourceManager

logger = logging.getLogger(__name__)

class ModuleRegistryError(Exception):
    """Base exception for module registry errors"""
    pass

class ModuleConfig(TypedDict, total=False):
    class_name: str
    module_name: str
    config: Dict[str, Any]
    status: str
    registered_time: float
    last_status_update: float
    dependencies: List[str]

class ModulesDict(TypedDict, total=False):
    modules: Dict[str, ModuleConfig]

class ConfigManager:
    """Handles loading and saving module configuration"""
    def __init__(self, config_path: str):
        self.config_path = config_path

    async def load_configuration(self) -> ModulesDict:
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            if os.path.exists(self.config_path):
                logger.info(f"Loading module configuration from {self.config_path}")
                with open(self.config_path, "r") as f:
                    config = json.load(f)
                return config
        except Exception as e:
            logger.error(f"Error loading module configuration: {e}")
        return {}

    async def save_configuration(self, modules: Dict[str, ModuleConfig]) -> None:
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            config = {
                "modules": modules,
                "last_updated": time.time()
            }
            with open(self.config_path, "w") as f:
                json.dump(config, f, indent=2)
            logger.info(f"Saved module configuration to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving module configuration: {e}")

class DependencyResolver:
    """Resolves module dependencies and computes startup order"""
    def __init__(self, modules: Dict[str, Any], dependencies: Dict[str, List[str]]):
        self.modules = modules
        self.dependencies = dependencies
        self.startup_order: List[str] = []

    def compute_startup_order(self) -> List[str]:
        visited: Set[str] = set()
        startup_order: List[str] = []

        def visit(module_id: str, path: Set[str] = None):
            if path is None:
                path = set()
            if module_id in path:
                raise ModuleRegistryError(f"Circular dependency detected: {' -> '.join(path)} -> {module_id}")
            if module_id in visited:
                return
            path.add(module_id)
            for dependency in self.dependencies.get(module_id, []):
                if dependency not in self.modules:
                    logger.warning(f"Missing dependency: {dependency} for module {module_id}")
                    continue
                visit(dependency, path.copy())
            visited.add(module_id)
            startup_order.append(module_id)

        for module_id in self.modules:
            if module_id not in visited:
                visit(module_id)
        self.startup_order = startup_order
        logger.info(f"Computed startup order: {', '.join(startup_order)}")
        return startup_order

class LifecycleManager:
    """Manages initialization and shutdown of modules"""
    def __init__(self, modules: Dict[str, Any], startup_order: List[str]):
        self.modules = modules
        self.startup_order = startup_order
        self.is_initialized = False

    async def initialize_modules(self):
        if self.is_initialized:
            logger.warning("Modules already initialized")
            return
        for module_name in self.startup_order:
            module = self.modules[module_name]
            if hasattr(module, 'initialize'):
                try:
                    init_method = getattr(module, 'initialize')
                    if inspect.iscoroutinefunction(init_method):
                        await init_method()
                    else:
                        init_method()
                    logger.info(f"Initialized module: {module_name}")
                except Exception as e:
                    logger.error(f"Failed to initialize module {module_name}: {e}")
        self.is_initialized = True

    async def shutdown_modules(self):
        if not self.is_initialized:
            logger.warning("Modules not initialized, nothing to shut down")
            return
        for module_name in reversed(self.startup_order):
            module = self.modules[module_name]
            if hasattr(module, 'shutdown'):
                try:
                    shutdown_method = getattr(module, 'shutdown')
                    if inspect.iscoroutinefunction(shutdown_method):
                        await shutdown_method()
                    else:
                        shutdown_method()
                    logger.info(f"Shutdown module: {module_name}")
                except Exception as e:
                    logger.error(f"Error shutting down module {module_name}: {e}")
        self.is_initialized = False

class MetricsTracker:
    """Tracks metrics related to modules and system"""
    def __init__(self, modules: Dict[str, Any], resource_manager: Optional[ResourceManager], metrics_collector: Any):
        self.modules = modules
        self.resource_manager = resource_manager
        self.metrics_collector = metrics_collector

    async def track_registration(self, module_id: str, module_class: Type) -> None:
        if self.metrics_collector:
            try:
                timestamp = time.time()
                await self.metrics_collector._add_metric_point(
                    "module_registrations",
                    1.0,
                    timestamp,
                    {"module_id": module_id, "class": module_class.__name__}
                )
                await self.metrics_collector._add_metric_point(
                    f"module_{module_id}_status",
                    1.0,
                    timestamp,
                    {"status": "registered"}
                )
            except Exception as e:
                logger.warning(f"Failed to add module registration metric: {e}")

    async def track_status_change(self, module_id: str, status: str) -> None:
        if self.metrics_collector:
            try:
                timestamp = time.time()
                status_code = {
                    "registered": 1.0,
                    "initializing": 2.0,
                    "active": 3.0,
                    "suspended": 4.0,
                    "error": 5.0,
                    "terminated": 6.0
                }.get(status, 0.0)
                await self.metrics_collector._add_metric_point(
                    f"module_{module_id}_status",
                    status_code,
                    timestamp,
                    {"status": status}
                )
            except Exception as e:
                logger.warning(f"Failed to add module status metric: {e}")

    async def track_shutdown(self) -> None:
        if self.metrics_collector:
            try:
                timestamp = time.time()
                await self.metrics_collector._add_metric_point(
                    "registry_shutdown",
                    1.0,
                    timestamp,
                    {"modules_count": len(self.modules)}
                )
            except Exception as e:
                logger.warning(f"Failed to add shutdown metric: {e}")
