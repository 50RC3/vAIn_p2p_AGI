import asyncio
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from core.constants import LOAD_BALANCING
from .edge_manager import EdgeManager

logger = logging.getLogger(__name__)

@dataclass
class NodeLoad:
    cpu: float
    memory: float
    bandwidth: float
    task_count: int

class LoadBalancer:
    def __init__(self):
        self.node_loads: Dict[str, NodeLoad] = {}
        self.lock = asyncio.Lock()
        self.edge_manager = EdgeManager()
        
    async def register_node(self, node_id: str, load: NodeLoad):
        async with self.lock:
            self.node_loads[node_id] = load
            
    async def get_best_node(self, task_requirements) -> Optional[str]:
        async with self.lock:
            best_node = None
            best_score = float('inf')
            
            for node_id, load in self.node_loads.items():
                if load.cpu > LOAD_BALANCING["max_load"]:
                    continue
                    
                score = self._calculate_score(load, task_requirements)
                if score < best_score:
                    best_score = score
                    best_node = node_id
                    
            return best_node
            
    def _calculate_score(self, load: NodeLoad, requirements) -> float:
        cpu_score = load.cpu * 0.4
        memory_score = load.memory * 0.3
        bandwidth_score = load.bandwidth * 0.3
        return cpu_score + memory_score + bandwidth_score
        
    async def rebalance_if_needed(self):
        async with self.lock:
            loads = [load for load in self.node_loads.values()]
            if not loads:
                return
                
            avg_load = sum(l.cpu for l in loads) / len(loads)
            max_load = max(l.cpu for l in loads)
            
            if max_load - avg_load > LOAD_BALANCING["rebalance_threshold"]:
                await self._trigger_rebalance()
                
    async def _trigger_rebalance(self):
        # Notify overloaded nodes to redistribute tasks
        logger.info("Triggering load rebalance")
        # Implementation details here

    async def allocate_task(self, task_id: str, required_capacity: NodeLoad) -> str:
        """Allocate task preferring edge nodes when available"""
        # Try edge nodes first
        edge_node = self.edge_manager.get_optimal_edge_node(task_id, required_capacity)
        if edge_node:
            self.edge_manager.assign_task(task_id, edge_node)
            return edge_node
            
        # Fall back to regular node allocation if no suitable edge node
        best_node = await self.get_best_node(required_capacity)
        if best_node:
            await self.register_node(best_node, required_capacity)
        return best_node
