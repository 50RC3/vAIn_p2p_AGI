import asyncio
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class NodeLoad:
    cpu: float
    memory: float
    bandwidth: float
    task_count: int

@dataclass
class NodeCapacity:
    cpu_available: float
    memory_available: float 
    bandwidth_available: float
    current_tasks: int

class LoadBalancer:
    def __init__(self):
        self.node_loads: Dict[str, NodeLoad] = {}
        self.node_capacities: Dict[str, NodeCapacity] = {}
        self.lock = asyncio.Lock()
        self.edge_manager = None  # Will be initialized when needed
        self.rebalance_threshold = 0.2  # 20% load difference triggers rebalance
        self.max_load = 0.8  # 80% maximum load per node
        
    async def register_node(self, node_id: str, capacity: NodeCapacity):
        """Register node capacity"""
        async with self.lock:
            self.node_capacities[node_id] = capacity
            
            # Also track current load
            self.node_loads[node_id] = NodeLoad(
                cpu=100 - capacity.cpu_available,
                memory=100 - capacity.memory_available,
                bandwidth=100 - capacity.bandwidth_available,
                task_count=capacity.current_tasks
            )
            
    async def get_best_node(self, task_requirements) -> Optional[str]:
        """Get best node for task based on available capacity"""
        async with self.lock:
            best_node = None
            best_score = float('inf')
            
            for node_id, capacity in self.node_capacities.items():
                # Skip overloaded nodes
                if capacity.cpu_available < 20:  # Less than 20% CPU available
                    continue
                    
                score = self._calculate_score(
                    NodeLoad(
                        cpu=100 - capacity.cpu_available,
                        memory=100 - capacity.memory_available,
                        bandwidth=100 - capacity.bandwidth_available,
                        task_count=capacity.current_tasks
                    ), 
                    task_requirements
                )
                
                if score < best_score:
                    best_score = score
                    best_node = node_id
                    
            return best_node
    
    def _calculate_score(self, load: NodeLoad, requirements) -> float:
        """Calculate node score based on load and requirements"""
        cpu_score = load.cpu * 0.4
        memory_score = load.memory * 0.3
        bandwidth_score = load.bandwidth * 0.3
        return cpu_score + memory_score + bandwidth_score
        
    async def rebalance_if_needed(self):
        """Check if rebalancing is needed and trigger if necessary"""
        async with self.lock:
            loads = [load for load in self.node_loads.values()]
            if not loads:
                return
                
            avg_load = sum(l.cpu for l in loads) / len(loads)
            max_load = max(l.cpu for l in loads)
            
            if max_load - avg_load > self.rebalance_threshold * 100:
                await self._trigger_rebalance()
                
    async def _trigger_rebalance(self):
        """Perform load rebalancing across nodes"""
        logger.info("Triggering load rebalance")
        # Implementation would redistribute tasks from highly loaded nodes
        # to lightly loaded ones
                
    async def allocate_task(self, task_id: str, required_capacity: NodeCapacity) -> str:
        """Allocate task to appropriate node"""
        # Try edge nodes first if available
        if self.edge_manager:
            edge_node = self.edge_manager.get_optimal_edge_node(task_id, required_capacity)
            if edge_node:
                return edge_node
            
        # Fall back to regular node allocation
        best_node = await self.get_best_node(required_capacity)
        return best_node
        
    def get_cluster_distribution(self) -> Dict[str, List[str]]:
        """Get distribution of nodes by cluster"""
        # This would work with cluster manager to organize nodes
        # For now, return empty result
        return {}
        
    async def adjust_capacity_thresholds(self, usage_metrics: Dict):
        """Dynamically adjust capacity thresholds based on usage patterns"""
        if 'average_cpu' in usage_metrics:
            # Adjust thresholds based on network-wide metrics
            self.max_load = min(0.9, max(0.6, 0.8 + (0.5 - usage_metrics['average_cpu']) * 0.2))
            logger.debug(f"Adjusted max load threshold to {self.max_load}")
