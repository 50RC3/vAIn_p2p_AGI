from typing import List, Dict
import asyncio

class TrainingCoordinator:
    def __init__(self, min_nodes: int = 3):
        self.min_nodes = min_nodes
        self.active_nodes = {}
        self.training_round = 0
        
    async def coordinate_round(self, nodes: List[str]) -> Dict:
        if len(nodes) < self.min_nodes:
            raise ValueError(f"Insufficient nodes: {len(nodes)} < {self.min_nodes}")
            
        self.training_round += 1
        selected_nodes = self._select_nodes(nodes)
        
        results = await asyncio.gather(*[
            self._train_on_node(node) 
            for node in selected_nodes
        ])
        
        return self._aggregate_results(results)
        
    def _select_nodes(self, nodes: List[str]) -> List[str]:
        # Node selection logic based on reputation and stake
        pass
