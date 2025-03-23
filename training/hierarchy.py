from typing import List, Optional, Dict, Any
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import time

logger = logging.getLogger(__name__)

@dataclass
class NodeMetrics:
    latency: float
    timestamp: float
    status: str = 'active'

class NodeCluster:
    def __init__(self, target_size: int = 10):
        self.nodes = []
        self.target_size = target_size
        self.max_depth = 3
        self.sub_clusters = []
        self.network_latency = []
        self.node_metrics: Dict[str, NodeMetrics] = {}
        self.last_rebalance = time()
        self.rebalance_interval = 300  # 5 minutes
        self.executor = ThreadPoolExecutor(max_workers=4)

    def add_node(self, node: Any) -> bool:
        """Add node and adjust hierarchy if needed"""
        try:
            if not self._validate_node(node):
                logger.error(f"Invalid node: {node}")
                return False

            # Update node metrics
            latency = self._measure_latency(node)
            self.node_metrics[node.id] = NodeMetrics(
                latency=latency,
                timestamp=time()
            )

            if len(self.nodes) < self.target_size:
                self.nodes.append(node)
                self._update_latency_matrix()
                return True

            if not self.sub_clusters and len(self.nodes) == self.target_size:
                self._split_cluster()

            return self._add_to_sub_cluster(node)

        except Exception as e:
            logger.error(f"Error adding node: {e}")
            return False

    def _validate_node(self, node: Any) -> bool:
        """Validate node has required attributes and capabilities"""
        required_attrs = ['id', 'status', 'network_info']
        return all(hasattr(node, attr) for attr in required_attrs)

    def _measure_latency(self, node: Any) -> float:
        """Measure network latency to node"""
        try:
            # Implement actual latency measurement
            latency = node.ping() if hasattr(node, 'ping') else 0.0
            return latency
        except Exception as e:
            logger.warning(f"Latency measurement failed: {e}")
            return float('inf')

    def _update_latency_matrix(self):
        """Update network latency matrix for all nodes"""
        n = len(self.nodes)
        self.network_latency = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                latency = self._get_latency(self.nodes[i], self.nodes[j])
                self.network_latency[i][j] = latency
                self.network_latency[j][i] = latency

    def _get_latency(self, node1: Any, node2: Any) -> float:
        """Get cached latency between nodes or measure if needed"""
        key = f"{node1.id}-{node2.id}"
        rev_key = f"{node2.id}-{node1.id}"

        metrics1 = self.node_metrics.get(node1.id)
        metrics2 = self.node_metrics.get(node2.id)

        if metrics1 and metrics2:
            return (metrics1.latency + metrics2.latency) / 2
        return self._measure_latency(node2)

    def _split_cluster(self):
        """Split cluster when it exceeds target size"""
        if len(self.sub_clusters) >= self.max_depth:
            self.target_size *= 2  # Grow cluster instead of adding depth
            logger.info(f"Increased target size to {self.target_size}")
            return

        try:
            # Create new sub-clusters based on network latency
            latency_matrix = np.array(self.network_latency)
            clusters = self._cluster_by_latency(latency_matrix)

            for cluster_nodes in clusters:
                new_cluster = NodeCluster(self.target_size)
                new_cluster.nodes = cluster_nodes
                self.sub_clusters.append(new_cluster)

            # Update metrics for new clusters
            for cluster in self.sub_clusters:
                cluster._update_latency_matrix()

        except Exception as e:
            logger.error(f"Cluster split failed: {e}")
            self.target_size *= 2  # Fallback: grow cluster

    def _add_to_sub_cluster(self, node: Any) -> bool:
        """Add node to most suitable sub-cluster"""
        if not self.sub_clusters:
            return False

        # Find sub-cluster with lowest average latency
        best_cluster = min(
            self.sub_clusters,
            key=lambda c: np.mean([self._get_latency(node, n) for n in c.nodes])
        )
        return best_cluster.add_node(node)

    def _cluster_by_latency(self, latency_matrix: np.ndarray) -> List[List]:
        """Group nodes by network latency using k-means"""
        from sklearn.cluster import KMeans
        k = max(2, len(self.nodes) // self.target_size)
        kmeans = KMeans(n_clusters=k)
        clusters = kmeans.fit_predict(latency_matrix)
        
        return [
            [self.nodes[i] for i in range(len(self.nodes)) if clusters[i] == j]
            for j in range(k)
        ]

    def cleanup(self):
        """Clean up resources"""
        self.executor.shutdown(wait=True)
        for cluster in self.sub_clusters:
            cluster.cleanup()
