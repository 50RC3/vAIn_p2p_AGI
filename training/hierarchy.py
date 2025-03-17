from typing import List
import numpy as np

class NodeCluster:
    def __init__(self, target_size=10):
        self.nodes = []
        self.target_size = target_size
        self.max_depth = 3
        self.sub_clusters = []
        self.network_latency = []

    def add_node(self, node) -> bool:
        """Add node and adjust hierarchy if needed"""
        if len(self.nodes) < self.target_size:
            self.nodes.append(node)
            return True
            
        if not self.sub_clusters and len(self.nodes) == self.target_size:
            self._split_cluster()
            
        return self._add_to_sub_cluster(node)

    def _split_cluster(self):
        """Split cluster when it exceeds target size"""
        if len(self.sub_clusters) >= self.max_depth:
            self.target_size *= 2  # Grow cluster instead of adding depth
            return
            
        # Create new sub-clusters based on network latency
        latency_matrix = np.array(self.network_latency)
        clusters = self._cluster_by_latency(latency_matrix)
        
        for cluster_nodes in clusters:
            new_cluster = NodeCluster(self.target_size)
            new_cluster.nodes = cluster_nodes
            self.sub_clusters.append(new_cluster)

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
