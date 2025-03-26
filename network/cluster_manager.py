import logging
import asyncio
from typing import Dict, Set, List, Optional
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

@dataclass
class ClusterMetrics:
    load: float = 0.0
    latency: float = 0.0
    capacity: float = 0.0
    connection_count: int = 0
    message_rate: float = 0.0
    error_rate: float = 0.0
    load_balance_score: float = 0.0  # Added field
    node_distribution: Dict[str, float] = field(default_factory=dict)  # Added field
    rebalance_frequency: float = 0.0  # Added field
    leader_node: Optional[str] = None
    node_reputations: Dict[str, float] = field(default_factory=dict)

class NetworkCluster:
    def __init__(self, cluster_id: str, max_size: int = 100):
        self.cluster_id = cluster_id
        self.nodes: Set[str] = set()
        self.max_size = max_size
        self.metrics = ClusterMetrics()
        self.sub_clusters: List[NetworkCluster] = []
        self.latency_matrix: Dict[str, Dict[str, float]] = {}
        self.coordinator_node: Optional[str] = None
        self.leader_node = None
        self.node_reputations: Dict[str, float] = {}

    def should_split(self) -> bool:
        """Check if cluster should be split based on size and metrics"""
        return (len(self.nodes) > self.max_size or 
                self.metrics.latency > 150 or  # 150ms latency threshold
                self.metrics.load > 0.8)       # 80% load threshold

    def update_leader(self) -> Optional[str]:
        """Select highest reputation node as leader"""
        if not self.nodes:
            return None
        self.leader_node = max(self.nodes, key=lambda x: self.node_reputations.get(x, 0))
        return self.leader_node

    def add_node_reputation(self, node_id: str, reputation: float):
        """Update node reputation"""
        self.node_reputations[node_id] = reputation
        # Update leader if new node has higher reputation
        if not self.leader_node or reputation > self.node_reputations.get(self.leader_node, 0):
            self.leader_node = node_id

class ClusterManager:
    def __init__(self, config: Dict):
        self.clusters: Dict[str, NetworkCluster] = {}
        self.node_to_cluster: Dict[str, str] = {}
        self.max_cluster_size = config.get('max_cluster_size', 100)
        self.min_cluster_size = config.get('min_cluster_size', 10)
        self.rebalance_threshold = config.get('rebalance_threshold', 0.2)
        self.latency_threshold = config.get('latency_threshold', 150)  # ms
        self.metrics_history: Dict[str, List[ClusterMetrics]] = defaultdict(list)
        self.node_reputations: Dict[str, float] = {}
        self.leader_update_interval = config.get('leader_update_interval', 300)  # 5 min

    async def add_node(self, node_id: str, latencies: Dict[str, float], reputation: float = 0.0) -> str:
        """Add node to most suitable cluster with reputation"""
        self.node_reputations[node_id] = reputation
        best_cluster = await self._find_best_cluster(node_id, latencies)
        
        if not best_cluster:
            # Create new cluster if no suitable one exists
            cluster_id = f"cluster_{len(self.clusters)}"
            self.clusters[cluster_id] = NetworkCluster(cluster_id, self.max_cluster_size)
            best_cluster = self.clusters[cluster_id]
            
        best_cluster.nodes.add(node_id)
        best_cluster.add_node_reputation(node_id, reputation)
        self.node_to_cluster[node_id] = best_cluster.cluster_id
        
        await self._update_cluster_metrics(best_cluster)
        
        if best_cluster.should_split():
            await self._split_cluster(best_cluster)
            
        return best_cluster.cluster_id

    async def _find_best_cluster(self, node_id: str, latencies: Dict[str, float]) -> Optional[NetworkCluster]:
        """Find best cluster for node based on latency and load"""
        best_score = float('inf')
        best_cluster = None
        
        for cluster in self.clusters.values():
            if len(cluster.nodes) >= cluster.max_size:
                continue
                
            # Calculate average latency to cluster nodes
            cluster_latencies = [latencies.get(n, float('inf')) 
                               for n in cluster.nodes]
            if not cluster_latencies:
                continue
            avg_latency = sum(cluster_latencies) / len(cluster_latencies)
            
            # Score based on latency and load
            score = (avg_latency * (1 + cluster.metrics.load))
            
            if score < best_score:
                best_score = score
                best_cluster = cluster
                
        return best_cluster

    async def _split_cluster(self, cluster: NetworkCluster):
        """Split cluster while preserving leader hierarchy"""
        if len(cluster.nodes) < self.min_cluster_size * 2:
            return
            
        # Build latency matrix for K-means
        nodes = list(cluster.nodes)
        latency_matrix = np.array([
            [cluster.latency_matrix.get(n1, {}).get(n2, float('inf'))
             for n2 in nodes]
            for n1 in nodes
        ])
        
        # Perform K-means clustering
        k = 2  # Split into 2 clusters initially
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(latency_matrix)
        
        # Create new clusters
        for i in range(k):
            new_cluster_id = f"{cluster.cluster_id}_{i}"
            new_cluster = NetworkCluster(new_cluster_id, self.max_cluster_size)
            
            # Add nodes to new cluster
            cluster_nodes = [nodes[j] for j in range(len(nodes)) 
                           if clusters[j] == i]
            new_cluster.nodes.update(cluster_nodes)
            
            # Copy reputations and select leader
            for node in cluster_nodes:
                rep = cluster.node_reputations.get(node, 0.0)
                new_cluster.add_node_reputation(node, rep)
            new_cluster.update_leader()
            
            # Update mappings
            self.clusters[new_cluster_id] = new_cluster
            for node in cluster_nodes:
                self.node_to_cluster[node] = new_cluster_id
                
        # Remove old cluster
        del self.clusters[cluster.cluster_id]

    async def _update_cluster_metrics(self, cluster: NetworkCluster):
        """Update cluster metrics based on node performance"""
        metrics = ClusterMetrics()
        
        node_loads = []
        for node in cluster.nodes:
            node_metrics = await self._get_node_metrics(node)
            metrics.load += node_metrics.load
            metrics.latency = max(metrics.latency, node_metrics.latency)
            metrics.connection_count += node_metrics.connection_count
            metrics.message_rate += node_metrics.message_rate
            metrics.error_rate += node_metrics.error_rate
            
            # Track individual node loads for distribution analysis
            if node_metrics.load_metrics:
                node_loads.append(node_metrics.load_metrics.current_load)
                metrics.node_distribution[node.id] = node_metrics.load_metrics.current_load
        
        # Calculate load balancing metrics
        node_count = len(cluster.nodes)
        if node_count > 0:
            metrics.load /= node_count
            metrics.message_rate /= node_count
            metrics.error_rate /= node_count
            
            # Calculate load balance score (0-1, higher is better)
            if node_loads:
                variance = np.var(node_loads)
                metrics.load_balance_score = 1.0 / (1.0 + variance)
            
            # Calculate rebalance frequency (rebalances per hour)
            metrics.rebalance_frequency = self._calculate_rebalance_frequency(cluster)
        
        cluster.metrics = metrics
        self.metrics_history[cluster.cluster_id].append(metrics)
