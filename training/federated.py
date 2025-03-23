import torch
from typing import List, Dict, Optional, Tuple
import copy
from torch import nn, optim
from torch.utils.data import DataLoader
from .federated_client import FederatedClient
from .aggregation import aggregate_models
import logging
from .compression import compress_gradients, decompress_gradients
from .hierarchy import NodeCluster
from .metrics import calculate_data_quality

logger = logging.getLogger(__name__)

class FederatedTrainingError(Exception):
    pass

class FederatedLearning:
    def __init__(self, config):
        if not hasattr(config, 'min_clients') or not hasattr(config, 'clients_per_round'):
            raise ValueError("Config missing required attributes: min_clients, clients_per_round")
            
        self.config = config
        self.clients = []  # Initialize empty client list
        self.global_model = None
        self.lr = getattr(config, 'learning_rate', 0.01)
        self.local_epochs = getattr(config, 'local_epochs', 1)
        self.rounds = getattr(config, 'rounds', 10)
        self.criterion = getattr(config, 'criterion', nn.CrossEntropyLoss())
        
        # Validate config parameters
        if self.config.min_clients < 2:
            raise ValueError("min_clients must be at least 2")
        if self.config.clients_per_round < self.config.min_clients:
            raise ValueError("clients_per_round must be >= min_clients")
            
        self.clusters = self._initialize_clusters()
        self.aggregation_threshold = 0.01  # Only sync gradients > 1%
        self.byzantine_threshold = min(0.33, getattr(config, 'byzantine_threshold', 0.33))
        self.min_byzantine_threshold = 0.15
        self.max_byzantine_threshold = 0.49
        self.byzantine_window = getattr(config, 'byzantine_window', 100)
        self.krum_neighbors = max(2, int(len(self.clients) * (1 - self.byzantine_threshold)))
        
        # Initialize tracking containers
        self.fraud_history = []
        self.error_accumulator = {}
        
        # Compression settings with validation
        self.compression_rate = max(0.01, min(0.3, getattr(config, 'compression_rate', 0.01)))
        self.min_compression = getattr(config, 'min_compression', 0.01)
        self.max_compression = getattr(config, 'max_compression', 0.3)
        self.local_cluster_size = max(2, min(10, getattr(config, 'local_cluster_size', 10)))
        self.data_quality_threshold = max(0.1, min(1.0, getattr(config, 'data_quality_threshold', 0.7)))
        
        # Initialize compression
        self.compression = AdaptiveCompression()
        
        logger.info("FederatedLearning initialized with %d clients minimum", self.config.min_clients)

    def _validate_state(self):
        """Validate internal state before operations"""
        if not self.clients:
            raise FederatedTrainingError("No clients registered")
        if not self.global_model and not hasattr(self.config, 'model_initializer'):
            raise FederatedTrainingError("No global model or model initializer defined")

    def _initialize_clusters(self) -> List[NodeCluster]:
        # Create hierarchical node clusters
        clusters = []
        # ...organize nodes into local clusters of 5-10 nodes each
        return clusters

    def register_client(self, client: FederatedClient):
        self.clients.append(client)
        
    def client_update(self, client_data: DataLoader) -> nn.Module:
        """Update client model using local data."""
        if not self.global_model:
            raise ValueError("Global model not initialized")
            
        local_model = copy.deepcopy(self.global_model)
        optimizer = optim.SGD(local_model.parameters(), lr=self.lr)

        for _ in range(self.local_epochs):
            for x, y in client_data:
                optimizer.zero_grad()
                output, _ = local_model(x)
                loss = self.criterion(output, y)
                loss.backward()
                optimizer.step()

        return local_model

    def train(self) -> nn.Module:
        """Execute federated training across all clients."""
        self._validate_state()
        
        try:
            if len(self.clients) < self.config.min_clients:
                msg = f"Insufficient clients: {len(self.clients)} < {self.config.min_clients}"
                logger.error(msg)
                raise FederatedTrainingError(msg)
            
            for round in range(self.rounds):
                logger.info(f"Starting training round {round + 1}/{self.rounds}")
                active_clients = self.select_clients()
                
                local_models = []
                for client in active_clients:
                    try:
                        local_model = self.client_update(client.data_loader)
                        local_models.append(local_model)
                    except Exception as e:
                        logger.error(f"Error training client: {e}")
                        continue
                        
                if not local_models:
                    msg = "No successful client updates in round"
                    logger.error(msg)
                    raise FederatedTrainingError(msg)
                    
                self.global_model = self._aggregate(local_models)
                logger.info(f"Round {round+1} complete")
                
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise FederatedTrainingError(f"Training failed: {str(e)}")
            
        logger.info("Training completed successfully")
        return self.global_model

    def _update_byzantine_threshold(self):
        """Dynamically adjust Byzantine threshold based on observed fraud"""
        if len(self.fraud_history) < self.byzantine_window:
            return
            
        recent_frauds = self.fraud_history[-self.byzantine_window:]
        fraud_rate = sum(recent_frauds) / len(recent_frauds)
        
        # Adjust threshold based on observed fraud rate
        new_threshold = min(
            self.max_byzantine_threshold,
            max(
                self.min_byzantine_threshold,
                fraud_rate * 1.5  # Set threshold 50% higher than observed rate
            )
        )
        
        self.byzantine_threshold = new_threshold
        self.krum_neighbors = max(2, int(len(self.clients) * (1 - self.byzantine_threshold)))

    def _compress_update(self, model_update: Dict[str, torch.Tensor]) -> Tuple[Dict, Dict]:
        compressed = {}
        errors = {}
        for key, tensor in model_update.items():
            # Get accumulated errors for this layer
            acc_error = self.error_accumulator.get(key, torch.zeros_like(tensor))
            # Add accumulated error to current update
            tensor = tensor + acc_error
            # Compress and track new errors
            compressed[key], errors[key] = compress_gradients(
                tensor, 
                self.compression_rate
            )
            self.error_accumulator[key] = errors[key]
        return compressed

    def _aggregate(self, local_models: List[nn.Module]) -> nn.Module:
        """Hierarchical aggregation of model updates"""
        try:
            # Update Byzantine threshold based on history
            self._update_byzantine_threshold()
            
            # Detect and filter malicious updates using Krum algorithm
            distances = self._compute_pairwise_distances(local_models)
            good_model_indices = self._krum_select(distances)
            
            # Record fraud metrics
            fraud_ratio = 1 - (len(good_model_indices) / len(local_models))
            self.fraud_history.append(fraud_ratio)
            
            filtered_models = [local_models[i] for i in good_model_indices]
            
            # Calculate data quality scores for each model
            quality_scores = [
                calculate_data_quality(model) 
                for model in filtered_models
            ]
            
            # Filter models below quality threshold
            quality_filtered = [
                (model, score) for model, score in zip(filtered_models, quality_scores)
                if score >= self.data_quality_threshold
            ]
            
            if not quality_filtered:
                raise ValueError("No models meet quality threshold")

            # Weighted aggregation based on quality scores
            models, scores = zip(*quality_filtered)
            weights = torch.softmax(torch.tensor(scores), dim=0)

            # Split nodes into local clusters
            clusters = [models[i:i + self.local_cluster_size] 
                        for i in range(0, len(models), self.local_cluster_size)]
            cluster_weights = [weights[i:i + self.local_cluster_size]
                             for i in range(0, len(weights), self.local_cluster_size)]

            # First level aggregation within clusters
            cluster_models = []
            for cluster, cluster_weight in zip(clusters, cluster_weights):
                updates = [model.state_dict() for model in cluster]
                compressed_updates = [
                    self.compression.compress_model_updates(update)[0] 
                    for update in updates
                ]
                # Aggregate compressed updates within cluster
                cluster_aggregate = self._aggregate_compressed_updates(compressed_updates)
                cluster_models.append(cluster_aggregate)
            
            # Final aggregation across clusters
            final_model = copy.deepcopy(local_models[0])
            final_update = self._aggregate_compressed_updates(cluster_models)
            final_model.load_state_dict(
                self.compression.decompress_model_updates(final_update)
            )
            return final_model
        except Exception as e:
            logger.error(f"Model aggregation failed: {str(e)}")
            raise FederatedTrainingError(f"Aggregation failed: {str(e)}")

    def _aggregate_compressed_updates(self, compressed_updates: List[Dict]) -> Dict:
        # Aggregate while preserving sparsity
        # ...existing code...

    def _compute_pairwise_distances(self, models: List[nn.Module]) -> torch.Tensor:
        n = len(models)
        distances = torch.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = self._model_distance(models[i], models[j])
                distances[i][j] = distances[j][i] = dist
                
        return distances

    def _model_distance(self, model1: nn.Module, model2: nn.Module) -> float:
        """Compute L2 distance between model parameters"""
        distance = 0.0
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            distance += torch.norm(p1 - p2).item()
        return distance

    def _krum_select(self, distances: torch.Tensor) -> List[int]:
        """Select non-Byzantine models using Krum algorithm"""
        n = distances.shape[0]
        scores = torch.zeros(n)
        
        for i in range(n):
            # Get distances to k nearest neighbors
            neighbor_distances = torch.topk(distances[i], self.krum_neighbors, largest=False).values
            scores[i] = torch.sum(neighbor_distances)
            
        # Select models with lowest scores (most similar to their neighbors)
        good_indices = torch.topk(scores, max(1, n - self.byzantine_threshold), largest=False).indices
        return good_indices.tolist()

    def select_clients(self) -> List[FederatedClient]:
        """Select subset of clients for training round."""
        if len(self.clients) <= self.config.min_clients:
            return self.clients
        return torch.randperm(len(self.clients))[:self.config.clients_per_round]

    def aggregate_models(self, local_models: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Single unified method for model aggregation"""
        if not local_models:
            raise ValueError("No local models to aggregate")
            
        aggregated_dict = {}
        for key in local_models[0].keys():
            aggregated_dict[key] = torch.mean(torch.stack([model[key] for model in local_models]), dim=0)
        return aggregated_dict

    def _adjust_compression_rate(self, round_metrics: Dict):
        """Dynamically adjust compression rate based on network conditions"""
        bandwidth_usage = round_metrics.get('bandwidth_usage', 0)
        accuracy_delta = round_metrics.get('accuracy_delta', 0)
        
        if accuracy_delta < -0.02:  # Accuracy dropping too fast
            self.compression_rate = max(
                self.compression_rate * 0.8,  # Reduce compression
                self.min_compression
            )
        elif bandwidth_usage > self.config.bandwidth_target:
            self.compression_rate = min(
                self.compression_rate * 1.2,  # Increase compression
                self.max_compression
            )

    def _track_training_progress(self, round_num: int, metrics: Dict[str, float]):
        """Track training progress and detect anomalies"""
        try:
            self._progress[round_num] = metrics
            
            # Check for training anomalies
            if round_num > 0:
                loss_delta = metrics['loss'] - self._progress[round_num-1]['loss']
                if abs(loss_delta) > self.config.loss_delta_threshold:
                    logger.warning(f"Large loss change detected in round {round_num}")
                    
                acc_delta = metrics['accuracy'] - self._progress[round_num-1]['accuracy'] 
                if acc_delta < -0.1:  # 10% drop in accuracy
                    logger.warning(f"Accuracy drop detected in round {round_num}")
                    
            # Update compression rate based on metrics
            self._adjust_compression_rate(metrics)
            
        except Exception as e:
            logger.error(f"Error tracking progress: {str(e)}")
