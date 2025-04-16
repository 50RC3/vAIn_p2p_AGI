import torch
from typing import List, Dict, Optional, Tuple
import copy
from torch import nn, optim
from torch.utils.data import DataLoader
from .federated_client import FederatedClient
import logging
from .compression import compress_gradients, decompress_gradients
from .privacy import add_differential_privacy_noise

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
        
        # Add AGI evolution tracking
        self.global_intelligence_score = 0.0
        self.evolution_history = []
        self.cognitive_metrics = {}
        self.meta_learning_state = {}
        
        # Add privacy and compression settings
        self.epsilon = getattr(config, 'privacy_epsilon', 1.0)
        self.delta = getattr(config, 'privacy_delta', 1e-5)
        self.clip_norm = getattr(config, 'gradient_clip_norm', 1.0)
        self.compression_rate = getattr(config, 'compression_rate', 0.1)
        self.error_feedback = {}

        # Add data sharding and parallel training coordination
        self.data_shards = []
        self.parallel_clients = []
        self.compression = AdaptiveCompression()

        # Add symbolic reasoning and cross-domain transfer
        self.symbolic_logic = PropositionalLogic()
        self.domain_transfer = CrossDomainTransfer()
        
        # Setup symbolic rules
        self._setup_symbolic_rules()

        logger.info("FederatedLearning initialized with %d clients minimum", self.config.min_clients)

    def _setup_symbolic_rules(self):
        self.symbolic_logic.add_rule(
            RuleType.UPDATE,
            "valid_update and not is_malicious"
        )
        self.symbolic_logic.add_rule(
            RuleType.DOMAIN,
            "domain_compatible and knowledge_transfer_safe"
        )

    def _validate_update(self, update: Dict) -> bool:
        try:
            self.symbolic_logic.set_variable("valid_update", 
                self._check_update_validity(update))
            self.symbolic_logic.set_variable("is_malicious",
                self._detect_malicious_update(update))
            return self.symbolic_logic.evaluate_expression(
                "valid_update and not is_malicious")
        except Exception as e:
            logger.error(f"Update validation failed: {str(e)}")
            return False

    def _check_update_validity(self, update: Dict) -> bool:
        # Add update validation logic
        return True

    def _detect_malicious_update(self, update: Dict) -> bool:
        # Add Byzantine detection logic
        return False

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
        
    def client_update(self, client_data: DataLoader, is_mobile: bool = False) -> Dict[str, torch.Tensor]:
        """Update client model with mobile optimization support"""
        if not self.global_model:
            raise ValueError("Global model not initialized")

        # Initialize mobile optimizer if needed        
        if is_mobile and not hasattr(self, 'mobile_optimizer'):
            self.mobile_optimizer = MobileOptimizer(compression_rate=0.1)

        local_model = copy.deepcopy(self.global_model)
        optimizer = optim.SGD(local_model.parameters(), lr=self.lr)

        initial_state = {k: v.clone() for k, v in self.global_model.state_dict().items()}

        for _ in range(self.local_epochs):
            for x, y in client_data:
                optimizer.zero_grad()
                output = local_model(x)
                loss = self.criterion(output, y)
                loss.backward()
                # Clip gradients for privacy
                nn.utils.clip_grad_norm_(local_model.parameters(), self.clip_norm)
                optimizer.step()

        # Compute and compress updates
        updates = {}
        for key, final_param in local_model.state_dict().items():
            updates[key] = final_param - initial_state[key]

        if is_mobile:
            # Use mobile-optimized compression
            return self.mobile_optimizer.compress_for_mobile(updates)
        else:
            # Use standard compression
            return self._compress_update(updates)

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

    def _compress_update(self, model_update: Dict[str, torch.Tensor>) -> Tuple[Dict, Dict]:
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

    async def update_global_intelligence(self, local_models: List[nn.Module]) -> float:
        """Track and update global intelligence score"""
        try:
            # Calculate cognitive improvements
            new_score = sum(self._evaluate_cognitive_abilities(model) 
                          for model in local_models) / len(local_models)
            
            # Track evolution
            self.evolution_history.append({
                'timestamp': time.time(),
                'score': new_score,
                'cognitive_metrics': self._measure_cognitive_metrics()
            })
            
            intelligence_delta = new_score - self.global_intelligence_score
            self.global_intelligence_score = new_score
            
            logger.info(f"Global intelligence updated: {new_score:.4f} "
                       f"(Δ: {intelligence_delta:+.4f})")
            
            return new_score
            
        except Exception as e:
            logger.error(f"Failed to update global intelligence: {str(e)}")
            return self.global_intelligence_score

    def _evaluate_cognitive_abilities(self, model: nn.Module) -> float:
        """Evaluate model's cognitive capabilities"""
        metrics = {}
        try:
            # Measure key cognitive abilities
            metrics['memory'] = self._test_memory_capacity(model)
            metrics['learning'] = self._test_learning_speed(model)
            metrics['reasoning'] = self._test_reasoning_ability(model)
            metrics['adaptation'] = self._test_adaptation_speed(model)
            
            # Weighted scoring of cognitive metrics
            weights = {
                'memory': 0.25,
                'learning': 0.25,
                'reasoning': 0.25,
                'adaptation': 0.25
            }
            
            score = sum(metric * weights[key] 
                       for key, metric in metrics.items())
                       
            self.cognitive_metrics = metrics
            return score
            
        except Exception as e:
            logger.error(f"Cognitive evaluation failed: {str(e)}")
            return 0.0

    def _aggregate(self, local_models: List[nn.Module]) -> nn.Module:
        """Hierarchical aggregation through cluster leaders"""
        try:
            # Update global intelligence first
            await self.update_global_intelligence(local_models)
            
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
            
            # Add meta-learning state
            self.meta_learning_state = {
                'global_score': self.global_intelligence_score,
                'evolution': self.evolution_history[-10:],  # Keep last 10 records
                'cognitive_state': self.cognitive_metrics
            }
            
            # Get cluster leaders and their models
            leader_models = []
            leader_weights = []
            
            for cluster in self.network.get_clusters():
                if not cluster.leader_node:
                    continue
                    
                leader_model = local_models[cluster.leader_node]
                leader_score = cluster.node_reputations[cluster.leader_node]
                
                leader_models.append(leader_model)
                leader_weights.append(leader_score)
            
            # Normalize weights
            leader_weights = torch.softmax(torch.tensor(leader_weights), dim=0)
            
            # Aggregate leader models
            final_model = copy.deepcopy(local_models[0])
            aggregated_state = self._aggregate_states(
                [model.state_dict() for model in leader_models],
                leader_weights
            )
            final_model.load_state_dict(aggregated_state)
            
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

    def aggregate_models(self, model_updates: List[Dict[str, torch.Tensor]], 
                        mobile_clients: Optional[List[int]] = None) -> Dict[str, torch.Tensor]:
        """Aggregate updates with mobile client handling"""
        if not model_updates:
            raise ValueError("No updates to aggregate")

        mobile_clients = mobile_clients or []
        decompressed_updates = []

        # Decompress updates based on client type
        for i, update in enumerate(model_updates):
            if i in mobile_clients:
                # Decompress mobile update
                decompressed = self.mobile_optimizer.decompress_mobile_update(update)
            else:
                # Standard decompression
                decompressed = self._decompress_update(update)
            decompressed_updates.append(decompressed)

        # Decompress updates and apply error correction
        decompressed_updates = []
        for update in model_updates:
            corrected_update = {}
            for key, compressed_value in update.items():
                # Get accumulated error for this key
                error = self.error_feedback.get(key, torch.zeros_like(compressed_value))
                # Decompress and add error correction
                corrected_update[key] = decompress_gradients(compressed_value) + error
                # Update error feedback
                self.error_feedback[key] = corrected_update[key] - compressed_value

            decompressed_updates.append(corrected_update)

        # Average the updates
        aggregated = {}
        for key in decompressed_updates[0].keys():
            aggregated[key] = torch.mean(torch.stack([
                update[key] for update in decompressed_updates
            ]), dim=0)

        return aggregated

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

    def shard_data(self, dataset):
        """Partition dataset across nodes"""
        num_shards = len(self.parallel_clients)
        shard_size = len(dataset) // num_shards
        indices = torch.randperm(len(dataset))
        
        self.data_shards = []
        for i in range(num_shards):
            start_idx = i * shard_size
            end_idx = start_idx + shard_size if i < num_shards-1 else len(dataset)
            shard_indices = indices[start_idx:end_idx]
            shard = torch.utils.data.Subset(dataset, shard_indices)
            self.data_shards.append(shard)

    async def train_parallel(self) -> nn.Module:
        """Execute federated training in parallel across nodes"""
        try:
            if not self.parallel_clients:
                raise ValueError("No parallel clients registered")

            # Assign data shards to clients
            self.shard_data(self.dataset)
            for i, client in enumerate(self.parallel_clients):
                client.data_shard = self.data_shards[i]
                client.shard_id = i

            # Train clients in parallel
            tasks = [client.train() for client in self.parallel_clients]
            results = await asyncio.gather(*tasks)

            # Aggregate updates
            aggregated_updates = self.aggregate_parallel_updates(results)
            
            # Apply aggregated updates to global model
            for key, update in aggregated_updates.items():
                self.global_model.state_dict()[key] += update

            return self.global_model

        except Exception as e:
            logger.error(f"Parallel training failed: {str(e)}")
            raise

    def aggregate_parallel_updates(self, client_results: List[Dict]) -> Dict:
        """Aggregate updates from parallel clients"""
        aggregated = {}
        num_clients = len(client_results)

        for key in self.global_model.state_dict().keys():
            updates = []
            for result in client_results:
                decompressed = self.compression.decompress_updates(result['updates'])
                updates.append(decompressed[key])
            
            # Average the updates
            aggregated[key] = sum(updates) / num_clients

        return aggregated
