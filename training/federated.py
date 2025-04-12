import torch
from typing import List, Dict, Optional, Tuple, Any
import copy
from torch import nn, optim
from torch.utils.data import DataLoader
from .federated_client import FederatedClient
import logging
from .compression import compress_gradients, decompress_gradients
from .privacy import add_differential_privacy_noise
from .hierarchy import NodeCluster
import json
logger = logging.getLogger(__name__)
from cryptography.fernet import Fernet
class FederatedTrainingError(Exception):metric import ed25519
    passmpression import compress_gradients, decompress_gradients
from .privacy import add_differential_privacy_noise
class FederatedLearning:odeCluster
    def __init__(self, config: Any) -> None:
        """Initialize federated learning coordinator.
        
        Args:ansmitterSystem:
            config: Configuration object with federation settings = None):
        """initial_levels is None:
        # Store configuration{"dopamine": 0.5, "serotonin": 0.5, "acetylcholine": 0.5}
        self.config = configl_levels
        
        # Set default valuesreward_signal: float, cognitive_load: float):
        self.clients = []mine"] = np.clip(self.levels["dopamine"] + 0.1 * reward_signal, 0, 1)
        self.global_model = None = np.clip(self.levels["serotonin"] - 0.05 * cognitive_load, 0, 1)
        self.levels["acetylcholine"] = np.clip(self.levels["acetylcholine"] + 0.05 * (1 - cognitive_load), 0, 1)
        # Extract config values with defaults
        self.min_clients = getattr(config, 'min_clients', 2)
        self.clients_per_round = getattr(config, 'clients_per_round', 10)
        self.rounds = getattr(config, 'rounds', 10)evels["dopamine"],
        self.learning_rate = getattr(config, 'learning_rate', 0.01)]),
        self.local_epochs = getattr(config, 'local_epochs', 1)choline"])
        }
        # Setup security features
        self.byzantine_threshold = getattr(config, 'byzantine_threshold', 0.33)
        self.compression_rate = getattr(config, 'compression_rate', 0.1)
        self.fernet = Fernet(encryption_key)
        # Initialize tracking structures19PrivateKey.generate()
        self.training_history = []lf.signing_key.public_key()
        self.error_accumulator = {}
        self.error_feedback = {}  # Store residual errors for error feedback
        message_id = str(time.time())
        # Add error handling metrics
        self.error_metrics = {
            "compression_errors": 0,
            "recovery_attempts": 0,),
            "successful_recoveries": 0
        }
        serialized = json.dumps(message_with_metadata).encode()
        # Initialize callback registryt(serialized)
        self.callbacks = {}gning_key.sign(encrypted)
        return {"encrypted_data": encrypted, "signature": signature}
        # Initialize specialized components
        self._setup_symbolic_rules()_message):
        encrypted_data = encoded_message["encrypted_data"]
        logger.info(f"FederatedLearning initialized with min_clients={self.min_clients}")
        
    def _setup_symbolic_rules(self):
        """Set up symbolic rules for update validation."""ted_data)
        try:pt Exception:
            # Define basic validation rulesture verification failed")
            self.expected_keys = []
            self.max_update_norm = 10.0(encrypted_data)
            rn json.loads(decrypted.decode())
            # Add more complex rules if available
            if hasattr(self, 'symbolic_logic'):
                self.symbolic_logic.add_rule(
                    "update_validation",
                    "no_nans AND within_norm_bounds AND has_required_keys"
                )
        except Exception as e:earning):
            logger.error(f"Error setting up symbolic rules: {str(e)}")
        self.nt_system = nt_system
    def _validate_update(self, update: Dict) -> bool:
        """Validate model updates to detect malicious behavior."""
        try:ms = self.nt_system.modulate_parameters()
            # Check if update has expected keys {params}")
            if not all(key in update for key in self.expected_keys):
                logger.warning("Update missing expected keys")
                return False:
                hod
            # Check for NaN valuesient_id: str, score: float):
            for key, tensor in update.items():
                if torch.isnan(tensor).any():
                    logger.warning(f"NaN values detected in {key}")
                    return Falsep_n: int):
                    
            # Check for unreasonably large updates
            for key, tensor in update.items():):
                if torch.abs(tensor).max() > self.max_update_norm:
                    logger.warning(f"Excessively large update detected in {key}")
                    return False
                    ation(self, client_id: str, score: float):
            return True self.reputation_scores:
            self.reputation_scores[client_id] = np.clip(self.reputation_scores[client_id] + score, 0, 1)
        except Exception as e:
            logger.error(f"Error validating update: {str(e)}") 0, 1)
            return False
    def get_top_clients(self, top_n=10):
    def _validate_state(self):d(self.reputation_scores.items(), key=lambda x: x[1], reverse=True)
        """Validate internal state before operations""":top_n]]
        if not self.clients:
            raise FederatedTrainingError("No clients registered")
        if not self.global_model and not hasattr(self.config, 'model_initializer'):
            raise FederatedTrainingError("No global model or model initializer defined")
    
    def _initialize_clusters(self) -> List[NodeCluster]:
        # Create hierarchical node clusters
        clusters = []fo(f"Aggregating {len(self.model_updates)} updates.")
        # ...organize nodes into local clusters of 5-10 nodes each
        return clustersleep(0.1)

    def register_client(self, client: FederatedClient):
        self.clients.append(client)
        await asyncio.sleep(0.1)
    def register_callback(self, event_name: str, callback_fn):
        """Register callback for specific events during federated training"""
        self.callbacks[event_name] = callback_fn
        logger.debug(f"Registered callback for event: {event_name}")
            "rounds_completed": 0,
    async def _call_callback(self, event_name: str, data: Dict):
        """Call registered callback with provided data"""
        if event_name in self.callbacks:
            callback = self.callbacks[event_name]
            try:ordinator:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)_interface
                else:syncio.Lock()
                    callback(data)me()
            except Exception as e:()
                logger.error(f"Error in callback {event_name}: {str(e)}")
    
    def client_update(self, client_data: DataLoader, is_mobile: bool = False) -> Dict[str, torch.Tensor]:
        """Update client model with mobile optimization support"""
        if not self.global_model:
            raise ValueError("Global model not initialized")
            if self.chatbot_interface.unsupervised_module and self.config.enable_unsupervised:
        # Initialize mobile optimizer if needed        usters()
        if is_mobile and not hasattr(self, 'mobile_optimizer'):
            self.mobile_optimizer = MobileOptimizer(compression_rate=0.1)
                    "clusters": getattr(self.chatbot_interface.unsupervised_module, 'n_clusters', 0),
        local_model = copy.deepcopy(self.global_model)atbot_interface.unsupervised_module, 'buffer', []))
        optimizer = optim.SGD(local_model.parameters(), lr=self.learning_rate)
                
        initial_state = {k: v.clone() for k, v in self.global_model.state_dict().items()}supervised:
                ss_metrics = await self._update_self_supervised()
        for _ in range(self.local_epochs): = ss_metrics
            for x, y in client_data:
                optimizer.zero_grad().rl_trainer and self.config.enable_reinforcement:
                output = local_model(x)om_history()
                loss = self.criterion(output, y)e.rl_trainer, 'get_training_stats'):
                loss.backward()inforcement"] = self.chatbot_interface.rl_trainer.get_training_stats()
                # Clip gradients for privacy
                nn.utils.clip_grad_norm_(local_model.parameters(), self.clip_norm)
                optimizer.step()await self._apply_cross_learning()
                metrics["cross_learning"] = cross_metrics
        # Compute and compress updates
        updates = {}etrics
        for key, final_param in local_model.state_dict().items():
            updates[key] = final_param - initial_state[key]
        await asyncio.sleep(0.1)
        if is_mobile:
            # Use mobile-optimized compression
            return self.mobile_optimizer.compress_for_mobile(updates)
        else: asyncio.sleep(0.1)
            # Use standard compression
            return self._compress_update(updates)
    async def _train_rl_from_history(self):
    def _compress_update(self, model_update: Dict[str, torch.Tensor]) -> Dict:

    async def _apply_cross_learning(self):
        await asyncio.sleep(0.1)
        return {"cross_learning_applied": True}

class FederatedTrainingError(Exception):
    pass

class FederatedLearning:
    def __init__(self, config: Any) -> None:
        """Initialize federated learning coordinator.
        
        Args:
            config: Configuration object with federation settings
        """
        # Store configuration
        self.config = config
        
        # Set default values
        self.clients = []
        self.global_model = None
        
        # Extract config values with defaults
        self.min_clients = getattr(config, 'min_clients', 2)
        self.clients_per_round = getattr(config, 'clients_per_round', 10)
        self.rounds = getattr(config, 'rounds', 10)
        self.learning_rate = getattr(config, 'learning_rate', 0.01)
        self.local_epochs = getattr(config, 'local_epochs', 1)
        
        # Setup security features
        self.byzantine_threshold = getattr(config, 'byzantine_threshold', 0.33)
        self.compression_rate = getattr(config, 'compression_rate', 0.1)
        
        # Initialize tracking structures
        self.training_history = []
        self.error_accumulator = {}
        self.error_feedback = {}  # Store residual errors for error feedback
        
        # Add error handling metrics
        self.error_metrics = {
            "compression_errors": 0,
            "recovery_attempts": 0,
            "successful_recoveries": 0
        }
        
        # Initialize callback registry
        self.callbacks = {}
        
        # Initialize specialized components
        self._setup_symbolic_rules()
        
        # Initialize neurotransmitter system for adaptive learning
        self.nt_system = NeurotransmitterSystem()
        
        # Initialize reputation system for client selection
        self.reputation_system = ClientReputationSystem()
        
        # Initialize secure messaging
        self.secure_protocol = SecureMessageProtocol(Fernet.generate_key())
        
        # Initialize model coordinator
        self.coordinator = ModelCoordinator()
        
        # Initialize cognitive evolution
        self.cognitive_evolution = CognitiveEvolution()
        
        # Track global intelligence and evolution
        self.global_intelligence_score = 0.5
        self.evolution_history = []
        self.cognitive_metrics = {}
        self.fraud_history = []
        self.byzantine_window = 5
        self.min_byzantine_threshold = 0.1
        self.max_byzantine_threshold = 0.5
        self.krum_neighbors = max(2, int(self.clients_per_round * (1 - self.byzantine_threshold)))
        self.local_cluster_size = 5
        self.data_quality_threshold = 0.7
        # Determine cognitive load and reward signal for this round
        logger.info(f"FederatedLearning initialized with min_clients={self.min_clients}")min(0.5 + round / (2 * self.rounds), 0.9)  # Gradually increase load
  # Default neutral value
    def _setup_symbolic_rules(self):
        """Set up symbolic rules for update validation."""
        try:system.adjust_levels(reward_signal, cognitive_load)
            # Define basic validation rules
            self.expected_keys = []
            self.max_update_norm = 10.0em.modulate_parameters()
            g_rate"]
            # Add more complex rules if available
            if hasattr(self, 'symbolic_logic'):
                self.symbolic_logic.add_rule(d round > 0:
                    "update_validation",reputation_system.get_top_clients(self.clients_per_round)
                    "no_nans AND within_norm_bounds AND has_required_keys"ents if client.client_id in top_client_ids]
                )n_clients:
        except Exception as e:allback to random selection
            logger.error(f"Error setting up symbolic rules: {str(e)}")
t_clients()
    # ...existing code...
                        # Track if client is mobile                        is_mobile = getattr(client, 'is_mobile', False)                        if is_mobile:                            mobile_clients.append(i)                                                    local_model = self.client_update(client.data_loader, is_mobile=is_mobile)                        local_models.append(local_model)                                                # Update client reputation based on model quality                        if hasattr(self, 'reputation_system') and hasattr(client, 'client_id'):                            # Simple quality score based on data size                            quality_score = min(1.0, len(client.data_loader) / 1000)                            self.reputation_system.update_reputation(client.client_id, quality_score)                                                except Exception as e:                        logger.error(f"Error training client: {e}")                        # Notify about client errors                        await self._call_callback("client_error", {                            "client_id": i,                            "round": round + 1,                            "error": str(e)                        })                        continue                                        if not local_models:                    msg = "No successful client updates in round"                    logger.error(msg)                    raise FederatedTrainingError(msg)                                # Update with mobile clients information                self.global_model = await self._aggregate(local_models, mobile_clients=mobile_clients)                                # Trigger cognitive evolution                await self.cognitive_evolution.evolve()                                # Update reward signal based on model improvement                if round > 0 and hasattr(self, 'prev_performance') and hasattr(self, 'current_performance'):                    improvement = self.current_performance - self.prev_performance                    reward_signal = np.clip(0.5 + improvement * 5, 0, 1)  # Scale improvement to 0-1                                    logger.info(f"Round {round+1} complete")                                # Call round completion callback                await self._call_callback("round_completed", {                    "round": round + 1,                    "total_rounds": self.rounds,                    "clients": len(active_clients),                    "error_metrics": self.error_metrics,                    "neurotransmitter_levels": self.nt_system.levels                })                        except Exception as e:            logger.error(f"Training failed: {str(e)}")            # Notify about training failure            await self._call_callback("training_failed", {                "error": str(e),                "error_metrics": self.error_metrics            })            raise FederatedTrainingError(f"Training failed: {str(e)}")                    logger.info("Training completed successfully")        # Notify about training completion        await self._call_callback("training_completed", {            "rounds_completed": self.rounds,            "error_metrics": self.error_metrics,            "final_neurotransmitter_levels": self.nt_system.levels,            "global_intelligence_score": self.global_intelligence_score        })                return self.global_model    def aggregate_models(self, model_updates: List[Dict[str, torch.Tensor]],                         mobile_clients: Optional[List[int]] = None) -> Dict[str, torch.Tensor]:        """Aggregate updates with mobile client handling and error feedback"""        if not model_updates:            raise ValueError("No updates to aggregate")        mobile_clients = mobile_clients or []        decompressed_updates = []        # Decompress updates based on client type        for i, update in enumerate(model_updates):            if i in mobile_clients:                # Decompress mobile update                decompressed = self.mobile_optimizer.decompress_mobile_update(update)            else:                # Standard decompression with error feedback                decompressed = self._decompress_update(update)            decompressed_updates.append(decompressed)        # Average the updates        aggregated = {}        for key in decompressed_updates[0].keys():            try:                tensors = [update[key] for update in decompressed_updates if key in update]                if tensors:                    aggregated[key] = torch.mean(torch.stack(tensors), dim=0)                                        # Update error residuals for this key                    if not hasattr(self, 'error_residuals'):                        self.error_residuals = {}                    self.error_residuals[key] = torch.zeros_like(aggregated[key])                                        # Track each client's residuals                    for i, update in enumerate(decompressed_updates):                        if key in update:                            # Add 1/n of the difference to error residuals                            self.error_residuals[key] += (                                update[key] - aggregated[key]                            ) / len(decompressed_updates)            except Exception as e:                logger.error(f"Error aggregating key {key}: {e}")                # Try to recover using data from previous round                if key in self.error_feedback:                    aggregated[key] = self.error_feedback[key]                    logger.info(f"Using feedback data for key {key}")        # Store aggregated updates as feedback for next round        self.error_feedback = aggregated                return aggregated    def _update_byzantine_threshold(self):        """Dynamically adjust Byzantine threshold based on observed fraud"""        if len(self.fraud_history) < self.byzantine_window:            return                    recent_frauds = self.fraud_history[-self.byzantine_window:]        fraud_rate = sum(recent_frauds) / len(recent_frauds)                # Adjust threshold based on observed fraud rate        new_threshold = min(            self.max_byzantine_threshold,            max(                self.min_byzantine_threshold,                fraud_rate * 1.5  # Set threshold 50% higher than observed rate            )        )                self.byzantine_threshold = new_threshold        self.krum_neighbors = max(2, int(len(self.clients) * (1 - self.byzantine_threshold)))    async def update_global_intelligence(self, local_models: List[nn.Module]) -> float:        """Track and update global intelligence score"""        try:            # Calculate cognitive improvements            new_score = sum(self._evaluate_cognitive_abilities(model)                           for model in local_models) / len(local_models)                        # Track evolution            self.evolution_history.append({                'timestamp': time.time(),                'score': new_score,                'cognitive_metrics': self._measure_cognitive_metrics()            })                        intelligence_delta = new_score - self.global_intelligence_score            self.global_intelligence_score = new_score                        logger.info(f"Global intelligence updated: {new_score:.4f} "                       f"(Δ: {intelligence_delta:+.4f})")                        return new_score                    except Exception as e:            logger.error(f"Failed to update global intelligence: {str(e)}")            return self.global_intelligence_score    def _evaluate_cognitive_abilities(self, model: nn.Module) -> float:        """Evaluate model's cognitive capabilities"""        metrics = {}        try:            # Measure key cognitive abilities            metrics['memory'] = self._test_memory_capacity(model)            metrics['learning'] = self._test_learning_speed(model)            metrics['reasoning'] = self._test_reasoning_ability(model)            metrics['adaptation'] = self._test_adaptation_speed(model)                        # Weighted scoring of cognitive metrics            weights = {                'memory': 0.25,                'learning': 0.25,                'reasoning': 0.25,                'adaptation': 0.25            }                        score = sum(metric * weights[key]                        for key, metric in metrics.items())                                   self.cognitive_metrics = metrics            return score                    except Exception as e:            logger.error(f"Cognitive evaluation failed: {str(e)}")            return 0.0    async def _aggregate(self, local_models: List[nn.Module], mobile_clients: Optional[List[int]] = None) -> nn.Module:        """Hierarchical aggregation through cluster leaders"""        try:            # Update global intelligence first            await self.update_global_intelligence(local_models)                        # Update Byzantine threshold based on history            self._update_byzantine_threshold()                        # Detect and filter malicious updates using Krum algorithm            distances = self._compute_pairwise_distances(local_models)            good_model_indices = self._krum_select(distances)                        # Record fraud metrics            fraud_ratio = 1 - (len(good_model_indices) / len(local_models))            self.fraud_history.append(fraud_ratio)                        filtered_models = [local_models[i] for i in good_model_indices]                        # Calculate data quality scores for each model            quality_scores = [                calculate_data_quality(model)                 for model in filtered_models            ]                        # Filter models below quality threshold            quality_filtered = [                (model, score) for model, score in zip(filtered_models, quality_scores)                if score >= self.data_quality_threshold            ]                        if not quality_filtered:                raise ValueError("No models meet quality threshold")            # Weighted aggregation based on quality scores            models, scores = zip(*quality_filtered)            weights = torch.softmax(torch.tensor(scores), dim=0)            # Split nodes into local clusters            clusters = [models[i:i + self.local_cluster_size]                         for i in range(0, len(models), self.local_cluster_size)]            cluster_weights = [weights[i:i + self.local_cluster_size]                             for i in range(0, len(weights), self.local_cluster_size)]            # First level aggregation within clusters            cluster_models = []            for cluster, cluster_weight in zip(clusters, cluster_weights):                updates = [model.state_dict() for model in cluster]                compressed_updates = [                    self.compression.compress_model_updates(update)[0]                     for update in updates                ]                # Aggregate compressed updates within cluster                cluster_aggregate = self._aggregate_compressed_updates(compressed_updates)                cluster_models.append(cluster_aggregate)                        # Final aggregation across clusters            final_model = copy.deepcopy(local_models[0])            final_update = self._aggregate_compressed_updates(cluster_models)            final_model.load_state_dict(                self.compression.decompress_model_updates(final_update)            )                        # Add meta-learning state            self.meta_learning_state = {                'global_score': self.global_intelligence_score,                'evolution': self.evolution_history[-10:],  # Keep last 10 records                'cognitive_state': self.cognitive_metrics            }                        # Get cluster leaders and their models            leader_models = []            leader_weights = []                        for cluster in self.network.get_clusters():                if not cluster.leader_node:                    continue                                    leader_model = local_models[cluster.leader_node]                leader_score = cluster.node_reputations[cluster.leader_node]                                leader_models.append(leader_model)                leader_weights.append(leader_score)                        # Normalize weights            leader_weights = torch.softmax(torch.tensor(leader_weights), dim=0)                        # Aggregate leader models            final_model = copy.deepcopy(local_models[0])            aggregated_state = self._aggregate_states(                [model.state_dict() for model in leader_models],                leader_weights            )            final_model.load_state_dict(aggregated_state)                        return final_model        except Exception as e:            logger.error(f"Model aggregation failed: {str(e)}")            raise FederatedTrainingError(f"Aggregation failed: {str(e)}")    def _aggregate_compressed_updates(self, compressed_updates: List[Dict]) -> Dict:        """Aggregate compressed updates while preserving sparsity"""        if not compressed_updates:            return {}                    aggregated = {}        num_updates = len(compressed_updates)                # Get all unique keys across updates        all_keys = set()        for update in compressed_updates:            all_keys.update(update.keys())                # Aggregate each key        for key in all_keys:            # Collect values for this key from all updates            tensors = []            for update in compressed_updates:                if key in update:                    tensors.append(update[key])                        if tensors:                # Average tensors for this key                aggregated[key] = sum(tensors) / len(tensors)                return aggregated    def _compute_pairwise_distances(self, models: List[nn.Module]) -> torch.Tensor:        """Compute pairwise distances between model parameters."""        n = len(models)        distances = torch.zeros((n, n))                for i in range(n):            for j in range(i + 1, n):                dist = self._model_distance(models[i], models[j])                distances[i][j] = distances[j][i] = dist                        return distances    def _model_distance(self, model1: nn.Module, model2: nn.Module) -> float:        """Compute L2 distance between model parameters"""        distance = 0.0        for p1, p2 in zip(model1.parameters(), model2.parameters()):            distance += torch.norm(p1 - p2).item()        return distance    def _krum_select(self, distances: torch.Tensor) -> List[int]:        """Select non-Byzantine models using Krum algorithm"""        n = distances.shape[0]        scores = torch.zeros(n)                for i in range(n):            # Get distances to k nearest neighbors            neighbor_distances = torch.topk(distances[i], self.krum_neighbors, largest=False).values            scores[i] = torch.sum(neighbor_distances)                    # Select models with lowest scores (most similar to their neighbors)        good_indices = torch.topk(scores, max(1, n - self.byzantine_threshold), largest=False).indices        return good_indices.tolist()    def select_clients(self) -> List[FederatedClient]:        """Select subset of clients for training round."""        if len(self.clients) <= self.min_clients:            return self.clients        return torch.randperm(len(self.clients))[:self.clients_per_round]    def _adjust_compression_rate(self, round_metrics: Dict):        """Dynamically adjust compression rate based on network conditions"""        bandwidth_usage = round_metrics.get('bandwidth_usage', 0)        accuracy_delta = round_metrics.get('accuracy_delta', 0)                if accuracy_delta < -0.02:  # Accuracy dropping too fast            self.compression_rate = max(                self.compression_rate * 0.8,  # Reduce compression                self.min_compression            )        elif bandwidth_usage > self.config.bandwidth_target:            self.compression_rate = min(                self.compression_rate * 1.2,  # Increase compression                self.max_compression            )    def _track_training_progress(self, round_num: int, metrics: Dict[str, float]):        """Track training progress and detect anomalies"""        try:            self._progress[round_num] = metrics                        # Check for training anomalies            if round_num > 0:                loss_delta = metrics['loss'] - self._progress[round_num-1]['loss']                if abs(loss_delta) > self.config.loss_delta_threshold:                    logger.warning(f"Large loss change detected in round {round_num}")                                    acc_delta = metrics['accuracy'] - self._progress[round_num-1]['accuracy']                 if acc_delta < -0.1:  # 10% drop in accuracy                    logger.warning(f"Accuracy drop detected in round {round_num}")                                # Update compression rate based on metrics            self._adjust_compression_rate(metrics)                    except Exception as e:            logger.error(f"Error tracking progress: {str(e)}")    def shard_data(self, dataset):        """Partition dataset across nodes"""        num_shards = len(self.parallel_clients)        shard_size = len(dataset) // num_shards        indices = torch.randperm(len(dataset))                self.data_shards = []        for i in range(num_shards):            start_idx = i * shard_size            end_idx = start_idx + shard_size if i < num_shards-1 else len(dataset)            shard_indices = indices[start_idx:end_idx]            shard = torch.utils.data.Subset(dataset, shard_indices)            self.data_shards.append(shard)    async def train_parallel(self) -> nn.Module:        """Execute federated training in parallel across nodes"""        try:            if not self.parallel_clients:                raise ValueError("No parallel clients registered")            # Assign data shards to clients
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
