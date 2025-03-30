"""Advanced learning system combining multiple learning paradigms in a unified framework."""
import asyncio
import logging
import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set, Union
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import random
import os
from pathlib import Path

from ..chatbot.rl_trainer import RLTrainer, RLConfig
from network.p2p_network import P2PNetwork
from ..chatbot.interface import ChatbotInterface
from models.simple_nn import TextEncoder
from ai.predictive.node_attention import InteractionLevel, InteractiveConfig, InteractiveSession

logger = logging.getLogger(__name__)

@dataclass
class LearningMetrics:
    """Tracks learning progress metrics"""
    total_supervised_samples: int = 0
    total_unsupervised_samples: int = 0
    total_reinforcement_samples: int = 0
    unsupervised_loss: float = 0.0
    supervised_loss: float = 0.0
    reinforcement_loss: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    training_cycles: int = 0
    improvement_rate: float = 0.0
    
    def update_metrics(self, metrics: Dict[str, float]) -> None:
        """Update learning metrics"""
        for key, value in metrics.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.last_updated = datetime.now()
        self.training_cycles += 1


class LearningError(Exception):
    """Exception raised for errors in the learning system."""
    pass


@dataclass
class LearningConfig:
    """Configuration for the learning system"""
    unsupervised_batch_size: int = 32
    supervised_batch_size: int = 16
    rl_batch_size: int = 8
    learning_rate: float = 1e-5
    unsupervised_ratio: float = 0.4  # Portion of training to be unsupervised
    contrastive_weight: float = 0.3  # Weight for contrastive learning
    training_interval: int = 600     # Seconds between training sessions
    min_samples_for_training: int = 50
    max_history_length: int = 10000  # Maximum samples to keep in history
    enable_peer_learning: bool = True
    contrastive_temperature: float = 0.07
    similarity_threshold: float = 0.85
    # New parameters for better memory management
    memory_cleanup_interval: int = 3600  # Clean up old samples every hour
    max_peer_samples: int = 5000  # Maximum peer samples to store
    save_path: str = "models/advanced_learning"
    save_interval: int = 10  # Save state every N training cycles
    # Interactive learning parameters
    interactive_mode: bool = False
    interactive_timeout: int = 30  # Timeout in seconds for interactive operations


class AdvancedLearning:
    """Advanced learning system combining self-supervised, unsupervised, and reinforcement learning"""
    
    def __init__(
        self, 
        chatbot: ChatbotInterface, 
        rl_trainer: RLTrainer,
        config: Optional[LearningConfig] = None,
        p2p_network: Optional[P2PNetwork] = None
    ):
        self.chatbot = chatbot
        self.rl_trainer = rl_trainer
        self.config = config or LearningConfig()
        self.p2p_network = p2p_network
        
        # Initialize text encoder for embedding generation
        self.text_encoder = TextEncoder()
        
        # Storage for different types of learning samples
        self.unsupervised_samples = []
        self.supervised_samples = []
        self.conversation_history = []
        self.peer_shared_samples = []
        
        # Track clusters of similar conversations
        self.conversation_clusters = defaultdict(list)
        self.cluster_centroids = {}
        self.next_cluster_id = 0
        
        # Metrics tracking
        self.metrics = LearningMetrics()
        
        # Training state
        self.is_training = False
        self.last_training_time = datetime.now()
        self.last_cleanup_time = datetime.now()
        self.training_lock = asyncio.Lock()
        
        # Interactive session
        self.session = None
        
        # Ensure save directory exists
        os.makedirs(self.config.save_path, exist_ok=True)
        
        logger.info("Advanced learning system initialized")
        
    async def add_unsupervised_sample(self, text: str) -> None:
        """Add an unsupervised learning sample (raw text without explicit feedback)"""
        if not text or len(text.strip()) < 5:
            return
            
        try:
            # Generate embedding for the text
            with torch.no_grad():
                embedding = self.text_encoder.encode(text)
                
            # Store sample with timestamp
            self.unsupervised_samples.append({
                "text": text,
                "embedding": embedding,
                "timestamp": datetime.now(),
                "type": "unsupervised"
            })
            
            # Limit collection size
            if len(self.unsupervised_samples) > self.config.max_history_length:
                self.unsupervised_samples.pop(0)
                
            # Update metrics
            self.metrics.total_unsupervised_samples += 1
            
            # Check if we should start unsupervised training
            await self._check_training_schedule()
            
            # Periodically clean up old samples
            await self._check_cleanup_schedule()
            
        except Exception as e:
            logger.error(f"Error adding unsupervised sample: {e}")
    
    async def add_conversation(self, message: str, response: str, feedback: Optional[float] = None) -> None:
        """Add a conversation turn with optional explicit feedback"""
        # Skip empty messages
        if not message or not response:
            return
            
        try:
            # Create conversation entry
            conversation = {
                "message": message,
                "response": response,
                "timestamp": datetime.now(),
                "feedback": feedback,
                "message_embedding": self.text_encoder.encode(message),
                "response_embedding": self.text_encoder.encode(response),
            }
            
            # Store conversation
            self.conversation_history.append(conversation)
            
            # If feedback provided, add as supervised sample
            if feedback is not None:
                self.supervised_samples.append({
                    "message": message,
                    "response": response,
                    "score": feedback,
                    "timestamp": datetime.now(),
                    "type": "supervised"
                })
                self.metrics.total_supervised_samples += 1
                
                # Also add to RL trainer if feedback is provided
                await self.rl_trainer.add_experience(message, response, feedback)
                self.metrics.total_reinforcement_samples += 1
                
                # Share with peers if network is available and feedback is good
                if self.p2p_network and self.config.enable_peer_learning and feedback > 0.7:
                    await self.share_with_peers(message, response, feedback)
            else:
                # No explicit feedback, treat as unsupervised
                await self.add_unsupervised_sample(f"{message} [SEP] {response}")
            
            # Assign to conversation cluster
            await self._assign_to_cluster(conversation)
            
            # Limit history size
            if len(self.conversation_history) > self.config.max_history_length:
                self.conversation_history.pop(0)
                
            # Check if we should start training
            await self._check_training_schedule()
            
        except Exception as e:
            logger.error(f"Error adding conversation: {e}")
    
    async def _assign_to_cluster(self, conversation: Dict) -> int:
        """Assign a conversation to a semantic cluster"""
        try:
            # Get combined embedding for message and response
            combined_embedding = (conversation["message_embedding"] + conversation["response_embedding"]) / 2
            
            # Find the closest cluster if any exist
            best_similarity = -1
            best_cluster = -1
            
            for cluster_id, centroid in self.cluster_centroids.items():
                similarity = torch.nn.functional.cosine_similarity(
                    combined_embedding.unsqueeze(0),
                    centroid.unsqueeze(0)
                ).item()
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster = cluster_id
                    
            # If similar enough to existing cluster, add to it
            if best_similarity > self.config.similarity_threshold:
                cluster_id = best_cluster
                self.conversation_clusters[cluster_id].append(conversation)
                
                # Update centroid with moving average
                cluster_size = len(self.conversation_clusters[cluster_id])
                new_centroid = self.cluster_centroids[cluster_id] * ((cluster_size - 1) / cluster_size)
                new_centroid += combined_embedding * (1 / cluster_size)
                self.cluster_centroids[cluster_id] = new_centroid
            else:
                # Create new cluster
                cluster_id = self.next_cluster_id
                self.next_cluster_id += 1
                self.conversation_clusters[cluster_id] = [conversation]
                self.cluster_centroids[cluster_id] = combined_embedding.clone()
                
            return cluster_id
            
        except Exception as e:
            logger.error(f"Error assigning to cluster: {e}")
            return -1
    
    async def share_with_peers(self, message: str, response: str, score: float) -> None:
        """Share valuable learning experiences with peers"""
        if not self.p2p_network:
            return
            
        try:
            await self.p2p_network.submit_interaction(message, response, score)
            logger.debug(f"Shared interaction with peers, score={score}")
        except Exception as e:
            logger.warning(f"Failed to share learning data with peers: {e}")
    
    async def receive_peer_sample(self, sample: Dict) -> None:
        """Process a learning sample shared by a peer"""
        # Validate sample
        required_fields = ["message", "response", "score"]
        if not all(field in sample for field in required_fields):
            logger.warning("Received invalid peer sample")
            return
            
        try:
            # Limit peer samples collection size
            if len(self.peer_shared_samples) >= self.config.max_peer_samples:
                # Remove oldest sample
                self.peer_shared_samples.pop(0)
                
            # Add to peer samples collection
            self.peer_shared_samples.append({
                **sample,
                "timestamp": datetime.now(),
                "peer_source": sample.get("peer_id", "unknown"),
                "type": "peer"
            })
            
            # If it's a high-quality sample, also add directly to supervised samples
            if sample.get("score", 0) > 0.7:
                self.supervised_samples.append({
                    "message": sample["message"],
                    "response": sample["response"],
                    "score": sample["score"],
                    "timestamp": datetime.now(),
                    "type": "peer_supervised",
                    "peer_source": sample.get("peer_id", "unknown")
                })
                
                # Also add high-quality samples to the RL trainer
                if sample.get("score", 0) > 0.8:
                    await self.rl_trainer.add_experience(
                        sample["message"], 
                        sample["response"],
                        float(sample["score"])
                    )
                    
        except Exception as e:
            logger.error(f"Error processing peer sample: {e}")
    
    async def _check_training_schedule(self) -> None:
        """Check if it's time to run a training session"""
        # Don't check if already training
        if self.is_training:
            return
            
        # Check if we have enough time since last training
        time_since_last = (datetime.now() - self.last_training_time).total_seconds()
        if time_since_last < self.config.training_interval:
            return
            
        # Check if we have enough samples
        total_samples = (
            len(self.supervised_samples) + 
            len(self.unsupervised_samples) + 
            len(self.rl_trainer.memory)
        )
        
        if total_samples < self.config.min_samples_for_training:
            return
            
        # Schedule a training session
        asyncio.create_task(self._run_combined_training())
    
    async def _check_cleanup_schedule(self) -> None:
        """Check if it's time to clean up old samples"""
        time_since_cleanup = (datetime.now() - self.last_cleanup_time).total_seconds()
        if time_since_cleanup > self.config.memory_cleanup_interval:
            await self._cleanup_old_samples()
            self.last_cleanup_time = datetime.now()
    
    async def _cleanup_old_samples(self) -> None:
        """Remove old samples to free memory"""
        try:
            # Sort by timestamp and keep only the most recent samples
            current_time = datetime.now()
            
            # Clean unsupervised samples older than 7 days
            cutoff = 7 * 24 * 3600  # 7 days in seconds
            self.unsupervised_samples = [
                sample for sample in self.unsupervised_samples
                if (current_time - sample["timestamp"]).total_seconds() < cutoff
            ]
            
            # Clean peer samples older than 14 days
            cutoff = 14 * 24 * 3600  # 14 days in seconds
            self.peer_shared_samples = [
                sample for sample in self.peer_shared_samples
                if (current_time - sample["timestamp"]).total_seconds() < cutoff
            ]
            
            # Keep the most recent conversation history
            if len(self.conversation_history) > self.config.max_history_length:
                self.conversation_history = self.conversation_history[-self.config.max_history_length:]
                
            # Clean small conversation clusters (likely noise)
            small_clusters = []
            for cluster_id, conversations in self.conversation_clusters.items():
                if len(conversations) < 3:  # Threshold for considering a cluster as noise
                    small_clusters.append(cluster_id)
                    
            for cluster_id in small_clusters:
                del self.conversation_clusters[cluster_id]
                if cluster_id in self.cluster_centroids:
                    del self.cluster_centroids[cluster_id]
                    
            logger.debug(f"Cleaned up old samples. Remaining: {len(self.unsupervised_samples)} unsupervised, "
                         f"{len(self.peer_shared_samples)} peer, {len(self.conversation_history)} conversations, "
                         f"{len(self.conversation_clusters)} clusters")
                         
        except Exception as e:
            logger.error(f"Error cleaning up samples: {e}")
    
    async def _run_combined_training(self) -> None:
        """Run a combined training session with multiple learning methods"""
        # Use the lock to prevent concurrent training sessions
        if not await self.training_lock.acquire():
            return
        
        try:
            self.is_training = True
            logger.info("Starting combined training session")
            
            # Start interactive session if configured
            if self.config.interactive_mode:
                await self._start_interactive_session()
            
            # 1. Self-supervised contrastive learning
            contrastive_loss = await self._run_contrastive_learning()
            
            # 2. Supervised response quality learning
            supervised_loss = await self._run_supervised_learning()
            
            # 3. Reinforcement learning update
            rl_loss = await self._run_reinforcement_learning()
            
            # Update metrics
            self.metrics.update_metrics({
                "unsupervised_loss": contrastive_loss,
                "supervised_loss": supervised_loss,
                "reinforcement_loss": rl_loss
            })
            
            # Calculate improvement rate
            self.metrics.improvement_rate = self._calculate_improvement()
            
            self.last_training_time = datetime.now()
            logger.info(
                f"Training complete. Metrics: "
                f"unsup_loss={contrastive_loss:.4f}, "
                f"sup_loss={supervised_loss:.4f}, "
                f"rl_loss={rl_loss:.4f}, "
                f"improvement={self.metrics.improvement_rate:.2f}"
            )
            
            # Save state periodically
            if self.metrics.training_cycles % self.config.save_interval == 0:
                await self._save_state()
                
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
        finally:
            # End interactive session if it was started
            if self.config.interactive_mode and self.session is not None:
                await self._end_interactive_session()
                
            self.is_training = False
            self.training_lock.release()
    
    async def _start_interactive_session(self) -> None:
        """Start interactive session for training with user feedback"""
        try:
            self.session = InteractiveSession(
                level=InteractionLevel.NORMAL,
                config=InteractiveConfig(
                    timeout=self.config.interactive_timeout,
                    persistent_state=True,
                    safe_mode=True
                )
            )
            await self.session.__aenter__()
        except Exception as e:
            logger.error(f"Failed to start interactive session: {e}")
            self.session = None
    
    async def _end_interactive_session(self) -> None:
        """End interactive session"""
        if self.session is not None:
            try:
                await self.session.__aexit__(None, None, None)
            except Exception as e:
                logger.error(f"Error closing interactive session: {e}")
            finally:
                self.session = None
    
    async def _run_contrastive_learning(self) -> float:
        """Run contrastive learning on unsupervised samples
        
        This implements self-supervised learning by creating positive/negative pairs.
        """
        if len(self.unsupervised_samples) < self.config.unsupervised_batch_size:
            return 0.0
            
        try:
            # Sample batch for contrastive learning
            batch_indices = random.sample(
                range(len(self.unsupervised_samples)),
                min(len(self.unsupervised_samples), self.config.unsupervised_batch_size)
            )
            
            # Extract embeddings for batch
            embeddings = torch.stack([
                self.unsupervised_samples[i]["embedding"] 
                for i in batch_indices
            ])
            
            # Create positive pairs by adding small noise to embeddings
            positive_embeddings = embeddings + torch.randn_like(embeddings) * 0.01
            
            # Calculate contrastive loss
            similarity = torch.matmul(embeddings, positive_embeddings.t())
            temperature = self.config.contrastive_temperature
            logits = similarity / temperature
            
            # Create labels (diagonal is positive pairs)
            labels = torch.arange(logits.size(0)).to(logits.device)
            
            # Calculate cross entropy loss
            contrastive_loss = torch.nn.functional.cross_entropy(logits, labels)
            
            # Apply loss to model through RL trainer
            await self.rl_trainer.apply_contrastive_loss(
                contrastive_loss * self.config.contrastive_weight
            )
            
            return contrastive_loss.item()
            
        except Exception as e:
            logger.error(f"Error in contrastive learning: {str(e)}")
            return 0.0
    
    async def _run_supervised_learning(self) -> float:
        """Run supervised learning on samples with explicit feedback"""
        if len(self.supervised_samples) < self.config.supervised_batch_size:
            return 0.0
            
        try:
            # Sample batch for supervised learning with priority for recent samples
            recent_weight = 0.7  # Weight for prioritizing recent samples
            timestamps = np.array([
                (datetime.now() - sample["timestamp"]).total_seconds() 
                for sample in self.supervised_samples
            ])
            
            # Calculate probabilities favoring recent samples
            max_age = max(timestamps) if timestamps.size > 0 else 1.0
            recency = 1.0 - (timestamps / max_age)
            weights = recent_weight * recency + (1.0 - recent_weight) * np.ones_like(recency)
            weights /= weights.sum()
            
            # Sample batch based on weights
            batch_indices = np.random.choice(
                len(self.supervised_samples),
                min(len(self.supervised_samples), self.config.supervised_batch_size),
                replace=False,
                p=weights
            )
            
            batch = [self.supervised_samples[i] for i in batch_indices]
            
            # Run supervised update through RL trainer
            supervised_loss = await self.rl_trainer.train_supervised_batch(batch)
            
            return supervised_loss
            
        except Exception as e:
            logger.error(f"Error in supervised learning: {str(e)}")
            return 0.0
    
    async def _run_reinforcement_learning(self) -> float:
        """Run reinforcement learning update"""
        try:
            # Check for confirmation in interactive mode
            if self.config.interactive_mode and self.session:
                should_continue = await self.session.get_confirmation(
                    "Continue with reinforcement learning update?",
                    timeout=self.config.interactive_timeout
                )
                if not should_continue:
                    logger.info("Reinforcement learning update skipped by user")
                    return 0.0
            
            rl_loss = await self.rl_trainer.train_step(
                batch_size=self.config.rl_batch_size
            )
            return rl_loss
        except Exception as e:
            logger.error(f"Error in reinforcement learning: {str(e)}")
            return 0.0
    
    def _calculate_improvement(self) -> float:
        """Calculate learning improvement rate"""
        if self.metrics.training_cycles <= 1:
            return 0.0
            
        # Weight recent losses more heavily
        recent_weight = 0.8
        
        # Calculate weighted average with previous improvement
        current_loss = (
            self.metrics.unsupervised_loss + 
            self.metrics.supervised_loss + 
            self.metrics.reinforcement_loss
        ) / 3.0
        
        # Simple change calculation (negative is better)
        improvement = -current_loss
        
        # Apply weighted average with previous improvement
        weighted_improvement = (
            recent_weight * improvement + 
            (1 - recent_weight) * self.metrics.improvement_rate
        )
        
        return weighted_improvement
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get learning system metrics"""
        return {
            "unsupervised_samples": len(self.unsupervised_samples),
            "supervised_samples": len(self.supervised_samples),
            "conversation_history": len(self.conversation_history),
            "peer_samples": len(self.peer_shared_samples),
            "clusters": len(self.conversation_clusters),
            "unsupervised_loss": float(self.metrics.unsupervised_loss),
            "supervised_loss": float(self.metrics.supervised_loss),
            "reinforcement_loss": float(self.metrics.reinforcement_loss),
            "training_cycles": self.metrics.training_cycles,
            "improvement_rate": float(self.metrics.improvement_rate),
            "last_training": self.last_training_time.isoformat()
        }
    
    async def get_cluster_info(self) -> Dict[str, Any]:
        """Get information about conversation clusters"""
        result = {}
        for cluster_id, conversations in self.conversation_clusters.items():
            # Get samples per cluster
            sample_count = len(conversations)
            
            # Calculate average feedback score if available
            feedback_scores = [conv["feedback"] for conv in conversations 
                               if conv.get("feedback") is not None]
            avg_feedback = sum(feedback_scores) / len(feedback_scores) if feedback_scores else None
            
            # Get most recent sample
            most_recent = max(conversations, key=lambda x: x["timestamp"])
            
            result[str(cluster_id)] = {
                "sample_count": sample_count,
                "avg_feedback": avg_feedback,
                "most_recent_timestamp": most_recent["timestamp"].isoformat(),
                "example_message": most_recent["message"][:100] + "..." if len(most_recent["message"]) > 100 else most_recent["message"]
            }
        return result
    
    async def _save_state(self) -> None:
        """Save learning state to disk"""
        try:
            save_path = Path(self.config.save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save metrics
            metrics_path = save_path / "metrics.pt"
            torch.save({
                "metrics": self.metrics.__dict__,
                "training_cycles": self.metrics.training_cycles,
                "improvement_rate": self.metrics.improvement_rate
            }, metrics_path)
            
            # Save cluster centroids
            centroids_path = save_path / "centroids.pt"
            torch.save({
                "centroids": self.cluster_centroids,
                "next_cluster_id": self.next_cluster_id
            }, centroids_path)
            
            logger.info(f"Saved learning state. Metrics: {self.metrics.training_cycles} training cycles")
            
        except Exception as e:
            logger.error(f"Error saving learning state: {e}")
    
    async def load_state(self) -> bool:
        """Load learning state from disk"""
        try:
            save_path = Path(self.config.save_path)
            
            # Load metrics
            metrics_path = save_path / "metrics.pt"
            if metrics_path.exists():
                metrics_dict = torch.load(metrics_path)
                for key, value in metrics_dict.get("metrics", {}).items():
                    if hasattr(self.metrics, key):
                        setattr(self.metrics, key, value)
            
            # Load cluster centroids
            centroids_path = save_path / "centroids.pt"
            if centroids_path.exists():
                centroids_dict = torch.load(centroids_path)
                self.cluster_centroids = centroids_dict.get("centroids", {})
                self.next_cluster_id = centroids_dict.get("next_cluster_id", 0)
            
            logger.info(f"Loaded learning state. Metrics: {self.metrics.training_cycles} training cycles")
            return True
            
        except Exception as e:
            logger.error(f"Error loading learning state: {e}")
            return False
    
    async def force_training_cycle(self) -> Dict[str, float]:
        """Force a training cycle immediately"""
        if self.is_training:
            return {"error": "Training already in progress"}
            
        try:
            await self._run_combined_training()
            return {
                "unsupervised_loss": float(self.metrics.unsupervised_loss),
                "supervised_loss": float(self.metrics.supervised_loss),
                "reinforcement_loss": float(self.metrics.reinforcement_loss),
                "improvement_rate": float(self.metrics.improvement_rate)
            }
        except Exception as e:
            logger.error(f"Error forcing training cycle: {e}")
            return {"error": str(e)}
