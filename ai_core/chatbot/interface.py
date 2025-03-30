# Standard library imports
import asyncio
import hashlib
import logging
import random
import re 
import time
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Protocol

# Third-party imports
import torch
import torch.nn.functional as F
from cachetools import LRUCache, TTLCache
from torch import nn

# Import the InputProcessor
try:
    # Try absolute import first (likely to work in more environments)
    from ai_core.chatbot.input_processor import InputProcessor
except ImportError:
    try:
        # Then try relative import
        from .input_processor import InputProcessor
    except ImportError:
        try:
            # Try with project prefix
            import sys
            sys.path.append('.')  # Add current directory to path
            from vAIn_p2p_AGI.ai_core.chatbot.input_processor import InputProcessor
        except ImportError:
            # If all imports fail, create a simple placeholder InputProcessor
            class InputProcessor:
                def __init__(self, input_size=512):
                    self.input_size = input_size
                
                def text_to_tensor(self, text):
                    # Placeholder implementation
                    import torch
                    return torch.zeros(self.input_size)
                    
                def tensor_to_text(self, tensor):
                    return "Processed text"
            print("Warning: InputProcessor module not found, using placeholder implementation")

# Try to import spaCy utilities
try:
    from ai_core.nlp.utils import load_spacy_model
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    
# Try to import NLTK utilities
try:
    from ai_core.nlp import (
        download_nltk_resources,
        initialize_nltk_components,
        analyze_text_nltk,
        get_nltk_sentiment,
        extract_keywords_nltk,
        determine_intent,
        NLTK_AVAILABLE
    )
except ImportError:
    NLTK_AVAILABLE = False

# Import learning modules
try:
    from ai_core.learning.unsupervised import UnsupervisedLearningModule
    from ai_core.learning.self_supervised import SelfSupervisedLearning
    try:
        from .rl_trainer import RLTrainer, RLConfig, TrainerState
    except ImportError:
        try:
            from ai_core.chatbot.rl_trainer import RLTrainer, RLConfig, TrainerState
        except ImportError:
            try:
                # Try with project prefix
                import sys
                sys.path.append('.')  # Add current directory to path
                from vAIn_p2p_AGI.ai_core.chatbot.rl_trainer import RLTrainer, RLConfig, TrainerState
            except ImportError:
                # Create placeholders if imports fail
                class RLConfig:
                    def __init__(self, **kwargs): pass
                class TrainerState:
                    def __init__(self): pass
                class RLTrainer:
                    def __init__(self, *args, **kwargs): pass
                    def register_callback(self, *args, **kwargs): pass
                    async def store_interaction(self, *args, **kwargs): pass
                print("Warning: rl_trainer module not found, using placeholder implementation")
    LEARNING_MODULES_AVAILABLE = True
except ImportError:
    LEARNING_MODULES_AVAILABLE = False

# Type aliases and protocols
ModelType = TypeVar('ModelType', bound=nn.Module)

class StorageProtocol(Protocol):
    async def get_model_version(self) -> str: ...
    async def store_feedback(self, feedback: Dict[str, Any]) -> None: ...
    async def persist_feedback(self, session_id: str, feedback: List[Dict[str, Any]]) -> None: ...
    """
    Persists feedback for a specific chat session.

    This asynchronous method stores the provided feedback data associated with
    the specified session ID in a persistent storage system.

    Args:
        session_id (str): The unique identifier for the chat session.
        feedback (List[Dict[str, Any]]): A list of feedback entries, where each entry
            is represented as a dictionary containing feedback data.

    Returns:
        None: This method doesn't return any value.

    Note:
        The method is asynchronous and should be awaited when called.
    """

StorageType = TypeVar('StorageType', bound=StorageProtocol)

logger = logging.getLogger(__name__)

@dataclass
class ChatResponse:
    id: str
    text: str
    confidence: float  
    timestamp: str
    error: bool

@dataclass
class UIEvent:
    type: str
    data: Any
    timestamp: float = field(default_factory=time.time)

@dataclass
class LearningConfig:
    """Configuration for learning modules"""
    # Self-supervised learning config
    enable_self_supervised: bool = True  # Enable by default
    self_supervised_model: str = "distilbert-base-uncased"  # Smaller model for better performance
    mask_probability: float = 0.15
    
    # Unsupervised learning config
    enable_unsupervised: bool = True  # Enable by default
    unsupervised_clusters: int = 10
    
    # Reinforcement learning config
    enable_reinforcement: bool = True  # Enable by default
    rl_learning_rate: float = 1e-4
    discount_factor: float = 0.95
    prioritized_replay: bool = True
    rl_update_interval: int = 5
    
    # General learning config
    batch_size: int = 8
    learning_rate: float = 2e-5
    save_interval: int = 500  # Save more frequently
    model_path: str = "./models"
    max_context_length: int = 512
    min_sample_length: int = 3  # Allow learning from shorter examples
    
    # Memory settings
    memory_size: int = 10000
    training_interval: int = 30  # Train more frequently (30 seconds)
    
    # Integration settings
    use_feedback_for_all: bool = True  # Use feedback across all learning types
    cross_learning: bool = True  # Enable knowledge sharing between learning types
    
    # New settings for improved learning
    adaptive_learning: bool = True  # Adjust learning based on performance
    min_samples_for_cluster_learning: int = 5  # Minimum samples per cluster before learning
    embedding_cache_size: int = 1000  # Size of embedding cache for improved efficiency
    learning_overlap_ratio: float = 0.3  # What portion of examples to share between learning methods
    prioritize_user_feedback: bool = True  # Prioritize examples with explicit feedback

class ChatbotInterface:
    event_handlers: Dict[str, Set[Callable[..., Any]]] = {}

    def __init__(self, model: ModelType, storage: StorageType,
                 max_history: int = 1000, max_input_len: int = 512,
                 interactive: Optional[Any] = None,
                 learning_config: Optional[LearningConfig] = None) -> None:
        self.model = model
        self.storage = storage
        self.history: List[Tuple[str, str]] = []
        self.feedback_scores: List[Dict[str, Any]] = []
        self.max_history = max_history
        self.max_input_len = max_input_len
        self.session_id: Optional[str] = None
        self.response_cache: LRUCache[str, ChatResponse] = LRUCache(maxsize=1000)
        self.feedback_cache: TTLCache[str, Dict[str, Any]] = TTLCache(maxsize=1000, ttl=3600)
        self.context_cache: TTLCache[str, Dict[str, Any]] = TTLCache(maxsize=1000, ttl=3600)
        self.state_change = asyncio.Event()
        self.ui_queue: asyncio.Queue[UIEvent] = asyncio.Queue()
        self.ui_task = asyncio.create_task(self._process_ui_events())
        self.cache_ttl = 3600  # 1 hour TTL default
        self.interactive = interactive
        
        # Initialize event handlers as instance attribute
        self.event_handlers = {}
        
        # Learning components
        self.learning_config = learning_config or LearningConfig()
        self.unsupervised_module = None
        self.self_supervised_module = None
        self.rl_trainer = None
        
        # Track whether each learning type is enabled
        self.learning_enabled = LEARNING_MODULES_AVAILABLE and (
            self.learning_config.enable_unsupervised or 
            self.learning_config.enable_self_supervised or 
            self.learning_config.enable_reinforcement
        )
        
        self.periodic_train_task = None
        
        if self.learning_enabled:
            self._initialize_learning_modules()
            # Create the periodic training task but don't reference it directly in __init__
            self.periodic_train_task = asyncio.create_task(self._periodic_train_wrapper())
        
        # Initialize spaCy if available
        self.spacy_available = SPACY_AVAILABLE
        if self.spacy_available:
            # Schedule spaCy model loading
            asyncio.create_task(self._initialize_nlp())
            
        # Initialize NLTK if available
        self.nltk_available = NLTK_AVAILABLE
        if self.nltk_available:
            # Schedule NLTK resources download
            asyncio.create_task(self._initialize_nltk())
        
        # Add InputProcessor for tensor handling
        self.input_processor = InputProcessor(
            input_size=self.model.input_size if hasattr(self.model, 'input_size') else 512
        )

    async def _initialize_nlp(self):
        """Initialize NLP components if available"""
        if self.spacy_available:
            try:
                await load_spacy_model()
                logger.info("spaCy NLP components initialized")
            except Exception as e:
                logger.error(f"Failed to initialize spaCy NLP components: {e}")
                self.spacy_available = False
                
    async def _initialize_nltk(self):
        """Initialize NLTK components if available"""
        if self.nltk_available:
            try:
                success = await download_nltk_resources()
                if success:
                    initialize_nltk_components()
                    logger.info("NLTK components initialized")
                else:
                    logger.warning("NLTK component initialization incomplete")
            except (ImportError, ValueError, RuntimeError) as e:
                logger.error(f"Failed to initialize NLTK components: {e}")
                self.nltk_available = False

    def _initialize_learning_modules(self) -> None:
        """Initialize learning modules based on configuration"""
        try:
            if not LEARNING_MODULES_AVAILABLE:
                logger.warning("Learning modules not available, disabling learning features")
                return
            
            # Ensure the imports are properly loaded at the start
            try:
                from ai_core.chatbot.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
            except ImportError:
                # Create minimal implementations if imports fail
                class ReplayBuffer:
                    def __init__(self, buffer_size=10000):
                        self.buffer = []
                    def __len__(self):
                        return len(self.buffer)
                        
                class PrioritizedReplayBuffer(ReplayBuffer):
                    def __init__(self, buffer_size=10000, alpha=0.6, beta=0.4):
                        super().__init__(buffer_size)
                        
            if self.learning_config.enable_unsupervised:
                logger.info("Initializing unsupervised learning module")
                self.unsupervised_module = UnsupervisedLearningModule(
                    embedding_dim=768,
                    n_clusters=self.learning_config.unsupervised_clusters,
                    buffer_size=max(200, self.learning_config.batch_size * 5)
                )
                self.unsupervised_module.initialize()
                
            if self.learning_config.enable_self_supervised:
                logger.info("Initializing self-supervised learning module")
                self.self_supervised_module = SelfSupervisedLearning(
                    model_name=self.learning_config.self_supervised_model,
                    learning_rate=self.learning_config.learning_rate,
                    mask_probability=self.learning_config.mask_probability,
                    save_interval=self.learning_config.save_interval,
                    model_path=f"{self.learning_config.model_path}/self_supervised"
                )
                
            if self.learning_config.enable_reinforcement:
                logger.info("Initializing reinforcement learning module")
                
                # Important: Create RL config the right way, removing embedder
                rl_config = RLConfig(
                    learning_rate=self.learning_config.rl_learning_rate,
                    gamma=self.learning_config.discount_factor,
                    memory_size=self.learning_config.memory_size,
                    batch_size=min(32, self.learning_config.batch_size * 2),
                    update_interval=self.learning_config.rl_update_interval
                )
                
                quality_model = nn.Sequential(
                    nn.Linear(768, 384),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(384, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1)
                )
                
                # Initialize with proper parameters to match constructor
                try:
                    self.rl_trainer = RLTrainer(
                        quality_model=quality_model,
                        config=rl_config
                    )
                except TypeError as e:
                    logger.error(f"RLTrainer initialization failed: {e}")
                    # Try alternative initialization without embedder argument
                    self.rl_trainer = RLTrainer(quality_model, rl_config)
                
                if self.rl_trainer:
                    self.rl_trainer.register_callback('update_completed', self._on_rl_update)
            
            # Initialize feedback tracking
            self.feedback_history = []
            self.feedback_by_cluster = {}
            self.embedding_cache = LRUCache(maxsize=self.learning_config.embedding_cache_size)
            self.embedding_cache_hits = 0
            self.embedding_cache_misses = 0
                
            logger.info("Learning modules initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize learning modules: {e}")
            self.learning_enabled = False

    async def _on_rl_update(self, data: Dict[str, float]) -> None:
        """Callback for RL training updates"""
        logger.debug(f"RL update: loss={data.get('loss', 'N/A')}, grad_step={data.get('grad_step', 'N/A')}")
        await self._notify_handlers('rl_update', data)

    async def _trigger_event(self, event_name: str, data: Any) -> None:
        """Trigger an event for registered handlers
        
        Args:
            event_name: The name of the event
            data: The data associated with the event
        """
        if event_name not in self.event_handlers:
            return
            
        for handler in self.event_handlers.get(event_name, set()):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except (ValueError, TypeError, RuntimeError, KeyError, AttributeError) as e:
                logger.error(f"Error in event handler for '{event_name}': {e}")
            except Exception as e:
                logger.critical(f"Unexpected error in event handler for '{event_name}': {e}", exc_info=True)
    
    def register_handler(self, event_name: str, handler: Callable[..., Any]) -> None:
        """Register a handler for an event
        
        Args:
            event_name: The name of the event
            handler: The function to call when the event is triggered
        """
        if event_name not in self.event_handlers:
            self.event_handlers[event_name] = set()
        self.event_handlers[event_name].add(handler)
    
    def subscribe(self, event_name: str, handler: Callable[..., Any]) -> None:
        """Subscribe to an event (alias for register_handler)
        
        Args:
            event_name: The name of the event
            handler: The function to call when the event is triggered
        """
        self.register_handler(event_name, handler)
        
    def unregister_handler(self, event_name: str, handler: Callable[..., Any]) -> None:
        """Unregister a handler for an event
        
        Args:
            event_name: The name of the event
            handler: The function to unregister
        """
        if event_name in self.event_handlers and handler in self.event_handlers[event_name]:
            self.event_handlers[event_name].remove(handler)

    async def _notify_handlers(self, event_name: str, data: Any) -> None:
        """Notify event handlers of an event - uses _trigger_event internally
        
        Args:
            event_name: The name of the event
            data: The data associated with the event
        """
        await self._trigger_event(event_name, data)

    async def start_session(self) -> str:
        """Start a new chat session
        
        Returns:
            str: The new session ID
        """
        try:
            # Clear any existing session first
            await self.clear_session()
            
            # Generate a new session ID
            session_id = f"session_{int(time.time())}_{random.randint(1000, 9999)}"
            self.session_id = session_id
            
            logger.info(f"Started new session: {session_id}")
            
            # Notify handlers
            await self._notify_handlers('session_started', session_id)
            
            return session_id
            
        except Exception as e:
            logger.error(f"Error starting session: {str(e)}", exc_info=True)
            raise

    async def clear_session(self) -> None:
        """Clear current session data and persist feedback."""
        try:
            if self.feedback_scores and self.session_id is not None:
                if hasattr(self.storage, 'persist_feedback'):
                    await self.storage.persist_feedback(self.session_id, self.feedback_scores)
                    
            self.history = []
            self.feedback_scores = [] 
            old_session = self.session_id
            self.session_id = None
            logger.info(f"Cleared session data for {old_session}")
            
            await self._notify_handlers('session_cleared', old_session)
            
        except Exception as e:
            logger.error(f"Error clearing session: {str(e)}", exc_info=True)
            raise

    async def _periodic_train_wrapper(self) -> None:
        """Wrapper around periodic train to handle exceptions gracefully"""
        try:
            await self._periodic_train()
        except asyncio.CancelledError:
            logger.info("Periodic training task cancelled")
        except Exception as e:
            logger.error(f"Periodic training task failed: {e}")

    async def cleanup(self) -> None:
        """Clean up resources before shutdown"""
        # Cancel periodic training task if running
        if hasattr(self, 'periodic_train_task') and self.periodic_train_task:
            self.periodic_train_task.cancel()
            try:
                await self.periodic_train_task
            except asyncio.CancelledError:
                pass
                
        # Cancel UI task if running
        if hasattr(self, 'ui_task') and self.ui_task:
            self.ui_task.cancel()
            try:
                await self.ui_task
            except asyncio.CancelledError:
                pass
                
        # Clear all message handlers
        if hasattr(self, 'event_handlers'):
            self.event_handlers.clear()
            
        # Clear caches to free memory
        if hasattr(self, 'response_cache'):
            self.response_cache.clear()
        if hasattr(self, 'feedback_cache'):
            self.feedback_cache.clear()
        if hasattr(self, 'context_cache'):
            self.context_cache.clear()
            
        logger.info("Chatbot interface cleaned up successfully")
    
    async def _periodic_train(self) -> None:
        """Periodically train models from accumulated history with improved coordination"""
        if not self.learning_enabled:
            return
            
        while True:
            try:
                # Wait for the configured interval
                await asyncio.sleep(self.learning_config.training_interval)
                
                # If there's enough history, do a training pass
                if len(self.history) > 3:  # Reduced threshold to enable more frequent learning
                    logger.debug(f"Running periodic training on {len(self.history)} history items")
                    
                    # Extract messages and responses for learning
                    messages = [msg for msg, _ in self.history[-50:]]  # Use recent history
                    responses = [resp for _, resp in self.history[-50:]]
                    
                    # If we have feedback scores, prioritize examples with higher feedback
                    prioritized_examples = []
                    if self.learning_config.prioritize_user_feedback and len(self.feedback_scores) > 0:
                        # Match history items with feedback scores
                        for i, ((msg, resp), feedback_item) in enumerate(zip(self.history, self.feedback_scores)):
                            if i >= min(len(self.history), len(self.feedback_scores)):
                                break
                                
                            # Extract score from feedback
                            if isinstance(feedback_item, dict) and 'score' in feedback_item:
                                score = feedback_item['score']
                                if score > 0.7:  # Only prioritize high-scoring examples
                                    prioritized_examples.append((msg, resp, score))
                    
                    # Combine all text for unsupervised learning
                    all_text = messages + responses
                    
                    # Include prioritized examples at the beginning
                    for msg, resp, _ in prioritized_examples:
                        if msg not in all_text:
                            all_text.insert(0, msg)
                        if resp not in all_text:
                            all_text.insert(1, resp)
                    
                    # Train unsupervised model if enabled
                    if self.learning_config.enable_unsupervised and self.unsupervised_module:
                        # Process both messages and responses for clustering
                        for text in all_text:
                            self.unsupervised_module.add_to_buffer(text)
                        unsupervised_success = await self.unsupervised_module.try_train_from_buffer()
                        if unsupervised_success:
                            logger.info("Unsupervised model updated from chat history")
                            
                            # Apply cluster-based feedback to RL model if available
                            if self.learning_config.use_feedback_for_all and hasattr(self, 'feedback_by_cluster'):
                                await self._apply_cluster_feedback_to_rl()
                    
                    # Train self-supervised model if enabled
                    if self.learning_config.enable_self_supervised and self.self_supervised_module:
                        # Process texts with interleaved user messages and responses
                        # Use user messages more for better learning (2:1 ratio)
                        training_texts = []
                        
                        # Add prioritized examples first
                        for msg, resp, score in prioritized_examples:
                            training_texts.append(msg)
                            # For very high scores, include the response multiple times
                            if score > 0.9:
                                training_texts.extend([resp, resp])
                            else:
                                training_texts.append(resp)
                                
                        # Add regular examples
                        training_texts.extend(messages + messages + responses)
                        random.shuffle(training_texts)  # Shuffle to avoid biased learning
                        
                        # Remove duplicates while preserving order
                        seen = set()
                        training_texts = [x for x in training_texts 
                                       if not (x in seen or seen.add(x))]
                        
                        total_trained = 0
                        for text in training_texts:
                            loss = await self.self_supervised_module.training_step(text)
                            if loss is not None:
                                total_trained += 1
                        
                        # Save model if we've processed enough examples
                        if total_trained > 0:
                            logger.info(f"Self-supervised model trained on {total_trained} examples")
                            if self.self_supervised_module.examples_processed % self.learning_config.save_interval == 0:
                                await self.self_supervised_module.save_model()
                    
                    # Train RL model based on history and feedback
                    if self.learning_config.enable_reinforcement and self.rl_trainer and self.self_supervised_module:
                        await self._train_rl_from_history()
                    
                    # Apply cross-learning between different approaches
                    if self.learning_config.cross_learning:
                        await self._apply_cross_learning()
                    
                    # Log cache efficiency statistics
                    if hasattr(self, 'embedding_cache_hits'):
                        total_requests = self.embedding_cache_hits + self.embedding_cache_misses
                        if total_requests > 0:
                            hit_rate = self.embedding_cache_hits / total_requests * 100
                            logger.debug(f"Embedding cache hit rate: {hit_rate:.1f}% ({self.embedding_cache_hits}/{total_requests})")
                    
                    # Notify about the training
                    await self._notify_handlers('periodic_training_completed', {
                        'history_size': len(self.history),
                        'messages_processed': len(all_text),
                        'timestamp': time.time()
                    })
                
            except asyncio.CancelledError:
                logger.info("Periodic training task cancelled")
                break
            except Exception as e:
                logger.error(f"Error during periodic training: {e}")
                # Don't break the loop on errors, just continue to the next iteration
                
    async def _train_rl_from_history(self):
        """Train reinforcement learning model from chat history with feedback"""
        if not self.rl_trainer or not self.self_supervised_module or len(self.history) < 3:
            return
            
        try:
            # Process recent history items that have feedback
            processed = 0
            
            # Match history items with feedback scores
            for i, ((msg, resp), feedback_item) in enumerate(zip(self.history, self.feedback_scores)):
                if i >= min(len(self.history), len(self.feedback_scores)):
                    break
                    
                # Extract score from feedback
                if isinstance(feedback_item, dict) and 'score' in feedback_item:
                    score = feedback_item['score']
                    
                    # Get embeddings - use cache for efficiency
                    msg_embedding = await self.get_embedding(msg)
                    resp_embedding = await self.get_embedding(resp)
                    
                    # Store interaction with explicit reward
                    if msg_embedding is not None and resp_embedding is not None:
                        await self.rl_trainer.store_interaction(
                            state=msg_embedding,
                            action=resp_embedding,
                            reward=score,
                            next_state=msg_embedding  # Using same state for simplicity
                        )
                        
                        processed += 1
                    
            if processed > 0:
                logger.info(f"Trained RL model on {processed} historical interactions with feedback")
                
        except Exception as e:
            logger.error(f"Error training RL from history: {e}")
            
    async def _apply_cluster_feedback_to_rl(self):
        """Apply cluster-based feedback to the RL model"""
        if not self.rl_trainer or not self.self_supervised_module or not hasattr(self, 'feedback_by_cluster'):
            return
            
        try:
            # For each cluster with feedback
            for cluster, feedback_items in self.feedback_by_cluster.items():
                if not feedback_items:
                    continue
                    
                # Calculate average feedback score for this cluster
                avg_score = sum(item['score'] for item in feedback_items) / len(feedback_items)
                
                # If we have stored messages for this cluster
                if hasattr(self, '_cluster_messages') and cluster in self._cluster_messages:
                    # Use a representative message from the cluster - use message from feedback if available
                    cluster_msg = next((item['message'] for item in feedback_items if 'message' in item), 
                                      self._cluster_messages[cluster][0])
                    
                    # Generate a synthetic response
                    if hasattr(self.model, 'generate_text'):
                        synthetic_response = self.model.generate_text(cluster_msg, max_length=30)
                    else:
                        synthetic_response = f"Response for cluster {cluster}"
                        
                    # Get embeddings - use cache
                    msg_embedding = await self.get_embedding(cluster_msg)
                    resp_embedding = await self.get_embedding(synthetic_response)
                    
                    if msg_embedding is not None and resp_embedding is not None:
                        # Add to RL training with the average cluster score
                        await self.rl_trainer.store_interaction(
                            state=msg_embedding,
                            action=resp_embedding,
                            reward=avg_score,
                            next_state=msg_embedding
                        )
                    
            logger.info(f"Applied feedback from {len(self.feedback_by_cluster)} clusters to RL model")
                    
        except Exception as e:
            logger.error(f"Error applying cluster feedback to RL: {e}")
            
    async def _apply_cross_learning(self):
        """Apply cross-learning between different learning approaches"""
        if not self.learning_config.cross_learning:
            return
            
        try:
            # Share information between unsupervised and self-supervised learning
            if self.unsupervised_module and self.self_supervised_module:
                # Get unsupervised cluster information
                if hasattr(self, '_cluster_messages'):
                    # For each cluster, train self-supervised model on representative examples
                    for cluster, messages in self._cluster_messages.items():
                        if not messages:
                            continue
                            
                        # Take a few examples from each cluster for training
                        sample_size = min(3, len(messages))
                        for msg in random.sample(messages, sample_size):
                            # Train self-supervised model on cluster examples
                            await self.self_supervised_module.training_step(msg)
                            
            # If we have RL model and feedback, use it to guide unsupervised clustering
            if self.rl_trainer and self.unsupervised_module and len(self.feedback_scores) > 0:
                # For high-scoring responses, reinforce those patterns in unsupervised learning
                high_scores = [(i, f['score']) for i, f in enumerate(self.feedback_scores) if f.get('score', 0) > 0.8]
                
                if high_scores:
                    # Get messages that led to high-scoring responses
                    for idx, score in high_scores[:5]:  # Use top 5 maximum
                        if idx < len(self.history):
                            msg, resp = self.history[idx]
                            # Add message multiple times to buffer to reinforce its pattern
                            for _ in range(3):  # Add 3 copies to increase weight
                                self.unsupervised_module.add_to_buffer(msg)
                                
            # Improve RL with self-supervised embeddings
            if self.rl_trainer and self.self_supervised_module and hasattr(self.rl_trainer, 'get_training_stats'):
                # Update embedding quality if RL is showing improvement
                stats = self.rl_trainer.get_training_stats()
                if 'avg_td_error' in stats and stats['avg_td_error'] < 0.5:
                    # RL is improving, get high-performing examples for self-supervised learning
                    if len(self.history) > 10:
                        # Get top samples by RL performance
                        for i in range(min(5, len(self.history))):
                            msg, resp = self.history[-(i+1)]  # Get from most recent
                            # Train self-supervised on both message and response
                            await self.self_supervised_module.training_step(msg)
                            await self.self_supervised_module.training_step(resp)
                                
        except Exception as e:
            logger.error(f"Error in cross-learning: {e}")

    def get_interaction_history(self) -> List[Tuple[str, str, float]]:
        """Return history with feedback scores"""
        return [(msg, resp, score['score']) 
                for (msg, resp), score in zip(self.history, self.feedback_scores)]

    async def _process_ui_events(self) -> None:
        """Process UI events from queue"""
        while True:
            try:
                event = await self.ui_queue.get()
                await self._handle_ui_event(event)
                self.ui_queue.task_done()
            except Exception as e:
                logger.error(f"Error processing UI event: {e}")

    async def _handle_ui_event(self, event: UIEvent) -> None:
        """Handle UI events with proper error handling"""
        try:
            if event.type == "user_input":
                response = await self.process_message(event.data)
                await self._notify_handlers("message_processed", response)
            elif event.type == "feedback":
                await self.store_feedback(event.data["response"], event.data["score"])
        except (KeyError, ValueError, RuntimeError) as e:
            logger.error("Error handling UI event: %s", str(e))
            await self._notify_handlers("error", str(e))
            
    async def store_feedback(self, response: str, score: float) -> None:
        """Store user feedback for a response
        
        Args:
            response: The response text that received feedback
            score: The feedback score (typically 0.0 to 1.0)
        """
        try:
            # Create feedback entry
            feedback = {
                "response": response,
                "score": float(score),
                "timestamp": time.time()
            }
            
            # Add to feedback scores
            self.feedback_scores.append(feedback)
            
            # Add to feedback cache using response hash
            response_hash = hashlib.md5(response[:50].encode()).hexdigest()
            self.feedback_cache[response_hash] = {"feedback": feedback}
            
            # Store in persistent storage if available
            await self.storage.store_feedback(feedback)
            
            # Update cluster feedback if available
            if hasattr(self, 'feedback_by_cluster') and hasattr(self, 'unsupervised_module'):
                # Try to determine which cluster this feedback belongs to
                if len(self.history) > 0:
                    last_message, _ = self.history[-1]
                    cluster = await self._process_unsupervised(last_message)
                    if cluster is not None:
                        if cluster not in self.feedback_by_cluster:
                            self.feedback_by_cluster[cluster] = []
                        self.feedback_by_cluster[cluster].append({
                            "message": last_message,
                            "response": response,
                            "score": score
                        })
            
            logger.info(f"Stored feedback with score {score} for response")
            await self._notify_handlers('feedback_stored', feedback)
            
        except Exception as e:
            logger.error(f"Error storing feedback: {e}")
    
    async def _generate_response(self, message: str, context: Optional[torch.Tensor]) -> Tuple[str, float]:
        """Generate a response to a message
        
        Args:
            message: The user's message
            context: Optional embedding context for the response
            
        Returns:
            Tuple[str, float]: The generated response text and confidence
        """
        try:
            # Generate response using the model
            if hasattr(self.model, 'generate_text') and callable(self.model.generate_text):
                response_text = await asyncio.to_thread(self.model.generate_text, message, max_length=100)
            else:
                response_text = f"Hello! I received your message: '{message}'. I'm still learning how to respond effectively."
            
            # Calculate confidence based on response length
            confidence = min(0.8, max(0.4, len(response_text) / 200))
            
            return response_text, confidence
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm sorry, I couldn't process that request properly.", 0.0

    def _clean_response(self, response_text: str) -> str:
        """Clean up response text"""
        return response_text.strip()

    async def process_message(self, message: str) -> ChatResponse:
        """Process user input message and generate response"""
        try:
            logger.info(f"Preprocessing message: '{message}'")
            
            # Process through NLP pipeline if available
            processed_input = await self._preprocess_input(message)
            
            # Generate embeddings for similarity search
            message_embedding = None
            if self.learning_enabled:
                message_embedding = await self.get_embedding(message)
                
                # Process unsupervised learning in background
                if self.unsupervised_module:
                    try:
                        # NOTE: await all coroutines properly!
                        cluster = await self._process_unsupervised(message, message_embedding)
                    except Exception as e:
                        logger.warning(f"Error in unsupervised processing: {e}")
            
            # Generate model response - ensure model outputs are sensible
            response_text, confidence = await self._generate_response(processed_input, message_embedding)
            
            # Clean up response text
            response_text = self._clean_response(response_text)
            
            # Create response object
            response_id = hashlib.md5(f"{message}{response_text}{time.time()}".encode()).hexdigest()
            response = ChatResponse(
                id=response_id,
                text=response_text,
                confidence=confidence,
                timestamp=datetime.now().isoformat(),
                error=False
            )
            
            # Store response in history for later learning
            if response and not response.error and hasattr(self, 'history'):
                self.history.append((message, response.text))
                
                # Use proper await here!
                if self.learning_enabled and self.self_supervised_module:
                    try:
                        # NOTE: await to prevent coroutine warning
                        await self._process_self_supervised(response.text)
                    except Exception as e:
                        logger.warning(f"Error in self-supervised learning: {e}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return ChatResponse(
                id=hashlib.md5(f"{message}{time.time()}".encode()).hexdigest(),
                text=f"Sorry, I encountered an error: {str(e)}",
                confidence=0.0,
                timestamp=datetime.now().isoformat(),
                error=True
            )

    async def _preprocess_input(self, message: str) -> str:
        """Preprocess input message"""
        return message.strip()
    
    async def get_embedding(self, text: str) -> Optional[torch.Tensor]:
        """Get embeddings for input text with caching"""
        if not self.learning_enabled:
            return None
            
        # Check cache first
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if hasattr(self, 'embedding_cache') and cache_key in self.embedding_cache:
            if hasattr(self, 'embedding_cache_hits'):
                self.embedding_cache_hits += 1
            return self.embedding_cache[cache_key]
        
        if hasattr(self, 'embedding_cache_misses'):
            self.embedding_cache_misses += 1
        
        # Get embeddings from self-supervised module
        if self.self_supervised_module:
            try:
                # Use awaitable properly
                embedding = await self.self_supervised_module.get_embedding(text)
                
                # Cache the result
                if hasattr(self, 'embedding_cache'):
                    self.embedding_cache[cache_key] = embedding
                    
                return embedding
            except Exception as e:
                logger.error(f"Error getting embeddings: {e}")
        
        return None

    async def _process_unsupervised(self, message: str, embedding: Optional[torch.Tensor] = None) -> Optional[int]:
        """Process message with unsupervised learning
        
        Args:
            message: The message to process
            embedding: Optional precomputed embedding
            
        Returns:
            Optional[int]: The cluster ID if successful, None otherwise
        """
        if not self.unsupervised_module:
            return None
            
        try:
            # Get embedding for the message if not provided
            embedding = embedding or await self.get_embedding(message)
            
            if embedding is None:
                return None
                
            # Send to unsupervised module for clustering
            cluster = await asyncio.to_thread(self.unsupervised_module.predict_cluster, embedding.cpu().numpy())
            
            # Track messages by cluster for better learning
            if not hasattr(self, '_cluster_messages'):
                self._cluster_messages = {}
                
            if cluster not in self._cluster_messages:
                self._cluster_messages[cluster] = []
                
            # Add to cluster history (avoid duplicates)
            if message not in self._cluster_messages[cluster]:
                self._cluster_messages[cluster].append(message)
                # Limit the size to avoid memory issues
                if len(self._cluster_messages[cluster]) > 50:
                    self._cluster_messages[cluster] = self._cluster_messages[cluster][-50:]
            
            return cluster
            
        except Exception as e:
            logger.warning(f"Error in unsupervised processing: {e}")
            return None
            
    async def _process_self_supervised(self, message: str) -> Optional[float]:
        """Process message with self-supervised learning
        
        Args:
            message: The message to process
            
        Returns:
            Optional[float]: The training loss if successful, None otherwise
        """
        if not self.self_supervised_module:
            return None
            
        try:
            # Check message length
            if len(message.split()) < self.learning_config.min_sample_length:
                return None
                
            # Train the self-supervised model on the message
            loss = await self.self_supervised_module.training_step(message)
            return loss
            
        except Exception as e:
            logger.warning(f"Error in self-supervised processing: {e}")
            return None