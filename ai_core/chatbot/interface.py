# Standard library imports
import asyncio
import hashlib
import logging
import re 
import time
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Protocol

# Third-party imports
import torch
from cachetools import LRUCache, TTLCache
from torch import nn

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
    text: str
    confidence: float  
    model_version: str
    latency: float
    error: Optional[str] = None

@dataclass
class UIEvent:
    type: str
    data: Any
    timestamp: float = field(default_factory=time.time)

class ChatbotInterface:
    event_handlers: Dict[str, Set[Callable[..., Any]]] = {}

    def __init__(self, model: ModelType, storage: StorageType,
                 max_history: int = 1000, max_input_len: int = 512,
                 interactive: Optional[Any] = None) -> None:
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

    async def start_session(self) -> str:
        """Start a new chat session."""
        import uuid
        self.session_id = str(uuid.uuid4())
        logger.info(f"Started new session {self.session_id}")
        return self.session_id

    async def subscribe(self, event_type: str, handler: Callable[..., Any]) -> None:
        """Subscribe to interface events"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = set()
        self.event_handlers[event_type].add(handler)

    async def _notify_handlers(self, event_type: str, data: Any) -> None:
        """Notify all handlers of an event"""
        if event_type in self.event_handlers:  
            for handler in self.event_handlers[event_type]:
                await asyncio.create_task(handler(data))

    async def _trigger_event(self, event_type: str, data: Any) -> None:
        """Trigger event handlers"""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                await handler(data)

    async def process_message(self, message: str, 
                            context: Optional[Dict[str, Any]] = None) -> ChatResponse:
        try:
            cache_key = self._generate_cache_key(message, context)
            
            cached_response = self.response_cache.get(cache_key)
            if cached_response and isinstance(cached_response, ChatResponse):
                await self._trigger_event('cache_hit', {'message': message})
                return cached_response

            await self._trigger_event('cache_miss', {'message': message})
            response = await self._generate_response(message, context)
            self.response_cache[cache_key] = response
            
            await self._trigger_event('response_generated', 
                                    {'message': message, 'response': response})
            return response

        except Exception as exc:
            error_msg = f"Error processing message: {str(exc)}"
            logger.error(error_msg, exc_info=True)
            await self._trigger_event('error', {'error': error_msg})
            raise RuntimeError(error_msg) from exc

    def _generate_cache_key(self, message: str, 
                          context: Optional[Dict[str, Any]]) -> str:
        # Generate deterministic cache key
        components = [message]
        if context:
            components.extend(f"{k}:{v}" for k, v in sorted(context.items()))
        return hashlib.sha256("|".join(components).encode()).hexdigest()

    async def _generate_response(self, message: str,
                               context: Optional[Dict[str, Any]] = None) -> ChatResponse:
        """Generate model response"""
        try:
            with torch.no_grad():
                input_tensor = self._preprocess_message(message)
                start_time = time.time()
                output = self.model(input_tensor) 
                latency = time.time() - start_time
                response = self._postprocess_output(output)

                try:
                    model_version = await self.storage.get_model_version()
                except AttributeError:
                    model_version = "unknown"

                return ChatResponse(
                    text=response,
                    confidence=float(output.max().item()),
                    model_version=model_version,
                    latency=latency
                )
        except torch.cuda.OutOfMemoryError as exc:
            error_msg = f"GPU memory exceeded: {str(exc)}"
            logger.critical(error_msg, exc_info=True)
            return ChatResponse(
                text="Server is overloaded, please try again later",
                confidence=0.0,
                model_version="error",
                latency=0.0,
                error=error_msg
            )
        except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
            error_msg = f"Response generation failed: {str(exc)}"
            logger.error(error_msg, exc_info=True)
            return ChatResponse(
                text="Error generating response",
                confidence=0.0,
                model_version="error",
                latency=0.0,
                error=error_msg
            )

    @lru_cache(maxsize=1000)
    def _preprocess_message(self, message: str) -> torch.Tensor:
        # Enhanced preprocessing with better normalization
        message = message.lower().strip()
        message = re.sub(r'\s+', ' ', message)
        message = re.sub(r'[^\w\s]', '', message)
        tokenized = torch.tensor([ord(c) for c in message], dtype=torch.long)
        return tokenized.unsqueeze(0)  # Add batch dimension
        
    def _postprocess_output(self, output: torch.Tensor) -> str:
        # Convert model output to human readable text
        probs = torch.softmax(output, dim=-1)
        tokens = torch.argmax(probs, dim=-1)
        return ''.join([chr(t) for t in tokens.squeeze()])

    async def store_feedback(self, response: ChatResponse, score: float) -> None:
        """Store user feedback for a response."""
        if not 0 <= score <= 1:
            raise ValueError("Score must be between 0 and 1")
            
        if not self.session_id:
            raise RuntimeError("No active session")
            
        try:
            # Use proper timestamp
            timestamp = datetime.now().timestamp()
            
            # Calculate response hash for caching
            response_hash = hashlib.md5(response.text.encode()).hexdigest()
            
            feedback = {
                'response': response,
                'score': float(score),
                'timestamp': timestamp,
                'session_id': self.session_id,
                'model_version': await self.storage.get_model_version(),
                'response_hash': response_hash
            }
            
            self.feedback_scores.append(feedback)
            # Cache feedback separately from response
            self.feedback_cache[response_hash] = {
                'feedback': feedback,
                'expires': timestamp + self.cache_ttl
            }
            
            # Update storage with feedback
            if hasattr(self.storage, 'store_feedback'):
                await self.storage.store_feedback(feedback)
                
            logger.debug("Stored feedback score %s for session %s", score, self.session_id)
            
            await self._notify_handlers('feedback_stored', feedback)
            
        except Exception as e:
            logger.error("Error storing feedback: %s", str(e), exc_info=True)
            raise

    async def clear_session(self) -> None:
        """Clear current session data and persist feedback."""
        try:
            if self.feedback_scores and self.session_id is not None:
                # Attempt to persist feedback before clearing
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
