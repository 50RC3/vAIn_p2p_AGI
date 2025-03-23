import torch
import logging
from typing import Dict, Any, Optional, List, Tuple, Callable, Set
from dataclasses import dataclass, field
from ..model_storage import ModelStorage
from functools import lru_cache
import hashlib
import re
from datetime import datetime
from asyncio import Event, create_task
import asyncio
import time

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
    def __init__(self, model: torch.nn.Module, storage: ModelStorage, 
                 max_history: int = 1000, max_input_len: int = 512):
        """Initialize chatbot interface with configurable limits.
        
        Args:
            model: The underlying ML model
            storage: Model storage interface
            max_history: Maximum number of interactions to store
            max_input_len: Maximum allowed input length
        """
        self.model = model
        self.storage = storage
        self.history = []
        self.feedback_scores = []
        self.max_history = max_history
        self.max_input_len = max_input_len
        self.session_id = None
        self.response_cache = {}
        self.cache_ttl = 3600  # 1 hour cache lifetime
        self.event_handlers: Dict[str, Set[Callable]] = {
            'message_processed': set(),
            'feedback_stored': set(),
            'session_cleared': set()
        }
        self.state_change = Event()
        self.ui_queue = asyncio.Queue()
        self.ui_task = asyncio.create_task(self._process_ui_events())

    def start_session(self) -> str:
        """Start a new chat session."""
        import uuid
        self.session_id = str(uuid.uuid4())
        logger.info(f"Started new session {self.session_id}")
        return self.session_id

    async def subscribe(self, event_type: str, handler: Callable):
        """Subscribe to interface events"""
        if event_type in self.event_handlers:
            self.event_handlers[event_type].add(handler)

    async def _notify_handlers(self, event_type: str, data: Any):
        """Notify all handlers of an event"""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                create_task(handler(data))

    async def process_message(self, message: str) -> ChatResponse:
        """Process an incoming message and return a response.
        
        Args:
            message: Input message string
            
        Returns:
            ChatResponse containing the model's response
            
        Raises:
            ValueError: If message is invalid
            RuntimeError: If model processing fails
        """
        if not self.session_id:
            self.start_session()

        try:
            # Input validation
            if not message or not isinstance(message, str):
                raise ValueError("Invalid message format")
            
            if len(message) > self.max_input_len:
                raise ValueError(f"Message exceeds max length of {self.max_input_len}")

            # Process input and generate response
            with torch.no_grad():
                input_tensor = self._preprocess_message(message)
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                output = self.model(input_tensor)
                end_time.record()
                torch.cuda.synchronize()
                
                response = self._postprocess_output(output)
                latency = start_time.elapsed_time(end_time)
                
                # Manage history size
                if len(self.history) >= self.max_history:
                    self.history.pop(0)
                    if self.feedback_scores:
                        self.feedback_scores.pop(0)
                        
                self.history.append((message, response))
                
                logger.debug(f"Processed message in {latency}ms")
                
                await self._notify_handlers('message_processed', response)
                
                return ChatResponse(
                    text=response,
                    confidence=float(output.max()),
                    model_version=self.storage.get_model_version(),
                    latency=latency
                )

        except ValueError as e:
            logger.warning(f"Input validation error: {str(e)}")
            return ChatResponse(
                text="",
                confidence=0.0,
                model_version=self.storage.get_model_version(),
                latency=0.0,
                error=str(e)
            )
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}", exc_info=True)
            return ChatResponse(
                text="",
                confidence=0.0,
                model_version=self.storage.get_model_version(), 
                latency=0.0,
                error="Internal processing error"
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
        
    async def store_feedback(self, response: ChatResponse, score: float):
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
                'model_version': self.storage.get_model_version(),
                'response_hash': response_hash
            }
            
            self.feedback_scores.append(feedback)
            
            # Cache response with feedback
            self.response_cache[response_hash] = {
                'feedback': feedback,
                'expires': timestamp + self.cache_ttl
            }
            
            # Update storage with feedback
            if hasattr(self.storage, 'store_feedback'):
                await self.storage.store_feedback(feedback)
                
            logger.debug(f"Stored feedback score {score} for session {self.session_id}")
            
            await self._notify_handlers('feedback_stored', feedback)
            
        except Exception as e:
            logger.error(f"Error storing feedback: {str(e)}", exc_info=True)
            raise

    async def clear_session(self):
        """Clear current session data and persist feedback."""
        try:
            if self.feedback_scores:
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

    async def _process_ui_events(self):
        """Process UI events from queue"""
        while True:
            try:
                event = await self.ui_queue.get()
                await self._handle_ui_event(event)
                self.ui_queue.task_done()
            except Exception as e:
                logger.error(f"Error processing UI event: {e}")

    async def _handle_ui_event(self, event: UIEvent):
        """Handle UI events with proper error handling"""
        try:
            if event.type == "user_input":
                response = await self.process_message(event.data)
                await self._notify_handlers("message_processed", response)
            elif event.type == "feedback":
                await self.store_feedback(event.data["response"], event.data["score"])
        except Exception as e:
            logger.error(f"Error handling UI event: {e}")
            await self._notify_handlers("error", str(e))
