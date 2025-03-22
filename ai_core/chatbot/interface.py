import torch
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from ..model_storage import ModelStorage

logger = logging.getLogger(__name__)

@dataclass
class ChatResponse:
    text: str
    confidence: float
    model_version: str
    latency: float
    error: Optional[str] = None

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

    def start_session(self) -> str:
        """Start a new chat session."""
        import uuid
        self.session_id = str(uuid.uuid4())
        logger.info(f"Started new session {self.session_id}")
        return self.session_id

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

    def _preprocess_message(self, message: str) -> torch.Tensor:
        # Convert message to model input format
        tokenized = torch.tensor([ord(c) for c in message], dtype=torch.long)
        return tokenized.unsqueeze(0)  # Add batch dimension
        
    def _postprocess_output(self, output: torch.Tensor) -> str:
        # Convert model output to human readable text
        probs = torch.softmax(output, dim=-1)
        tokens = torch.argmax(probs, dim=-1)
        return ''.join([chr(t) for t in tokens.squeeze()])
        
    def store_feedback(self, response: ChatResponse, score: float):
        """Store user feedback for a response.
        
        Args:
            response: The ChatResponse that received feedback
            score: Feedback score between 0 and 1
        """
        if not 0 <= score <= 1:
            raise ValueError("Score must be between 0 and 1")
            
        try:
            self.feedback_scores.append({
                'response': response,
                'score': score,
                'timestamp': torch.cuda.current_device(),
                'session_id': self.session_id
            })
            logger.debug(f"Stored feedback score {score}")
        except Exception as e:
            logger.error(f"Error storing feedback: {str(e)}")

    def clear_session(self):
        """Clear current session data."""
        self.history = []
        self.feedback_scores = []
        self.session_id = None
        logger.info("Cleared session data")

    def get_interaction_history(self) -> List[Tuple[str, str, float]]:
        """Return history with feedback scores"""
        return [(msg, resp, score['score']) 
                for (msg, resp), score in zip(self.history, self.feedback_scores)]
