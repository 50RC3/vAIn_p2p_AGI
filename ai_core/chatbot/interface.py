import torch
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from ..model_storage import ModelStorage

@dataclass
class ChatResponse:
    text: str
    confidence: float
    model_version: str
    latency: float

class ChatbotInterface:
    def __init__(self, model: torch.nn.Module, storage: ModelStorage):
        self.model = model
        self.storage = storage
        self.history = []
        self.feedback_scores = []
        
    async def process_message(self, message: str) -> ChatResponse:
        """Main interaction point for node-model communication"""
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
            
            self.history.append((message, response))
            
            return ChatResponse(
                text=response,
                confidence=float(output.max()),
                model_version=self.storage.get_model_version(),
                latency=latency
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
        """Collects user feedback for reinforcement learning"""
        self.feedback_scores.append({
            'response': response,
            'score': score,
            'timestamp': torch.cuda.current_device()
        })
        
    def get_interaction_history(self) -> List[Tuple[str, str, float]]:
        """Return history with feedback scores"""
        return [(msg, resp, score['score']) 
                for (msg, resp), score in zip(self.history, self.feedback_scores)]
