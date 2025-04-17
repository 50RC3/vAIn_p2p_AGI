import asyncio
import logging
import time
import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelStorage:
    """Storage for model data, weights, and metadata."""
    
    def __init__(self, base_path: str = "model_storage"):
        """Initialize the model storage."""
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
        self.feedback_file = os.path.join(base_path, "feedback.json")
        logger.info(f"ModelStorage initialized with base path: {base_path}")
        
    async def get_model_version(self) -> str:
        """Get the current model version."""
        # Return a predefined version or read from a file in a real implementation
        return self.version
        
    async def store_feedback(self, feedback: Dict[str, Any]) -> None:
        """Store a single feedback entry."""
        # Convert complex objects to serializable format
        serializable_feedback = {
            'session_id': feedback.get('session_id', 'unknown'),
            'score': feedback.get('score', 0.0),
            'timestamp': feedback.get('timestamp', time.time()),
            'response_hash': feedback.get('response_hash', ''),
            'model_version': feedback.get('model_version', self.version),
        }
        
        # Add response data if available
        if 'response' in feedback:
            response = feedback['response']
            serializable_feedback['response_text'] = getattr(response, 'text', '')
            serializable_feedback['response_confidence'] = getattr(response, 'confidence', 0.0)
            serializable_feedback['response_latency'] = getattr(response, 'latency', 0.0)
        
        logger.debug(f"Storing feedback: {serializable_feedback}")
        
        # Implementation would typically store this in a database
        # For simplicity, we'll just log it
        logger.info(f"Feedback stored: score={serializable_feedback['score']}")
        
    async def persist_feedback(self, session_id: str, feedback: List[Dict[str, Any]]) -> None:
        """Persist feedback for a specific session."""
        try:
            # Convert feedback to serializable format
            serializable_feedback = []
            for entry in feedback:
                converted = {
                    'session_id': session_id,
                    'score': entry.get('score', 0.0),
                    'timestamp': entry.get('timestamp', time.time()),
                    'response_hash': entry.get('response_hash', ''),
                    'model_version': entry.get('model_version', self.version),
                }
                
                if 'response' in entry:
                    response = entry['response']
                    converted['response_text'] = getattr(response, 'text', '')
                    converted['response_confidence'] = getattr(response, 'confidence', 0.0)
                    converted['response_latency'] = getattr(response, 'latency', 0.0)
                
                serializable_feedback.append(converted)
            
            # Load existing feedback if file exists
            existing_data = []
            if self.feedback_file.exists():
                with open(self.feedback_file, 'r') as f:
                    try:
                        existing_data = json.load(f)
                    except json.JSONDecodeError:
                        logger.warning(f"Error reading feedback file, starting with empty data")
            
            # Append new feedback and save
            combined_data = existing_data + serializable_feedback
            with open(self.feedback_file, 'w') as f:
                json.dump(combined_data, f, indent=2)
                
            logger.info(f"Persisted {len(serializable_feedback)} feedback entries for session {session_id}")
            
        except Exception as e:
            logger.error(f"Error persisting feedback: {e}")
            raise

    def save_feedback(self, user_id: str, message: str, response: str, rating: int) -> bool:
        """Save user feedback for a response."""
        feedback = self._load_feedback()
        
        # Add new feedback
        if user_id not in feedback:
            feedback[user_id] = []
            
        feedback[user_id].append({
            "message": message,
            "response": response,
            "rating": rating,
            "timestamp": str(datetime.datetime.now())
        })
        
        # Save updated feedback
        with open(self.feedback_file, 'w') as f:
            json.dump(feedback, f, indent=2)
            
        return True
        
    def _load_feedback(self) -> Dict[str, Any]:
        """Load existing feedback data."""
        if not os.path.exists(self.feedback_file):
            return {}
            
        try:
            with open(self.feedback_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading feedback: {e}")
            return {}