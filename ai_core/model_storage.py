import asyncio
import logging
import time
import json
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelStorage:
    """A simple model storage implementation."""
    
    def __init__(self, storage_dir: str = "model_storage"):
        """Initialize the model storage with a directory path."""
        self.storage_dir = Path(storage_dir)
        self.version = "0.1.0"
        self.feedback_file = self.storage_dir / "feedback.json"
        
        # Create storage directory if it doesn't exist
        self.storage_dir.mkdir(exist_ok=True, parents=True)
        
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