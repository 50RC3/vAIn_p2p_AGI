import logging
import time
import json
import pickle
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
        
        logger.debug("Storing feedback: %s", serializable_feedback)
        
        # Implementation would typically store this in a database
        # For simplicity, we'll just log it
        logger.info("Feedback stored: score=%s", serializable_feedback['score'])
        
    async def persist_feedback(self, session_id: str, feedback: List[Dict[str, Any]]) -> None:
        """Persist feedback for a specific session."""
        try:
            # Convert feedback to serializable format
            serializable_feedback = []
            for entry in feedback:
                # Clean each feedback entry
                clean_entry = {
                    'session_id': entry.get('session_id', session_id),
                    'score': entry.get('score', 0.0),
                    'timestamp': entry.get('timestamp', time.time()),
                    'response_hash': entry.get('response_hash', ''),
                    'model_version': entry.get('model_version', self.version),
                }
                
                # Add response data if available
                if 'response' in entry:
                    response = entry['response']
                    clean_entry['response_text'] = getattr(response, 'text', '')
                    clean_entry['response_confidence'] = getattr(response, 'confidence', 0.0)
                    clean_entry['response_latency'] = getattr(response, 'latency', 0.0)
                    
                serializable_feedback.append(clean_entry)
                
            # Load existing feedback if file exists
            existing_data = []
            if self.feedback_file.exists():
                try:
                    with open(self.feedback_file, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                except json.JSONDecodeError:
                    logger.warning("Could not decode existing feedback file, creating new one")
                    existing_data = []
                    
            # Append new feedback and save
            combined_data = existing_data + serializable_feedback
            with open(self.feedback_file, 'w', encoding='utf-8') as f:
                json.dump(combined_data, f, indent=2)
                
            logger.info("Persisted %s feedback entries for session %s", len(serializable_feedback), session_id)
            
        except Exception as e:
            logger.error("Error persisting feedback: %s", e)
            raise

    async def store_model_state(self, model_id: str, state: Dict[str, Any]) -> None:
        """Store model state in storage
        
        Args:
            model_id: ID of the model
            state: The state to store as a dictionary
        """
        try:
            # Ensure consistent state format
            state_dict = state
            if not isinstance(state, dict):
                # If ModelState object was passed, convert to dict
                state_dict = {
                    "value": getattr(state, "value", 0.0),
                    "status": getattr(state, "status", "unknown"),
                    "timestamp": getattr(state, "timestamp", time.time()),
                    "model_id": model_id
                }
                # Add error information if available
                if hasattr(state, "error") and state.error:
                    state_dict["error"] = state.error
            
            # Store in a state file
            state_file = self.storage_dir / f"{model_id}_state.json"
            
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state_dict, f, indent=2)
            
            logger.info("Stored state for model %s: %s", model_id, state_dict.get('status', 'unknown'))
        
        except Exception as e:
            logger.error("Error storing model state for %s: %s", model_id, e)
            raise

    async def load_model_state(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Load model state from storage
        
        Args:
            model_id: ID of the model
            
        Returns:
            Model state dictionary or None if not found
        """
        try:
            state_file = self.storage_dir / f"{model_id}_state.json"
            
            if not state_file.exists():
                logger.debug("No state file found for model %s", model_id)
                return None
                
            with open(state_file, 'r', encoding='utf-8') as f:
                state_dict = json.load(f)
                
            logger.info("Loaded state for model %s", model_id)
            return state_dict
            
        except json.JSONDecodeError:
            logger.error("Invalid state file format for model %s", model_id)
            return None
            
        except Exception as e:
            logger.error("Error loading model state for %s: %s", model_id, e)
            return None

    async def save_model(self, model_id: str, model_data: Any) -> bool:
        """Save model data to storage
        
        Args:
            model_id: ID of the model
            model_data: The model data to save (could be a state, weights, etc.)
        
        Returns:
            bool: True if save was successful
        """
        try:
            # Create a dedicated directory for this model if it doesn't exist
            model_dir = self.storage_dir / model_id
            model_dir.mkdir(exist_ok=True, parents=True)
            
            # Determine the type of model data and save appropriately
            if hasattr(model_data, "__dict__"):
                # If the object has a __dict__, save as JSON
                data_dict = {k: v for k, v in model_data.__dict__.items() 
                            if not k.startswith('_') and not callable(v)}
                
                with open(model_dir / "model_data.json", 'w', encoding='utf-8') as f:
                    json.dump(data_dict, f, indent=2)
                    
            elif hasattr(model_data, "state_dict"):
                # If it's a PyTorch model or has a state_dict method
                try:
                    import torch
                    state_dict = model_data.state_dict()
                    torch.save(state_dict, model_dir / "model_weights.pt")
                except ImportError:
                    logger.warning("PyTorch not available, saving as pickle instead")
                    with open(model_dir / "model_data.pkl", 'wb') as f:
                        pickle.dump(model_data, f)
                
            else:
                # For other types, try pickle
                with open(model_dir / "model_data.pkl", 'wb') as f:
                    pickle.dump(model_data, f)
                    
            logger.info("Model %s saved successfully", model_id)
            return True
            
        except Exception as e:
            logger.error("Error saving model %s: %s", model_id, e)
            return False

    async def retrieve_model(self, model_id: str) -> Optional[Any]:
        """Retrieve a model from storage
        
        Args:
            model_id: ID of the model
            
        Returns:
            Model object or None if not found
        """
        try:
            model_dir = self.storage_dir / model_id
            
            if not model_dir.exists():
                logger.debug("Model directory %s not found", model_id)
                return None
                
            # Try to load as PyTorch model first
            weights_file = model_dir / "model_weights.pt"
            if weights_file.exists():
                try:
                    import torch
                    return torch.load(weights_file)
                except ImportError:
                    logger.warning("PyTorch not available, can't load model weights")
                except Exception as e:
                    logger.error("Error loading PyTorch model %s: %s", model_id, e)
            
            # Try JSON data
            json_file = model_dir / "model_data.json"
            if json_file.exists():
                with open(json_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
                    
            # Try pickle data
            pkl_file = model_dir / "model_data.pkl"
            if pkl_file.exists():
                with open(pkl_file, 'rb') as f:
                    return pickle.load(f)
                    
            logger.warning("Model %s exists but no recognizable format found", model_id)
            return None
            
        except Exception as e:
            logger.error("Error retrieving model %s: %s", model_id, e)
            return None