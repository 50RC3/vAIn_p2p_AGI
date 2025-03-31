import json
import asyncio
import logging
from typing import Dict, Any, List, Optional
import time
import hashlib
from datetime import datetime

from ai_core.chatbot.interface import ChatbotInterface, ChatResponse
from ai_core.chatbot.mobile_interface import MobileChatInterface, MobileConfig, LearningConfig

logger = logging.getLogger(__name__)

class AndroidChatInterface(MobileChatInterface):
    """
    Specialized chat interface for Android that handles the conversion between
    Python ChatResponse objects and JSON format suitable for Android MessageItems.
    """
    
    def __init__(self, model, storage, 
                 max_history: int = 50,  # Further reduced for Android
                 mobile_config: Optional[MobileConfig] = None,
                 learning_config: Optional[LearningConfig] = None):
        """Initialize with Android-specific configurations"""
        super().__init__(model, storage, max_history, mobile_config, learning_config)
        self.pending_messages = {}
        self.offline_cache = {}
        self.network_available = True
        
    async def process_message_android(self, message: str, 
                                    user_id: str = "android_user") -> Dict[str, Any]:
        """
        Process a message from an Android client and return JSON-ready response
        
        Args:
            message: The message text from the user
            user_id: Unique identifier for the user (defaults to "android_user")
            
        Returns:
            Dictionary ready to be converted to JSON for Android
        """
        start_time = time.time()
        
        try:
            # Track the message request
            message_id = hashlib.md5(f"{message}{time.time()}".encode()).hexdigest()
            self.pending_messages[message_id] = {
                "message": message,
                "timestamp": time.time(),
                "user_id": user_id,
                "status": "processing"
            }
            
            # Check for network connectivity
            if not self.network_available:
                return self._create_offline_response(message, message_id)
            
            # Process message with mobile optimizations
            response = await self.process_message(message)
            
            # Convert to Android-friendly format
            result = {
                "id": response.id,
                "text": response.text,
                "confidence": response.confidence,
                "timestamp": datetime.now().isoformat(),
                "error": response.error,
                "processing_time": time.time() - start_time
            }
            
            # Update pending message status
            self.pending_messages[message_id]["status"] = "completed"
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing Android message: {e}")
            return {
                "id": hashlib.md5(f"error{time.time()}".encode()).hexdigest(),
                "text": f"Sorry, I encountered an error: {str(e)}",
                "confidence": 0.0,
                "timestamp": datetime.now().isoformat(),
                "error": True,
                "processing_time": time.time() - start_time
            }
            
    def _create_offline_response(self, message: str, message_id: str) -> Dict[str, Any]:
        """Create response for offline mode"""
        # Cache message for later processing
        self.offline_cache[message_id] = {
            "content": message,
            "timestamp": datetime.now().isoformat(),
            "synced": False
        }
        
        return {
            "id": message_id,
            "text": "Your message has been saved and will be processed when back online.",
            "confidence": 1.0,
            "timestamp": datetime.now().isoformat(),
            "error": False,
            "offline": True
        }
            
    async def get_pending_message_status(self, message_id: str) -> Dict[str, Any]:
        """Get status of a pending message"""
        if message_id in self.pending_messages:
            return self.pending_messages[message_id]
        return {"status": "not_found"}
    
    async def sync_offline_messages(self) -> List[Dict[str, Any]]:
        """Process cached offline messages and return results"""
        if not self.network_available:
            return []
            
        results = []
        for msg_id, msg_data in self.offline_cache.items():
            if msg_data["synced"]:
                continue
                
            try:
                response = await self.process_message(msg_data["content"])
                result = {
                    "id": response.id,
                    "text": response.text,
                    "confidence": response.confidence,
                    "timestamp": datetime.now().isoformat(),
                    "original_message_id": msg_id,
                    "error": response.error
                }
                results.append(result)
                msg_data["synced"] = True
                
            except Exception as e:
                logger.error(f"Failed to process offline message: {e}")
                
        # Clean synced messages
        self.offline_cache = {
            k: v for k, v in self.offline_cache.items() if not v["synced"]
        }
        
        return results
    
    def set_network_status(self, available: bool) -> None:
        """Update network availability status"""
        self.network_available = available
