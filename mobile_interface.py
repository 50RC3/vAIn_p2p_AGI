from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class OfflineMessage:
    content: str
    timestamp: datetime
    synced: bool = False

class MobileChatInterface(ChatbotInterface):
    def __init__(self):
        super().__init__()
        self.offline_cache: Dict[str, OfflineMessage] = {}
        self.sync_manager = SyncManager()
        
    async def process_message_mobile(self, message: str) -> ChatResponse:
        """Handle mobile message processing with offline support
        
        Args:
            message: The user's message to process
            
        Returns:
            ChatResponse: The processed response
            
        Raises:
            NetworkError: If network connection fails during processing
        """
        try:
            if not self.network_monitor.is_connected():
                return self._handle_offline_message(message)
            
            # Try to sync any cached messages first
            await self._sync_offline_messages()
            return await self.process_message(message)
            
        except NetworkError as e:
            # Fallback to offline mode if network fails during processing
            return self._handle_offline_message(message)

    def _handle_offline_message(self, message: str) -> ChatResponse:
        """Cache message for later sync and provide offline response"""
        msg_id = str(uuid.uuid4())
        self.offline_cache[msg_id] = OfflineMessage(
            content=message,
            timestamp=datetime.now()
        )
        return ChatResponse(
            content="Message saved for processing when back online",
            offline_mode=True
        )

    async def _sync_offline_messages(self) -> None:
        """Attempt to sync cached offline messages"""
        if not self.offline_cache:
            return
            
        for msg_id, offline_msg in self.offline_cache.items():
            if offline_msg.synced:
                continue
                
            try:
                await self.process_message(offline_msg.content)
                offline_msg.synced = True
            except NetworkError:
                break

        # Clean up synced messages
        self.offline_cache = {
            k: v for k, v in self.offline_cache.items() 
            if not v.synced
        }
