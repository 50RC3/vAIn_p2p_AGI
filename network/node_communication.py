import asyncio
import aiohttp
from typing import Dict, Any

class NodeCommunication:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.message_queue = asyncio.Queue()
        
    async def send_message(self, target_node: str, message: Dict[str, Any]):
        async with aiohttp.ClientSession() as session:
            await session.post(f"http://{target_node}/message", json=message)
