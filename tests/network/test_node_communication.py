import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from aiohttp import ClientError, ClientTimeout
from network.node_communication import NodeCommunication
from core.interactive_utils import InteractiveSession, InteractionTimeout

@pytest.fixture
async def node_comm():
    comm = NodeCommunication("test_node", interactive=True)
    yield comm
    await comm.cleanup()

@pytest.fixture
def mock_session():
    session = AsyncMock()
    response = AsyncMock()
    response.status = 200
    session.post.return_value.__aenter__.return_value = response
    return session

class TestNodeCommunication:
    
    @pytest.mark.asyncio
    async def test_successful_message_send(self, node_comm, mock_session):
        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await node_comm.send_message("target_node", {"test": "data"})
            assert result == True
            mock_session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self, node_comm):
        session = AsyncMock()
        session.post.side_effect = [
            asyncio.TimeoutError(),
            AsyncMock(status=200).__aenter__.return_value
        ]
        
        with patch('aiohttp.ClientSession', return_value=session):
            with patch.object(InteractiveSession, 'get_confirmation', return_value=True):
                result = await node_comm.send_message("target_node", {"test": "data"})
                assert result == True
                assert session.post.call_count == 2

    @pytest.mark.asyncio
    async def test_rate_limiting(self, node_comm):
        session = AsyncMock()
        response = AsyncMock()
        response.status = 200
        session.post.return_value.__aenter__.return_value = response

        with patch('aiohttp.ClientSession', return_value=session):
            # Send multiple messages quickly
            tasks = [
                node_comm.send_message("target_node", {"test": f"msg{i}"})
                for i in range(5)
            ]
            results = await asyncio.gather(*tasks)
            assert all(results)  # All messages should eventually send
            
            # Check rate limiting worked
            calls = session.post.call_args_list
            timestamps = [call.timestamp for call in calls]
            for t1, t2 in zip(timestamps, timestamps[1:]):
                assert t2 - t1 >= 0.001  # Messages should be spaced out

    @pytest.mark.asyncio
    async def test_connection_pool_reuse(self, node_comm):
        sessions = []
        
        async def mock_client_session():
            session = AsyncMock()
            response = AsyncMock()
            response.status = 200
            session.post.return_value.__aenter__.return_value = response
            sessions.append(session)
            return session

        with patch('aiohttp.ClientSession', side_effect=mock_client_session):
            # Send multiple messages to same node
            for _ in range(3):
                await node_comm.send_message("target_node", {"test": "data"})
            
            # Should reuse same session
            assert len(sessions) == 1

    @pytest.mark.asyncio
    async def test_session_expiry(self, node_comm):
        with patch('asyncio.get_event_loop') as mock_loop:
            mock_loop.return_value.time.side_effect = [0, 301]  # Force TTL expiry
            
            session1 = AsyncMock()
            session2 = AsyncMock()
            
            with patch('aiohttp.ClientSession', side_effect=[session1, session2]):
                await node_comm.send_message("target_node", {"test": "msg1"})
                await node_comm.send_message("target_node", {"test": "msg2"})
                
                # Should create new session after expiry
                assert session1.close.called
                assert session2.post.called

    @pytest.mark.asyncio
    async def test_interactive_retry_prompt(self, node_comm):
        session = AsyncMock()
        session.post.side_effect = ClientError()
        
        user_responses = [True, True, False]  # Retry twice then give up
        
        with patch('aiohttp.ClientSession', return_value=session):
            with patch.object(InteractiveSession, 'get_confirmation') as mock_confirm:
                mock_confirm.side_effect = user_responses
                
                result = await node_comm.send_message("target_node", {"test": "data"})
                assert result == False
                assert mock_confirm.call_count == 3
                assert session.post.call_count == 3

    @pytest.mark.asyncio
    async def test_cleanup(self, node_comm):
        # Setup active sessions and workers
        session1 = AsyncMock()
        session2 = AsyncMock()
        
        with patch('aiohttp.ClientSession', side_effect=[session1, session2]):
            await node_comm.send_message("node1", {"test": "data"})
            await node_comm.send_message("node2", {"test": "data"})
            
            await node_comm.cleanup()
            
            # Verify cleanup
            assert session1.close.called
            assert session2.close.called
            assert len(node_comm._worker_tasks) == 0
            assert node_comm._session_pool == {}
