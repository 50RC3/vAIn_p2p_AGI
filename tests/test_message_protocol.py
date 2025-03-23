import pytest
import asyncio
from cryptography.fernet import Fernet
from cryptography.exceptions import InvalidKey, InvalidSignature
from network.message_protocol import SecureMessageProtocol

@pytest.fixture
def encryption_key():
    return Fernet.generate_key()

@pytest.fixture
def protocol(encryption_key):
    return SecureMessageProtocol(encryption_key, interactive=False)

@pytest.fixture
def test_message():
    return {
        "type": "test",
        "content": "Hello World",
        "timestamp": 1234567890
    }

@pytest.mark.asyncio
async def test_message_encoding(protocol, test_message):
    # Test basic encoding
    encoded = protocol.encode_message(test_message)
    assert isinstance(encoded, dict)
    assert all(k in encoded for k in ['data', 'signature', 'public_key'])
    assert all(isinstance(v, bytes) for v in encoded.values())

@pytest.mark.asyncio
async def test_message_decoding(protocol, test_message):
    # Test complete encode-decode cycle
    encoded = protocol.encode_message(test_message)
    decoded = await protocol.decode_message_interactive(encoded)
    assert decoded == test_message

@pytest.mark.asyncio
async def test_invalid_signature(protocol, test_message):
    # Test tampering detection
    encoded = protocol.encode_message(test_message)
    encoded['signature'] = b'invalid_signature'
    
    with pytest.raises(InvalidSignature):
        await protocol.decode_message_interactive(encoded)

@pytest.mark.asyncio
async def test_invalid_message_format(protocol):
    # Test handling of malformed messages
    invalid_message = {
        'data': b'invalid_data',
        'signature': b'invalid_signature',
        'public_key': b'invalid_key'
    }
    
    result = await protocol.decode_message_interactive(invalid_message)
    assert result is None

@pytest.mark.asyncio
async def test_large_message_handling(protocol):
    # Test large message handling
    large_message = {
        "type": "test",
        "content": "x" * (2 * 1024 * 1024)  # 2MB of data
    }
    
    # Should return None in non-interactive mode for large messages
    encoded = await protocol.encode_message_interactive(large_message)
    assert encoded is None

@pytest.mark.asyncio
async def test_metrics_tracking(protocol, test_message):
    # Test metrics updating
    encoded = protocol.encode_message(test_message)
    await protocol.decode_message_interactive(encoded)
    
    metrics = await protocol.get_metrics()
    assert metrics['messages_processed'] > 0
    assert metrics['verification_failures'] == 0

@pytest.mark.asyncio
async def test_shutdown_handling(protocol, test_message):
    # Test graceful shutdown
    protocol.request_shutdown()
    assert protocol._interrupt_requested is True

@pytest.mark.asyncio
async def test_multiple_protocol_instances(encryption_key):
    # Test message exchange between different protocol instances
    protocol1 = SecureMessageProtocol(encryption_key, interactive=False)
    protocol2 = SecureMessageProtocol(encryption_key, interactive=False)
    
    message = {"type": "test", "content": "cross-protocol test"}
    encoded = protocol1.encode_message(message)
    decoded = await protocol2.decode_message_interactive(encoded)
    
    assert decoded == message
