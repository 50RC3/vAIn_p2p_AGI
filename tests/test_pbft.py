import unittest
import asyncio
import json
import sys
import time
from unittest.mock import Mock, MagicMock, patch

# Add the project root to the path
sys.path.insert(0, '/home/runner/work/vAIn_p2p_AGI/vAIn_p2p_AGI')

try:
    # Import consensus modules directly
    sys.path.insert(0, '/home/runner/work/vAIn_p2p_AGI/vAIn_p2p_AGI/network/consensus')
    from messages import SignedMessage, MessageType
    from pbft import PBFTNode, LogEntry
    from view_change import ViewState
    from byzantine_detection import MisbehaviorTracker
    
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, PublicFormat, NoEncryption
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

class TestSignedMessage(unittest.TestCase):
    def setUp(self):
        self.private_key = ed25519.Ed25519PrivateKey.generate()
        self.public_key_pem = self.private_key.public_key().public_bytes(
            Encoding.PEM, PublicFormat.SubjectPublicKeyInfo
        )

    def test_message_creation_and_verification(self):
        """Test creating and verifying a signed message"""
        payload = {"test": "data", "value": 42}
        msg = SignedMessage.make(
            typ="PREPARE",
            view=1,
            seq=2,
            sender="node_1",
            payload=payload,
            priv=self.private_key
        )
        
        # Verify the message
        try:
            msg.verify(self.public_key_pem)
        except Exception as e:
            self.fail(f"Message verification failed: {e}")
        
        # Check message fields
        self.assertEqual(msg.typ, "PREPARE")
        self.assertEqual(msg.view, 1)
        self.assertEqual(msg.seq, 2)
        self.assertEqual(msg.sender, "node_1")
        self.assertEqual(msg.payload, payload)

    def test_message_verification_failure(self):
        """Test that verification fails with wrong key"""
        wrong_key = ed25519.Ed25519PrivateKey.generate()
        wrong_public_key_pem = wrong_key.public_key().public_bytes(
            Encoding.PEM, PublicFormat.SubjectPublicKeyInfo
        )
        
        payload = {"test": "data"}
        msg = SignedMessage.make(
            typ="COMMIT",
            view=0,
            seq=1,
            sender="node_2",
            payload=payload,
            priv=self.private_key
        )
        
        with self.assertRaises(Exception):
            msg.verify(wrong_public_key_pem)

class TestViewState(unittest.TestCase):
    def test_quorum_detection(self):
        """Test view change quorum detection"""
        view_state = ViewState()
        
        # Test N=4, f=1 scenario
        N, f = 4, 1
        
        # Not enough votes initially
        self.assertFalse(view_state.has_quorum(N, f))
        
        # Add one vote
        view_state.collect("node_1", 1)
        self.assertFalse(view_state.has_quorum(N, f))
        
        # Add second vote (still need 2f+1 = 3)
        view_state.collect("node_2", 1)
        self.assertFalse(view_state.has_quorum(N, f))
        
        # Add third vote - should reach quorum
        view_state.collect("node_3", 1)
        self.assertTrue(view_state.has_quorum(N, f))

class TestMisbehaviorTracker(unittest.TestCase):
    def test_byzantine_detection(self):
        """Test Byzantine node detection"""
        tracker = MisbehaviorTracker(threshold=3)
        
        # Initially no node is Byzantine
        self.assertFalse(tracker.is_byzantine("node_1"))
        
        # Flag some misbehavior
        tracker.flag("node_1", "bad_signature")
        tracker.flag("node_1", "equivocation")
        self.assertFalse(tracker.is_byzantine("node_1"))  # Still below threshold
        
        # One more flag should trigger Byzantine detection
        tracker.flag("node_1", "bad_signature")
        self.assertTrue(tracker.is_byzantine("node_1"))

class TestPBFTNode(unittest.TestCase):
    def setUp(self):
        # Create keys for nodes
        self.keys = {}
        self.public_keys = {}
        for i in range(4):  # 4 nodes, f=1
            node_id = f"node_{i}"
            private_key = ed25519.Ed25519PrivateKey.generate()
            public_key_pem = private_key.public_key().public_bytes(
                Encoding.PEM, PublicFormat.SubjectPublicKeyInfo
            )
            self.keys[node_id] = private_key
            self.public_keys[node_id] = public_key_pem
        
        # Mock send function
        self.sent_messages = []
        def mock_send(msg, peer):
            self.sent_messages.append((msg, peer))
        
        # Mock get_pubkey function
        def mock_get_pubkey(node_id):
            return self.public_keys[node_id]
        
        # Mock is_primary function
        def mock_is_primary(view, group):
            return group[view % len(group)] == "node_0"
        
        # Create PBFT node
        self.node = PBFTNode(
            node_id="node_0",
            peers=["node_1", "node_2", "node_3"],
            f=1,
            send=mock_send,
            get_pubkey=mock_get_pubkey,
            is_primary=mock_is_primary
        )
        
        # Set up private key wrapper
        def priv_wrapper(typ, view, seq, payload):
            return SignedMessage.make(typ, view, seq, "node_0", payload, self.keys["node_0"])
        self.node.set_priv_wrapper(priv_wrapper)

    def test_node_initialization(self):
        """Test PBFT node initialization"""
        self.assertEqual(self.node.id, "node_0")
        self.assertEqual(self.node.peers, ["node_1", "node_2", "node_3"])
        self.assertEqual(self.node.N, 4)
        self.assertEqual(self.node.f, 1)
        self.assertEqual(self.node.view, 0)
        self.assertEqual(self.node.seq, 0)

    def test_preprepare_handling(self):
        """Test PREPREPARE message handling"""
        # Create a PREPREPARE message
        payload = {"proposal": "test_request"}
        msg = SignedMessage.make(
            typ="PREPREPARE",
            view=0,
            seq=1,
            sender="node_0",
            payload=payload,
            priv=self.keys["node_0"]
        )
        
        # Clear sent messages
        self.sent_messages.clear()
        
        # Handle the message
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.node.on_message(msg))
        finally:
            loop.close()
        
        # Should broadcast PREPARE to all peers
        self.assertEqual(len(self.sent_messages), 3)  # 3 peers
        for sent_msg, peer in self.sent_messages:
            self.assertEqual(sent_msg.typ, "PREPARE")
            self.assertIn(peer, ["node_1", "node_2", "node_3"])

    def test_prepare_quorum(self):
        """Test PREPARE phase quorum"""
        # Set up a log entry
        digest = "test_digest"
        self.node.log[1] = LogEntry("test_request", digest, set(), set(), set())
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Simulate PREPARE messages from 2f nodes (should trigger COMMIT)
            for i in range(1, 3):  # node_1 and node_2
                prepare_msg = SignedMessage.make(
                    typ="PREPARE",
                    view=0,
                    seq=1,
                    sender=f"node_{i}",
                    payload={"digest": digest},
                    priv=self.keys[f"node_{i}"]
                )
                
                self.sent_messages.clear()
                loop.run_until_complete(self.node.on_message(prepare_msg))
        finally:
            loop.close()
        
        # After 2f PREPARE messages, should broadcast COMMIT
        commit_broadcasts = [msg for msg, peer in self.sent_messages if msg.typ == "COMMIT"]
        self.assertTrue(len(commit_broadcasts) > 0)

    def test_primary_selection(self):
        """Test primary selection logic"""
        # node_0 should be primary for view 0
        primary = self.node._primary_id()
        self.assertEqual(primary, "node_0")
        
        # Change view and test
        self.node.view = 1
        primary = self.node._primary_id()
        self.assertEqual(primary, "node_1")

if __name__ == '__main__':
    unittest.main()