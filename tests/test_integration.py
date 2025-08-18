import unittest
import sys
import os
import json
import time
import base64
from unittest.mock import Mock, patch

# Add the project root to the path
sys.path.insert(0, '/home/runner/work/vAIn_p2p_AGI/vAIn_p2p_AGI')

try:
    from security.attestation import AttestationManager, Quote
    # Direct imports for consensus modules
    sys.path.insert(0, '/home/runner/work/vAIn_p2p_AGI/vAIn_p2p_AGI/network/consensus')
    from messages import SignedMessage
    from pbft import PBFTNode
    
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes
    import datetime
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

class TestIntegration(unittest.TestCase):
    """Integration tests demonstrating how attestation and PBFT work together"""
    
    def setUp(self):
        # Create test root certificate
        self.root_key = ed25519.Ed25519PrivateKey.generate()
        root_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Test Root CA")])
        
        self.root_cert = x509.CertificateBuilder().subject_name(
            root_name
        ).issuer_name(
            root_name
        ).public_key(
            self.root_key.public_key()
        ).serial_number(1).not_valid_before(
            datetime.datetime.now(datetime.timezone.utc)
        ).not_valid_after(
            datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=365)
        ).sign(self.root_key, None)
        
        root_pem = self.root_cert.public_bytes(Encoding.PEM)
        
        # Create attestation manager
        self.attestation_manager = AttestationManager([root_pem])
        
        # Create test node keys for consensus
        self.node_keys = {}
        self.node_pubkeys = {}
        for i in range(4):
            node_id = f"node_{i}"
            key = ed25519.Ed25519PrivateKey.generate()
            pubkey_pem = key.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
            self.node_keys[node_id] = key
            self.node_pubkeys[node_id] = pubkey_pem

    def test_attestation_then_consensus(self):
        """Test full flow: attestation verification followed by consensus participation"""
        
        # Step 1: Node attestation phase
        node_id = "node_1"
        ak_key = ed25519.Ed25519PrivateKey.generate()
        ak_pub_pem = ak_key.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
        
        # Create a valid quote
        quote_data = b"hardware_measurement_data"
        signature = ak_key.sign(quote_data)
        hash_obj = hashes.Hash(hashes.SHA256())
        hash_obj.update(quote_data)
        pcr_digest = hash_obj.finalize()  # PCR should match quote hash
        nonce = os.urandom(16)
        
        quote_json = {
            "ak_pub_pem": ak_pub_pem.decode(),
            "quote_b64": base64.b64encode(quote_data).decode(),
            "sig_b64": base64.b64encode(signature).decode(),
            "pcr_digest_hex": pcr_digest.hex(),
            "nonce_b64": base64.b64encode(nonce).decode(),
            "ts": time.time(),
            "cert_chain_pem": None
        }
        
        packed_quote = json.dumps(quote_json).encode()
        quote = self.attestation_manager.parse_quote(packed_quote)
        
        # Verify the quote (this would happen during handshake)
        try:
            self.attestation_manager.verify_quote(node_id, quote, nonce)
            attestation_passed = True
        except Exception as e:
            attestation_passed = False
            print(f"Attestation failed: {e}")
        
        self.assertTrue(attestation_passed, "Node attestation should pass")
        
        # Step 2: Node joins consensus after successful attestation
        if attestation_passed:
            # Create a mock send function
            sent_messages = []
            def mock_send(msg, peer):
                sent_messages.append((msg, peer))
            
            # Create PBFT node
            pbft_node = PBFTNode(
                node_id=node_id,
                peers=["node_2", "node_3", "node_4"],
                f=1,
                send=mock_send,
                get_pubkey=lambda node: self.node_pubkeys[node],
                is_primary=lambda view, group: group[view % len(group)] == node_id
            )
            
            # Set up private key wrapper for signing consensus messages
            def priv_wrapper(typ, view, seq, payload):
                return SignedMessage.make(typ, view, seq, node_id, payload, self.node_keys[node_id])
            pbft_node.set_priv_wrapper(priv_wrapper)
            
            # Test that the node can participate in consensus
            self.assertEqual(pbft_node.id, node_id)
            self.assertEqual(pbft_node.N, 4)
            self.assertEqual(pbft_node.f, 1)
            
            # Verify the node can create and verify messages
            test_msg = SignedMessage.make(
                "PREPARE", 0, 1, node_id, {"test": "data"}, self.node_keys[node_id]
            )
            
            # Verify the message can be verified with the public key
            try:
                test_msg.verify(self.node_pubkeys[node_id])
                message_verification_passed = True
            except Exception as e:
                message_verification_passed = False
                print(f"Message verification failed: {e}")
            
            self.assertTrue(message_verification_passed, "Consensus messages should be verifiable")

    def test_multiple_node_attestation_and_consensus(self):
        """Test multiple nodes going through attestation and then participating in consensus"""
        
        attested_nodes = []
        
        # Step 1: Multiple nodes complete attestation
        for i in range(4):
            node_id = f"node_{i}"
            ak_key = ed25519.Ed25519PrivateKey.generate()
            ak_pub_pem = ak_key.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
            
            # Create quote for each node
            quote_data = f"hardware_data_for_{node_id}".encode()
            signature = ak_key.sign(quote_data)
            hash_obj = hashes.Hash(hashes.SHA256())
            hash_obj.update(quote_data)
            pcr_digest = hash_obj.finalize()
            nonce = os.urandom(16)
            
            quote_json = {
                "ak_pub_pem": ak_pub_pem.decode(),
                "quote_b64": base64.b64encode(quote_data).decode(),
                "sig_b64": base64.b64encode(signature).decode(),
                "pcr_digest_hex": pcr_digest.hex(),
                "nonce_b64": base64.b64encode(nonce).decode(),
                "ts": time.time(),
                "cert_chain_pem": None
            }
            
            packed_quote = json.dumps(quote_json).encode()
            quote = self.attestation_manager.parse_quote(packed_quote)
            
            try:
                self.attestation_manager.verify_quote(node_id, quote, nonce)
                attested_nodes.append(node_id)
            except Exception as e:
                print(f"Attestation failed for {node_id}: {e}")
        
        # All nodes should pass attestation
        self.assertEqual(len(attested_nodes), 4, "All nodes should pass attestation")
        
        # Step 2: Verify all attested nodes have their AKs registered
        for node_id in attested_nodes:
            ak = self.attestation_manager.get_ak(node_id)
            self.assertIsNotNone(ak, f"AK should be registered for {node_id}")
        
        # Step 3: Create a consensus network with attested nodes
        consensus_network = {}
        for node_id in attested_nodes:
            other_peers = [n for n in attested_nodes if n != node_id]
            
            sent_messages = []
            def make_send_for_node(sender):
                def send_func(msg, peer):
                    sent_messages.append((sender, msg, peer))
                return send_func
            
            pbft_node = PBFTNode(
                node_id=node_id,
                peers=other_peers,
                f=1,
                send=make_send_for_node(node_id),
                get_pubkey=lambda node: self.node_pubkeys[node],
                is_primary=lambda view, group: group[view % len(group)] == "node_0"
            )
            
            def make_priv_wrapper(node):
                def wrapper(typ, view, seq, payload):
                    return SignedMessage.make(typ, view, seq, node, payload, self.node_keys[node])
                return wrapper
            
            pbft_node.set_priv_wrapper(make_priv_wrapper(node_id))
            consensus_network[node_id] = (pbft_node, sent_messages)
        
        # Verify all nodes are properly set up for consensus
        self.assertEqual(len(consensus_network), 4, "Should have 4 consensus nodes")
        
        # Test cross-node message verification
        node_0 = consensus_network["node_0"][0]
        test_msg = SignedMessage.make(
            "PREPARE", 0, 1, "node_0", {"test": "cross_verification"}, self.node_keys["node_0"]
        )
        
        # Other nodes should be able to verify node_0's messages
        try:
            test_msg.verify(self.node_pubkeys["node_0"])
            cross_verification_passed = True
        except Exception:
            cross_verification_passed = False
        
        self.assertTrue(cross_verification_passed, "Cross-node message verification should work")

if __name__ == '__main__':
    unittest.main()