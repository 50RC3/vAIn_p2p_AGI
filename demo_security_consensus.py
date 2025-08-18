#!/usr/bin/env python3
"""
Demonstration script showing how to use the new security/attestation and network/consensus modules.

This script shows:
1. Setting up hardware attestation for nodes
2. Creating a PBFT consensus network  
3. Basic consensus operations with authenticated messages
"""

import sys
import os
import json
import time
import base64
from typing import Dict, List

# Add the project root to the path
sys.path.insert(0, '/home/runner/work/vAIn_p2p_AGI/vAIn_p2p_AGI')

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

def create_test_root_cert():
    """Create a test root certificate for attestation"""
    root_key = ed25519.Ed25519PrivateKey.generate()
    root_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Test Root CA")])
    
    root_cert = x509.CertificateBuilder().subject_name(
        root_name
    ).issuer_name(
        root_name
    ).public_key(
        root_key.public_key()
    ).serial_number(1).not_valid_before(
        datetime.datetime.now(datetime.timezone.utc)
    ).not_valid_after(
        datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=365)
    ).sign(root_key, None)
    
    return root_cert.public_bytes(Encoding.PEM)

def create_hardware_quote(node_id: str, nonce: bytes) -> bytes:
    """Simulate creating a hardware attestation quote"""
    # In real implementation, this would interact with TPM
    ak_key = ed25519.Ed25519PrivateKey.generate()
    ak_pub_pem = ak_key.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
    
    # Simulate hardware measurement data
    quote_data = f"hardware_measurement_for_{node_id}".encode()
    signature = ak_key.sign(quote_data)
    
    # Calculate PCR digest 
    hash_obj = hashes.Hash(hashes.SHA256())
    hash_obj.update(quote_data)
    pcr_digest = hash_obj.finalize()
    
    quote_json = {
        "ak_pub_pem": ak_pub_pem.decode(),
        "quote_b64": base64.b64encode(quote_data).decode(),
        "sig_b64": base64.b64encode(signature).decode(),
        "pcr_digest_hex": pcr_digest.hex(),
        "nonce_b64": base64.b64encode(nonce).decode(),
        "ts": time.time(),
        "cert_chain_pem": None
    }
    
    return json.dumps(quote_json).encode()

class DemoConsensusNetwork:
    """Demonstration consensus network using PBFT"""
    
    def __init__(self, node_ids: List[str]):
        self.node_ids = node_ids
        self.attestation_manager = AttestationManager([create_test_root_cert()])
        self.nodes: Dict[str, PBFTNode] = {}
        self.node_keys = {}
        self.node_pubkeys = {}
        self.message_log = []
        
        # Generate keys for each node
        for node_id in node_ids:
            key = ed25519.Ed25519PrivateKey.generate()
            pubkey_pem = key.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
            self.node_keys[node_id] = key
            self.node_pubkeys[node_id] = pubkey_pem
    
    def attest_node(self, node_id: str) -> bool:
        """Perform hardware attestation for a node"""
        print(f"🔐 Attesting node {node_id}...")
        
        # Generate challenge nonce
        nonce = os.urandom(16)
        print(f"  Challenge nonce: {nonce.hex()[:16]}...")
        
        # Node provides hardware quote
        quote_data = create_hardware_quote(node_id, nonce)
        quote = self.attestation_manager.parse_quote(quote_data)
        
        # Verify the quote
        try:
            self.attestation_manager.verify_quote(node_id, quote, nonce)
            print(f"  ✅ Attestation successful for {node_id}")
            return True
        except Exception as e:
            print(f"  ❌ Attestation failed for {node_id}: {e}")
            return False
    
    def setup_consensus_node(self, node_id: str):
        """Set up a PBFT consensus node after successful attestation"""
        print(f"🏗️  Setting up consensus for {node_id}...")
        
        other_peers = [n for n in self.node_ids if n != node_id]
        
        def send_message(msg: SignedMessage, peer: str):
            self.message_log.append({
                'from': node_id,
                'to': peer,
                'type': msg.typ,
                'view': msg.view,
                'seq': msg.seq,
                'timestamp': time.time()
            })
            print(f"  📤 {node_id} → {peer}: {msg.typ} (view={msg.view}, seq={msg.seq})")
        
        def get_pubkey(node: str) -> bytes:
            return self.node_pubkeys[node]
        
        def is_primary(view: int, group: List[str]) -> bool:
            return group[view % len(group)] == node_id
        
        pbft_node = PBFTNode(
            node_id=node_id,
            peers=other_peers,
            f=(len(self.node_ids) - 1) // 3,  # f = (N-1)/3 for PBFT
            send=send_message,
            get_pubkey=get_pubkey,
            is_primary=is_primary
        )
        
        # Set up message signing
        def priv_wrapper(typ, view, seq, payload):
            return SignedMessage.make(typ, view, seq, node_id, payload, self.node_keys[node_id])
        pbft_node.set_priv_wrapper(priv_wrapper)
        
        self.nodes[node_id] = pbft_node
        print(f"  ✅ Consensus node ready: {node_id} (N={pbft_node.N}, f={pbft_node.f})")
    
    def demonstrate_consensus(self):
        """Demonstrate basic consensus operations"""
        print(f"\n🎯 Demonstrating consensus with {len(self.nodes)} nodes...")
        
        if len(self.nodes) < 4:
            print("❌ Need at least 4 nodes for meaningful PBFT demonstration")
            return
        
        # Show network status
        for node_id, node in self.nodes.items():
            primary_status = "PRIMARY" if node.is_primary_fn(node.view, [node.id] + node.peers) else "REPLICA"
            print(f"  📊 {node_id}: {primary_status} (view={node.view})")
        
        # Show message authentication
        print(f"\n🔏 Message authentication example:")
        test_node = list(self.nodes.keys())[0]
        test_message = SignedMessage.make(
            "PREPARE", 0, 1, test_node, {"example": "data"}, self.node_keys[test_node]
        )
        
        print(f"  📝 Created message: {test_message.typ} from {test_message.sender}")
        print(f"  🔑 Message digest: {test_message.digest[:16]}...")
        
        # Verify with different nodes
        for node_id in self.node_ids:
            try:
                test_message.verify(self.node_pubkeys[test_node])
                print(f"  ✅ {node_id} verified {test_node}'s message")
            except Exception as e:
                print(f"  ❌ {node_id} failed to verify: {e}")

def main():
    """Main demonstration function"""
    print("🚀 vAIn P2P AGI - Security & Consensus Demonstration\n")
    print("This demonstrates the new security/attestation and network/consensus modules:")
    print("- Hardware attestation with TPM-style quotes")
    print("- PBFT consensus with Ed25519 message authentication")
    print("- Integration between attestation and consensus\n")
    
    # Create a 4-node network (minimum for f=1 Byzantine fault tolerance)
    node_ids = ["node_0", "node_1", "node_2", "node_3"]
    network = DemoConsensusNetwork(node_ids)
    
    print(f"🌐 Creating consensus network with {len(node_ids)} nodes\n")
    
    # Phase 1: Hardware Attestation
    print("=== PHASE 1: HARDWARE ATTESTATION ===")
    attested_nodes = []
    for node_id in node_ids:
        if network.attest_node(node_id):
            attested_nodes.append(node_id)
    
    print(f"\n✅ Attestation complete: {len(attested_nodes)}/{len(node_ids)} nodes attested\n")
    
    # Phase 2: Consensus Setup
    print("=== PHASE 2: CONSENSUS SETUP ===")
    for node_id in attested_nodes:
        network.setup_consensus_node(node_id)
    
    print(f"\n✅ Consensus setup complete\n")
    
    # Phase 3: Demonstrate Operations
    print("=== PHASE 3: CONSENSUS DEMONSTRATION ===")
    network.demonstrate_consensus()
    
    print(f"\n📈 Network Statistics:")
    print(f"  Attested nodes: {len(attested_nodes)}")
    print(f"  Consensus nodes: {len(network.nodes)}")
    print(f"  Messages sent: {len(network.message_log)}")
    print(f"  Byzantine fault tolerance: f={((len(network.nodes) - 1) // 3) if network.nodes else 0}")
    
    print(f"\n🎉 Demonstration complete!")
    print(f"\nThe modules are ready for integration into the vAIn P2P AGI system.")

if __name__ == "__main__":
    main()