import unittest
import os
import sys
import json
import time
import base64
from unittest.mock import Mock, patch

# Add the project root to the path
sys.path.insert(0, '/home/runner/work/vAIn_p2p_AGI/vAIn_p2p_AGI')

try:
    from security.attestation import AttestationManager, AttestationError, Quote
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, PublicFormat, NoEncryption
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes
    import datetime
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

class TestAttestationManager(unittest.TestCase):
    def setUp(self):
        # Create a self-signed root certificate for testing
        self.root_key = ed25519.Ed25519PrivateKey.generate()
        root_name = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, "Test Root CA"),
        ])
        
        self.root_cert = x509.CertificateBuilder().subject_name(
            root_name
        ).issuer_name(
            root_name
        ).public_key(
            self.root_key.public_key()
        ).serial_number(
            1
        ).not_valid_before(
            datetime.datetime.now(datetime.timezone.utc)
        ).not_valid_after(
            datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=365)
        ).sign(self.root_key, None)  # Ed25519 requires None algorithm
        
        root_pem = self.root_cert.public_bytes(Encoding.PEM)
        
        # Create an attestation manager with the test root
        self.attestation_manager = AttestationManager([root_pem])
        
        # Create a test AK key
        self.ak_key = ed25519.Ed25519PrivateKey.generate()
        self.ak_pub_pem = self.ak_key.public_key().public_bytes(
            Encoding.PEM, PublicFormat.SubjectPublicKeyInfo
        )

    def test_register_and_get_ak(self):
        """Test AK registration and retrieval"""
        node_id = "test_node_1"
        
        # Test AK not found initially
        self.assertIsNone(self.attestation_manager.get_ak(node_id))
        
        # Register AK
        self.attestation_manager.register_ak(node_id, self.ak_pub_pem)
        
        # Test AK can be retrieved
        retrieved_ak = self.attestation_manager.get_ak(node_id)
        self.assertEqual(retrieved_ak, self.ak_pub_pem)

    def test_ak_expiration(self):
        """Test AK expiration"""
        node_id = "test_node_2"
        
        # Register AK with very short TTL
        now = time.time()
        self.attestation_manager.register_ak(node_id, self.ak_pub_pem, now)
        
        # Mock time to be past expiration
        with patch('time.time', return_value=now + self.attestation_manager._ak_ttl + 1):
            self.assertIsNone(self.attestation_manager.get_ak(node_id))

    def test_quote_parsing(self):
        """Test quote parsing from JSON"""
        # Create a test quote
        quote_data = b"test_quote_data"
        signature = self.ak_key.sign(quote_data)
        pcr_digest = hashes.Hash(hashes.SHA256()).finalize()
        nonce = os.urandom(16)
        
        quote_json = {
            "ak_pub_pem": self.ak_pub_pem.decode(),
            "quote_b64": base64.b64encode(quote_data).decode(),
            "sig_b64": base64.b64encode(signature).decode(),
            "pcr_digest_hex": pcr_digest.hex(),
            "nonce_b64": base64.b64encode(nonce).decode(),
            "ts": time.time(),
            "cert_chain_pem": None
        }
        
        packed = json.dumps(quote_json).encode()
        quote = self.attestation_manager.parse_quote(packed)
        
        self.assertEqual(quote.ak_pub_pem, self.ak_pub_pem)
        self.assertEqual(quote.quote_bytes, quote_data)
        self.assertEqual(quote.signature, signature)
        self.assertEqual(quote.pcr_digest, pcr_digest)
        self.assertEqual(quote.nonce, nonce)

    def test_quote_verification_nonce_mismatch(self):
        """Test quote verification fails with wrong nonce"""
        node_id = "test_node_3"
        
        # Create a test quote
        quote_data = b"test_quote_data"
        signature = self.ak_key.sign(quote_data)
        pcr_digest = hashes.Hash(hashes.SHA256()).finalize()
        wrong_nonce = os.urandom(16)
        expected_nonce = os.urandom(16)
        
        quote = Quote(
            ak_pub_pem=self.ak_pub_pem,
            quote_bytes=quote_data,
            signature=signature,
            pcr_digest=pcr_digest,
            nonce=wrong_nonce,
            ts=time.time(),
            cert_chain_pem=None
        )
        
        from security.attestation.attestation_manager import QuoteError
        with self.assertRaises(QuoteError) as ctx:
            self.attestation_manager.verify_quote(node_id, quote, expected_nonce)
        
        self.assertIn("Nonce mismatch", str(ctx.exception))

    def test_quote_verification_timestamp_skew(self):
        """Test quote verification fails with timestamp skew"""
        node_id = "test_node_4"
        
        # Create a test quote with old timestamp
        quote_data = b"test_quote_data"
        signature = self.ak_key.sign(quote_data)
        pcr_digest = hashes.Hash(hashes.SHA256()).finalize()
        nonce = os.urandom(16)
        old_timestamp = time.time() - 200  # 200 seconds ago
        
        quote = Quote(
            ak_pub_pem=self.ak_pub_pem,
            quote_bytes=quote_data,
            signature=signature,
            pcr_digest=pcr_digest,
            nonce=nonce,
            ts=old_timestamp,
            cert_chain_pem=None
        )
        
        from security.attestation.attestation_manager import QuoteError
        with self.assertRaises(QuoteError) as ctx:
            self.attestation_manager.verify_quote(node_id, quote, nonce)
        
        self.assertIn("timestamp outside allowed skew", str(ctx.exception))

if __name__ == '__main__':
    unittest.main()