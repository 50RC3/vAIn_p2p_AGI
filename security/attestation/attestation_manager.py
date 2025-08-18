# security/attestation/attestation_manager.py
from dataclasses import dataclass
from typing import Optional, Dict, Sequence, Tuple
import hashlib, json, time
from cryptography.hazmat.primitives.serialization import load_pem_public_key
from cryptography import x509
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, padding
from cryptography.hazmat.primitives import hashes

try:
    # Optional: TPM2 Software Stack Python bindings (recommended)
    # pip install tpm2-pytss
    from tpm2_pytss import ESYS_TR, TSS2_Exception  # type: ignore
    HAVE_TPM = True
except Exception:
    HAVE_TPM = False

@dataclass
class Quote:
    ak_pub_pem: bytes                  # Attestation Key public cert/PEM
    quote_bytes: bytes                 # Raw quote (vendor format)
    signature: bytes                   # Signature over quote_bytes by AK
    pcr_digest: bytes                  # Expected PCR digest (sha256)
    nonce: bytes                       # Anti-replay nonce
    ts: float                          # Sender timestamp
    cert_chain_pem: Optional[bytes]    # Optional AK -> root chain

class AttestationError(Exception): ...
class CertificateError(AttestationError): ...
class QuoteError(AttestationError): ...
class TPMUnavailable(AttestationError): ...

class AttestationManager:
    """
    Verifies: certificate chain (optional), quote signature, nonce freshness, and PCR digest.
    Store/rotate AKs per-node. Designed to run w/ or w/o TPM libs (software verify fallback).
    """
    def __init__(self, roots_pem: Sequence[bytes], ak_store_ttl_s: int = 90*24*3600):
        self._roots = [x509.load_pem_x509_certificate(r) for r in roots_pem]
        self._ak_db: Dict[str, Tuple[bytes, float]] = {}  # node_id -> (ak_pub_pem, exp_ts)
        self._ak_ttl = ak_store_ttl_s

    # ---------- Chain validation ----------
    def validate_cert_chain(self, leaf_pem: bytes, chain_pem: Optional[bytes]) -> None:
        """Best-effort X.509 path build & signature checks (no CRL/OCSP here)."""
        leaf = x509.load_pem_x509_certificate(leaf_pem)
        intermediates = []
        if chain_pem:
            for block in chain_pem.split(b"-----END CERTIFICATE-----"):
                block = block.strip()
                if block:
                    intermediates.append(
                        x509.load_pem_x509_certificate(block + b"\n-----END CERTIFICATE-----\n")
                    )
        # Naive builder: try each root; ensure signatures form a path
        issuers = {c.subject.rfc4514_string(): c for c in intermediates + self._roots}
        cur = leaf
        depth = 0
        while True:
            depth += 1
            if depth > 6:
                raise CertificateError("Chain too deep / loop suspected")
            if any(cur.issuer == r.subject for r in self._roots):
                # verify leaf->root signatures
                self._verify_cert_sig(cur, self._find_issuer(cur, intermediates + self._roots))
                break
            issuer = self._find_issuer(cur, intermediates)
            if issuer is None:
                raise CertificateError("Issuer not found in provided chain/roots")
            self._verify_cert_sig(cur, issuer)
            cur = issuer

    @staticmethod
    def _find_issuer(cert: x509.Certificate, pool: Sequence[x509.Certificate]):
        for c in pool:
            if cert.issuer == c.subject:
                return c
        return None

    @staticmethod
    def _verify_cert_sig(cert: x509.Certificate, issuer: x509.Certificate) -> None:
        pub = issuer.public_key()
        try:
            if isinstance(pub, ec.EllipticCurvePublicKey):
                pub.verify(cert.signature, cert.tbs_certificate_bytes,
                           ec.ECDSA(cert.signature_hash_algorithm))
            else:
                # RSA (padding/alg inferred)
                pub.verify(cert.signature, cert.tbs_certificate_bytes,
                           padding.PKCS1v15(), cert.signature_hash_algorithm)
        except Exception as e:
            raise CertificateError(f"Cert signature invalid: {e}")

    # ---------- AK lifecycle ----------
    def register_ak(self, node_id: str, ak_pub_pem: bytes, now: Optional[float] = None):
        ts = now or time.time()
        self._ak_db[node_id] = (ak_pub_pem, ts + self._ak_ttl)

    def rotate_ak(self, node_id: str, new_ak_pub_pem: bytes):
        self.register_ak(node_id, new_ak_pub_pem)

    def get_ak(self, node_id: str) -> Optional[bytes]:
        rec = self._ak_db.get(node_id)
        if not rec:
            return None
        ak, exp = rec
        if time.time() > exp:
            self._ak_db.pop(node_id, None)
            return None
        return ak

    # ---------- Quote parsing & verification ----------
    def parse_quote(self, packed: bytes) -> Quote:
        """
        Accepts vendor-agnostic JSON envelope for simplicity:
        {
          "ak_pub_pem": "...PEM...",
          "quote_b64": "...",
          "sig_b64": "...",
          "pcr_digest_hex": "...",
          "nonce_b64": "...",
          "ts": 1710000000.123,
          "cert_chain_pem": "PEM...PEM"
        }
        """
        try:
            obj = json.loads(packed.decode("utf-8"))
            return Quote(
                ak_pub_pem=obj["ak_pub_pem"].encode(),
                quote_bytes=_b64d(obj["quote_b64"]),
                signature=_b64d(obj["sig_b64"]),
                pcr_digest=bytes.fromhex(obj["pcr_digest_hex"]),
                nonce=_b64d(obj["nonce_b64"]),
                ts=float(obj["ts"]),
                cert_chain_pem=obj.get("cert_chain_pem", None).encode()
                    if obj.get("cert_chain_pem") else None
            )
        except Exception as e:
            raise QuoteError(f"Malformed quote payload: {e}")

    def verify_quote(self, node_id: str, quote: Quote, expected_nonce: bytes,
                     max_skew_s: int = 120) -> None:
        # 1) Time window / nonce
        if abs(time.time() - quote.ts) > max_skew_s:
            raise QuoteError("Quote timestamp outside allowed skew")
        if quote.nonce != expected_nonce:
            raise QuoteError("Nonce mismatch")

        # 2) Optional chain validation
        if quote.cert_chain_pem:
            self.validate_cert_chain(quote.ak_pub_pem, quote.cert_chain_pem)

        # 3) Signature check (AK over quote_bytes)
        ak_pub = load_pem_public_key(quote.ak_pub_pem)
        try:
            if isinstance(ak_pub, ed25519.Ed25519PublicKey):
                ak_pub.verify(quote.signature, quote.quote_bytes)
            elif isinstance(ak_pub, ec.EllipticCurvePublicKey):
                ak_pub.verify(quote.signature, quote.quote_bytes, ec.ECDSA(hashes.SHA256()))
            else:
                raise QuoteError("Unsupported AK public key type")
        except Exception as e:
            raise QuoteError(f"Quote signature invalid: {e}")

        # 4) PCR digest check (vendor-specific parser omitted; require hash match)
        calc = hashlib.sha256(quote.quote_bytes).digest()
        if calc != quote.pcr_digest:
            raise QuoteError("PCR digest mismatch")

        # 5) Persist AK (trust on first use OR require out-of-band allowlist)
        if self.get_ak(node_id) is None:
            self.register_ak(node_id, quote.ak_pub_pem)

# helpers
import base64
def _b64d(s: str) -> bytes: return base64.b64decode(s.encode())