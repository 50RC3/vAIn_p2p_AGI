# network/consensus/messages.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict
import json, time, hashlib
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import load_pem_public_key

MessageType = str  # "PREPREPARE" | "PREPARE" | "COMMIT" | "VIEW_CHANGE" | "NEW_VIEW" | "CLIENT_REQ"

@dataclass(frozen=True)
class SignedMessage:
    typ: MessageType
    view: int
    seq: int
    digest: str           # hex sha256 of client request / proposal
    sender: str           # node_id
    payload: Dict[str, Any]
    ts: float
    sig: bytes            # ed25519 signature

    @staticmethod
    def make(typ: MessageType, view: int, seq: int, sender: str, payload: Dict[str, Any],
             priv: Ed25519PrivateKey) -> "SignedMessage":
        body = {
            "typ": typ, "view": view, "seq": seq, "sender": sender,
            "payload": payload, "ts": time.time()
        }
        # Derive digest from payload['client_req'] or 'proposal'
        blob = json.dumps(payload, sort_keys=True).encode()
        dg = hashlib.sha256(blob).hexdigest()
        data = json.dumps({**body, "digest": dg}, sort_keys=True).encode()
        sig = priv.sign(data)
        return SignedMessage(typ, view, seq, dg, sender, payload, body["ts"], sig)

    def verify(self, pub_pem: bytes) -> None:
        pub = load_pem_public_key(pub_pem)
        data = json.dumps({
            "typ": self.typ, "view": self.view, "seq": self.seq, "sender": self.sender,
            "payload": self.payload, "ts": self.ts, "digest": self.digest
        }, sort_keys=True).encode()
        assert isinstance(pub, Ed25519PublicKey)
        pub.verify(self.sig, data)  # raises on failure