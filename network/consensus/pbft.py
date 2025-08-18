# network/consensus/pbft.py
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Set, Optional, List, Any, Callable
import asyncio, hashlib
try:
    from .messages import SignedMessage
    from .view_change import ViewState
    from .byzantine_detection import MisbehaviorTracker
except ImportError:
    # Fallback for direct execution
    from messages import SignedMessage
    from view_change import ViewState
    from byzantine_detection import MisbehaviorTracker

@dataclass
class LogEntry:
    request: Any
    digest: str
    preprepares: Set[str]
    prepares: Set[str]
    commits: Set[str]
    decided: bool = False

class PBFTNode:
    """
    Minimal PBFT core:
      - PRE-PREPARE (primary proposes)
      - PREPARE (replicas ack proposal)
      - COMMIT (replicas commit upon quorum)
      - VIEW CHANGE (on timeout) + leader election
    Authentication: Ed25519 signatures on all messages.
    """
    def __init__(self, node_id: str, peers: List[str], f: int,
                 send: Callable[[SignedMessage, str], None],
                 get_pubkey: Callable[[str], bytes],
                 is_primary: Callable[[int, List[str]], bool]):
        self.id = node_id
        self.peers = peers
        self.N = len(peers) + 1  # incl self
        self.f = f               # tolerate f faults (N >= 3f+1)
        self.view = 0
        self.seq = 0
        self.log: Dict[int, LogEntry] = {}
        self.view_state = ViewState()
        self.mis = MisbehaviorTracker()
        self.send = send
        self.get_pubkey = get_pubkey
        self.is_primary_fn = is_primary
        self.timeout_s = 2.0  # tune per environment

    # ---------- Client request entry ----------
    async def on_client_request(self, request: Any, sign: Callable[[bytes], bytes], priv_wrapper) -> None:
        self.seq += 1
        digest = hashlib.sha256(repr(request).encode()).hexdigest()
        self.log[self.seq] = LogEntry(request, digest, set(), set(), set())
        if self.is_primary_fn(self.view, [self.id] + self.peers):
            msg = priv_wrapper("PREPREPARE", self.view, self.seq, {"proposal": request})
            for p in self.peers:
                self.send(msg, p)
            await self._arm_timer(self.seq)
        else:
            # Non-primary forwards to primary
            primary = self._primary_id()
            self.send(priv_wrapper("CLIENT_REQ", self.view, self.seq, {"request": request}), primary)

    # ---------- Message handlers ----------
    async def on_message(self, m: SignedMessage):
        # Verify sig
        try:
            m.verify(self.get_pubkey(m.sender))
        except Exception:
            self.mis.flag(m.sender, "bad_signature")
            return
        if m.typ == "PREPREPARE":
            await self._on_preprepare(m)
        elif m.typ == "PREPARE":
            await self._on_prepare(m)
        elif m.typ == "COMMIT":
            await self._on_commit(m)
        elif m.typ == "VIEW_CHANGE":
            await self._on_view_change(m)
        elif m.typ == "NEW_VIEW":
            await self._on_new_view(m)

    async def _on_preprepare(self, m: SignedMessage):
        s = m.seq
        dg = m.digest
        le = self.log.setdefault(s, LogEntry(m.payload["proposal"], dg, set(), set(), set()))
        if le.digest != dg:
            self.mis.flag(m.sender, "equivocation_preprepare")
            return
        le.preprepares.add(m.sender)
        # broadcast PREPARE
        prep = self._wrap("PREPARE", s, {"digest": dg})
        for p in self.peers:
            self.send(prep, p)
        await self._arm_timer(s)

    async def _on_prepare(self, m: SignedMessage):
        s = m.seq
        le = self.log.get(s)
        if not le or le.digest != m.digest:
            return
        le.prepares.add(m.sender)
        if len(le.prepares) >= (2*self.f):  # quorum for prepare (excl self)
            com = self._wrap("COMMIT", s, {"digest": le.digest})
            for p in self.peers:
                self.send(com, p)

    async def _on_commit(self, m: SignedMessage):
        s = m.seq
        le = self.log.get(s)
        if not le or le.digest != m.digest:
            return
        le.commits.add(m.sender)
        if len(le.commits) >= (2*self.f + 1):  # incl self once we add it
            le.decided = True
            # TODO: deliver to application

    # ---------- View changes / leader election ----------
    async def _arm_timer(self, seq: int):
        async def _timeout():
            await asyncio.sleep(self.timeout_s)
            le = self.log.get(seq)
            if not le or le.decided:
                return
            # start view change
            vc = self._wrap("VIEW_CHANGE", seq, {"view": self.view + 1})
            for p in self.peers:
                self.send(vc, p)
        asyncio.create_task(_timeout())

    async def _on_view_change(self, m: SignedMessage):
        if m.payload.get("view", self.view) <= self.view:
            return
        self.view_state.collect(m.sender, m.payload["view"])
        if self.view_state.has_quorum(self.N, self.f):
            self.view = m.payload["view"]
            # new primary broadcasts NEW_VIEW
            if self.is_primary_fn(self.view, [self.id] + self.peers):
                nv = self._wrap("NEW_VIEW", 0, {"view": self.view})
                for p in self.peers:
                    self.send(nv, p)

    async def _on_new_view(self, m: SignedMessage):
        if m.payload.get("view", self.view) > self.view:
            self.view = m.payload["view"]

    # ---------- utils ----------
    def _primary_id(self) -> str:
        group = [self.id] + self.peers
        return group[self.view % len(group)]

    def _wrap(self, typ: str, seq: int, payload: dict) -> SignedMessage:
        return self._sign(typ, self.view, seq, payload)

    def _sign(self, typ: str, view: int, seq: int, payload: dict) -> SignedMessage:
        return self._priv_wrapper(typ, view, seq, payload)

    def set_priv_wrapper(self, wrapper: callable):
        self._priv_wrapper = wrapper