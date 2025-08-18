# network/consensus/byzantine_detection.py
from collections import Counter
from typing import Dict

class MisbehaviorTracker:
    def __init__(self, threshold:int=3):
        self.counts: Dict[str, Counter] = {}
        self.threshold = threshold

    def flag(self, node_id: str, reason: str):
        c = self.counts.setdefault(node_id, Counter())
        c[reason] += 1

    def is_byzantine(self, node_id: str) -> bool:
        c = self.counts.get(node_id)
        return bool(c and sum(c.values()) >= self.threshold)