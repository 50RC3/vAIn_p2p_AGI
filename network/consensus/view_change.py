# network/consensus/view_change.py
from collections import defaultdict

class ViewState:
    def __init__(self):
        self._votes = defaultdict(set)  # view -> {node_ids}

    def collect(self, sender: str, view: int):
        self._votes[view].add(sender)

    def has_quorum(self, N: int, f: int) -> bool:
        # PBFT: need 2f+1 to move safely, with N >= 3f+1
        for view, voters in self._votes.items():
            if len(voters) >= (2*f + 1):
                return True
        return False