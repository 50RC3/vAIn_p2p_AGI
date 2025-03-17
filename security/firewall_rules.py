from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class FirewallRule:
    protocol: str
    port: int
    direction: str
    action: str
    priority: int

class FirewallManager:
    def __init__(self):
        self.rules: List[FirewallRule] = []
        self._setup_default_rules()
        
    def _setup_default_rules(self):
        self.rules.extend([
            FirewallRule("TCP", 8545, "INBOUND", "ALLOW", 100),  # Ethereum RPC
            FirewallRule("TCP", 30303, "BOTH", "ALLOW", 100),    # P2P
            FirewallRule("UDP", 30303, "BOTH", "ALLOW", 100)     # P2P Discovery
        ])
        
    def add_rule(self, rule: FirewallRule):
        self.rules.append(rule)
        self._sort_rules()
