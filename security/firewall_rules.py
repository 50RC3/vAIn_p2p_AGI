import logging
import asyncio
from typing import List, Dict, Optional
from dataclasses import dataclass
from core.constants import InteractionLevel, INTERACTION_TIMEOUTS
from core.interactive_utils import InteractiveSession, InteractiveConfig

logger = logging.getLogger(__name__)

@dataclass
class FirewallRule:
    protocol: str
    port: int
    direction: str
    action: str
    priority: int
    
    def validate(self) -> bool:
        """Validate rule parameters"""
        try:
            if self.protocol not in ["TCP", "UDP"]:
                return False
            if not 1 <= self.port <= 65535:
                return False
            if self.direction not in ["INBOUND", "OUTBOUND", "BOTH"]:
                return False
            if self.action not in ["ALLOW", "DENY"]:
                return False
            if not 0 <= self.priority <= 1000:
                return False
            return True
        except Exception:
            return False

class FirewallManager:
    def __init__(self, interactive: bool = True):
        self.rules: List[FirewallRule] = []
        self.interactive = interactive
        self.session: Optional[InteractiveSession] = None
        self._interrupt_requested = False
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

    async def add_rule_interactive(self, rule: FirewallRule) -> bool:
        """Add rule with interactive validation and confirmation"""
        try:
            if self.interactive:
                self.session = InteractiveSession(
                    level=InteractionLevel.NORMAL,
                    config=InteractiveConfig(
                        timeout=INTERACTION_TIMEOUTS["default"],
                        persistent_state=True,
                        safe_mode=True
                    )
                )

            async with self.session:
                if not rule.validate():
                    logger.error("Invalid firewall rule parameters")
                    return False

                # Check for rule conflicts
                conflicts = self._check_rule_conflicts(rule)
                if conflicts and self.interactive:
                    proceed = await self.session.confirm_with_timeout(
                        f"\nRule conflicts with {len(conflicts)} existing rules. Continue?",
                        timeout=INTERACTION_TIMEOUTS["confirmation"]
                    )
                    if not proceed:
                        return False

                self.add_rule(rule)
                return True

        except Exception as e:
            logger.error(f"Failed to add firewall rule: {str(e)}")
            return False
        finally:
            if self.session:
                await self.session.__aexit__(None, None, None)

    def _check_rule_conflicts(self, new_rule: FirewallRule) -> List[FirewallRule]:
        """Check for conflicting rules"""
        conflicts = []
        for rule in self.rules:
            if (rule.protocol == new_rule.protocol and 
                rule.port == new_rule.port and
                rule.direction in [new_rule.direction, "BOTH"] and
                rule.action != new_rule.action):
                conflicts.append(rule)
        return conflicts

    def _sort_rules(self):
        """Sort rules by priority"""
        self.rules.sort(key=lambda x: x.priority)

    async def bulk_update_rules_interactive(self, new_rules: List[FirewallRule]) -> bool:
        """Update multiple rules with progress tracking"""
        try:
            if self.interactive:
                self.session = InteractiveSession(
                    level=InteractionLevel.NORMAL,
                    config=InteractiveConfig(
                        timeout=INTERACTION_TIMEOUTS["batch"],
                        persistent_state=True,
                        safe_mode=True
                    )
                )

            async with self.session:
                invalid_rules = [r for r in new_rules if not r.validate()]
                if invalid_rules:
                    logger.error(f"Found {len(invalid_rules)} invalid rules")
                    return False

                if self.interactive:
                    proceed = await self.session.confirm_with_timeout(
                        f"\nUpdate {len(new_rules)} firewall rules?",
                        timeout=INTERACTION_TIMEOUTS["confirmation"]
                    )
                    if not proceed:
                        return False

                self.rules = new_rules
                self._sort_rules()
                return True

        except Exception as e:
            logger.error(f"Bulk update failed: {str(e)}")
            return False
        finally:
            if self.session:
                await self.session.__aexit__(None, None, None)

    def request_shutdown(self):
        """Request graceful shutdown"""
        self._interrupt_requested = True
