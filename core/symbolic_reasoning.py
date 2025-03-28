import logging
from typing import Dict, Any, Set
from enum import Enum

logger = logging.getLogger(__name__)

class RuleType(Enum):
    DOMAIN = "domain"
    UPDATE = "update"
    MEMORY = "memory"

class PropositionalLogic:
    def __init__(self):
        self.variables: Dict[str, bool] = {}
        self.rules: Dict[RuleType, Set[str]] = {
            RuleType.DOMAIN: set(),
            RuleType.UPDATE: set(),
            RuleType.MEMORY: set()
        }

    def add_variable(self, name: str, value: bool = True) -> None:
        self.variables[name] = value

    def add_rule(self, rule_type: RuleType, expression: str) -> None:
        self.rules[rule_type].add(expression)

    def evaluate_expression(self, expr: str) -> bool:
        try:
            return eval(expr, {"__builtins__": {}}, self.variables)
        except Exception as e:
            logger.error(f"Failed to evaluate expression {expr}: {str(e)}")
            return False

    def set_variable(self, name: str, value: bool) -> None:
        self.variables[name] = value
