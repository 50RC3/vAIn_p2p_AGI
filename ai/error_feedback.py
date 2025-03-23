import logging
from typing import Dict, Any, Optional, List
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
from core.interactive_utils import InteractiveSession, InteractiveConfig
from core.constants import INTERACTION_TIMEOUTS
from ai.exceptions import BaseAIError

logger = logging.getLogger(__name__)

@dataclass
class ErrorMetrics:
    severity: float  # 0-1 scale
    impact: float  # 0-1 scale 
    frequency: int
    timestamp: datetime
    source: str
    remediation_attempts: int

class ErrorFeedbackManager:
    def __init__(self, interactive: bool = True):
        self.interactive = interactive
        self.error_history: Dict[str, List[ErrorMetrics]] = {}
        self.malicious_patterns: Dict[str, int] = {}
        self.severity_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        }
        self.impact_weights = {
            'critical_system': 1.0,
            'model_integrity': 0.8,
            'performance': 0.6,
            'data_quality': 0.7
        }
        
    async def process_error(self, error: BaseAIError, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process an error and determine appropriate feedback response"""
        try:
            error_metrics = self._analyze_error(error, context)
            
            if self._detect_malicious_pattern(error_metrics):
                await self._handle_malicious_behavior(error_metrics)
                return {'action': 'block', 'reason': 'malicious_behavior_detected'}
                
            severity_score = self._calculate_severity_score(error_metrics)
            impact_score = self._calculate_impact_score(error_metrics)
            
            response = await self._determine_response(severity_score, impact_score)
            self._update_history(error_metrics)
            
            return response
            
        except Exception as e:
            logger.error(f"Error feedback processing failed: {str(e)}")
            return {'action': 'default_fallback', 'error': str(e)}

    def _analyze_error(self, error: BaseAIError, context: Dict[str, Any]) -> ErrorMetrics:
        """Analyze error characteristics and context"""
        severity = self._calculate_base_severity(error)
        impact = self._assess_impact(error, context)
        
        return ErrorMetrics(
            severity=severity,
            impact=impact,
            frequency=self._get_error_frequency(error),
            timestamp=datetime.now(),
            source=context.get('source', 'unknown'),
            remediation_attempts=0
        )

    def _detect_malicious_pattern(self, metrics: ErrorMetrics) -> bool:
        """Detect potential malicious behavior patterns"""
        source = metrics.source
        if source not in self.malicious_patterns:
            self.malicious_patterns[source] = 0
            
        recent_errors = [
            e for e in self.error_history.get(source, [])
            if datetime.now() - e.timestamp < timedelta(minutes=5)
        ]
        
        if len(recent_errors) > 10:
            self.malicious_patterns[source] += 1
            
        high_severity_ratio = sum(1 for e in recent_errors if e.severity > 0.8) / max(len(recent_errors), 1)
        if high_severity_ratio > 0.7:
            self.malicious_patterns[source] += 1
            
        return self.malicious_patterns[source] >= 3

    async def _handle_malicious_behavior(self, metrics: ErrorMetrics) -> None:
        """Handle detected malicious behavior"""
        if self.interactive:
            session = InteractiveSession(
                config=InteractiveConfig(
                    timeout=INTERACTION_TIMEOUTS["emergency"],
                    safe_mode=True
                )
            )
            
            async with session:
                if await session.get_confirmation(
                    f"\nMalicious behavior detected from {metrics.source}. Block source?",
                    timeout=INTERACTION_TIMEOUTS["emergency"]
                ):
                    # Add blocking logic here
                    logger.warning(f"Blocked malicious source: {metrics.source}")

    def _calculate_severity_score(self, metrics: ErrorMetrics) -> float:
        """Calculate weighted severity score"""
        base_score = metrics.severity
        frequency_factor = min(metrics.frequency / 10, 1)
        time_factor = self._calculate_time_factor(metrics.timestamp)
        
        return min(base_score * (1 + frequency_factor) * time_factor, 1.0)

    def _calculate_impact_score(self, metrics: ErrorMetrics) -> float:
        """Calculate weighted impact score"""
        return min(metrics.impact * self.impact_weights.get(metrics.source, 0.5), 1.0)

    async def _determine_response(self, severity: float, impact: float) -> Dict[str, Any]:
        """Determine appropriate response based on severity and impact"""
        combined_score = (severity + impact) / 2
        
        if combined_score > self.severity_thresholds['high']:
            return {
                'action': 'critical_intervention',
                'severity': severity,
                'impact': impact,
                'requires_human': True
            }
        elif combined_score > self.severity_thresholds['medium']:
            return {
                'action': 'automated_mitigation',
                'severity': severity,
                'impact': impact,
                'requires_human': False
            }
        else:
            return {
                'action': 'monitor',
                'severity': severity,
                'impact': impact,
                'requires_human': False
            }

    def _update_history(self, metrics: ErrorMetrics) -> None:
        """Update error history with new metrics"""
        if metrics.source not in self.error_history:
            self.error_history[metrics.source] = []
            
        self.error_history[metrics.source].append(metrics)
        self._prune_history(metrics.source)

    def _prune_history(self, source: str) -> None:
        """Remove old entries from error history"""
        cutoff = datetime.now() - timedelta(hours=24)
        self.error_history[source] = [
            e for e in self.error_history[source]
            if e.timestamp > cutoff
        ]
