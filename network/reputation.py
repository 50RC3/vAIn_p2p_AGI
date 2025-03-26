from typing import Dict, Optional, Set, Tuple, List, NamedTuple, DefaultDict
from dataclasses import dataclass
import time
import logging
import asyncio
from tqdm import tqdm
from core.constants import INTERACTION_TIMEOUTS, InteractionLevel
from core.interactive_utils import InteractiveSession, InteractiveConfig
from collections import defaultdict, Counter
from time import monotonic
import ipaddress
import numpy as np
import os
from .memory_monitor import MemoryMonitor

@dataclass
class ReputationMetrics:
    score: float
    last_update: float
    total_contributions: int
    last_validation: float = 0.0
    validation_failures: int = 0
    total_validations: int = 0

@dataclass
class SuspiciousActivity:
    timestamp: float
    activity_type: str  # sybil, collusion, validation_failure, etc
    evidence: Dict[str, any]
    severity: float

class PendingChange(NamedTuple):
    score_delta: float
    timestamp: float
    reason: str

class ReputationManager:
    def __init__(self, decay_factor: float = 0.95, min_reputation: float = -100, 
                 interactive: bool = True, **kwargs):
        self.reputation_scores: Dict[str, ReputationMetrics] = {}
        self.decay_factor = decay_factor
        self.min_reputation = min_reputation
        self.logger = logging.getLogger('ReputationManager')
        self.interactive = interactive
        self.session: Optional[InteractiveSession] = None
        self._interrupt_requested = False
        self._progress_file = "reputation_progress.json"
        self.revalidation_config = {
            'period': 3600,  # Revalidate every hour
            'failure_threshold': 3,  # Max failures before penalty
            'penalty_factor': 0.5,  # 50% reputation reduction on validation failure
            'recovery_rate': 0.1,  # Slow recovery after failures
            'min_validations': 5,  # Minimum validations before stable reputation
            'max_skip_period': 86400  # Maximum time between validations (1 day)
        }
        self._revalidation_task = None
        
        # Add cooling period settings
        self.cooling_config = {
            'period': 300,  # 5 minute cooling period
            'batch_interval': 60,  # Process pending changes every minute
            'negligible_change': 0.1,  # Small changes processed immediately
            'cache_ttl': 60,  # Cache validation results for 1 minute
            'max_pending': 100  # Maximum pending changes per peer
        }
        
        # Add pending changes and cache
        self.pending_changes: Dict[str, List[PendingChange]] = defaultdict(list)
        self._validation_cache: Dict[str, Tuple[bool, float]] = {}
        self._pending_task: Optional[asyncio.Task] = None

        # Add malicious behavior detection settings
        self.malicious_config = {
            'sybil': {
                'ip_threshold': 5,  # Max nodes per IP subnet
                'creation_interval': 3600,  # Min seconds between new nodes
                'similarity_threshold': 0.9  # Behavioral similarity threshold
            },
            'collusion': {
                'interaction_window': 86400,  # 24h window for interaction analysis
                'agreement_threshold': 0.95,  # Suspicious agreement threshold
                'min_interactions': 10  # Minimum interactions to analyze
            },
            'analysis_interval': 3600  # Run analysis hourly
        }
        
        # Add tracking structures
        self.ip_mappings: DefaultDict[str, Set[str]] = defaultdict(set)  # IP -> node_ids
        self.node_ips: Dict[str, str] = {}  # node_id -> IP
        self.interaction_history: DefaultDict[str, List[Tuple[str, float, bool]]] = defaultdict(list)  # node -> [(interacted_with, timestamp, agreed)]
        self.suspicious_activities: DefaultDict[str, List[SuspiciousActivity]] = defaultdict(list)
        self._analysis_task: Optional[asyncio.Task] = None

        # Add dynamic decay configuration
        self.decay_config = {
            'base_factor': decay_factor,
            'min_factor': 0.8,  # Faster decay for problematic nodes
            'max_factor': 0.98,  # Slower decay for reliable nodes
            'adjustment_rate': 0.02,  # How quickly to adjust decay factor
            'failure_penalty': 0.05,  # Decrease factor by 5% per failure
            'success_bonus': 0.01,   # Increase factor by 1% per success
            'history_window': 10     # Number of validations to consider
        }
        
        # Track per-node decay factors
        self.node_decay_factors: Dict[str, float] = defaultdict(
            lambda: self.decay_config['base_factor']
        )

        # Add batch processing configuration
        self.batch_config = {
            'max_batch_size': kwargs.get('max_batch_size', 1000),
            'min_batch_size': kwargs.get('min_batch_size', 50),
            'max_parallel': kwargs.get('max_parallel', min(os.cpu_count() * 2, 16)),
            'target_latency': kwargs.get('target_latency', 0.5)
        }
        
        # Batch processing state
        self.batch_queue = asyncio.PriorityQueue()
        self.processing_tasks = set()
        self.batch_stats = {
            'processed': 0,
            'avg_latency': 0.0,
            'last_adjustment': time.time()
        }

        # Initialize memory monitor
        self.memory_monitor = MemoryMonitor(
            threshold=kwargs.get('memory_threshold', 0.9),
            check_interval=kwargs.get('memory_check_interval', 60)
        )

        # Add memory callback
        async def on_high_memory(usage: float):
            # Reduce batch size to manage memory
            self.batch_config['max_batch_size'] = max(
                self.batch_config['min_batch_size'],
                int(self.batch_config['max_batch_size'] * 0.8)
            )
            self.logger.warning(
                f"Reduced max batch size to {self.batch_config['max_batch_size']} "
                f"due to high memory usage ({usage:.1%})"
            )

        # Start memory monitoring in start() method
        self._memory_callback = on_high_memory

    def update_reputation(self, node_id: str, score_delta: float):
        """Update node's reputation score with dynamic decay."""
        current_time = time.time()
        if node_id not in self.reputation_scores:
            self.reputation_scores[node_id] = ReputationMetrics(0.0, current_time, 0)
            
        metrics = self.reputation_scores[node_id]
        # Apply dynamic time decay
        time_diff = current_time - metrics.last_update
        decay_factor = self.node_decay_factors[node_id]
        decayed_score = metrics.score * (decay_factor ** (time_diff / 86400))  # Daily decay
        
        # Update metrics
        new_score = max(decayed_score + score_delta, self.min_reputation)
        self.reputation_scores[node_id] = ReputationMetrics(
            score=new_score,
            last_update=current_time,
            total_contributions=metrics.total_contributions + 1,
            last_validation=metrics.last_validation,
            validation_failures=metrics.validation_failures,
            total_validations=metrics.total_validations
        )
        
    def get_reputation(self, node_id: str) -> float:
        """Get current reputation score for a node."""
        if node_id not in self.reputation_scores:
            return 0.0
        return self.reputation_scores[node_id].score
        
    def get_top_nodes(self, n: int = 10) -> Dict[str, float]:
        """Get top N nodes by reputation score."""
        return dict(sorted(
            [(k, v.score) for k, v in self.reputation_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )[:n])

    async def start(self):
        """Start background tasks"""
        await self.start_revalidation_loop()
        self._pending_task = asyncio.create_task(self._process_pending_changes())
        self._analysis_task = asyncio.create_task(self._analyze_malicious_behavior())
        await self.memory_monitor.start_monitoring(self._memory_callback)

    async def update_reputation_interactive(self, node_id: str, 
                                         score_delta: float, reason: str = "") -> bool:
        """Queue reputation update with cooling period"""
        try:
            # Process small changes immediately
            if abs(score_delta) <= self.cooling_config['negligible_change']:
                self.update_reputation(node_id, score_delta)
                return True
                
            # Check pending changes limit
            if len(self.pending_changes[node_id]) >= self.cooling_config['max_pending']:
                self.logger.warning(f"Too many pending changes for {node_id}")
                return False
                
            # Queue the change
            self.pending_changes[node_id].append(PendingChange(
                score_delta=score_delta,
                timestamp=monotonic(),
                reason=reason
            ))
            
            if self.interactive:
                self.logger.info(f"Queued reputation change for {node_id}: {score_delta}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to queue reputation update: {str(e)}")
            return False

    async def bulk_update_interactive(self, updates: Dict[str, float]) -> bool:
        """Process multiple reputation updates with progress tracking"""
        if not updates:
            return True

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
                # Restore progress if available
                processed: Set[str] = set()
                if self.interactive:
                    saved_progress = await self.session._load_progress()
                    if (saved_progress):
                        processed = set(saved_progress.get("processed_nodes", []))
                        self.logger.info(f"Restored progress: {len(processed)} nodes")

                remaining_updates = {k: v for k, v in updates.items() 
                                  if k not in processed}

                if self.interactive:
                    pbar = tqdm(total=len(remaining_updates), 
                              desc="Processing Reputation Updates")

                for node_id, delta in remaining_updates.items():
                    if self._interrupt_requested:
                        break

                    success = await self.update_reputation_interactive(node_id, delta)
                    if success:
                        processed.add(node_id)
                        if self.interactive:
                            pbar.update(1)
                            await self.session._save_progress({
                                "processed_nodes": list(processed)
                            })

                if self.interactive:
                    pbar.close()

                return len(processed) == len(updates)

        except Exception as e:
            self.logger.error(f"Bulk reputation update failed: {str(e)}")
            return False
        finally:
            if self.session:
                await self.session.__aexit__(None, None, None)

    async def start_revalidation_loop(self):
        """Start periodic peer revalidation"""
        self._revalidation_task = asyncio.create_task(self._periodic_revalidation())
        self.logger.info("Started periodic peer revalidation")

    async def _periodic_revalidation(self):
        """Run periodic revalidation of peers"""
        while not self._interrupt_requested:
            try:
                current_time = time.time()
                peers_to_validate = [
                    peer_id for peer_id, metrics in self.reputation_scores.items()
                    if (current_time - metrics.last_validation) >= self.revalidation_config['period']
                ]

                if peers_to_validate:
                    await self._revalidate_peers(peers_to_validate)
                
                await asyncio.sleep(self.revalidation_config['period'])

            except Exception as e:
                self.logger.error(f"Error in revalidation loop: {str(e)}")
                await asyncio.sleep(60)  # Retry after delay

    async def _revalidate_peers(self, peer_ids: List[str]):
        """Revalidate a batch of peers"""
        try:
            if self.interactive:
                self.session = InteractiveSession(
                    level=InteractionLevel.NORMAL,
                    config=InteractiveConfig(
                        timeout=INTERACTION_TIMEOUTS["batch"],
                        persistent_state=True
                    )
                )

            async with self.session:
                if self.interactive:
                    pbar = tqdm(total=len(peer_ids), desc="Revalidating Peers")

                for peer_id in peer_ids:
                    if self._interrupt_requested:
                        break

                    validation_result = await self._validate_peer(peer_id)
                    await self._update_validation_metrics(peer_id, validation_result)

                    if self.interactive:
                        pbar.update(1)

                if self.interactive:
                    pbar.close()

        except Exception as e:
            self.logger.error(f"Peer revalidation failed: {str(e)}")
        finally:
            if self.session:
                await self.session.__aexit__(None, None, None)

    async def _validate_peer(self, peer_id: str) -> bool:
        """Validate peer with caching"""
        current_time = monotonic()
        
        # Check cache
        if peer_id in self._validation_cache:
            result, timestamp = self._validation_cache[peer_id]
            if current_time - timestamp < self.cooling_config['cache_ttl']:
                return result
                
        # Perform validation
        metrics = self.reputation_scores[peer_id]
        current_time = time.time()

        # Check validation frequency
        if (current_time - metrics.last_validation) > self.revalidation_config['max_skip_period']:
            self.logger.warning(f"Peer {peer_id} exceeded maximum validation interval")
            return False

        # Custom validation logic here - integrate with node verification
        # This should be implemented based on your specific requirements
        result = True
        
        # Cache result
        self._validation_cache[peer_id] = (result, current_time)
        return result

    async def _update_validation_metrics(self, peer_id: str, success: bool):
        """Update peer metrics and decay factor based on validation result"""
        if peer_id not in self.reputation_scores:
            return

        metrics = self.reputation_scores[peer_id]
        current_time = time.time()

        metrics.last_validation = current_time
        metrics.total_validations += 1

        # Update decay factor based on validation result
        current_factor = self.node_decay_factors[peer_id]
        config = self.decay_config

        if not success:
            metrics.validation_failures += 1
            # Decrease decay factor for failures
            new_factor = max(
                config['min_factor'],
                current_factor * (1 - config['failure_penalty'])
            )
            if metrics.validation_failures >= self.revalidation_config['failure_threshold']:
                # Apply reputation penalty
                penalty = -abs(metrics.score * self.revalidation_config['penalty_factor'])
                self.update_reputation(peer_id, penalty)
                self.logger.warning(
                    f"Applied reputation penalty to {peer_id} for validation failures. "
                    f"Decay factor decreased to {new_factor:.3f}"
                )
        else:
            # Increase decay factor for successful validations
            new_factor = min(
                config['max_factor'],
                current_factor * (1 + config['success_bonus'])
            )
            # Handle recovery from previous failures
            if metrics.validation_failures > 0:
                self.update_reputation(
                    peer_id,
                    self.revalidation_config['recovery_rate'] * metrics.validation_failures
                )
                metrics.validation_failures = max(0, metrics.validation_failures - 1)

        self.node_decay_factors[peer_id] = new_factor

    async def _process_pending_changes(self):
        """Process pending reputation changes with optimized batching"""
        while not self._interrupt_requested:
            try:
                current_time = monotonic()
                batch = []
                priorities = []

                # Collect and prioritize changes
                for node_id, changes in list(self.pending_changes.items()):
                    ready_changes = [
                        change for change in changes
                        if current_time - change.timestamp >= self.cooling_config['period']
                    ]
                    
                    if ready_changes:
                        # Calculate priority based on total impact and age
                        total_impact = sum(abs(c.score_delta) for c in ready_changes)
                        max_age = max(current_time - c.timestamp for c in ready_changes)
                        priority = total_impact * (1 + max_age / self.cooling_config['period'])
                        
                        batch.append((node_id, ready_changes))
                        priorities.append(priority)

                        # Remove processed changes
                        self.pending_changes[node_id] = [
                            c for c in changes if c not in ready_changes
                        ]

                if not batch:
                    await asyncio.sleep(self.cooling_config['batch_interval'])
                    continue

                # Sort by priority
                batch = [x for _, x in sorted(zip(priorities, batch), reverse=True)]
                
                # Split into sub-batches for parallel processing
                sub_batches = self._create_sub_batches(batch)
                
                # Process sub-batches in parallel
                start_time = time.time()
                tasks = []
                for sub_batch in sub_batches:
                    if len(self.processing_tasks) >= self.batch_config['max_parallel']:
                        # Wait for a task to complete if at parallel limit
                        done, _ = await asyncio.wait(
                            self.processing_tasks, 
                            return_when=asyncio.FIRST_COMPLETED
                        )
                        self.processing_tasks.difference_update(done)
                    
                    task = asyncio.create_task(self._process_sub_batch(sub_batch))
                    self.processing_tasks.add(task)
                    tasks.append(task)

                # Wait for all sub-batches to complete
                await asyncio.gather(*tasks)
                
                # Update batch statistics
                process_time = time.time() - start_time
                self._update_batch_stats(len(batch), process_time)
                
                # Adjust batch parameters based on performance
                self._adjust_batch_parameters()

            except Exception as e:
                self.logger.error(f"Error processing pending changes: {str(e)}")
                await asyncio.sleep(60)

    def _create_sub_batches(self, batch: List[Tuple[str, List[PendingChange]]]) -> List[List]:
        """Split batch into optimal sub-batches for parallel processing"""
        target_size = max(
            self.batch_config['min_batch_size'],
            min(
                len(batch) // self.batch_config['max_parallel'],
                self.batch_config['max_batch_size']
            )
        )
        
        return [
            batch[i:i + target_size] 
            for i in range(0, len(batch), target_size)
        ]

    async def _process_sub_batch(self, sub_batch: List[Tuple[str, List[PendingChange]]]):
        """Process a sub-batch of reputation updates"""
        try:
            for node_id, changes in sub_batch:
                total_delta = sum(c.score_delta for c in changes)
                self.update_reputation(node_id, total_delta)
                
                if self.interactive:
                    self.logger.debug(
                        f"Processed {len(changes)} changes for {node_id}, "
                        f"total delta: {total_delta:.3f}"
                    )
        finally:
            if asyncio.current_task() in self.processing_tasks:
                self.processing_tasks.remove(asyncio.current_task())

    def _update_batch_stats(self, batch_size: int, process_time: float):
        """Update batch processing statistics"""
        self.batch_stats['processed'] += batch_size
        
        # Exponential moving average for latency
        alpha = 0.1
        self.batch_stats['avg_latency'] = (
            (1 - alpha) * self.batch_stats['avg_latency'] + 
            alpha * (process_time / batch_size)
        )

    def _adjust_batch_parameters(self):
        """Dynamically adjust batch parameters based on performance"""
        current_time = time.time()
        if current_time - self.batch_stats['last_adjustment'] < 60:
            return

        latency = self.batch_stats['avg_latency']
        target = self.batch_config['target_latency']
        
        # Adjust max_batch_size based on latency
        if latency > target * 1.2:  # Too slow
            self.batch_config['max_batch_size'] = max(
                self.batch_config['min_batch_size'],
                int(self.batch_config['max_batch_size'] * 0.8)
            )
        elif latency < target * 0.8:  # Too fast
            self.batch_config['max_batch_size'] = min(
                self.batch_config['max_batch_size'] * 1.2,
                1000  # Hard upper limit
            )

        self.batch_stats['last_adjustment'] = current_time

    async def register_node(self, node_id: str, ip_address: str) -> bool:
        """Register a new node with IP tracking"""
        try:
            # Validate IP
            ip = ipaddress.ip_address(ip_address)
            subnet = str(ipaddress.ip_network(f"{ip}/24", strict=False))
            
            # Check Sybil indicators
            if len(self.ip_mappings[subnet]) >= self.malicious_config['sybil']['ip_threshold']:
                await self._record_suspicious_activity(node_id, "sybil", {
                    "subnet": subnet,
                    "existing_nodes": list(self.ip_mappings[subnet])
                })
                return False
                
            current_time = time.time()
            subnet_registrations = [
                node for node in self.ip_mappings[subnet]
                if node in self.reputation_scores and
                current_time - self.reputation_scores[node].last_update < 
                self.malicious_config['sybil']['creation_interval']
            ]
            
            if subnet_registrations:
                await self._record_suspicious_activity(node_id, "sybil", {
                    "subnet": subnet,
                    "recent_registrations": subnet_registrations
                })
                return False
            
            # Register node
            self.ip_mappings[subnet].add(node_id)
            self.node_ips[node_id] = ip_address
            return True

        except Exception as e:
            self.logger.error(f"Node registration failed: {str(e)}")
            return False

    async def record_interaction(self, node1: str, node2: str, agreed: bool):
        """Record interaction between nodes for collusion detection"""
        current_time = time.time()
        self.interaction_history[node1].append((node2, current_time, agreed))
        self.interaction_history[node2].append((node1, current_time, agreed))

    async def _analyze_malicious_behavior(self):
        """Periodic analysis of malicious behavior patterns with decay factor adjustment"""
        while not self._interrupt_requested:
            try:
                current_time = time.time()
                window_start = current_time - self.malicious_config['collusion']['interaction_window']
                
                # Analyze interactions for collusion
                for node_id, interactions in self.interaction_history.items():
                    recent_interactions = [
                        i for i in interactions if i[1] > window_start
                    ]
                    
                    if len(recent_interactions) < self.malicious_config['collusion']['min_interactions']:
                        continue
                        
                    # Group by interacting node
                    peer_interactions = defaultdict(list)
                    for peer, _, agreed in recent_interactions:
                        peer_interactions[peer].append(agreed)
                        
                    # Check for suspicious agreement patterns
                    for peer, agreements in peer_interactions.items():
                        if len(agreements) < self.malicious_config['collusion']['min_interactions']:
                            continue
                            
                        agreement_rate = sum(agreements) / len(agreements)
                        if agreement_rate > self.malicious_config['collusion']['agreement_threshold']:
                            await self._record_suspicious_activity(node_id, "collusion", {
                                "peer": peer,
                                "agreement_rate": agreement_rate,
                                "interactions": len(agreements)
                            })
                
                # Analyze behavioral patterns for Sybil detection
                await self._analyze_behavioral_patterns()
                
                # Adjust decay factors based on behavior patterns
                await self._adjust_decay_factors()
                
                await asyncio.sleep(self.malicious_config['analysis_interval'])
                
            except Exception as e:
                self.logger.error(f"Malicious behavior analysis failed: {str(e)}")
                await asyncio.sleep(60)

    async def _adjust_decay_factors(self):
        """Adjust decay factors based on recent behavior patterns"""
        try:
            current_time = time.time()
            for node_id, activities in self.suspicious_activities.items():
                # Get recent suspicious activities
                recent_activities = [
                    a for a in activities 
                    if current_time - a.timestamp < self.malicious_config['collusion']['interaction_window']
                ]
                
                if recent_activities:
                    # Calculate average severity of recent suspicious activities
                    avg_severity = sum(a.severity for a in recent_activities) / len(recent_activities)
                    
                    # Decrease decay factor based on severity
                    current_factor = self.node_decay_factors[node_id]
                    new_factor = max(
                        self.decay_config['min_factor'],
                        current_factor * (1 - avg_severity * self.decay_config['adjustment_rate'])
                    )
                    
                    self.node_decay_factors[node_id] = new_factor
                    if new_factor < current_factor:
                        self.logger.warning(
                            f"Decreased decay factor for {node_id} to {new_factor:.3f} "
                            f"due to suspicious activities"
                        )

        except Exception as e:
            self.logger.error(f"Decay factor adjustment failed: {str(e)}")

    async def _analyze_behavioral_patterns(self):
        """Analyze node behavioral patterns for Sybil detection"""
        try:
            # Group nodes by subnet
            subnet_nodes = defaultdict(list)
            for node_id, ip in self.node_ips.items():
                subnet = str(ipaddress.ip_network(f"{ip}/24", strict=False))
                subnet_nodes[subnet].append(node_id)
                
            # Analyze behavior similarity within subnets
            for subnet, nodes in subnet_nodes.items():
                if len(nodes) < 2:
                    continue
                    
                # Compare interaction patterns
                for i, node1 in enumerate(nodes):
                    for node2 in nodes[i+1:]:
                        similarity = await self._calculate_behavior_similarity(node1, node2)
                        if similarity > self.malicious_config['sybil']['similarity_threshold']:
                            await self._record_suspicious_activity(node1, "sybil", {
                                "similar_node": node2,
                                "subnet": subnet,
                                "similarity": similarity
                            })
        
        except Exception as e:
            self.logger.error(f"Behavioral pattern analysis failed: {str(e)}")

    async def _calculate_behavior_similarity(self, node1: str, node2: str) -> float:
        """Calculate behavioral similarity between two nodes"""
        try:
            # Get recent interactions
            current_time = time.time()
            window_start = current_time - self.malicious_config['collusion']['interaction_window']
            
            interactions1 = [
                (peer, agreed) for peer, ts, agreed in self.interaction_history[node1]
                if ts > window_start
            ]
            interactions2 = [
                (peer, agreed) for peer, ts, agreed in self.interaction_history[node2]
                if ts > window_start
            ]
            
            # Compare interaction peers
            peers1 = set(i[0] for i in interactions1)
            peers2 = set(i[0] for i in interactions2)
            peer_similarity = len(peers1.intersection(peers2)) / max(len(peers1), len(peers2)) if peers1 or peers2 else 0
            
            # Compare agreement patterns
            common_peers = peers1.intersection(peers2)
            if not common_peers:
                return peer_similarity
                
            agreements1 = {peer: agreed for peer, agreed in interactions1 if peer in common_peers}
            agreements2 = {peer: agreed for peer, agreed in interactions2 if peer in common_peers}
            
            agreement_similarity = sum(
                agreements1[peer] == agreements2[peer] for peer in common_peers
            ) / len(common_peers)
            
            return (peer_similarity + agreement_similarity) / 2
            
        except Exception as e:
            self.logger.error(f"Behavior similarity calculation failed: {str(e)}")
            return 0.0

    async def _record_suspicious_activity(self, node_id: str, activity_type: str, evidence: Dict):
        """Record suspicious activity with severity calculation"""
        severity = {
            "sybil": 0.8,
            "collusion": 0.6,
            "validation_failure": 0.4
        }.get(activity_type, 0.3)
        
        activity = SuspiciousActivity(
            timestamp=time.time(),
            activity_type=activity_type,
            evidence=evidence,
            severity=severity
        )
        
        self.suspicious_activities[node_id].append(activity)
        
        # Apply immediate reputation penalty
        penalty = -abs(self.reputation_scores[node_id].score * severity)
        await self.update_reputation_interactive(
            node_id,
            penalty,
            f"suspicious_{activity_type}"
        )

    def request_shutdown(self):
        """Request graceful shutdown of reputation updates"""
        self._interrupt_requested = True
        self.logger.info("Shutdown requested for reputation manager")

    async def _cleanup(self):
        """Cleanup resources"""
        try:
            # Clean up any temporary files or resources
            if self.session:
                await self.session.__aexit__(None, None, None)
            self.session = None
        except Exception as e:
            self.logger.error(f"Cleanup failed: {str(e)}")

    async def cleanup(self):
        """Enhanced cleanup with memory monitor"""
        self._interrupt_requested = True
        if self._pending_task:
            self._pending_task.cancel()
            try:
                await self._pending_task
            except asyncio.CancelledError:
                pass
        if self._revalidation_task:
            self._revalidation_task.cancel()
            try:
                await self._revalidation_task
            except asyncio.CancelledError:
                pass
        if self._analysis_task:
            self._analysis_task.cancel()
            try:
                await self._analysis_task
            except asyncio.CancelledError:
                pass
        self.memory_monitor.stop()
        await self._cleanup()
