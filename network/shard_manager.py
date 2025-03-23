import asyncio
import logging
import time
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import mmh3  # MurmurHash3 for consistent hashing
from core.constants import SHARD_CONFIG, INTERACTION_TIMEOUTS
from .node_communication import NodeCommunication

logger = logging.getLogger(__name__)

# Peer states and timeouts
PEER_STATES = {
    'ACTIVE': 'active',
    'INACTIVE': 'inactive',
    'FAILING': 'failing',
    'EXPIRED': 'expired'
}

PEER_TIMEOUTS = {
    'HEARTBEAT': 300,      # 5 minutes without heartbeat -> inactive
    'INACTIVITY': 1800,    # 30 minutes inactive -> failing
    'EXPIRATION': 3600,    # 1 hour failing -> expired
    'GRACE_PERIOD': 60     # 1 minute grace period for network issues
}

# Add shard management constants
SHARD_METRICS = {
    'LOAD_THRESHOLD': 0.8,     # Trigger resharding at 80% load
    'MIN_SHARDS': 16,          # Minimum number of shards
    'MAX_SHARDS': 1024,        # Maximum number of shards
    'MIN_REPLICAS': 3,         # Minimum replicas per shard
    'MAX_REPLICAS': 5,         # Maximum replicas per shard
    'REBALANCE_INTERVAL': 300  # Check load every 5 minutes
}

class ShardManager:
    def __init__(self):
        self.shards = defaultdict(dict)  # shard_id -> {peer_id -> peer_info}
        self.shard_replicas = {}  # shard_id -> [replica_nodes]
        self.ring = []  # Consistent hashing ring
        self.lock = asyncio.Lock()
        self.replication_retries = 3
        self.replication_timeout = INTERACTION_TIMEOUTS["peer_update"]
        self.node_comm = NodeCommunication(node_id="shard_manager", interactive=True)
        self.peer_states = {}  # Track peer states
        
        # Add zone and metrics tracking
        self.zones = defaultdict(set)  # zone -> {replica_nodes}
        self.shard_metrics = defaultdict(dict)  # shard_id -> metrics
        self.last_rebalance = time.time()
        self.rebalancing = False
        
    async def get_shard_id(self, peer_id: str) -> int:
        """Get shard ID for a peer using consistent hashing"""
        return mmh3.hash(peer_id) % SHARD_CONFIG["num_shards"]
        
    async def add_peer(self, peer_id: str, peer_info: Dict) -> bool:
        """Add peer to appropriate shard"""
        async with self.lock:
            shard_id = await self.get_shard_id(peer_id)
            if len(self.shards[shard_id]) >= SHARD_CONFIG["max_peers_per_shard"]:
                await self._rebalance_shards()
                shard_id = await self.get_shard_id(peer_id)
                
            self.shards[shard_id][peer_id] = peer_info
            await self._replicate_shard(shard_id)
            return True
            
    async def get_peer(self, peer_id: str) -> Optional[Dict]:
        """Get peer info from appropriate shard"""
        shard_id = await self.get_shard_id(peer_id)
        return self.shards[shard_id].get(peer_id)
        
    async def _check_shard_health(self) -> Dict[str, float]:
        """Calculate shard health metrics"""
        metrics = {}
        for shard_id, shard in self.shards.items():
            active_peers = sum(1 for p in shard.values() 
                             if not self._is_peer_expired(p))
            load = active_peers / SHARD_CONFIG["max_peers_per_shard"]
            replica_health = len(self.shard_replicas.get(shard_id, [])) / SHARD_METRICS['MIN_REPLICAS']
            metrics[shard_id] = min(load, replica_health)
        return metrics

    async def _should_reshard(self) -> Tuple[bool, str]:
        """Determine if resharding is needed based on metrics"""
        if self.rebalancing:
            return False, "Already rebalancing"
            
        if time.time() - self.last_rebalance < SHARD_METRICS['REBALANCE_INTERVAL']:
            return False, "Too soon to rebalance"
            
        metrics = await self._check_shard_health()
        if not metrics:
            return False, "No active shards"
            
        avg_load = sum(metrics.values()) / len(metrics)
        max_load = max(metrics.values())
        
        if max_load > SHARD_METRICS['LOAD_THRESHOLD']:
            return True, f"High load detected: {max_load:.2f}"
            
        if avg_load < SHARD_METRICS['LOAD_THRESHOLD'] / 2 and len(self.shards) > SHARD_METRICS['MIN_SHARDS']:
            return True, f"Low average load: {avg_load:.2f}"
            
        return False, "Healthy load levels"

    async def _rebalance_shards(self):
        """Enhanced rebalancing with load-based triggers"""
        try:
            should_reshard, reason = await self._should_reshard()
            if not should_reshard:
                logger.debug(f"Skipping rebalance: {reason}")
                return

            self.rebalancing = True
            logger.info(f"Starting shard rebalance: {reason}")

            # Calculate new shard count based on load
            total_peers = sum(len(shard) for shard in self.shards.values())
            target_load = SHARD_METRICS['LOAD_THRESHOLD'] * 0.7  # Target 70% of threshold
            new_shard_count = min(
                SHARD_METRICS['MAX_SHARDS'],
                max(SHARD_METRICS['MIN_SHARDS'],
                    int(total_peers / (SHARD_CONFIG["max_peers_per_shard"] * target_load))
                )
            )

            if new_shard_count == len(self.shards):
                return

            # Rebalance peers across new shards
            new_shards = defaultdict(dict)
            for shard in self.shards.values():
                for peer_id, peer_info in shard.items():
                    new_shard_id = mmh3.hash(peer_id) % new_shard_count
                    new_shards[new_shard_id][peer_id] = peer_info

            # Update shard configuration
            self.shards = new_shards
            SHARD_CONFIG["num_shards"] = new_shard_count
            self.last_rebalance = time.time()

            # Rebalance replicas
            await self._rebalance_replicas()

        except Exception as e:
            logger.error(f"Shard rebalancing failed: {e}")
            raise
        finally:
            self.rebalancing = False

    async def _rebalance_replicas(self):
        """Rebalance replica nodes with zone awareness"""
        for shard_id in self.shards:
            current_replicas = set(self.shard_replicas.get(shard_id, []))
            
            # Calculate required replicas per zone
            zones = list(self.zones.keys())
            replicas_per_zone = max(
                1, min(SHARD_METRICS['MAX_REPLICAS'] // len(zones),
                      SHARD_METRICS['MIN_REPLICAS'])
            )
            
            # Select replicas from each zone
            new_replicas = set()
            for zone in zones:
                zone_nodes = self.zones[zone]
                available = zone_nodes - current_replicas
                selected = set(sorted(available)[:replicas_per_zone])
                new_replicas.update(selected)
            
            # Update replica list
            self.shard_replicas[shard_id] = list(new_replicas)
            await self._replicate_shard(shard_id)
            
    async def _replicate_shard(self, shard_id: int):
        """Replicate shard data to backup nodes"""
        if shard_id in self.shard_replicas:
            failed_replicas = []
            for replica_node in self.shard_replicas[shard_id]:
                try:
                    success = await self._send_shard_data(shard_id, replica_node)
                    if not success:
                        failed_replicas.append(replica_node)
                except Exception as e:
                    logger.error(f"Failed to replicate shard {shard_id} to {replica_node}: {e}")
                    failed_replicas.append(replica_node)
            
            # Remove failed replicas
            for node in failed_replicas:
                await self.remove_replica_node(shard_id, node)

    async def _send_shard_data(self, shard_id: int, replica_node: str):
        """Send shard data to a replica node with retry logic"""
        retries = 0
        while retries < self.replication_retries:
            try:
                shard_data = self.shards[shard_id]
                # Create async timeout context
                async with asyncio.timeout(self.replication_timeout):
                    # Simulate network call - replace with actual RPC
                    await self._rpc_send_data(replica_node, {
                        "shard_id": shard_id,
                        "data": shard_data,
                        "timestamp": time.time()
                    })
                return True
            except asyncio.TimeoutError:
                logger.warning(f"Replication timeout to {replica_node}, attempt {retries + 1}")
                retries += 1
            except Exception as e:
                logger.error(f"Replication error: {str(e)}")
                retries += 1
            await asyncio.sleep(1)  # Brief delay between retries
        return False

    async def _rpc_send_data(self, node: str, data: Dict):
        """Send shard data to replica node using NodeCommunication"""
        try:
            message = {
                "type": "shard_replication",
                "payload": data,
                "version": "1.0",
                "timestamp": time.time()
            }
            
            success = await self.node_comm.send_message_interactive(
                target_node=node,
                message=message
            )
            
            if not success:
                raise Exception(f"Failed to send shard data to {node}")
                
            return True
            
        except Exception as e:
            logger.error(f"RPC error while sending data to {node}: {str(e)}")
            raise

    async def add_replica_node(self, shard_id: int, node: str):
        """Add a new replica node for a shard"""
        async with self.lock:
            if shard_id not in self.shard_replicas:
                self.shard_replicas[shard_id] = []
            if node not in self.shard_replicas[shard_id]:
                self.shard_replicas[shard_id].append(node)
                await self._replicate_shard(shard_id)

    async def remove_replica_node(self, shard_id: int, node: str):
        """Remove a replica node from a shard"""
        async with self.lock:
            if shard_id in self.shard_replicas and node in self.shard_replicas[shard_id]:
                self.shard_replicas[shard_id].remove(node)
                logger.info(f"Removed replica node {node} from shard {shard_id}")

    async def cleanup_shards(self):
        """Remove expired peers from shards"""
        async with self.lock:
            for shard_id, shard in self.shards.items():
                expired = []
                for peer_id, peer_info in shard.items():
                    if self._is_peer_expired(peer_info):
                        expired.append(peer_id)
                for peer_id in expired:
                    del shard[peer_id]
                if expired:
                    await self._replicate_shard(shard_id)
                    
    def _is_peer_expired(self, peer_info: Dict) -> bool:
        """Enhanced peer expiration check based on network activity"""
        current_time = time.time()
        peer_id = peer_info.get('peer_id')
        
        # Get last network interaction timestamps
        last_heartbeat = peer_info.get('last_heartbeat', 0)
        last_activity = peer_info.get('last_activity', 0)
        last_state_change = peer_info.get('last_state_change', 0)
        
        # Calculate time deltas
        time_since_heartbeat = current_time - last_heartbeat
        time_since_activity = current_time - last_activity
        current_state = self.peer_states.get(peer_id, PEER_STATES['ACTIVE'])
        
        # State transition logic
        if current_state == PEER_STATES['ACTIVE']:
            if time_since_heartbeat > PEER_TIMEOUTS['HEARTBEAT']:
                self._update_peer_state(peer_id, PEER_STATES['INACTIVE'])
                return False
                
        elif current_state == PEER_STATES['INACTIVE']:
            if time_since_activity > PEER_TIMEOUTS['INACTIVITY']:
                self._update_peer_state(peer_id, PEER_STATES['FAILING'])
                return False
                
        elif current_state == PEER_STATES['FAILING']:
            if current_time - last_state_change > PEER_TIMEOUTS['EXPIRATION']:
                self._update_peer_state(peer_id, PEER_STATES['EXPIRED'])
                return True
                
        # Check hard expiration
        expires_at = peer_info.get('expires_at', 0)
        if expires_at and current_time > expires_at + PEER_TIMEOUTS['GRACE_PERIOD']:
            self._update_peer_state(peer_id, PEER_STATES['EXPIRED'])
            return True
            
        return False

    def _update_peer_state(self, peer_id: str, new_state: str):
        """Update peer state with timestamp"""
        self.peer_states[peer_id] = new_state
        shard_id = self.get_shard_id(peer_id)
        if peer_id in self.shards[shard_id]:
            self.shards[shard_id][peer_id]['last_state_change'] = time.time()
            logger.info(f"Peer {peer_id} state changed to {new_state}")

    async def record_peer_activity(self, peer_id: str, activity_type: str):
        """Record peer network activity"""
        shard_id = await self.get_shard_id(peer_id)
        if peer_id in self.shards[shard_id]:
            current_time = time.time()
            self.shards[shard_id][peer_id]['last_activity'] = current_time
            if activity_type == 'heartbeat':
                self.shards[shard_id][peer_id]['last_heartbeat'] = current_time
                if self.peer_states.get(peer_id) != PEER_STATES['ACTIVE']:
                    self._update_peer_state(peer_id, PEER_STATES['ACTIVE'])

    async def cleanup(self):
        """Cleanup resources on shutdown"""
        self.node_comm.request_shutdown()
