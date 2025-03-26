from typing import Dict, Any
import os

# Production security settings
SECURITY_CONFIG = {
    "min_tls_version": "1.3",
    "key_rotation_days": 90,
    "max_failed_auth_attempts": 5,
    "auth_lockout_minutes": 30,
    "required_hash_rounds": 12,
    "max_concurrent_connections": 10000,
    "rate_limit_requests": 100,  # Per minute per IP
    "ddos_threshold": 1000,  # Requests per second
}

# Production node requirements
NODE_REQUIREMENTS = {
    "min_cpu_cores": 4,
    "min_ram_gb": 8,
    "min_storage_gb": 100,
    "min_bandwidth_mbps": 100,
    "max_latency_ms": 100,
    "min_uptime_percent": 99.9
}

# Production tier thresholds
TIER_CONFIG = {
    "tier_1": {
        "min_reputation": 0,
        "min_stake": "1000",  # In smallest token unit
        "voting_power": 1
    },
    "tier_2": {
        "min_reputation": 500,
        "min_stake": "10000",
        "voting_power": 2
    },
    "tier_3": {
        "min_reputation": 1000,
        "min_stake": "50000",
        "voting_power": 3
    }
}

# Resource scaling settings
SCALING_CONFIG = {
    "max_nodes_per_region": 1000,
    "min_nodes_per_region": 10,
    "scale_up_threshold": 0.8,  # 80% resource utilization
    "scale_down_threshold": 0.3,  # 30% resource utilization
    "scale_up_increment": 2,  # Double nodes
    "scale_down_increment": 0.5,  # Halve nodes
}

# Consensus mechanism weights
CONSENSUS_CONFIG = {
    "pow_weight": 0.2,      # 20% weight for proof of work
    "stake_weight": 0.4,    # 40% weight for proof of stake
    "contribution_weight": 0.4,  # 40% weight for proof of contribution
    
    "pow": {
        "min_difficulty": 1000,
        "target_block_time": 600,  # 10 minutes
        "adjustment_factor": 0.25
    },
    
    "contribution": {
        "compute_weight": 0.4,     # GPU/CPU contribution weight
        "storage_weight": 0.3,     # Storage contribution weight
        "bandwidth_weight": 0.3,   # Bandwidth contribution weight
        "min_compute": 100,        # Minimum GFLOPS
        "min_storage": 100,        # Minimum GB
        "min_bandwidth": 10        # Minimum Mbps
    }
}

# General production settings
PRODUCTION_CONFIG = {
    'max_nodes': 1000,
    'batch_timeout': 30,
    'retry_limit': 3,
    'memory_threshold': 0.9,
    'backup_interval': 3600,
    'metrics_retention_days': 30
}

def validate_production_config() -> bool:
    """Validate all production configuration settings"""
    try:
        for key, value in NODE_REQUIREMENTS.items():
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"Invalid {key}: {value}")
        
        for tier, config in TIER_CONFIG.items():
            if config["min_reputation"] < 0:
                raise ValueError(f"Invalid reputation for {tier}")
            if int(config["min_stake"]) <= 0:
                raise ValueError(f"Invalid stake for {tier}")
        
        # Validate PRODUCTION_CONFIG
        if not 0 < PRODUCTION_CONFIG['max_nodes'] <= 10000:
            raise ValueError("max_nodes must be between 1 and 10000")
        if not 0 < PRODUCTION_CONFIG['batch_timeout'] <= 300:
            raise ValueError("batch_timeout must be between 1 and 300 seconds")
        if not 0 < PRODUCTION_CONFIG['retry_limit'] <= 10:
            raise ValueError("retry_limit must be between 1 and 10")
        if not 0 < PRODUCTION_CONFIG['memory_threshold'] <= 1:
            raise ValueError("memory_threshold must be between 0 and 1")
        if not 0 < PRODUCTION_CONFIG['backup_interval'] <= 86400:
            raise ValueError("backup_interval must be between 1 and 86400 seconds")
        if not 0 < PRODUCTION_CONFIG['metrics_retention_days'] <= 365:
            raise ValueError("metrics_retention_days must be between 1 and 365")
                
        return True
    except Exception as e:
        raise RuntimeError(f"Production config validation failed: {e}")
