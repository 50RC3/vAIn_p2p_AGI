"""Predictive models for node interactions and stability."""
from typing import List, Dict, Any, Optional

try:
    from .node_attention import (
        NodeAttentionLayer,
        RLAgent,
        BayesianOptimizer,
        PredictiveModel
    )

    __all__: List[str] = [
        'NodeAttentionLayer',
        'RLAgent',
        'BayesianOptimizer', 
        'PredictiveModel'
    ]

except ImportError as e:
    import logging
    logging.warning(f"Could not import predictive models: {e}")
    # Define stub classes to prevent undefined references
    class NodeAttentionLayerStub: pass
    class RLAgentStub: pass 
    class BayesianOptimizerStub: pass
    class PredictiveModelStub: pass
    
    # Use the stub classes instead
    NodeAttentionLayer = NodeAttentionLayerStub
    RLAgent = RLAgentStub
    BayesianOptimizer = BayesianOptimizerStub
    PredictiveModel = PredictiveModelStub
    __all__ = []
