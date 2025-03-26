import numpy as np
import heapq
import random
import logging
import asyncio
import psutil
from typing import Dict, List, Tuple, Optional
from bayes_opt import BayesianOptimization
from core.constants import InteractionLevel, INTERACTION_TIMEOUTS
from core.interactive_utils import InteractiveSession, InteractiveConfig

logger = logging.getLogger(__name__)

class NodeAttentionError(Exception):
    """Custom exception for node attention errors."""
    pass

class NodeAttentionLayer:
    """Implements an attention-based mechanism for selecting relevant nodes."""
    def __init__(self, node_id: str, attention_capacity: int = 10, weights: Optional[Dict[str, float]] = None):
        self.node_id = node_id
        self.attention_capacity = max(1, min(attention_capacity, 100))  # Bound capacity
        self.weights = weights if weights else {
            "reputation": 5.0,
            "uptime": 0.1,
            "latency": 1.0,
            "availability": 2.0
        }
        self.session: Optional[InteractiveSession] = None
        self._interrupt_requested = False
        self.metrics: Dict[str, float] = {"total_calls": 0, "success_rate": 0.0}
        logger.info(f"Initialized NodeAttentionLayer for {node_id}")

    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """Update attention weights with validation."""
        if not all(w > 0 for w in new_weights.values()):
            raise NodeAttentionError("All weights must be positive")
        self.weights.update(new_weights)
        logger.info(f"Updated weights for {self.node_id}")

    async def compute_attention(self, neighbor_states: Dict[str, Dict[str, float]]) -> List[Tuple[str, float]]:
        """Computes attention scores with resource monitoring and async support."""
        try:
            self.session = InteractiveSession(
                level=InteractionLevel.NORMAL,
                config=InteractiveConfig(
                    timeout=INTERACTION_TIMEOUTS["batch"],
                    memory_threshold=0.9,
                    persistent_state=True
                )
            )

            async with self.session:
                # Check system resources
                if not await self._check_resources():
                    raise NodeAttentionError("Insufficient resources")

                if not neighbor_states:
                    raise NodeAttentionError("Empty neighbor states")

                attention_scores = {}
                for neighbor, state in neighbor_states.items():
                    try:
                        score = await self._calculate_attention_score(state)
                        attention_scores[neighbor] = score
                    except Exception as e:
                        logger.warning(f"Error calculating score for {neighbor}: {e}")
                        continue

                self.metrics["total_calls"] += 1
                self.metrics["success_rate"] = (
                    len(attention_scores) / len(neighbor_states)
                ) * 100

                return heapq.nlargest(
                    self.attention_capacity,
                    attention_scores.items(),
                    key=lambda x: x[1]
                )

        except Exception as e:
            logger.error(f"Error computing attention: {e}")
            raise NodeAttentionError(f"Attention computation failed: {e}")
        finally:
            if self.session:
                await self.session.__aexit__(None, None, None)

    async def _check_resources(self) -> bool:
        """Monitor system resources."""
        try:
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                logger.warning("High memory usage detected")
                if self.session:
                    return await self.session.get_confirmation(
                        "Continue despite high memory usage?",
                        timeout=INTERACTION_TIMEOUTS["emergency"]
                    )
                return False
            return True
        except Exception as e:
            logger.error(f"Resource check failed: {e}")
            return False

    async def _calculate_attention_score(self, state: Dict[str, float]) -> float:
        """Calculate attention score with async support."""
        try:
            # Normalize values
            normalized_state = {
                k: v / max(self.weights.values())
                for k, v in state.items()
                if k in self.weights
            }
            return sum(
                normalized_state.get(attr, 0) * weight 
                for attr, weight in self.weights.items()
            )
        except Exception as e:
            logger.error(f"Score calculation failed: {e}")
            return 0.0

    def get_metrics(self) -> Dict[str, float]:
        """Return attention layer metrics."""
        return self.metrics.copy()

class RLAgent:
    """Reinforcement Learning Agent for node selection."""
    def __init__(self, exploration_rate: float = 0.1):
        self.exploration_rate = exploration_rate
        self.q_table: Dict[str, float] = {}

    async def select_node(self, attention_scores: List[Tuple[str, float]]) -> str:
        """Async node selection with validation."""
        if not attention_scores:
            raise NodeAttentionError("Empty attention scores")
        try:
            if random.uniform(0, 1) < self.exploration_rate:
                return random.choice([score[0] for score in attention_scores])
            return max(attention_scores, key=lambda x: self.q_table.get(x[0], 0))[0]
        except Exception as e:
            logger.error(f"Node selection failed: {e}")
            raise NodeAttentionError(f"Selection failed: {e}")

class BayesianOptimizer:
    """Bayesian Optimizer for tuning attention weights."""
    def __init__(self, node_attention_layer: NodeAttentionLayer):
        self.node_attention_layer = node_attention_layer
        self.optimizer = BayesianOptimization(
            f=self._objective_function,
            pbounds={k: (0.1, 10.0) for k in node_attention_layer.weights.keys()},
            random_state=42
        )

    async def _objective_function(self, **params) -> float:
        """Async objective function with error handling."""
        try:
            weights = {k: float(v) for k, v in params.items()}
            self.node_attention_layer.update_weights(weights)
            scores = []
            for _ in range(5):
                attention_result = await self.node_attention_layer.compute_attention({
                    "test-node": weights
                })
                scores.append(attention_result[0][1] if attention_result else 0)
            return np.mean(scores)
        except Exception as e:
            logger.error(f"Optimization objective failed: {e}")
            return 0.0

    async def optimize_weights(self) -> None:
        """Async optimization with proper cleanup."""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.optimizer.maximize(init_points=5, n_iter=10)
            )
            self.node_attention_layer.update_weights(self.optimizer.max["params"])
        except Exception as e:
            logger.error(f"Weight optimization failed: {e}")
            raise NodeAttentionError(f"Optimization failed: {e}")

class PredictiveModel:
    """Predictive model for node interactions."""
    def __init__(self, node_id: str):
        self.attention_layer = NodeAttentionLayer(node_id=node_id)
        self.rl_agent = RLAgent()
        self.optimizer = BayesianOptimizer(self.attention_layer)

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        try:
            if hasattr(self.attention_layer, 'session') and self.attention_layer.session:
                await self.attention_layer.session.__aexit__(exc_type, exc_val, exc_tb)
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    async def predict_interactions(self, neighbor_states: Dict[str, Dict[str, float]]) -> str:
        """Predict interactions and select the best node."""
        attention_scores = await self.attention_layer.compute_attention(neighbor_states)
        return await self.rl_agent.select_node(attention_scores)

    async def optimize_attention_weights(self) -> None:
        """Optimize attention weights using Bayesian optimization."""
        await self.optimizer.optimize_weights()

    async def average_attention_score(self, neighbor_states: Dict[str, Dict[str, float]]) -> float:
        """Calculate the average attention score for given neighbor states."""
        attention_scores = await self.attention_layer.compute_attention(neighbor_states)
        return np.mean([score for _, score in attention_scores])

if __name__ == "__main__":
    async def main():
        async with PredictiveModel(node_id="NODE-123") as model:
            neighbor_states = {
                "NODE-456": {"reputation": 0.8, "uptime": 30, "latency": 10, "availability": 0.95},
                "NODE-789": {"reputation": 0.95, "uptime": 60, "latency": 5, "availability": 0.99},
                "NODE-012": {"reputation": 0.75, "uptime": 40, "latency": 15, "availability": 0.85},
            }
            selected_node = await model.predict_interactions(neighbor_states)
            print(f"Selected Node: {selected_node}")
            
            await model.optimize_attention_weights()
            print(f"Optimized weights: {model.attention_layer.weights}")
            
            score = await model.average_attention_score(neighbor_states) 
            print(f"Average attention score: {score}")

    asyncio.run(main())
