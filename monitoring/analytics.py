import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import logging
from tqdm import tqdm
from core.interactive_utils import InteractiveSession, InteractionLevel, InteractiveConfig
from core.constants import INTERACTION_TIMEOUTS

logger = logging.getLogger(__name__)

class NetworkAnalytics:
    def __init__(self, log_path: str, interactive: bool = True):
        self.log_path = log_path
        self.metrics_history = []
        self.interactive = interactive
        self.session: Optional[InteractiveSession] = None

    def log_metrics(self, metrics: Dict[str, float]):
        metrics['timestamp'] = datetime.now()
        self.metrics_history.append(metrics)
        
    def get_network_health(self) -> Dict[str, float]:
        df = pd.DataFrame(self.metrics_history)
        return {
            'avg_cpu_usage': df['cpu_usage'].mean(),
            'avg_memory_usage': df['memory_usage'].mean(),
            'peak_network_load': df['network_io'].max()
        }

    async def log_metrics_interactive(self, metrics: Dict[str, float]) -> bool:
        """Log metrics with interactive progress and validation"""
        try:
            if self.interactive:
                self.session = InteractiveSession(
                    level=InteractionLevel.NORMAL,
                    config=InteractiveConfig(
                        timeout=INTERACTION_TIMEOUTS["save"],
                        persistent_state=True
                    )
                )

            async with self.session:
                metrics['timestamp'] = datetime.now()
                self.metrics_history.append(metrics)
                
                # Periodic persistence
                if len(self.metrics_history) % 100 == 0:
                    await self._save_progress(metrics)
                    
                return True

        except Exception as e:
            logger.error(f"Failed to log metrics: {str(e)}")
            return False
        finally:
            if self.session:
                await self.session.__aexit__(None, None, None)

    async def get_network_health_interactive(self) -> Optional[Dict[str, float]]:
        """Get network health with progress tracking"""
        try:
            if self.interactive:
                self.session = InteractiveSession(
                    level=InteractionLevel.NORMAL,
                    config=InteractiveConfig(timeout=30)
                )

            async with self.session:
                with tqdm(total=3, desc="Analyzing Network Health") as pbar:
                    df = pd.DataFrame(self.metrics_history)
                    pbar.update(1)

                    health_metrics = {
                        'avg_cpu_usage': df['cpu_usage'].mean(),
                        'avg_memory_usage': df['memory_usage'].mean(),
                        'peak_network_load': df['network_io'].max()
                    }
                    pbar.update(1)

                    # Validate metrics
                    if not all(0 <= v <= 100 for v in [health_metrics['avg_cpu_usage'], 
                                                      health_metrics['avg_memory_usage']]):
                        raise ValueError("Invalid metric values detected")

                    await self._save_progress(health_metrics)
                    pbar.update(1)

                    return health_metrics

        except Exception as e:
            logger.error(f"Failed to get network health: {str(e)}")
            return None
        finally:
            if self.session:
                await self.session.__aexit__(None, None, None)

    async def _save_progress(self, data: Dict) -> None:
        if self.session:
            await self.session.save_progress(data)
