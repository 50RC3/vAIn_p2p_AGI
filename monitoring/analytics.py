import pandas as pd
from datetime import datetime
from typing import Dict, List

class NetworkAnalytics:
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.metrics_history = []
        
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
