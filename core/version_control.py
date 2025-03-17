import hashlib
from datetime import datetime
from typing import Dict, Optional

class ModelVersionControl:
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.version_history = {}
        
    def save_version(self, model_state: Dict, metadata: Dict) -> str:
        version_hash = self._compute_state_hash(model_state)
        self.version_history[version_hash] = {
            'timestamp': datetime.now(),
            'metadata': metadata,
            'parent_hash': metadata.get('parent_hash')
        }
        return version_hash
        
    def _compute_state_hash(self, state: Dict) -> str:
        state_bytes = str(sorted(state.items())).encode()
        return hashlib.sha256(state_bytes).hexdigest()
