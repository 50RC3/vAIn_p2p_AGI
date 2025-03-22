import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class VersionInfo:
    hash: str
    timestamp: datetime
    metadata: Dict
    parent_hash: Optional[str]

class ModelVersionControl:
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.version_history = {}
        self._load_history()
        
    def save_version(self, model_state: Dict, metadata: Dict) -> str:
        """Save a new model version with validation"""
        try:
            self._validate_state(model_state)
            self._validate_metadata(metadata)
            
            version_hash = self._compute_state_hash(model_state)
            timestamp = datetime.now()
            
            version_info = VersionInfo(
                hash=version_hash,
                timestamp=timestamp,
                metadata=metadata,
                parent_hash=metadata.get('parent_hash')
            )
            
            self.version_history[version_hash] = version_info
            self._save_version_file(version_hash, model_state, version_info)
            
            logger.info(f"Saved version {version_hash[:8]} at {timestamp}")
            return version_hash
            
        except Exception as e:
            logger.error(f"Failed to save version: {str(e)}")
            raise

    def interactive_save(self) -> None:
        """Interactive version saving with enhanced validation and UX"""
        MAX_RETRIES = 3
        INPUT_TIMEOUT = 300  # 5 minutes timeout
        
        def get_input_with_timeout(prompt: str, timeout: int = INPUT_TIMEOUT) -> str:
            """Get input with timeout"""
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Input timeout exceeded")
            
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout)
                value = input(prompt).strip()
                signal.alarm(0)
                return value
            except TimeoutError:
                print("\nInput timeout exceeded")
                raise
                
        try:
            print("\nSave New Model Version")
            print("=" * 50)
            print("Note: Press Ctrl+C at any time to cancel\n")
            
            # Get model state file with retries
            for attempt in range(MAX_RETRIES):
                try:
                    state_path = get_input_with_timeout("Enter path to model state file: ")
                    if not state_path:
                        print("Path cannot be empty")
                        continue
                        
                    path = Path(state_path)
                    if not path.exists():
                        print(f"File not found: {state_path}")
                        if attempt < MAX_RETRIES - 1:
                            print("Please try again")
                            continue
                        raise FileNotFoundError("Maximum retry attempts exceeded")
                        
                    print("\nLoading and validating model state...")
                    try:
                        with open(path) as f:
                            model_state = json.load(f)
                        self._validate_state(model_state)
                        print("Model state validated successfully")
                        break
                    except (json.JSONDecodeError, ValueError) as e:
                        print(f"Invalid model state file: {e}")
                        if attempt < MAX_RETRIES - 1:
                            print("Please try again")
                            continue
                        raise
                        
                except TimeoutError:
                    if attempt < MAX_RETRIES - 1:
                        print("Please try again")
                        continue
                    raise
                    
            # Get metadata with validation
            print("\nEnter Version Metadata")
            print("-" * 30)
            
            metadata = {}
            
            description = get_input_with_timeout("Description (required): ")
            if not description:
                raise ValueError("Description is required")
            metadata['description'] = description
            
            author = get_input_with_timeout("Author (required): ")
            if not author:
                raise ValueError("Author is required")
            metadata['author'] = author
            
            metadata['parent_hash'] = get_input_with_timeout("Parent version hash (optional): ")
            
            # Optional additional metadata
            while True:
                extra_key = get_input_with_timeout("\nAdd additional metadata? (key or enter to skip): ")
                if not extra_key:
                    break
                metadata[extra_key] = get_input_with_timeout(f"Value for {extra_key}: ")
            
            # Show summary and confirm
            print("\nVersion Summary")
            print("=" * 50)
            print(f"Model state file: {state_path}")
            print("\nMetadata:")
            for k, v in metadata.items():
                print(f"  {k}: {v}")
                
            if get_input_with_timeout("\nSave this version? (y/N): ").lower() != 'y':
                print("Version save cancelled")
                return
                
            print("\nSaving version...")
            version_hash = self.save_version(model_state, metadata)
            print(f"\nSuccessfully saved version: {version_hash}")
            print(f"Timestamp: {self.version_history[version_hash].timestamp}")
            
        except KeyboardInterrupt:
            print("\n\nVersion save cancelled by user")
        except TimeoutError:
            print("\n\nVersion save cancelled due to timeout")
        except Exception as e:
            logger.error(f"Interactive save failed: {str(e)}")
            print(f"\nError saving version: {str(e)}")
            raise

    def get_version_info(self, version_hash: str) -> VersionInfo:
        """Get version info with validation"""
        if version_hash not in self.version_history:
            raise ValueError(f"Version {version_hash} not found")
        return self.version_history[version_hash]

    def compare_versions(self, hash1: str, hash2: str) -> Dict:
        """Compare two versions"""
        v1 = self.get_version_info(hash1)
        v2 = self.get_version_info(hash2)
        return {
            'time_diff': v2.timestamp - v1.timestamp,
            'metadata_diff': self._dict_diff(v1.metadata, v2.metadata)
        }

    def rollback(self, version_hash: str) -> None:
        """Rollback to a previous version"""
        if version_hash not in self.version_history:
            raise ValueError(f"Version {version_hash} not found")
        
        # Load version state
        version_path = self.storage_path / f"{version_hash}.json"
        if not version_path.exists():
            raise FileNotFoundError(f"Version file missing: {version_hash}")
            
        logger.info(f"Rolling back to version {version_hash[:8]}")
        # Implementation of actual rollback logic here

    def _validate_state(self, state: Dict) -> None:
        """Validate model state"""
        if not isinstance(state, dict):
            raise ValueError("Model state must be a dictionary")
        if not state:
            raise ValueError("Model state cannot be empty")

    def _validate_metadata(self, metadata: Dict) -> None:
        """Validate version metadata"""
        required = ['description', 'author']
        missing = [f for f in required if f not in metadata]
        if missing:
            raise ValueError(f"Missing required metadata fields: {missing}")

    def _save_version_file(self, version_hash: str, state: Dict, info: VersionInfo) -> None:
        """Save version data to file"""
        version_path = self.storage_path / f"{version_hash}.json"
        version_data = {
            'state': state,
            'info': {
                'timestamp': info.timestamp.isoformat(),
                'metadata': info.metadata,
                'parent_hash': info.parent_hash
            }
        }
        with open(version_path, 'w') as f:
            json.dump(version_data, f, indent=2)

    def _load_history(self) -> None:
        """Load version history from files"""
        try:
            for path in self.storage_path.glob('*.json'):
                with open(path) as f:
                    data = json.load(f)
                    version_hash = path.stem
                    self.version_history[version_hash] = VersionInfo(
                        hash=version_hash,
                        timestamp=datetime.fromisoformat(data['info']['timestamp']),
                        metadata=data['info']['metadata'],
                        parent_hash=data['info']['parent_hash']
                    )
        except Exception as e:
            logger.error(f"Failed to load version history: {str(e)}")
            raise

    def _compute_state_hash(self, state: Dict) -> str:
        state_bytes = str(sorted(state.items())).encode()
        return hashlib.sha256(state_bytes).hexdigest()

    @staticmethod
    def _dict_diff(d1: Dict, d2: Dict) -> Dict:
        """Compare two dictionaries"""
        diff = {}
        all_keys = set(d1.keys()) | set(d2.keys())
        for k in all_keys:
            if k not in d1:
                diff[k] = ('added', d2[k])
            elif k not in d2:
                diff[k] = ('removed', d1[k]) 
            elif d1[k] != d2[k]:
                diff[k] = ('changed', d1[k], d2[k])
        return diff
