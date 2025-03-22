import torch
import ipfshttpclient
from typing import Dict, Optional, Tuple, Any, Callable
from pathlib import Path
import tempfile
import uuid
import logging
from functools import lru_cache
import time
from retry import retry
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ModelStorage:
    def __init__(self, ipfs_host: str = "/ip4/127.0.0.1/tcp/5001", 
                 chunk_size: int = 64 * 1024 * 1024,  # 64MB chunks
                 progress_callback: Optional[Callable[[str, float], None]] = None):
        self.ipfs = ipfshttpclient.connect(ipfs_host)
        self.chunk_size = chunk_size
        self.progress_callback = progress_callback
        self._model_cache = {}
        self._verify_connection()
        
    def _verify_connection(self):
        """Verify IPFS connection is working"""
        try:
            self.ipfs.id()
        except Exception as e:
            logger.error("Failed to connect to IPFS node: %s", e)
            raise ConnectionError("IPFS node not available") from e

    def store_model(self, model: torch.nn.Module, metadata: Dict[str, Any], 
                   interactive: bool = True) -> Tuple[str, str]:
        """Store model weights and metadata on IPFS with progress tracking."""
        if not isinstance(model, torch.nn.Module):
            raise ValueError("Model must be a torch.nn.Module instance")
        if not isinstance(metadata, dict):
            raise ValueError("Metadata must be a dictionary")
            
        temp_dir = Path(tempfile.gettempdir())
        temp_path = temp_dir / f"model_{uuid.uuid4()}.pt"
            
        try:
            # Add version tracking and validation info
            metadata['version'] = metadata.get('version', 0) + 1
            metadata['timestamp'] = time.time()
            metadata['model_size'] = Path(temp_path).stat().st_size
            metadata['checksum'] = self._calculate_checksum(temp_path)
            
            if interactive:
                logger.info(f"Saving model to temporary file: {temp_path}")
            torch.save(model.state_dict(), temp_path)
            
            # Handle large files in chunks with progress tracking
            model_hash = self._store_chunked_file(temp_path, interactive)
            metadata["model_hash"] = model_hash
            metadata_hash = self.ipfs.add_json(metadata)
            
            if interactive:
                logger.info(f"Successfully stored model {model_hash} with metadata {metadata_hash}")
            return model_hash, metadata_hash
        except Exception as e:
            logger.error(f"Error storing model: {e}", exc_info=True)
            raise
        finally:
            temp_path.unlink(missing_ok=True)

    @retry(tries=3, delay=1, backoff=2, logger=logger)
    def load_model(self, ipfs_hash: str, interactive: bool = True) -> Optional[Dict]:
        """Load model weights and metadata from IPFS with progress tracking."""
        if not ipfs_hash:
            raise ValueError("IPFS hash cannot be empty")
            
        # Check cache first
        if ipfs_hash in self._model_cache:
            logger.info(f"Cache hit for model {ipfs_hash}")
            return self._model_cache[ipfs_hash]
            
        try:
            if interactive:
                logger.info(f"Loading model metadata from {ipfs_hash}")
            metadata = self.ipfs.get_json(ipfs_hash)
            model_hash = metadata.get("model_hash")
            if not model_hash:
                raise ValueError("Invalid metadata: missing model_hash")
                
            temp_dir = Path(tempfile.gettempdir())
            local_path = temp_dir / f"model_{uuid.uuid4()}.pt"
            
            try:
                self._load_chunked_file(model_hash, local_path, interactive)
                
                # Verify checksum
                if metadata.get('checksum'):
                    current_checksum = self._calculate_checksum(local_path)
                    if current_checksum != metadata['checksum']:
                        raise ValueError("Model file checksum verification failed")
                
                state_dict = torch.load(local_path)
                metadata["state_dict"] = state_dict
                
                # Cache with TTL
                self._cache_model(ipfs_hash, metadata)
                return metadata
            finally:
                local_path.unlink(missing_ok=True)
        except (ipfshttpclient.exceptions.Error, IOError, RuntimeError) as e:
            logger.error(f"Error loading model {ipfs_hash}: {e}", exc_info=True)
            return None

    @lru_cache(maxsize=100)
    def get_metadata(self, metadata_hash: str) -> Optional[Dict]:
        """Retrieve and cache metadata for a model."""
        if not metadata_hash:
            raise ValueError("Metadata hash cannot be empty")
            
        try:
            return self.ipfs.get_json(metadata_hash)
        except Exception as e:
            logger.error("Error loading metadata %s: %s", metadata_hash, e)
            return None
            
    def _store_chunked_file(self, file_path: Path, interactive: bool = True) -> str:
        """Store large files in chunks with progress tracking."""
        file_size = file_path.stat().st_size
        if file_size <= self.chunk_size:
            if interactive:
                logger.info("Uploading model as single file")
            return self.ipfs.add(str(file_path))["Hash"]
            
        # Split into chunks with progress bar
        chunks = []
        total_chunks = (file_size + self.chunk_size - 1) // self.chunk_size
        
        with open(file_path, 'rb') as f:
            with tqdm(total=total_chunks, disable=not interactive,
                     desc="Uploading model chunks") as pbar:
                while chunk := f.read(self.chunk_size):
                    chunk_hash = self.ipfs.add_bytes(chunk)["Hash"]
                    chunks.append(chunk_hash)
                    pbar.update(1)
                    if self.progress_callback:
                        progress = (len(chunks) / total_chunks) * 100
                        self.progress_callback("upload", progress)
                
        return self.ipfs.add_json({"chunks": chunks})["Hash"]
        
    def _load_chunked_file(self, file_hash: str, target_path: Path, 
                          interactive: bool = True):
        """Load a potentially chunked file with progress tracking."""
        try:
            chunks_meta = self.ipfs.get_json(file_hash)
            if isinstance(chunks_meta, dict) and "chunks" in chunks_meta:
                chunks = chunks_meta["chunks"]
                
                with open(target_path, 'wb') as f:
                    with tqdm(total=len(chunks), disable=not interactive,
                             desc="Downloading model chunks") as pbar:
                        for chunk_hash in chunks:
                            chunk_data = self.ipfs.cat(chunk_hash)
                            f.write(chunk_data)
                            pbar.update(1)
                            if self.progress_callback:
                                progress = (pbar.n / pbar.total) * 100
                                self.progress_callback("download", progress)
            else:
                if interactive:
                    logger.info("Downloading model as single file")
                self.ipfs.get(file_hash, target=str(target_path))
        except Exception as e:
            logger.error("Error loading chunked file: %s", e)
            raise

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of file"""
        import hashlib
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _cache_model(self, ipfs_hash: str, metadata: Dict, ttl: int = 3600):
        """Cache model with TTL"""
        self._model_cache[ipfs_hash] = {
            'data': metadata,
            'expires': time.time() + ttl
        }

    def clear_expired_cache(self):
        """Clear expired items from cache"""
        now = time.time()
        expired = [k for k, v in self._model_cache.items() 
                  if v['expires'] < now]
        for k in expired:
            del self._model_cache[k]