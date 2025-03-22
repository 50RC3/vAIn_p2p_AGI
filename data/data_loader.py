import torch
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, List, Optional, Dict, Any
import os
import logging
import signal
from tqdm import tqdm
import psutil
from pathlib import Path
import json
import time

logger = logging.getLogger(__name__)

class InteractionTimeout(Exception):
    """Raised when an interactive operation times out"""
    pass

class vAInDataset(Dataset):
    def __init__(self, data_path: str, transform=None, verify: bool = True):
        self.data_path = Path(data_path)
        self.transform = transform
        self.verify = verify
        self.data: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.stats: Dict = {}
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {data_path}")
            
        self.interactive_load()
        
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.data[idx]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def _load_data(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Load data with progress tracking and validation"""
        try:
            logger.info(f"Loading data from {self.data_path}")
            
            # Monitor memory usage
            initial_mem = psutil.Process().memory_info().rss / 1024 / 1024
            
            data = []
            total_files = len(list(self.data_path.glob('*.pt')))
            
            with tqdm(total=total_files, desc="Loading data") as pbar:
                for file in self.data_path.glob('*.pt'):
                    try:
                        batch = torch.load(file)
                        if self.verify:
                            self._verify_batch(batch)
                        data.append(batch)
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"Error loading {file}: {str(e)}")
                        if not self.interactive_handle_error(file, e):
                            continue
                            
            final_mem = psutil.Process().memory_info().rss / 1024 / 1024
            self.stats = {
                'total_samples': len(data),
                'memory_usage_mb': final_mem - initial_mem,
                'corrupt_files': [],
                'skipped_files': []
            }
            
            logger.info(f"Loaded {len(data)} samples, using {self.stats['memory_usage_mb']:.1f}MB memory")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise

    def _get_input_with_timeout(self, prompt: str, timeout: int = 30) -> str:
        """Get user input with timeout"""
        def timeout_handler(signum, frame):
            raise InteractionTimeout("Input timeout exceeded")

        try:
            # Set timeout handler
            original_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)

            try:
                response = input(prompt).strip()
                return response
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, original_handler)
        except InteractionTimeout:
            logger.warning(f"Input timeout after {timeout}s")
            raise

    def interactive_load(self) -> None:
        """Interactive data loading with configuration"""
        start_time = time.time()
        try:
            logger.info("Starting interactive data loading session")
            print("\nInteractive Data Loading")
            print("=" * 50)
            
            try:
                verify_input = self._get_input_with_timeout(
                    "Verify data integrity? (y/N): ",
                    timeout=30
                )
                self.verify = verify_input.lower() == 'y'
            except InteractionTimeout:
                logger.info("Using default verification setting: False")
                self.verify = False
            
            if self.verify:
                print("\nVerification will check tensor shapes and values")
                logger.info("Data verification enabled")

            # Monitor memory
            initial_mem = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Load data with progress tracking
            self.data = self._load_data()
            
            final_mem = psutil.Process().memory_info().rss / 1024 / 1024
            mem_used = final_mem - initial_mem
            
            # Update stats
            self.stats.update({
                'load_time_seconds': time.time() - start_time,
                'memory_usage_mb': mem_used,
                'verification_enabled': self.verify
            })
            
            # Save stats
            stats_file = self.data_path / 'dataset_stats.json'
            with open(stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
            
            logger.info(
                f"Data loading completed in {self.stats['load_time_seconds']:.1f}s, "
                f"using {mem_used:.1f}MB memory"
            )
                
        except KeyboardInterrupt:
            logger.warning("Data loading cancelled by user")
            self._cleanup()
            raise
        except Exception as e:
            logger.error(f"Interactive loading failed: {str(e)}")
            self._cleanup()
            raise

    def interactive_handle_error(self, file: Path, error: Exception) -> bool:
        """Interactively handle data loading errors with timeout"""
        logger.error(f"Error loading {file.name}: {str(error)}")
        print(f"\nError loading {file.name}: {str(error)}")
        
        try:
            action = self._get_input_with_timeout(
                "(S)kip, (R)etry, or (A)bort? ",
                timeout=30
            ).lower()
        except InteractionTimeout:
            logger.info("Action timeout - defaulting to Skip")
            action = 's'
            
        if action == 'r':
            logger.info(f"Retrying load of {file.name}")
            try:
                batch = torch.load(file)
                if self.verify and not self._verify_batch(batch):
                    self.stats['corrupt_files'].append(str(file))
                    return False
                return True
            except Exception as e:
                logger.error(f"Retry failed for {file.name}: {str(e)}")
                self.stats['corrupt_files'].append(str(file))
                return False
        elif action == 'a':
            logger.info("User requested abort")
            raise error
        else:  # Skip
            logger.info(f"Skipping file {file.name}")
            self.stats['skipped_files'].append(str(file))
            return False

    def _cleanup(self) -> None:
        """Clean up resources and memory"""
        logger.info("Cleaning up resources")
        if hasattr(torch, 'cuda'):
            torch.cuda.empty_cache()
        
        # Clear data if load was interrupted
        self.data = []
        
        # Force garbage collection
        import gc
        gc.collect()

    def _verify_batch(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> bool:
        """Verify data batch integrity"""
        try:
            x, y = batch
            if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
                raise ValueError("Batch must contain PyTorch tensors")
            if x.shape[0] != y.shape[0]:
                raise ValueError(f"Batch size mismatch: {x.shape[0]} vs {y.shape[0]}")
            if torch.isnan(x).any() or torch.isnan(y).any():
                raise ValueError("Batch contains NaN values")
            return True
        except Exception as e:
            logger.error(f"Batch verification failed: {str(e)}")
            return False
