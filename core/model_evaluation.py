import torch
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import time
import psutil
import logging
from tqdm import tqdm
import signal
import json
from pathlib import Path
import asyncio
from contextlib import contextmanager
from .interactive_utils import InteractiveSession, InteractionLevel, InteractiveConfig

@dataclass
class EvaluationMetrics:
    accuracy: float
    loss: float
    f1_score: float
    latency: float
    memory_usage: float

@dataclass
class EvaluationConfig:
    """Configuration for interactive evaluation"""
    timeout: int = 30  # Seconds
    max_retries: int = 3
    progress_file: str = "eval_progress.json"
    batch_timeout: int = 300  # Max seconds per batch
    min_memory_threshold: float = 0.1  # GB
    max_memory_threshold: float = 0.9  # GB
    interactive_level: InteractionLevel = InteractionLevel.NORMAL

class ModelEvaluator:
    def __init__(self, model: torch.nn.Module, device: str = 'cuda', 
                 config: Optional[EvaluationConfig] = None):
        self.model = model
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss()
        self.logger = logging.getLogger(__name__)
        self._interrupt_requested = False
        self.config = config or EvaluationConfig()
        self._progress_path = Path("./progress").joinpath(self.config.progress_file)
        signal.signal(signal.SIGINT, self._handle_interrupt)

    def _handle_interrupt(self, signum, frame):
        self._interrupt_requested = True
        self.logger.info("Interrupt requested, completing current batch...")

    @contextmanager
    def evaluation_session(self):
        """Context manager for evaluation sessions"""
        self._interrupt_requested = False
        self.model.eval()
        try:
            yield
        finally:
            torch.cuda.empty_cache()

    def _compute_accuracy(self, data_loader) -> float:
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total

    def _compute_loss(self, data_loader) -> float:
        total_loss = 0
        num_batches = 0
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                num_batches += 1
        return total_loss / num_batches

    def _compute_f1_score(self, data_loader) -> float:
        # ... rest of implementation
        pass

    def _measure_inference_latency(self, data_loader) -> float:
        start_time = time.time()
        with torch.no_grad():
            for inputs, _ in data_loader:
                inputs = inputs.to(self.device)
                _ = self.model(inputs)
        end_time = time.time()
        return (end_time - start_time) * 1000  # Convert to ms

    def _measure_memory_usage(self) -> float:
        return psutil.Process().memory_info().rss / 1024 / 1024  # Convert to MB

    def evaluate(self, data_loader) -> EvaluationMetrics:
        self.model.eval()
        metrics = {
            'accuracy': self._compute_accuracy(data_loader),
            'loss': self._compute_loss(data_loader),
            'f1_score': self._compute_f1_score(data_loader),
            'latency': self._measure_inference_latency(data_loader),
            'memory_usage': self._measure_memory_usage()
        }
        return EvaluationMetrics(**metrics)

    async def _save_progress(self, batch_metrics: List[Dict]):
        """Save evaluation progress for recovery"""
        try:
            self._progress_path.parent.mkdir(exist_ok=True)
            temp_path = self._progress_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump({
                    'timestamp': time.time(),
                    'batch_metrics': batch_metrics,
                    'device': self.device,
                    'total_batches': self._total_batches
                }, f)
            temp_path.replace(self._progress_path)
        except Exception as e:
            self.logger.error(f"Failed to save progress: {str(e)}")

    async def _load_progress(self) -> List[Dict]:
        """Load saved evaluation progress"""
        try:
            if self._progress_path.exists():
                with open(self._progress_path) as f:
                    data = json.load(f)
                    if time.time() - data['timestamp'] < 3600:  # 1 hour expiry
                        return data['batch_metrics']
        except Exception as e:
            self.logger.error(f"Failed to load progress: {str(e)}")
        return []

    def _cleanup(self):
        """Clean up resources and temporary files"""
        try:
            if self._progress_path.exists():
                self._progress_path.unlink()
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        except Exception as e:
            self.logger.error(f"Cleanup failed: {str(e)}")

    async def interactive_evaluate(self, data_loader, show_progress: bool = True) -> Optional[EvaluationMetrics]:
        """Interactive model evaluation with progress tracking and error handling"""
        session = None
        try:
            session = InteractiveSession(
                self.config.interactive_level,
                InteractiveConfig(
                    timeout=self.config.batch_timeout,
                    persistent_state=True,
                    safe_mode=True
                )
            )
            
            async with session:
                self._total_batches = len(data_loader)
                saved_progress = await session._load_progress()
                
                batch_metrics = []
                if saved_progress:
                    batch_metrics = saved_progress.get('metrics', [])
                    logger.info(f"Restored progress: {len(batch_metrics)} batches")

                try:
                    with self.evaluation_session():
                        print("\nStarting Interactive Model Evaluation")
                        print("=" * 50)
                        print(f"Device: {self.device}")
                        print(f"Total batches: {self._total_batches}")
                        
                        if show_progress:
                            pbar = tqdm(total=self._total_batches, desc="Evaluating")

                        # Track batch-level metrics with persistence
                        batch_metrics: List[Dict[str, float]] = saved_progress or []

                        for i, (inputs, labels) in enumerate(data_loader):
                            if i < len(batch_metrics):  # Resume from saved progress
                                if show_progress:
                                    pbar.update(1)
                                continue

                            if self._interrupt_requested:
                                await self._save_progress(batch_metrics)
                                print("\nEvaluation interrupted, progress saved")
                                break

                            try:
                                # Check memory before batch
                                memory_usage = self._measure_memory_usage()
                                if memory_usage > self.config.max_memory_threshold * 1024:  # Convert to MB
                                    raise RuntimeError("Memory usage exceeds threshold")

                                async with session.timeout(self.config.batch_timeout):
                                    batch_result = await self._process_batch(inputs, labels)
                                    batch_metrics.append(batch_result)

                                    if show_progress:
                                        pbar.update(1)
                                        pbar.set_postfix({
                                            'loss': f"{batch_result['loss']:.4f}",
                                            'acc': f"{batch_result['accuracy']:.4f}"
                                        })

                                    # Periodic progress save
                                    if i % 10 == 0:  # Save every 10 batches
                                        await self._save_progress(batch_metrics)

                            except asyncio.TimeoutError:
                                self.logger.error(f"Batch {i} timed out")
                                continue
                            except RuntimeError as e:
                                if "out of memory" in str(e):
                                    self.logger.error("GPU OOM, attempting recovery...")
                                    if hasattr(torch.cuda, 'empty_cache'):
                                        torch.cuda.empty_cache()
                                raise

                        if show_progress:
                            pbar.close()

                        if not self._interrupt_requested and batch_metrics:
                            return await self._finalize_evaluation(batch_metrics, data_loader)

                except Exception as e:
                    self.logger.error(f"Evaluation failed: {str(e)}")
                    if batch_metrics:
                        await self._save_progress(batch_metrics)
                        self.logger.info(f"Progress saved at batch {len(batch_metrics)}")
                    raise
                finally:
                    self._cleanup()

        except Exception as e:
            logger.error(f"Interactive evaluation failed: {e}")
            raise
        finally:
            if session:
                await session.__aexit__(None, None, None)

    async def _process_batch(self, inputs, labels) -> Dict[str, float]:
        """Process a single evaluation batch"""
        batch_start = time.time()
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            
        return {
            'loss': loss.item(),
            'accuracy': (predicted == labels).sum().item() / labels.size(0),
            'latency': (time.time() - batch_start) * 1000,
            'memory': self._measure_memory_usage()
        }

    async def _finalize_evaluation(self, batch_metrics: List[Dict], data_loader) -> EvaluationMetrics:
        """Compute final evaluation metrics"""
        metrics = {
            'accuracy': sum(b['accuracy'] for b in batch_metrics) / len(batch_metrics),
            'loss': sum(b['loss'] for b in batch_metrics) / len(batch_metrics),
            'f1_score': self._compute_f1_score(data_loader),
            'latency': sum(b['latency'] for b in batch_metrics) / len(batch_metrics),
            'memory_usage': max(b['memory'] for b in batch_metrics)
        }

        print("\nEvaluation Summary")
        print("=" * 50)
        for metric, value in metrics.items():
            print(f"{metric.replace('_', ' ').title()}: {value:.4f}")

        return EvaluationMetrics(**metrics)
