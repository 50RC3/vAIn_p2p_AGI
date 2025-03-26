from typing import List, Dict, Optional, Any
import asyncio
import logging
from datetime import datetime
import json
from pathlib import Path
from tqdm import tqdm
from .interactive_utils import InteractiveSession, InteractionTimeout, InteractionLevel

logger = logging.getLogger(__name__)

class TrainingCoordinator:
    def __init__(self, min_nodes: int = 3,
                 interactive_level: InteractionLevel = InteractionLevel.NORMAL,
                 progress_dir: str = "./progress",
                 validation_timeout: int = 300):  # 5 minute default timeout
        self.min_nodes = min_nodes
        self.active_nodes = {}
        self.training_round = 0
        self.interactive_mode = False
        self.progress_bar = None
        self.interactive_level = interactive_level
        self.progress_dir = Path(progress_dir)
        self.progress_dir.mkdir(exist_ok=True)
        self._progress_file = self.progress_dir / "training_progress.json"
        self._interrupt_requested = False
        self._setup_logging()
        self.validation_timeout = validation_timeout
        self.validation_retries = 3

    def _setup_logging(self) -> None:
        """Configure logging for the coordinator"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    async def coordinate_round(self, nodes: List[str], interactive: bool = False) -> Dict:
        """Coordinate a training round with interactive controls and progress tracking"""
        session = None
        try:
            self._validate_nodes(nodes)
            self.interactive_mode = interactive
            
            if self.interactive_mode:
                session = InteractiveSession(self.interactive_level)
                await session.__aenter__()
                await self._interactive_confirmation()
            
            self.training_round += 1
            logger.info(f"Starting training round {self.training_round}")
            
            # Try to restore progress
            saved_progress = await self._load_progress()
            if saved_progress:
                logger.info("Restored previous progress")
                aggregated = saved_progress
            else:
                selected_nodes = self._select_nodes(nodes)
                logger.info(f"Selected {len(selected_nodes)} nodes for training")
                
                if self.interactive_mode:
                    self.progress_bar = tqdm(total=len(selected_nodes), desc="Training Progress")
                
                results = []
                for node in selected_nodes:
                    if self._interrupt_requested:
                        logger.info("Training interrupted by user")
                        break
                        
                    try:
                        result = await self._train_on_node(node)
                        results.append(result)
                        await self._save_progress(results)
                    except Exception as e:
                        logger.error(f"Node {node} training failed: {str(e)}")
                        if not await self._handle_node_failure(node, session):
                            break
                
                aggregated = self._aggregate_results(results)
                
            if self.progress_bar:
                self.progress_bar.close()
                
            self._log_round_completion(aggregated)
            await self._cleanup_progress()
            return aggregated
            
        except Exception as e:
            logger.error(f"Training round {self.training_round} failed: {str(e)}")
            await self._save_progress(None, error=str(e))
            if self.progress_bar:
                self.progress_bar.close()
            raise
        finally:
            if session:
                await session.__aexit__(None, None, None)

    def _validate_nodes(self, nodes: List[str]) -> None:
        """Validate node list and requirements"""
        if not nodes:
            raise ValueError("Node list cannot be empty")
        if len(nodes) < self.min_nodes:
            raise ValueError(f"Insufficient nodes: {len(nodes)} < {self.min_nodes}")
        if len(set(nodes)) != len(nodes):
            raise ValueError("Duplicate nodes detected")

    async def _interactive_confirmation(self) -> None:
        """Get user confirmation with retry logic and timeout"""
        session = InteractiveSession(self.interactive_level)
        max_retries = 3
        
        async with session:
            for attempt in range(max_retries):
                try:
                    print("\nTraining Round Configuration")
                    print("-" * 30)
                    print(f"Round Number: {self.training_round + 1}")
                    print(f"Minimum Nodes: {self.min_nodes}")
                    print(f"Interactive Level: {self.interactive_level.name}")
                    
                    confirmed = await session.confirm_with_timeout(
                        "\nProceed with training round?",
                        timeout=30,
                        default=False
                    )
                    if not confirmed:
                        raise InterruptedError("Training round cancelled by user")
                    break
                except InteractionTimeout:
                    if attempt == max_retries - 1:
                        raise InterruptedError("Confirmation timeout exceeded maximum retries")
                    print("\nConfirmation timeout, please try again")
                    continue

    async def _train_on_node(self, node: str) -> Dict:
        """Execute training on a node with monitoring"""
        try:
            logger.debug(f"Starting training on node {node}")
            # Simulate training - replace with actual implementation
            await asyncio.sleep(2)
            result = {"node": node, "status": "success", "timestamp": datetime.now().isoformat()}
            
            if self.progress_bar:
                self.progress_bar.update(1)
                
            logger.debug(f"Completed training on node {node}")
            return result
            
        except Exception as e:
            logger.error(f"Training failed on node {node}: {str(e)}")
            if self.progress_bar:
                self.progress_bar.update(1)
            raise

    def _select_nodes(self, nodes: List[str]) -> List[str]:
        """Select nodes based on reputation and stake"""
        try:
            # Add actual node selection logic here
            selected = nodes[:self.min_nodes] 
            logger.info(f"Selected nodes: {', '.join(selected)}")
            return selected
        except Exception as e:
            logger.error(f"Node selection failed: {str(e)}")
            raise

    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """Aggregate training results with validation"""
        try:
            if not results:
                raise ValueError("No valid results to aggregate")
                
            aggregated = {
                "round": self.training_round,
                "timestamp": datetime.now().isoformat(),
                "node_count": len(results),
                "successful_nodes": len([r for r in results if r.get("status") == "success"]),
                "results": results
            }
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Results aggregation failed: {str(e)}")
            raise

    def _log_round_completion(self, results: Dict) -> None:
        """Log training round completion metrics"""
        logger.info(
            f"Training round {self.training_round} completed: "
            f"{results['successful_nodes']}/{results['node_count']} nodes successful"
        )

    async def _save_progress(self, results: Optional[List[Dict]], error: Optional[str] = None) -> None:
        """Save current training progress"""
        try:
            progress = {
                'round': self.training_round,
                'timestamp': datetime.now().isoformat(),
                'results': results,
                'error': error
            }
            async with aiofiles.open(self._progress_file, 'w') as f:
                await f.write(json.dumps(progress))
        except Exception as e:
            logger.error(f"Failed to save progress: {str(e)}")

    async def _load_progress(self) -> Optional[Dict]:
        """Load saved training progress if available"""
        try:
            if self._progress_file.exists():
                async with aiofiles.open(self._progress_file) as f:
                    content = await f.read()
                    return json.loads(content)
        except Exception as e:
            logger.error(f"Failed to load progress: {str(e)}")
        return None

    async def _cleanup_progress(self) -> None:
        """Clean up progress files after successful completion"""
        try:
            if self._progress_file.exists():
                self._progress_file.unlink()
        except Exception as e:
            logger.error(f"Failed to cleanup progress: {str(e)}")

    async def _handle_node_failure(self, node: str, session: Optional[InteractiveSession]) -> bool:
        """Handle node failure with interactive retry option"""
        if not session or self.interactive_level == InteractionLevel.NONE:
            return False
            
        try:
            should_retry = await session.confirm_with_timeout(
                f"\nNode {node} failed. Retry with another node?",
                timeout=30,
                default=True
            )
            return should_retry
        except InteractionTimeout:
            logger.warning("Node failure handling timeout, skipping retry")
            return False

    async def coordinate_validation(self, model: Any, validators: List[str]) -> Dict:
        """Coordinate model validation across multiple validators"""
        try:
            result = await asyncio.wait_for(
                self._validate_with_retry(model, validators),
                timeout=self.validation_timeout
            )
            return result
        except asyncio.TimeoutError:
            await self._handle_timeout()
            raise
        except Exception as e:
            await self._handle_error(e)
            raise

    async def _validate_with_retry(self, model: Any, validators: List[str]) -> Dict:
        """Execute validation with retry logic"""
        for attempt in range(self.validation_retries):
            try:
                results = await asyncio.gather(
                    *[self._validate_on_node(model, node) for node in validators],
                    return_exceptions=True
                )
                
                successful = [r for r in results if isinstance(r, dict)]
                if successful:
                    return self._aggregate_validation_results(successful)
                    
                logger.warning(f"Validation attempt {attempt + 1} failed, retrying...")
            except Exception as e:
                logger.error(f"Validation error on attempt {attempt + 1}: {str(e)}")
                if attempt == self.validation_retries - 1:
                    raise

        raise RuntimeError("Validation failed after all retry attempts")

    async def _validate_on_node(self, model: Any, node: str) -> Dict:
        """Execute validation on a single node"""
        try:
            logger.debug(f"Starting validation on node {node}")
            # Implementation would integrate with actual validation logic
            result = {
                "node": node,
                "status": "success",
                "metrics": {},
                "timestamp": datetime.now().isoformat()
            }
            logger.debug(f"Completed validation on node {node}")
            return result
        except Exception as e:
            logger.error(f"Validation failed on node {node}: {str(e)}")
            raise

    async def _handle_timeout(self) -> None:
        """Handle validation timeout"""
        logger.error("Validation timeout exceeded")
        if self.interactive_mode:
            await self._notify_timeout()

    async def _handle_error(self, error: Exception) -> None:
        """Handle validation errors"""
        logger.error(f"Validation error: {str(error)}")
        if self.interactive_mode:
            await self._notify_error(error)

    def _aggregate_validation_results(self, results: List[Dict]) -> Dict:
        """Aggregate validation results from multiple nodes"""
        return {
            "timestamp": datetime.now().isoformat(),
            "validator_count": len(results),
            "successful_validations": len([r for r in results if r["status"] == "success"]),
            "results": results
        }
