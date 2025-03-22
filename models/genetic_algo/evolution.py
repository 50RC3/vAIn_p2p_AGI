import torch
import numpy as np
import logging
import asyncio
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
from core.interactive_utils import InteractiveSession, InteractionLevel, InteractiveConfig, INTERACTION_TIMEOUTS

logger = logging.getLogger(__name__)

class GeneticOptimizer:
    def __init__(self, population_size: int, mutation_rate: float, interactive: bool = True):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.interactive = interactive
        self.session = None
        self._interrupt_requested = False
        self._progress = {}
        
    async def evolve_interactive(self, population: List[torch.nn.Module], 
                               fitness_scores: torch.Tensor) -> Optional[List[torch.nn.Module]]:
        """Interactive evolution with progress tracking and error handling"""
        try:
            if self.interactive:
                self.session = InteractiveSession(
                    level=InteractionLevel.NORMAL,
                    config=InteractiveConfig(
                        timeout=INTERACTION_TIMEOUTS["batch"],
                        persistent_state=True,
                        safe_mode=True,
                        progress_tracking=True
                    )
                )

            async with self.session:
                # Validate inputs
                if not self._validate_inputs(population, fitness_scores):
                    return None

                print("\nStarting Genetic Evolution")
                print("=" * 50)
                print(f"Population size: {self.population_size}")
                print(f"Mutation rate: {self.mutation_rate}")

                try:
                    # Restore previous progress if available
                    saved_progress = await self.session._load_progress()
                    if saved_progress:
                        logger.info("Restoring from previous evolution state")
                        population = saved_progress.get('population', population)

                    # Selection with progress
                    parents = await self._select_parents_interactive(population, fitness_scores)
                    if self._interrupt_requested:
                        return None

                    # Crossover with monitoring
                    offspring = await self._crossover_interactive(parents)
                    if self._interrupt_requested:
                        return None

                    # Mutation with safety checks
                    mutated = await self._mutate_interactive(offspring)
                    
                    # Save final state
                    await self._save_progress(mutated)
                    return mutated

                except Exception as e:
                    logger.error(f"Evolution error: {str(e)}")
                    await self._save_progress(population)
                    raise

        except Exception as e:
            logger.error(f"Interactive evolution failed: {str(e)}")
            raise
        finally:
            if self.session:
                await self.session.__aexit__(None, None, None)

    async def _select_parents_interactive(self, population: List[torch.nn.Module], 
                                       fitness_scores: torch.Tensor) -> List[torch.nn.Module]:
        """Interactive parent selection with monitoring"""
        try:
            # Monitor memory usage
            if hasattr(torch.cuda, 'memory_allocated'):
                mem_before = torch.cuda.memory_allocated()

            # Selection logic with progress tracking
            selected = []
            with tqdm(total=len(population)//2, desc="Selecting Parents") as pbar:
                for i in range(0, len(population), 2):
                    if self._interrupt_requested:
                        break
                    
                    # Add selection logic here
                    parents = self._select_tournament(population, fitness_scores)
                    selected.extend(parents)
                    pbar.update(1)

            return selected

        except Exception as e:
            logger.error(f"Parent selection failed: {str(e)}")
            raise

    async def _crossover_interactive(self, parents: List[torch.nn.Module]) -> List[torch.nn.Module]:
        """Interactive crossover with safety checks"""
        try:
            offspring = []
            with tqdm(total=len(parents)//2, desc="Performing Crossover") as pbar:
                for i in range(0, len(parents), 2):
                    if self._interrupt_requested:
                        break

                    # Add crossover logic here
                    child1, child2 = self._crossover_pair(parents[i], parents[i+1])
                    offspring.extend([child1, child2])
                    pbar.update(1)

            return offspring

        except Exception as e:
            logger.error(f"Crossover failed: {str(e)}")
            raise

    async def _mutate_interactive(self, population: List[torch.nn.Module]) -> List[torch.nn.Module]:
        """Interactive mutation with progress tracking"""
        try:
            mutated = []
            with tqdm(total=len(population), desc="Applying Mutations") as pbar:
                for individual in population:
                    if self._interrupt_requested:
                        break

                    # Add mutation logic here
                    mutated_individual = self._mutate_individual(individual)
                    mutated.append(mutated_individual)
                    pbar.update(1)

            return mutated

        except Exception as e:
            logger.error(f"Mutation failed: {str(e)}")
            raise

    def _validate_inputs(self, population: List[torch.nn.Module], 
                        fitness_scores: torch.Tensor) -> bool:
        """Validate input parameters and data"""
        try:
            if len(population) != self.population_size:
                raise ValueError(f"Population size mismatch: {len(population)} != {self.population_size}")
            
            if len(population) != len(fitness_scores):
                raise ValueError("Population and fitness scores length mismatch")
                
            if not all(isinstance(ind, torch.nn.Module) for ind in population):
                raise TypeError("All individuals must be torch.nn.Module instances")
                
            return True

        except Exception as e:
            logger.error(f"Input validation failed: {str(e)}")
            return False

    async def _save_progress(self, population: List[torch.nn.Module]) -> None:
        """Save evolution progress"""
        if self.session and self.session.config.persistent_state:
            self._progress['population'] = population
            await self.session._save_progress()

    def cleanup(self):
        """Cleanup resources"""
        self._interrupt_requested = False
        self._progress.clear()
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
