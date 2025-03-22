from typing import Optional
from ..interactive_utils import InteractiveSession, InteractiveConfig

class InteractiveFeature:
    """Base template for adding interactive features"""
    
    def __init__(self, config: Optional[InteractiveConfig] = None):
        self.config = config or InteractiveConfig()
        
    async def run_interactive(self) -> bool:
        """Run feature with interactive controls"""
        async with InteractiveSession(config=self.config) as session:
            try:
                if not await session.confirm("Start interactive session?"):
                    return False
                    
                # Execute feature steps with interaction
                await self._execute_steps(session)
                
                return True
                
            except Exception as e:
                logger.error(f"Interactive session failed: {str(e)}")
                if self.config.error_recovery:
                    await self._attempt_recovery(session)
                return False
                
    async def _execute_steps(self, session: InteractiveSession):
        """Implement feature-specific interactive steps"""
        raise NotImplementedError
        
    async def _attempt_recovery(self, session: InteractiveSession):
        """Implement feature-specific recovery"""
        raise NotImplementedError
