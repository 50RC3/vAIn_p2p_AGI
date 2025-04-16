from typing import Optional, Dict, Any
import logging
from ..interactive_utils import InteractiveSession, InteractiveConfig
from ..component_integration import ComponentIntegration

logger = logging.getLogger(__name__)

class InteractiveFeature:
    """Base template for adding interactive features"""
    
    def __init__(self, config: Optional[InteractiveConfig] = None):
        self.config = config or InteractiveConfig()
        self.integration: Optional[ComponentIntegration] = None
        
    async def run_interactive(self) -> bool:
        """Run feature with interactive controls"""
        async with InteractiveSession(config=self.config) as session:
            try:
                if not await session.prompt_yes_no("Start interactive session?"):
                    return False
                    
                # Execute feature steps with interaction
                await self._execute_steps(session)
                
                return True
                
            except Exception as e:
                logger.error(f"Interactive session failed: {str(e)}")
                if self.config.auto_recovery:
                    await self._attempt_recovery(session)
                return False
    
    async def connect_integration(self, integration: ComponentIntegration) -> None:
        """Connect to component integration"""
        self.integration = integration
        logger.debug("Connected to component integration")
                
    async def _execute_steps(self, session: InteractiveSession):
        """Implement feature-specific interactive steps"""
        raise NotImplementedError
        
    async def _attempt_recovery(self, session: InteractiveSession):
        """Implement feature-specific recovery"""
        raise NotImplementedError
    
    async def get_component(self, component_id: str) -> Any:
        """Get a component from the integration"""
        if not self.integration:
            raise RuntimeError("Not connected to integration")
        
        return await self.integration.get_component(component_id)
    
    async def _check_system_health(self) -> Dict[str, str]:
        """Check the health of system components"""
        if not self.integration:
            raise RuntimeError("Not connected to integration")
            
        return await self.integration._check_components_health()
