import logging
import os
from typing import Dict, Any, Optional, Type

from ai_core.model_storage import ModelStorage
from ai_core.chatbot.module_integration import ModuleIntegration, ModuleIntegrationConfig
from ai_core.chatbot.learning_coordinator import LearningCoordinatorConfig
from ai_core.chatbot.rl_trainer import RLConfig
from ai_core.chatbot.interface import ChatbotInterface, LearningConfig
from ai_core.chatbot.mobile_interface import MobileChatInterface, MobileConfig

logger = logging.getLogger(__name__)

class ModuleIntegrationFactory:
    """Factory for creating module integration instances"""
    
    @staticmethod
    async def create_integration(
        config: Optional[Dict[str, Any]] = None, 
        model_storage: Optional[ModelStorage] = None
    ) -> Optional[ModuleIntegration]:
        """Create a new module integration instance with all necessary components"""
        try:
            # Create default config if none provided
            if not config:
                config = {}
            
            # Build module integration config
            integration_config = ModuleIntegrationConfig(
                device=config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
                resource_monitoring_interval=config.get("resource_monitoring_interval", 60),
                memory_threshold=config.get("memory_threshold", 85.0),
                enable_distributed=config.get("enable_distributed", False),
                checkpoint_dir=config.get("checkpoint_dir", "./checkpoints"),
                log_level=config.get("log_level", "INFO")
            )

            # Create module integration
            integration = ModuleIntegration(integration_config)
            
            # Create model storage if not provided
            if not model_storage:
                model_storage = ModelStorage(
                    base_dir=config.get("model_dir", "./models"),
                    use_versioning=config.get("use_versioning", True)
                )
            
            # Create learning coordinator config
            learning_config = LearningCoordinatorConfig(
                enable_self_supervised=config.get("enable_self_supervised", True),
                enable_unsupervised=config.get("enable_unsupervised", True),
                enable_reinforcement=config.get("enable_reinforcement", True),
                model_path=config.get("model_path", "./models"),
                stats_save_interval=config.get("stats_save_interval", 100),
                min_sample_length=config.get("min_sample_length", 5),
                max_buffer_size=config.get("max_buffer_size", 1000),
                self_supervised_model=config.get("self_supervised_model", "bert-base-uncased"),
                unsupervised_clusters=config.get("unsupervised_clusters", 10),
                batch_size=config.get("batch_size", 8),
                save_interval=config.get("save_interval", 500)
            )
            
            # Create RL config if reinforcement learning is enabled
            rl_config = None
            if config.get("enable_reinforcement", True):
                rl_config = RLConfig(
                    device=integration_config.device,
                    lr=config.get("rl_learning_rate", 1e-4),
                    discount_factor=config.get("discount_factor", 0.99),
                    memory_size=config.get("memory_size", 10000),
                    batch_size=config.get("rl_batch_size", 64),
                    update_interval=config.get("update_interval", 10),
                    prioritized_replay=config.get("prioritized_replay", True),
                    alpha=config.get("alpha", 0.6),
                    beta=config.get("beta", 0.4),
                    epsilon=config.get("epsilon", 0.01),
                    grad_clip=config.get("grad_clip", 1.0)
                )
            
            # Initialize module integration
            success = await integration.initialize(
                model_storage=model_storage,
                learning_config=learning_config,
                rl_config=rl_config
            )
            
            if not success:
                logger.error("Failed to initialize module integration")
                return None
                
            logger.info("Module integration created and initialized successfully")
            return integration
            
        except Exception as e:
            logger.error(f"Failed to create module integration: {e}")
            return None
    
    @staticmethod
    def create_chatbot_interface(
        model: torch.nn.Module,
        storage: Any,
        interface_type: str = "standard",
        config: Optional[Dict[str, Any]] = None
    ) -> Optional[ChatbotInterface]:
        """Create a chatbot interface based on type"""
        try:
            if not config:
                config = {}
                
            # Create learning config
            learning_config = LearningConfig(
                enable_self_supervised=config.get("enable_self_supervised", False),
                enable_unsupervised=config.get("enable_unsupervised", False),
                self_supervised_model=config.get("self_supervised_model", "bert-base-uncased"),
                unsupervised_clusters=config.get("unsupervised_clusters", 10),
                batch_size=config.get("batch_size", 8),
                learning_rate=config.get("learning_rate", 2e-5),
                mask_probability=config.get("mask_probability", 0.15),
                save_interval=config.get("save_interval", 1000),
                model_path=config.get("model_path", "./models"),
                max_context_length=config.get("max_context_length", 512),
                min_sample_length=config.get("min_sample_length", 5)
            )
            
            if interface_type.lower() == "mobile":
                # Create mobile config
                mobile_config = MobileConfig(
                    min_batch_size=config.get("min_batch_size", 8),
                    max_batch_size=config.get("max_batch_size", 32),
                    battery_threshold=config.get("battery_threshold", 0.2),
                    network_threshold=config.get("network_threshold", 0.5),
                    enable_power_saving=config.get("enable_power_saving", True),
                    enable_learning=config.get("enable_learning", False),
                    learning_offline_only=config.get("learning_offline_only", True),
                    compressed_learning=config.get("compressed_learning", True),
                    max_buffer_size=config.get("max_buffer_size", 100)
                )
                
                return MobileChatInterface(
                    model=model,
                    storage=storage,
                    max_history=config.get("max_history", 100),
                    mobile_config=mobile_config,
                    learning_config=learning_config
                )
            else:
                # Standard interface
                return ChatbotInterface(
                    model=model,
                    storage=storage,
                    max_history=config.get("max_history", 1000),
                    max_input_len=config.get("max_input_len", 512),
                    interactive=config.get("interactive", None),
                    learning_config=learning_config
                )
                
        except Exception as e:
            logger.error(f"Failed to create chatbot interface: {e}")
            return None
