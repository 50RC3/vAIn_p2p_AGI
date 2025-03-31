import pytest
import torch
import asyncio
import logging
from unittest.mock import MagicMock, patch
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ai_core.learning.advanced_learning import AdvancedLearning, LearningConfig, LearningMetrics
from ai_core.chatbot.rl_trainer import RLTrainer, RLConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestCognitiveLearning:
    @pytest.fixture
    def mock_chatbot(self):
        """Create a mock chatbot interface"""
        chatbot = MagicMock()
        chatbot.history = [
            ("Hello, how are you?", "I'm doing well, thank you for asking."),
            ("Tell me about machine learning", "Machine learning is a field of AI...")
        ]
        chatbot.feedback_scores = [
            {"score": 0.8},
            {"score": 0.9}
        ]
        return chatbot
    
    @pytest.fixture
    def mock_rl_trainer(self):
        """Create a mock RL trainer"""
        rl_trainer = MagicMock(spec=RLTrainer)
        rl_trainer.get_training_stats.return_value = {
            "avg_reward": 0.7,
            "avg_td_error": 0.3,
            "training_steps": 100
        }
        return rl_trainer
    
    @pytest.fixture
    def learning_config(self):
        """Create a learning configuration"""
        return LearningConfig(
            unsupervised_batch_size=4,
            supervised_batch_size=2,
            rl_batch_size=2,
            learning_rate=0.001,
            training_interval=5,  # Short interval for testing
            min_samples_for_training=1,  # Allow training with just one sample
            max_history_length=100,
            save_path="models/test_learning",
            interactive_mode=False
        )
    
    @pytest.fixture
    def mock_p2p_network(self):
        """Create a mock P2P network"""
        network = MagicMock()
        network.broadcast.return_value = True
        network.get_connected_peers.return_value = ["peer1", "peer2"]
        return network
    
    @pytest.fixture
    def advanced_learning(self, mock_chatbot, mock_rl_trainer, learning_config, mock_p2p_network):
        """Create an AdvancedLearning instance"""
        return AdvancedLearning(
            chatbot=mock_chatbot,
            rl_trainer=mock_rl_trainer,
            config=learning_config,
            p2p_network=mock_p2p_network
        )
    
    @pytest.mark.asyncio
    async def test_learning_flow(self, advanced_learning):
        """Test the complete learning flow"""
        # Add some conversation samples
        await advanced_learning.add_conversation_sample(
            message="What is federated learning?",
            response="Federated learning is a machine learning approach...",
            score=0.9
        )
        
        # Check that samples were added
        assert len(advanced_learning.conversation_history) == 1
        
        # Run training
        with patch.object(advanced_learning, '_run_unsupervised_learning', return_value=0.5) as mock_unsupervised:
            with patch.object(advanced_learning, '_run_supervised_learning', return_value=0.3) as mock_supervised:
                with patch.object(advanced_learning, '_run_reinforcement_learning', return_value=0.2) as mock_rl:
                    with patch.object(advanced_learning, '_run_contrastive_learning', return_value=0.1) as mock_contrastive:
                        await advanced_learning._run_combined_training()
        
        # Check that all training methods were called
        mock_unsupervised.assert_called_once()
        mock_supervised.assert_called_once()
        mock_rl.assert_called_once()
        mock_contrastive.assert_called_once()
        
        # Check metrics were updated
        assert advanced_learning.metrics.training_cycles == 1
        
    @pytest.mark.asyncio
    async def test_peer_learning(self, advanced_learning, mock_p2p_network):
        """Test peer-based learning"""
        # Create a sample to share
        sample = {
            "message": "What is AGI?",
            "response": "AGI refers to Artificial General Intelligence...",
            "score": 0.95,
            "timestamp": 1632481632.0
        }
        
        # Share with peers
        await advanced_learning.share_with_peers(
            message=sample["message"],
            response=sample["response"],
            score=sample["score"]
        )
        
        # Verify P2P network was called
        mock_p2p_network.broadcast.assert_called_once()
        
        # Receive a peer sample
        await advanced_learning.receive_peer_sample(sample)
        
        # Check that peer sample was added
        assert len(advanced_learning.peer_shared_samples) == 1
        
    @pytest.mark.asyncio
    async def test_metrics_tracking(self, advanced_learning):
        """Test metrics tracking"""
        # Get initial metrics
        initial_metrics = await advanced_learning.get_metrics()
        
        # Run a training cycle to update metrics
        with patch.object(advanced_learning, '_run_unsupervised_learning', return_value=0.4):
            with patch.object(advanced_learning, '_run_supervised_learning', return_value=0.3):
                with patch.object(advanced_learning, '_run_reinforcement_learning', return_value=0.2):
                    await advanced_learning._run_combined_training()
        
        # Get updated metrics
        updated_metrics = await advanced_learning.get_metrics()
        
        # Verify metrics were updated
        assert updated_metrics["training_cycles"] == 1
        assert updated_metrics["unsupervised_loss"] < 1.0  # Should be a reasonable number
        assert updated_metrics["supervised_loss"] < 1.0
        assert updated_metrics["reinforcement_loss"] < 1.0
