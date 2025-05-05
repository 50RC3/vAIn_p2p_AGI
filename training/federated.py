"""
Federated Learning Module for P2P AGI

This module provides functionality for federated learning across 
distributed nodes in the network, with privacy-preserving features.
"""

import asyncio
import logging
import time
import numpy as np
import torch
from typing import Dict, List, Any, Tuple, Optional, Union
import json
import hashlib
import os

# Import compression utilities
try:
    from .utils.compression import compress_gradients, decompress_gradients
except ImportError:
    def compress_gradients(grads):
        """Fallback compression function"""
        return grads
        
    def decompress_gradients(grads):
        """Fallback decompression function"""
        return grads

# Import crypto if available
try:
    import ed25519
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logging.warning("ed25519 not available, using basic signing")

logger = logging.getLogger(__name__)

class FederatedLearningManager:
    """Manages federated learning operations across the network"""
    
    def __init__(self, config, node_id, p2p_network=None):
        """Initialize the federated learning manager
        
        Args:
            config: Configuration settings
            node_id: Unique ID for this node
            p2p_network: P2P network interface
        """
        self.config = config
        self.node_id = node_id
        self.p2p_network = p2p_network
        self.models = {}
        self.aggregation_task = None
        self.active = False
        self.is_coordinator = False
        
        # Initialize crypto keys
        self._initialize_crypto()
        
        # Metrics tracking
        self.metrics = {
            "rounds_completed": 0,
            "contributions": 0, 
            "models_updated": 0,
            "last_update": time.time()
        }
        
        # Initialize neurotransmitter system for intrinsic motivation
        self.neurotransmitter_system = NeurotransmitterSystem()
        
        # Initialize reputation system
        self.reputation_system = ClientReputationSystem()
        
        # Initialize secure messaging
        self.secure_protocol = SecureMessageProtocol()
        
        # Initialize model coordinator
        self.model_coordinator = ModelCoordinator()
        
        # Initialize cognitive evolution system
        self.cognitive_evolution = CognitiveEvolution()
        
        logger.info("Federated learning manager initialized")
    
    def _initialize_crypto(self):
        """Initialize cryptographic keys for secure model updates"""
        try:
            if CRYPTO_AVAILABLE:
                # Generate signing keypair
                self.signing_private_key, self.signing_public_key = ed25519.create_keypair()
                
                # Generate encryption key (simplified)
                self.encryption_key = os.urandom(32)
            else:
                # Fallback to basic security
                self.signing_private_key = hashlib.sha256(f"{self.node_id}:{time.time()}".encode()).digest()
                self.signing_public_key = hashlib.sha256(self.signing_private_key).digest()
                self.encryption_key = hashlib.sha256(f"{self.node_id}:enc:{time.time()}".encode()).digest()
                
            logger.debug("Cryptographic keys initialized")
        except Exception as e:
            logger.error(f"Error initializing crypto: {e}", exc_info=True)
    
    def sign_message(self, message):
        """Sign a message with the node's private key
        
        Args:
            message: Message to sign
            
        Returns:
            bytes: The signature
        """
        try:
            if CRYPTO_AVAILABLE:
                return self.signing_private_key.sign(message)
            else:
                # Simple fallback signing method
                message_hash = hashlib.sha256(message).digest()
                signature = hashlib.sha256(message_hash + self.signing_private_key).digest()
                return signature
        except Exception as e:
            logger.error(f"Error signing message: {e}", exc_info=True)
            return None
    
    def verify_signature(self, message, signature, public_key):
        """Verify a signature
        
        Args:
            message: Original message
            signature: Signature to verify
            public_key: Public key to use for verification
            
        Returns:
            bool: True if signature is valid
        """
        try:
            if CRYPTO_AVAILABLE:
                verifier = ed25519.VerifyingKey(public_key)
                return verifier.verify(message, signature)
            else:
                # Simple fallback verification
                expected = hashlib.sha256(hashlib.sha256(message).digest() + 
                                        hashlib.sha256(public_key).digest()).digest()
                return signature == expected
        except Exception:
            return False
    
    def encrypt_data(self, data):
        """Encrypt data before transmission
        
        Args:
            data: Data to encrypt
            
        Returns:
            bytes: Encrypted data
        """
        # Simple XOR encryption (for demonstration - use proper encryption in production)
        try:
            serialized = json.dumps(data).encode()
            key_bytes = self.encryption_key
            # Repeat key to match length
            key_expanded = key_bytes * (len(serialized) // len(key_bytes) + 1)
            key_expanded = key_expanded[:len(serialized)]
            
            # XOR encryption
            encrypted = bytes([a ^ b for a, b in zip(serialized, key_expanded)])
            return encrypted
        except Exception as e:
            logger.error(f"Error encrypting data: {e}", exc_info=True)
            return None
    
    def decrypt_data(self, encrypted_data, key=None):
        """Decrypt received data
        
        Args:
            encrypted_data: Encrypted data to decrypt
            key: Optional key to use (defaults to self.encryption_key)
            
        Returns:
            dict: Decrypted data
        """
        if key is None:
            key = self.encryption_key
            
        try:
            # Repeat key to match length
            key_expanded = key * (len(encrypted_data) // len(key) + 1)
            key_expanded = key_expanded[:len(encrypted_data)]
            
            # XOR decryption
            decrypted = bytes([a ^ b for a, b in zip(encrypted_data, key_expanded)])
            return json.loads(decrypted.decode())
        except Exception as e:
            logger.error(f"Error decrypting data: {e}", exc_info=True)
            return None
    
    async def start(self):
        """Start the federated learning process"""
        if self.active:
            return
            
        self.active = True
        if self.p2p_network:
            # Register for federated learning events
            self.p2p_network.register_handler("fl_update", self._handle_model_update)
            self.p2p_network.register_handler("fl_aggregate", self._handle_aggregation_request)
            
        # Start background aggregation task
        self.aggregation_task = asyncio.create_task(self._aggregation_loop())
        logger.info("Federated learning manager started")
    
    async def stop(self):
        """Stop the federated learning process"""
        self.active = False
        if self.aggregation_task:
            self.aggregation_task.cancel()
            try:
                await self.aggregation_task
            except asyncio.CancelledError:
                pass
        logger.info("Federated learning manager stopped")
    
    async def register_model(self, model_name, model, initial_params=None):
        """Register a model for federated learning
        
        Args:
            model_name: Name of the model
            model: PyTorch model or compatible object
            initial_params: Initial parameters (optional)
        """
        if model_name in self.models:
            logger.warning(f"Model {model_name} already registered, updating")
            
        self.models[model_name] = {
            "model": model,
            "version": 1,
            "last_update": time.time(),
            "updates_received": 0,
            "parameters": initial_params or self._extract_parameters(model),
            "metadata": {}
        }
        logger.info(f"Registered model {model_name} for federated learning")
    
    def _extract_parameters(self, model):
        """Extract parameters from a model
        
        Args:
            model: Model to extract parameters from
            
        Returns:
            list: Model parameters
        """
        try:
            if hasattr(model, 'state_dict'):
                # PyTorch model
                return {k: v.cpu().numpy() for k, v in model.state_dict().items()}
            elif hasattr(model, 'get_weights'):
                # Keras/TF model
                return model.get_weights()
            else:
                # Try generic approach
                return list(model.parameters())
        except Exception as e:
            logger.error(f"Error extracting parameters: {e}", exc_info=True)
            return []
    
    async def contribute_update(self, model_name):
        """Contribute a model update to the federation
        
        Args:
            model_name: Name of the model to update
            
        Returns:
            bool: Success status
        """
        if model_name not in self.models:
            logger.error(f"Model {model_name} not registered")
            return False
            
        try:
            # Get current model parameters
            current_params = self._extract_parameters(self.models[model_name]["model"])
            
            # Prepare update package
            update = {
                "model_name": model_name,
                "version": self.models[model_name]["version"],
                "parameters": compress_gradients(current_params),
                "timestamp": time.time(),
                "node_id": self.node_id,
                "metrics": {
                    "training_samples": self.metrics.get("training_samples", 0),
                    "loss": self.metrics.get("loss", 0),
                    "accuracy": self.metrics.get("accuracy", 0)
                }
            }
            
            # Sign the update
            update_bytes = json.dumps(update).encode()
            signature = self.sign_message(update_bytes)
            
            # Send to network
            if self.p2p_network:
                await self.p2p_network.broadcast("fl_update", {
                    "update": update,
                    "signature": signature.hex() if isinstance(signature, bytes) else signature
                })
                
                self.metrics["contributions"] += 1
                logger.info(f"Contributed update for model {model_name}")
                return True
            else:
                logger.error("No P2P network available for update contribution")
                return False
                
        except Exception as e:
            logger.error(f"Error contributing update: {e}", exc_info=True)
            return False
    
    async def _handle_model_update(self, message):
        """Handle an incoming model update
        
        Args:
            message: Update message
        """
        try:
            update = message.get("update")
            signature = message.get("signature")
            
            if not update or not signature:
                logger.warning("Received invalid update format")
                return
                
            model_name = update.get("model_name")
            if model_name not in self.models:
                logger.debug(f"Received update for unregistered model {model_name}")
                return
                
            # Verify signature (in production, would verify against known public key)
            # For now just log that verification would happen here
            logger.debug("Would verify signature here in production")
            
            # Process the update
            self._process_model_update(update)
            
        except Exception as e:
            logger.error(f"Error handling model update: {e}")
    
    def _process_model_update(self, update):
        """Process a model update
        
        Args:
            update: Model update to process
        """
        model_name = update.get("model_name")
        parameters = update.get("parameters")
        
        if parameters:
            # Decompress parameters if needed
            parameters = decompress_gradients(parameters)
            
            # Check if we should apply this update
            if self.reputation_system.should_accept_update(update.get("node_id", "unknown"), update):
                try:
                    # For simplicity, we're just replacing parameters
                    # In practice, you'd want to do proper parameter averaging or aggregation
                    self.models[model_name]["parameters"] = parameters
                    self.models[model_name]["version"] += 1
                    self.models[model_name]["last_update"] = time.time()
                    self.models[model_name]["updates_received"] += 1
                    
                    # Update metrics
                    self.metrics["models_updated"] += 1
                    
                    logger.info(f"Applied update to model {model_name}")
                    
                    # Apply update to model if possible
                    self._apply_parameters_to_model(model_name, parameters)
                    
                except Exception as e:
                    logger.error(f"Error applying update to model {model_name}: {e}", exc_info=True)
    
    def _apply_parameters_to_model(self, model_name, parameters):
        """Apply parameters to a model
        
        Args:
            model_name: Name of the model
            parameters: Parameters to apply
        """
        model = self.models[model_name]["model"]
        
        try:
            if hasattr(model, 'load_state_dict'):
                # PyTorch model
                state_dict = {k: torch.tensor(v) for k, v in parameters.items()}
                model.load_state_dict(state_dict)
            elif hasattr(model, 'set_weights'):
                # Keras/TF model
                model.set_weights(parameters)
            else:
                logger.warning(f"Don't know how to apply parameters to model {model_name}")
        except Exception as e:
            logger.error(f"Error applying parameters: {e}", exc_info=True)
    
    async def _aggregation_loop(self):
        """Background task for model aggregation"""
        while self.active:
            try:
                # Sleep between aggregation rounds
                await asyncio.sleep(self.config.get("aggregation_interval", 300))
                
                if self.is_coordinator:
                    # Trigger aggregation for all registered models
                    for model_name in self.models:
                        await self._aggregate_model(model_name)
                        
                    # Update metrics
                    self.metrics["rounds_completed"] += 1
                    self.metrics["last_update"] = time.time()
                    
            except asyncio.CancelledError:
                logger.info("Aggregation loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in aggregation loop: {e}", exc_info=True)
                await asyncio.sleep(60)  # Back off on error
    
    async def _aggregate_model(self, model_name):
        """Aggregate updates for a model
        
        Args:
            model_name: Name of the model to aggregate
        """
        try:
            # In a real implementation, this would request parameters from all peers
            # and perform federated averaging or another aggregation method
            logger.info(f"Would aggregate model {model_name} here")
            
            # Update cognitive load for this model
            cognitive_load = self.cognitive_evolution.update_model_complexity(model_name, 
                                                                           self.models[model_name])
            
            # Update neurotransmitter levels based on learning progress
            reward_signal = self.metrics.get("accuracy", 0.5)
            self.neurotransmitter_system.update(reward_signal, cognitive_load)
            
            # Sample learning rate from dopamine level
            learning_rate = self.neurotransmitter_system.get_learning_rate()
            logger.debug(f"Dopamine-modulated learning rate: {learning_rate}")
            
            # Broadcast aggregated model if we're the coordinator
            if self.is_coordinator and self.p2p_network:
                await self.p2p_network.broadcast("fl_aggregate", {
                    "model_name": model_name,
                    "version": self.models[model_name]["version"],
                    "timestamp": time.time()
                })
                
        except Exception as e:
            logger.error(f"Error aggregating model {model_name}: {e}", exc_info=True)
    
    async def _handle_aggregation_request(self, message):
        """Handle an aggregation request
        
        Args:
            message: Aggregation request message
        """
        try:
            model_name = message.get("model_name")
            if model_name in self.models:
                # Fetch latest model if needed
                if message.get("version", 0) > self.models[model_name]["version"]:
                    await self._request_latest_model(model_name, message.get("node_id"))
        except Exception as e:
            logger.error(f"Error handling aggregation request: {e}", exc_info=True)
    
    async def _request_latest_model(self, model_name, target_node=None):
        """Request the latest version of a model
        
        Args:
            model_name: Name of the model
            target_node: Optional node to request from (None for broadcast)
        """
        if not self.p2p_network:
            logger.error("No P2P network available for model request")
            return
            
        try:
            request = {
                "model_name": model_name,
                "current_version": self.models[model_name]["version"] if model_name in self.models else 0,
                "node_id": self.node_id,
                "timestamp": time.time()
            }
            
            if target_node:
                await self.p2p_network.send_to_node(target_node, "fl_model_request", request)
            else:
                await self.p2p_network.broadcast("fl_model_request", request)
                
            logger.debug(f"Requested latest version of model {model_name}")
        except Exception as e:
            logger.error(f"Error requesting model: {e}", exc_info=True)
    
    def get_metrics(self):
        """Get current metrics
        
        Returns:
            dict: Current metrics
        """
        return self.metrics
