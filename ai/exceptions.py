# ai/exceptions.py
"""Exception classes used throughout the AI module."""
from typing import Optional, Dict, Any  # imported but keeping for future use

class AIException(Exception):
    """Base exception for all AI related errors."""

class ModelInitializationError(AIException):
    """Exception raised when model initialization fails."""

class TrainingError(AIException):
    """Exception raised when training fails."""

class InferenceError(AIException):
    """Exception raised when inference fails."""

class DataPreparationError(AIException):
    """Exception raised when data preparation fails."""

class AggregationError(AIException):
    """Exception raised when model aggregation fails."""

class ValidationError(AIException):
    """Exception raised when validation fails."""

class CompressionError(AIException):
    """Exception raised when compression fails."""

class EncryptionError(AIException):
    """Exception raised when encryption fails."""

class CommunicationError(AIException):
    """Exception raised when communication fails."""

class FederatedLearningError(AIException):
    """Exception raised when federated learning process fails."""

class SecurityError(AIException):
    """Exception raised when security check fails."""

class PrivacyViolationError(AIException):
    """Exception raised when privacy constraints are violated."""
