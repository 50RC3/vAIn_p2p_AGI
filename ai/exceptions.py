class BaseAIError(Exception):
    """Base exception class for AI-related errors"""
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

class AggregationError(BaseAIError):
    """Raised when model aggregation fails"""
    def __init__(self, message: str, details: dict = None):
        super().__init__(f"Model aggregation failed: {message}", details)

class TrainingError(BaseAIError):
    """Raised when local training fails"""
    def __init__(self, message: str, details: dict = None):
        super().__init__(f"Local training failed: {message}", details)

class ValidationError(BaseAIError):
    """Raised when model validation fails"""
    def __init__(self, message: str, details: dict = None):
        super().__init__(f"Model validation failed: {message}", details)

class ModelLoadError(BaseAIError):
    """Raised when model loading fails"""
    def __init__(self, message: str, details: dict = None):
        super().__init__(f"Model loading failed: {message}", details)

class ResourceError(BaseAIError):
    """Raised when resource allocation/management fails"""
    def __init__(self, message: str, details: dict = None):
        super().__init__(f"Resource error: {message}", details)

class ConsensusError(BaseAIError):
    """Raised when consensus cannot be reached"""
    def __init__(self, message: str, details: dict = None):
        super().__init__(f"Consensus error: {message}", details)
