import torch
import logging
from typing import Optional, Dict, Any, Union, List

logger = logging.getLogger(__name__)

class InputProcessor:
    """
    Handles text processing for neural network input/output.
    Converts between text and tensor representations.
    """
    
    def __init__(self, 
                 input_size: int = 512,
                 max_sequence_length: int = 1024,
                 tokenizer: Optional[Any] = None):
        """
        Initialize the InputProcessor with parameters.
        
        Args:
            input_size: Dimensionality of the input embeddings
            max_sequence_length: Maximum length of input sequences
            tokenizer: Optional tokenizer instance
        """
        self.input_size = input_size
        self.max_sequence_length = max_sequence_length
        self.tokenizer = tokenizer
        
    def text_to_tensor(self, text: str) -> torch.Tensor:
        """
        Convert text input to tensor representation suitable for the model.
        
        Args:
            text: Input text string
            
        Returns:
            torch.Tensor: Tensor representation of the input text
        """
        try:
            # If we have a tokenizer, use it
            if self.tokenizer and hasattr(self.tokenizer, 'encode'):
                tokens = self.tokenizer.encode(text, 
                                              max_length=self.max_sequence_length,
                                              truncation=True,
                                              padding='max_length',
                                              return_tensors='pt')
                return torch.as_tensor(tokens)
                
            # Simple fallback implementation using character encoding
            chars = [ord(c) / 255.0 for c in text[:self.max_sequence_length]]
            # Pad to input_size
            chars = chars + [0] * (self.input_size - len(chars))
            return torch.tensor([chars], dtype=torch.float32)
            
        except Exception as e:
            logger.error(f"Error in text_to_tensor: {e}")
            # Return a default tensor in case of error
            return torch.zeros((1, self.input_size), dtype=torch.float32)
            
    def tensor_to_text(self, tensor: torch.Tensor) -> str:
        """
        Convert tensor output from the model back to text.
        
        Args:
            tensor: Output tensor from model
            
        Returns:
            str: Text representation of the output tensor
        """
        try:
            # If we have a tokenizer, use it
            if self.tokenizer and hasattr(self.tokenizer, 'decode'):
                if len(tensor.shape) > 1 and tensor.shape[0] == 1:
                    # Remove batch dimension if present
                    tensor = tensor.squeeze(0)
                return self.tokenizer.decode(tensor, skip_special_tokens=True)
                
            # Simple fallback implementation using character decoding
            if len(tensor.shape) > 1:
                # Just take the first sequence if it's batched
                tensor = tensor[0]
                
            # Convert back to characters
            chars = [chr(int(c * 255)) for c in tensor if c > 0]
            return ''.join(chars)
            
        except Exception as e:
            logger.error(f"Error in tensor_to_text: {e}")
            return "Error: Could not convert tensor to text"
    
    def preprocess_input(self, text: str) -> str:
        """
        Preprocess input text before conversion to tensor.
        
        Args:
            text: Raw input text
            
        Returns:
            str: Preprocessed text
        """
        # Trim whitespace
        text = text.strip()
        # Remove duplicate spaces
        text = ' '.join(text.split())
        return text
    
    def postprocess_output(self, text: str) -> str:
        """
        Postprocess output text after conversion from tensor.
        
        Args:
            text: Raw output text from model
            
        Returns:
            str: Cleaned and formatted output text
        """
        # Trim whitespace
        text = text.strip()
        # Fix common formatting issues
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        return text
