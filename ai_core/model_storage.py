import torch
import ipfshttpclient
import json
from typing import Dict, Optional
from pathlib import Path

class ModelStorage:
    def __init__(self, ipfs_host: str = "/ip4/127.0.0.1/tcp/5001"):
        self.ipfs = ipfshttpclient.connect(ipfs_host)
        
    def store_model(self, model: torch.nn.Module, metadata: Dict) -> str:
        """Store model weights and metadata on IPFS."""
        # Save model state
        temp_path = Path("temp_model.pt")
        torch.save(model.state_dict(), temp_path)
        
        # Upload to IPFS
        model_hash = self.ipfs.add(str(temp_path))["Hash"]
        
        # Store metadata
        metadata["model_hash"] = model_hash
        metadata_hash = self.ipfs.add_json(metadata)
        
        temp_path.unlink()  # Cleanup
        return metadata_hash
        
    def load_model(self, model: torch.nn.Module, ipfs_hash: str) -> Optional[torch.nn.Module]:
        """Load model weights from IPFS."""
        try:
            # Define the local filename for the downloaded model
            local_model_path = Path("downloaded_model.pt")
            
            # Download model state from IPFS
            self.ipfs.get(ipfs_hash, target=str(local_model_path))
            
            # Load state into the model
            model.load_state_dict(torch.load(local_model_path))
            
            # Cleanup the downloaded model file
            local_model_path.unlink()
            
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None