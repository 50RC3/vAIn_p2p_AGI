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
        state_dict = model.state_dict()
        temp_path = Path("temp_model.pt")
        torch.save(state_dict, temp_path)
        
        # Upload to IPFS
        with open(temp_path, "rb") as f:
            model_hash = self.ipfs.add(f)["Hash"]
            
        # Store metadata
        metadata["model_hash"] = model_hash
        metadata_hash = self.ipfs.add_json(metadata)
        
        temp_path.unlink()  # Cleanup
        return metadata_hash
        
    def load_model(self, model: torch.nn.Module, ipfs_hash: str) -> Optional[torch.nn.Module]:
        """Load model weights from IPFS."""
        try:
            # Download model state
            self.ipfs.get(ipfs_hash)
            model.load_state_dict(torch.load(ipfs_hash))
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
