import torch
from typing import List, Dict

def aggregate_models(models: List[Dict[str, torch.Tensor]], 
                    weights: List[float] = None) -> Dict[str, torch.Tensor]:
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
        
    aggregated_dict = {}
    for key in models[0].keys():
        aggregated_dict[key] = sum(w * model[key] for w, model in zip(weights, models))
    
    return aggregated_dict
