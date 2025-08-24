from pathlib import Path
from typing import Tuple, Iterator, TypeAlias
import numpy as np
import torch
import time
from torch.nn import Parameter, Module

"""
Utilities
"""

GPTLayerProperty: TypeAlias = dict[tuple[str, int], Parameter | np.ndarray]

def get_weight_matrix_iterator(model, only_hidden: bool = True) -> GPTLayerProperty:
    """
    Get all weight matrices in the model as a GPTLayerProperty dict.
    If only_hidden=True, only returns hidden matrices (from blocks, 2D+, no embeddings)
    following the same logic as Muon optimizer parameter selection.
    """
    result = {}
    
    def extract_layer_info(name: str) -> tuple[str, int]:
        """Extract param type and layer number from parameter name."""
        # Example: "blocks.0.mlp.c_fc.weight" -> ("MLP Input", 0)
        # Example: "blocks.5.attn.c_attn.weight" -> splits into Q/K/V
        parts = name.split('.')
        if len(parts) >= 2 and parts[0] == "blocks":
            layer_num = int(parts[1])
            if "attn" in name:
                if "qkvo_w" in name:
                    # For attention, we need to determine Q/K/V/O from the parameter itself
                    return "Attention", layer_num  # Will be split later
                elif "c_proj" in name:  # Keep for backward compatibility
                    return "Attention O", layer_num
            elif "mlp" in name:
                if "fc_w" in name:
                    return "MLP Input", layer_num
                elif "proj_w" in name:
                    return "MLP Output", layer_num
                elif "c_fc" in name:  # Keep for backward compatibility
                    return "MLP Input", layer_num
                elif "c_proj" in name:  # Keep for backward compatibility
                    return "MLP Output", layer_num
        return "Unknown", -1
    
    if only_hidden:
        # Follow Muon's logic: blocks parameters, 2D+, no embeddings
        for name, param in model.named_parameters():
            if ("blocks." in name and 
                param.ndim >= 2 and 
                "embed" not in name):
                param_type, layer_num = extract_layer_info(name)
                if param_type != "Unknown":
                    if param_type == "Attention" and "qkvo_w" in name:
                        # Split qkvo_w into Q, K, V, O (shape: [4, 1024, 1024])
                        weight = param.data  # shape: [4, d_model, d_model]
                        q_weight = weight[0]  # Q: [d_model, d_model]
                        k_weight = weight[1]  # K: [d_model, d_model] 
                        v_weight = weight[2]  # V: [d_model, d_model]
                        o_weight = weight[3]  # O: [d_model, d_model]
                        result[("Attention Q", layer_num)] = q_weight
                        result[("Attention K", layer_num)] = k_weight
                        result[("Attention V", layer_num)] = v_weight
                        result[("Attention O", layer_num)] = o_weight
                    elif param_type == "Attention" and "c_attn" in name:
                        # Keep backward compatibility for old format
                        weight = param.data
                        d_model = weight.shape[1]
                        q_weight = weight[:d_model, :]
                        k_weight = weight[d_model:2*d_model, :]
                        v_weight = weight[2*d_model:, :]
                        result[("Attention Q", layer_num)] = q_weight
                        result[("Attention K", layer_num)] = k_weight
                        result[("Attention V", layer_num)] = v_weight
                    else:
                        result[(param_type, layer_num)] = param
    else:
        for name, param in model.named_parameters():
            param_type, layer_num = extract_layer_info(name)
            if param_type != "Unknown":
                if param_type == "Attention" and "qkvo_w" in name:
                    # Split qkvo_w into Q, K, V, O (shape: [4, 1024, 1024])
                    weight = param.data  # shape: [4, d_model, d_model]
                    q_weight = weight[0]  # Q: [d_model, d_model]
                    k_weight = weight[1]  # K: [d_model, d_model] 
                    v_weight = weight[2]  # V: [d_model, d_model]
                    o_weight = weight[3]  # O: [d_model, d_model]
                    result[("Attention Q", layer_num)] = q_weight
                    result[("Attention K", layer_num)] = k_weight
                    result[("Attention V", layer_num)] = v_weight
                    result[("Attention O", layer_num)] = o_weight
                elif param_type == "Attention" and "c_attn" in name:
                    # Keep backward compatibility for old format
                    weight = param.data
                    d_model = weight.shape[1]
                    q_weight = weight[:d_model, :]
                    k_weight = weight[d_model:2*d_model, :]
                    v_weight = weight[2*d_model:, :]
                    result[("Attention Q", layer_num)] = q_weight
                    result[("Attention K", layer_num)] = k_weight
                    result[("Attention V", layer_num)] = v_weight
                else:
                    result[(param_type, layer_num)] = param
            else:
                result[(name, -1)] = param
    
    return result

def apply_fn_to_all_layers(model, fn, run_name: str, only_hidden: bool = True):
    """
    Apply a function to all layers in the model.
    """
    layer_properties = get_weight_matrix_iterator(model, only_hidden)
    return {
        key: fn(key, param, run_name)
        for key, param in layer_properties.items()
    }

def get_research_log_path(metric_name: str, run_name: str, pattern: str, **kwargs) -> Path:
    return Path("research_logs") / metric_name / run_name / pattern.format(**kwargs)