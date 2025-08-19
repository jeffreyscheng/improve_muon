from pathlib import Path
from typing import Tuple, Iterator, TypeAlias
import numpy as np
import torch
import time
from torch.nn import Parameter, Module

"""
Utilities
"""

NamedParameter: TypeAlias = Tuple[str, Parameter]

def get_weight_matrix_iterator(model, only_hidden: bool = True):
    """
    Get an iterator over all weight matrices in the model.
    If only_hidden=True, only returns hidden matrices (from blocks, 2D+, no embeddings)
    following the same logic as Muon optimizer parameter selection.
    """
    if only_hidden:
        # Follow Muon's logic: blocks parameters, 2D+, no embeddings
        for name, param in model.named_parameters():
            if ("blocks." in name and 
                param.ndim >= 2 and 
                "embed" not in name):
                yield name, param
    else:
        for name, param in model.named_parameters():
            yield name, param

def apply_fn_to_all_weights(model, fn, run_name: str, only_hidden: bool = True):
    """
    Apply a function to all weights in the model.
    """
    return {
        name: fn((name, param), run_name)
        for name, param in get_weight_matrix_iterator(model, only_hidden)
    }

def get_research_log_path(metric_name: str, run_name: str, pattern: str, **kwargs) -> Path:
    return Path("research_logs") / metric_name / run_name / pattern.format(**kwargs)