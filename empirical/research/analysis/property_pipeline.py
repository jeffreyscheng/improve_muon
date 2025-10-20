"""
Property-based computation pipeline for gradient analysis.

This module implements a declarative approach to specifying and executing
chains of computations over layer properties. The key insight is that
gradient analysis is just a dependency graph of transformations, and we
can separate the "what" (property specifications) from the "how" (execution).
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Any, Tuple
import time
import logging
import torch.distributed as dist
from empirical.research.analysis.logging_utilities import log_from_rank
from collections import defaultdict, deque

from empirical.research.analysis.model_utilities import GPTLayerProperty


@dataclass
class PropertySpec:
    """Specification for a single property computation.
    
    A PropertySpec declares:
    1. What property to compute (name)
    2. What inputs it needs (dependency names)  
    3. How to compute it (pure transform function)
    """
    name: str
    inputs: List[str]
    transform: Callable[[Dict[str, Any]], Any]
    
    def __post_init__(self):
        """Validate the specification."""
        if not self.name:
            raise ValueError("Property name cannot be empty")
        if not callable(self.transform):
            raise ValueError("Transform must be callable")


class PropertyPipeline:
    """Executes property computations in dependency order.
    
    The pipeline takes a list of PropertySpec objects and:
    1. Builds a dependency graph
    2. Topologically sorts the specifications  
    3. Executes transforms in the correct order
    4. Handles both single-layer and multi-layer execution
    """
    
    def __init__(self, specs: List[PropertySpec]):
        """Initialize pipeline with dependency-sorted specs."""
        self.specs = self._topological_sort(specs)
        self._validate_dependencies()
    
    def _topological_sort(self, specs: List[PropertySpec]) -> List[PropertySpec]:
        """Sort specs by dependency order using Kahn's algorithm."""
        # Build adjacency lists
        spec_by_name = {spec.name: spec for spec in specs}
        in_degree = defaultdict(int)
        graph = defaultdict(list)
        
        # Initialize in-degrees and graph
        for spec in specs:
            in_degree[spec.name] = 0
        
        for spec in specs:
            for dep in spec.inputs:
                if dep in spec_by_name:  # Internal dependency
                    graph[dep].append(spec.name)
                    in_degree[spec.name] += 1
        
        # Kahn's algorithm
        queue = deque([name for name in in_degree if in_degree[name] == 0])
        sorted_names = []
        
        while queue:
            name = queue.popleft()
            sorted_names.append(name)
            
            for neighbor in graph[name]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if len(sorted_names) != len(specs):
            raise ValueError("Circular dependency detected in property specs")
        
        return [spec_by_name[name] for name in sorted_names]
    
    def _validate_dependencies(self):
        """Ensure all internal dependencies are satisfied."""
        available = set()
        
        for spec in self.specs:
            # Check that all dependencies are either available or external
            for dep in spec.inputs:
                if dep not in available and not self._is_external_property(dep):
                    raise ValueError(f"Unsatisfied dependency: {spec.name} needs {dep}")
            available.add(spec.name)
    
    def _is_external_property(self, prop_name: str) -> bool:
        """Check if a property is provided externally (not computed in pipeline)."""
        # These are the "root" properties provided by the caller
        external_props = {
            "checkpoint_weights",
            "per_minibatch_gradient",
            "noise_sigma",  # provided from serialized checkpoints, not computed here
        }
        return prop_name in external_props
    
    def execute_for_layer(self, initial_props: Dict[str, Any], layer_key: Tuple[str, int] | None = None) -> Dict[str, Any]:
        """Execute all transforms for a single layer.
        
        Args:
            initial_props: Dictionary containing root properties (weights, gradients)
            
        Returns:
            Dictionary containing all computed properties
        """
        props = initial_props.copy()
        
        rank = dist.get_rank() if dist.is_initialized() else 0
        for spec in self.specs:
            # Extract inputs for this transform as positional arguments
            try:
                input_values = [props[key] for key in spec.inputs]
            except KeyError as e:
                raise KeyError(f"Property {spec.name} missing input {e}")
            
            # Execute transform with positional arguments
            t0 = time.time()
            props[spec.name] = spec.transform(*input_values)
            dt_ms = (time.time() - t0) * 1000.0
            if layer_key is not None:
                log_from_rank(f"pipeline: {spec.name} for {layer_key} in {dt_ms:.1f} ms", rank)
            else:
                log_from_rank(f"pipeline: {spec.name} in {dt_ms:.1f} ms", rank)
        
        return props
    
    def execute_for_all_layers(
        self, 
        layer_props: GPTLayerProperty,
        progress_callback: Callable[[int, int], None] = None
    ) -> GPTLayerProperty:
        """Execute pipeline across all model layers.
        
        Args:
            layer_props: Initial layer properties (weights + gradients)
            progress_callback: Optional callback for progress tracking
            
        Returns:
            Layer properties with all computed analysis results
        """
        results = {}
        total_layers = len(layer_props)
        
        for i, (layer_key, props) in enumerate(layer_props.items()):
            results[layer_key] = self.execute_for_layer(props, layer_key)
            
            if progress_callback:
                progress_callback(i + 1, total_layers)
        
        return results
    
    def get_property_names(self) -> List[str]:
        """Get names of all properties computed by this pipeline."""
        return [spec.name for spec in self.specs]
    
    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """Get the dependency graph for visualization/debugging."""
        return {spec.name: spec.inputs for spec in self.specs}
