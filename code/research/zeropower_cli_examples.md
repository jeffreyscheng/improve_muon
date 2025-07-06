# Zeropower Testing CLI Examples

The `zeropower_testing.py` script now supports a sophisticated CLI interface for testing different orthogonalization methods on MLP vs attention layers.

## Key Features

1. **Removed tanh_element backend** - it was not useful for orthogonalization
2. **Split optimizers** - separate Muon optimizers for MLP and attention layers  
3. **Added classic_newton_schulz** - simple f(x)=1.5x-0.5x^3 iteration with configurable iterations
4. **CLI interface** - specify different methods and hyperparameters for different layer types

## Usage Examples

### Basic Usage (same method for all layers)
```bash
python zeropower_testing.py --zeropower-method newton_schulz
```

### Use different methods for MLP vs attention
```bash
# Strong whitening for MLP, gentle for attention  
python zeropower_testing.py \
  --mlp-method svd_polar \
  --attn-method newton_schulz
```

### Configure hyperparameters
```bash
# Classic Newton-Schulz with more iterations for MLP
python zeropower_testing.py \
  --mlp-method classic_newton_schulz \
  --mlp-hyperparams '{"num_iters": 20}' \
  --attn-method newton_schulz
```

### Advanced configuration
```bash
# Different methods and hyperparameters for each layer type
python zeropower_testing.py \
  --mlp-method classic_newton_schulz \
  --mlp-hyperparams '{"num_iters": 25}' \
  --attn-method svd_polar \
  --attn-hyperparams '{}'
```

## Available Backends

1. **`newton_schulz`** - The original quintic Newton-Schulz (default)
2. **`svd_polar`** - SVD-based polar decomposition (best for strong whitening)
3. **`classic_newton_schulz`** - Simple f(x)=1.5x-0.5x^3 iteration
   - Hyperparameter: `num_iters` (default: 15)
4. **`tanh_matrix`** - Matrix tanh (currently falls back to newton_schulz)

## Hyperparameter Format

Hyperparameters are passed as JSON strings. Examples:

- `'{}'` - no hyperparameters (use defaults)
- `'{"num_iters": 20}'` - set num_iters to 20 for classic_newton_schulz
- `'{"alpha": 5.0}'` - set alpha parameter (for methods that support it)

## Rationale for Split Optimizers

**MLP layers** have lower noise floors and benefit from strong whitening of small singular values (around 1e-8), helping with gradient conditioning.

**Attention layers** have much higher noise floors, so whitening anything smaller than 1e-3 just amplifies noise without benefit.

The split optimizer approach allows different orthogonalization strategies optimized for each layer type's characteristics.

## Output

The system logs show which methods are being used:
```
ZEROPOWER METHOD: newton_schulz (hyperparams: {})
MLP METHOD: svd_polar (hyperparams: {})  
ATTENTION METHOD: newton_schulz (hyperparams: {})
```

This makes it easy to track which configuration is being tested in each run. 