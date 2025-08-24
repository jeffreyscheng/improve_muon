# Docker Setup for improve_muon

This document outlines the Docker setup for running `minimal_medium.py` successfully on an 8x H100 system using archived PyTorch with FlexAttention compatibility.

> **Quick Setup**: For automated installation, use `./docker_setup.sh` which handles all steps below automatically.

## System Requirements

- **GPUs**: 8x NVIDIA H100 80GB HBM3
- **CUDA Version**: 12.8 (Driver), 12.6 (Toolkit)  
- **Python**: 3.12.7 (in Docker container)
- **Disk Space**: ~25GB+ available for Docker build and data
- **Docker**: With NVIDIA Container Toolkit

---

## 🔥 **ARCHIVED PYTORCH SOLUTION FOR FLEXATTENTION**

### Issue with Latest PyTorch

The latest PyTorch nightly builds (2.9.0.dev20250808+cu126) have a compatibility issue with FlexAttention in distributed training:

```
RuntimeError('Detected that you are using FX to symbolically trace a dynamo-optimized function. This is not supported at the moment.')
```

This prevents the ~10 minute compilation phase from completing and training from starting.

### ✅ **SOLUTION: Use Archived PyTorch Version**

**Working PyTorch Version**: `torch-2.7.0.dev20250208+cu126`

### Step 1: Dockerfile.archived (Already Provided)

The repository includes `Dockerfile.archived` with the working configuration:

```dockerfile
FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.12.7
ENV PATH=/usr/local/bin:$PATH

RUN apt update && apt install -y --no-install-recommends build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev curl git libncursesw5-dev xz-utils tk-dev libxml2-dev \
    libxmlsec1-dev libffi-dev liblzma-dev \
    && apt clean && rm -rf /var/lib/apt/lists/*

RUN curl -O https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar -xzf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    cd .. && \
    rm -rf Python-${PYTHON_VERSION} Python-${PYTHON_VERSION}.tgz

RUN ln -s /usr/local/bin/python3.12 /usr/local/bin/python && \
    ln -s /usr/local/bin/pip3.12 /usr/local/bin/pip

WORKDIR /improve_muon

RUN python -m pip install --upgrade pip

# Install only basic requirements without torch first
RUN pip install tqdm numpy huggingface_hub

# Install specific archived PyTorch version that works with FlexAttention
RUN pip install https://github.com/YouJiacheng/pytorch-nightly-whl-archive/releases/download/v2.7.0.dev20250208/torch-2.7.0.dev20250208+cu126-cp312-cp312-manylinux_2_28_x86_64.whl --no-deps

# Try installing triton from PyPI that's compatible with this torch version
RUN pip install triton==3.2.0 --no-deps || pip install triton --no-deps || echo "Triton installation failed, continuing..."

# Install remaining NVIDIA dependencies
RUN pip install nvidia-cuda-nvrtc-cu12==12.6.77 nvidia-cuda-runtime-cu12==12.6.77 nvidia-cuda-cupti-cu12==12.6.80 nvidia-cudnn-cu12==9.5.1.17 nvidia-cublas-cu12==12.6.4.1 nvidia-cufft-cu12==11.3.0.4 nvidia-curand-cu12==10.3.7.77 nvidia-cusolver-cu12==11.7.1.2 nvidia-cusparse-cu12==12.5.4.2 nvidia-cusparselt-cu12==0.6.3 nvidia-nccl-cu12==2.21.5 nvidia-nvtx-cu12==12.6.77 nvidia-nvjitlink-cu12==12.6.85

# Install remaining torch dependencies
RUN pip install sympy==1.13.1 networkx jinja2 fsspec filelock typing-extensions setuptools

CMD ["bash"]
ENTRYPOINT []
```

### Step 2: Build Archived PyTorch Image

```bash
docker build -f Dockerfile.archived -t nanogpt-archived .
```

**Expected warnings**: You may see pip dependency warnings about `pytorch-triton` version mismatches. These can be ignored as the training still works.

### Expected Build Timeline

- **0-5 minutes**: Base image download and system packages
- **5-15 minutes**: Python compilation from source
- **15-20 minutes**: PyTorch and dependencies installation

### Key Differences from Standard Docker Approach

| Aspect | Standard Dockerfile | Archived PyTorch Dockerfile |
|--------|---------------------|------------------------------|
| PyTorch Version | Latest nightly (2.9.0.dev20250808) | Archived (2.7.0.dev20250208) |
| FlexAttention | ❌ FX tracing error | ✅ Works perfectly |
| Triton | Latest version | Compatible 3.2.0 version |
| Installation | Single step | Multi-step with --no-deps |

### Troubleshooting Docker Build

1. **Build Failures**: If the archived PyTorch wheel download fails, check the GitHub release page for availability
2. **Memory Issues**: Ensure sufficient system RAM during Docker build (needs ~8GB+)
3. **GPU Access**: Verify `nvidia-docker` runtime is properly configured with `docker run --gpus all`
4. **Disk Space**: The archived build requires ~20GB total disk space

---

## 🚀 **RUNNING MINIMAL_MEDIUM.PY**

The `improve_muon` repository contains `minimal_medium.py`, which is an exact reproduction of `train_gpt_medium.py` but refactored into modular components.

### Prerequisites

1. **Training Data**: Download all 103 shards of FineWeb data:
   ```bash
   docker run --gpus all --rm -v $(pwd):/workspace -w /workspace nanogpt-archived \
     python data/cached_fineweb10B.py
   ```

2. **Docker Image**: Use the `nanogpt-archived` image built above.

### ✅ **Running minimal_medium.py**

```bash
docker run --gpus all --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  -e PYTHONPATH=/workspace \
  nanogpt-archived \
  torchrun --standalone --nproc_per_node=8 empirical/research/minimal_medium.py
```

### Key Differences from train_gpt_medium.py

| Aspect | train_gpt_medium.py | minimal_medium.py |
|--------|---------------------|-------------------|
| Code Structure | Monolithic ~600 lines | Modular 12 main lines |
| Components | All in one file | Separated into modules |
| Zeropower Function | Hardcoded Newton-Schulz | Swappable via `get_zeropower_function()` |
| Logging | Inline print statements | Pluggable logger system |
| Python Path | Not needed | Requires `PYTHONPATH=/workspace` |
| Training Data | Direct path access | Uses data/fineweb10B directory |

### Success Indicators

**During Compilation Phase (~10-15 minutes):**
1. **GPU Initialization**: All 8 H100s show ~52GB memory allocation
2. **Distributed Processes**: 8 Python processes active across GPUs
3. **Compilation Phase**: `inductor_output_code` messages appear
4. **No Import Errors**: `empirical` module imports successfully
5. **No FX Tracing Errors**: The archived PyTorch prevents runtime errors

**During Training Phase:**
1. **Memory Usage**: ~52GB per GPU (not 8GB - that indicates a problem)
2. **Training Speed**: ~230-240ms per step
3. **Validation Loss**: Should decrease from ~10.8 toward target of 2.92
4. **Step Logging**: Regular step progress with timing information

### Expected Timeline

- **0-5 minutes**: Environment setup and model initialization
- **5-15 minutes**: Kernel compilation phase (the expensive compilation overhead)
- **15+ minutes**: Actual training begins with step logging
- **25-30 minutes**: Training completes (GPT-2 Medium track target)

### Troubleshooting

**ModuleNotFoundError: No module named 'empirical'**
- **Solution**: Add `-e PYTHONPATH=/workspace` to the docker run command

**Low GPU memory usage (8GB instead of 52GB)**
- **Issue**: Data loader or model configuration problem
- **Check**: Ensure all 103 data shards are downloaded
- **Verify**: Training should use 65K sequence length per GPU

**FX tracing errors during compilation**
- **Issue**: Using wrong PyTorch version
- **Solution**: Ensure using `nanogpt-archived` image with PyTorch 2.7.0.dev20250208

**Docker image not found**
- **Solution**: Build the image first with `docker build -f Dockerfile.archived -t nanogpt-archived .`

### Architecture Benefits

The modular `minimal_medium.py` provides:
- **Swappable Components**: Easy to experiment with different zeropower functions, optimizers, loggers
- **Clean Separation**: Training logic separated from model/data/logging code  
- **Research Friendly**: 12 lines of main logic vs hundreds in monolithic version
- **Same Performance**: Identical training behavior and speed as train_gpt_medium.py

---

## 📊 **Performance Expectations**

### Memory Usage
- **Compilation Phase**: ~52GB per GPU (increases gradually during compilation)
- **Training Phase**: ~52GB per GPU (consistent)
- **Warning Signs**: If usage is <20GB per GPU, check data loading and sequence lengths

### Training Performance
- **Compilation Time**: 10-15 minutes (first run only)
- **Step Time**: ~230-240ms per step
- **Target Loss**: 2.92 for GPT-2 Medium track
- **Total Time**: ~25-30 minutes for complete training run

### Validation Loss Trajectory
Expected progression:
```
step:0/5960 val_loss:10.826
step:125/5960 val_loss:4.325
step:250/5960 val_loss:3.878
step:500/5960 val_loss:3.585
...
step:5960/5960 val_loss:~2.92
```

---

## 🔧 **Alternative Commands**

### Interactive Container
```bash
docker run --gpus all --rm -it \
  -v $(pwd):/workspace \
  -w /workspace \
  -e PYTHONPATH=/workspace \
  nanogpt-archived bash
```

### Monitor Training
```bash
# GPU utilization
nvidia-smi

# Training logs (check most recent file)
tail -f logs/$(ls logs/ | tail -1)

# Check data
ls -la data/fineweb10B/fineweb_train_*.bin | wc -l  # Should be 103
```

### Test PyTorch Installation
```bash
docker run --gpus all --rm nanogpt-archived python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
print('✅ Docker GPU access working!')
"
```

---

## 🏃 **TORCH.COMPILE OPTIMIZATION**

### Persistent Compilation Cache

PyTorch's `torch.compile` creates optimized kernels but recompiles them on every Docker run. To avoid 10+ minute recompilation:

**Mount persistent cache directory:**
```bash
# Create cache directory on host
mkdir -p torch_compile_cache

# Mount cache in Docker runs
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $(pwd):/improve_muon \
  -v $(pwd)/torch_compile_cache:/tmp/torchinductor_root \
  -e PYTHONPATH=/improve_muon \
  nanogpt-archived \
  torchrun --standalone --nproc_per_node=8 script.py
```

**Benefits:**
- **First run**: 10+ minutes compilation (creates cache)
- **Subsequent runs**: ~30 seconds startup (uses cached kernels)
- **Cache location**: `./torch_compile_cache/` persists between Docker runs

**Usage for analysis scripts:**
- Training runs create cached kernels
- Analysis scripts reuse same kernels
- Eliminates redundant FlexAttention compilation

## Critical for Distributed Analysis

**Problem**: Running analysis scripts on the host (without Docker cache mount) causes compilation explosion:
- Each of 8 ranks tries to compile the same kernels simultaneously
- Creates 200+ torch inductor compile workers
- Leads to distributed deadlocks and process explosion
- Makes analysis scripts unusable

**Solution**: Always use Docker with persistent cache mount for analysis:
```bash
# WRONG - runs on host, causes compilation chaos
torchrun --standalone --nproc_per_node=8 analysis_script.py

# CORRECT - uses persistent cache, coordinates compilation properly
docker run --gpus all --rm \
  -v $(pwd):/workspace \
  -v $(pwd)/torch_compile_cache:/tmp/torchinductor_root \
  -w /workspace \
  -e PYTHONPATH=/workspace \
  nanogpt-archived \
  torchrun --standalone --nproc_per_node=8 analysis_script.py
```

**Why this works:**
- All 8 ranks share the same compilation cache directory
- Cached kernels from training are reused immediately
- No redundant compilation across ranks
- Proper file system coordination prevents race conditions