#!/bin/bash

# Complete Docker Setup Script for improve_muon on 8xH100 Lambda Labs
# This script replicates the full working setup from scratch

set -e  # Exit on any error

echo "=========================================="
echo "Docker Setup for improve_muon (8xH100)"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

# Check if running as root for Docker installation
check_sudo() {
    if [[ $EUID -eq 0 ]]; then
        warn "Running as root. This is fine for Docker installation but not recommended for training."
    fi
}

# Step 1: Install Docker
install_docker() {
    log "Step 1: Installing Docker..."
    
    # Remove old Docker packages
    sudo apt-get remove -y docker docker-engine docker.io containerd runc 2>/dev/null || true
    
    # Update package index
    sudo apt-get update
    sudo apt-get upgrade -y
    
    # Install required packages
    sudo apt-get install -y ca-certificates curl gnupg lsb-release
    
    # Add Docker GPG key
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg
    
    # Add Docker repository
    echo \
      "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    # Install Docker
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    
    # Start Docker service
    sudo systemctl start docker
    sudo systemctl enable docker
    
    # Fix Docker permissions
    sudo usermod -aG docker $USER
    sudo chmod 666 /var/run/docker.sock
    
    # Verify installation
    docker --version
    docker compose version
    
    log "Docker installation completed successfully!"
}

# Step 2: Verify system requirements
verify_system() {
    log "Step 2: Verifying system requirements..."
    
    # Check disk space (need at least 25GB free)
    DISK_FREE=$(df / | awk 'NR==2 {print int($4/1024/1024)}')
    if [ $DISK_FREE -lt 25 ]; then
        error "Insufficient disk space. Need at least 25GB free, have ${DISK_FREE}GB"
    fi
    log "Disk space check passed: ${DISK_FREE}GB available"
    
    # Check GPU availability
    if ! nvidia-smi > /dev/null 2>&1; then
        error "nvidia-smi not found. NVIDIA drivers not properly installed."
    fi
    
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
    if [ "$GPU_COUNT" != "8" ]; then
        warn "Expected 8 GPUs, found ${GPU_COUNT}. Training may not work as expected."
    else
        log "GPU check passed: 8 H100 GPUs detected"
    fi
    
    # Check if we're in the right directory
    if [ ! -f "Dockerfile.archived" ]; then
        error "Dockerfile.archived not found. Please run this script from the improve_muon directory."
    fi
    log "Directory check passed: Dockerfile.archived found"
}

# Step 3: Build Docker image
build_docker_image() {
    log "Step 3: Building Docker image (this will take 10-15 minutes)..."
    
    # Check if image already exists
    if docker images nanogpt-archived | grep -q nanogpt-archived; then
        warn "Docker image 'nanogpt-archived' already exists. Skipping build."
        log "To rebuild, run: docker rmi nanogpt-archived"
        return 0
    fi
    
    # Build the image
    log "Building Docker image from Dockerfile.archived..."
    docker build -f Dockerfile.archived -t nanogpt-archived .
    
    # Verify the build
    if docker images nanogpt-archived | grep -q nanogpt-archived; then
        log "Docker image built successfully!"
        docker images nanogpt-archived
    else
        error "Docker image build failed"
    fi
}

# Step 4: Test Docker GPU access
test_docker_gpu() {
    log "Step 4: Testing Docker GPU access..."
    
    log "Testing PyTorch and CUDA in Docker container..."
    docker run --gpus all --rm nanogpt-archived python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU 0: {torch.cuda.get_device_name(0)}')
print('✅ Docker GPU access working!')
"
    log "Docker GPU test completed successfully!"
}

# Step 5: Download training data
download_data() {
    log "Step 5: Downloading training data (all 103 shards, ~20GB)..."
    
    # Check if data already exists
    if [ -d "data/fineweb10B" ] && [ $(ls data/fineweb10B/fineweb_train_*.bin 2>/dev/null | wc -l) -gt 100 ]; then
        warn "Training data appears to already exist. Skipping download."
        log "Found $(ls data/fineweb10B/fineweb_train_*.bin 2>/dev/null | wc -l) training files"
        return 0
    fi
    
    log "Downloading all 103 training shards plus validation data..."
    log "This will download ~20GB of data and may take 10-30 minutes depending on connection speed."
    
    # Download data using Docker container
    docker run --gpus all --rm -v $(pwd):/workspace -w /workspace nanogpt-archived \
        python data/cached_fineweb10B.py
    
    # Verify download
    TRAIN_FILES=$(ls data/fineweb10B/fineweb_train_*.bin 2>/dev/null | wc -l)
    VAL_FILES=$(ls data/fineweb10B/fineweb_val_*.bin 2>/dev/null | wc -l)
    
    if [ $TRAIN_FILES -eq 103 ] && [ $VAL_FILES -eq 1 ]; then
        log "Data download completed successfully!"
        log "Training files: ${TRAIN_FILES}"
        log "Validation files: ${VAL_FILES}"
        log "Total data size: $(du -sh data/fineweb10B/ | cut -f1)"
    else
        error "Data download incomplete. Expected 103 training + 1 validation file, got ${TRAIN_FILES} + ${VAL_FILES}"
    fi
}

# Step 6: Run a quick training test
run_training_test() {
    log "Step 6: Running a quick training test (optional)..."
    
    read -p "Do you want to run a quick training test? This will compile kernels (~10 min) and run a few training steps. (y/N): " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "Skipping training test."
        return 0
    fi
    
    log "Starting training test with minimal_medium.py..."
    log "This will take ~10-15 minutes for compilation, then show training progress."
    log "You can Ctrl+C to stop the test once you see training steps running."
    
    # Run training in background so we can monitor
    timeout 1200 docker run --gpus all --rm \
      -v $(pwd):/workspace \
      -w /workspace \
      -e PYTHONPATH=/workspace \
      nanogpt-archived \
      torchrun --standalone --nproc_per_node=8 empirical/research/minimal_medium.py || true
    
    log "Training test completed (or timed out after 20 minutes)."
}

# Step 7: Provide usage instructions
show_usage() {
    log "Step 7: Setup completed! Here's how to use it:"
    
    echo
    echo "=========================================="
    echo "Usage Instructions"
    echo "=========================================="
    echo
    echo "🐳 Docker Image: nanogpt-archived"
    echo "📊 Training Data: data/fineweb10B/ (103 shards, ~20GB)"
    echo "🖥️  GPUs: 8x H100 80GB HBM3"
    echo
    echo "💻 Run training:"
    echo "docker run --gpus all --rm \\"
    echo "  -v \$(pwd):/workspace \\"
    echo "  -w /workspace \\"
    echo "  -e PYTHONPATH=/workspace \\"
    echo "  nanogpt-archived \\"
    echo "  torchrun --standalone --nproc_per_node=8 empirical/research/minimal_medium.py"
    echo
    echo "📈 Monitor GPU usage:"
    echo "nvidia-smi"
    echo
    echo "📋 Check logs:"
    echo "tail -f logs/\$(ls logs/ | tail -1)"
    echo
    echo "🔧 Interactive container:"
    echo "docker run --gpus all --rm -it \\"
    echo "  -v \$(pwd):/workspace \\"
    echo "  -w /workspace \\"
    echo "  -e PYTHONPATH=/workspace \\"
    echo "  nanogpt-archived bash"
    echo
    echo "⚡ Expected performance:"
    echo "- Memory usage: ~52GB per GPU"
    echo "- Compilation time: ~10-15 minutes (first run only)"
    echo "- Training speed: ~230-240ms per step"
    echo "- Target loss: 2.92 for GPT-2 Medium track"
    echo
}

# Main execution
main() {
    log "Starting Docker setup for improve_muon..."
    
    check_sudo
    install_docker
    verify_system
    build_docker_image
    test_docker_gpu
    download_data
    run_training_test
    show_usage
    
    log "🎉 Setup completed successfully!"
    log "You can now run GPT-2 Medium training with Docker on your 8xH100 cluster."
}

# Run main function
main "$@"