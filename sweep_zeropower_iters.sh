#!/bin/bash

# Zeropower Newton-Schulz Iteration Sweep
# Sweeps MLP iterations from [15, 20, 30, 40] and attention iterations from [15, 20, 30, 40]

echo "Starting zeropower iteration sweep..."
echo "MLP iterations: 15, 20, 30, 40"
echo "Attention iterations: 15, 20, 30, 40"
echo "Total runs: 4 * 4 = 16"
echo ""

# Counter for tracking progress
run_count=0
total_runs=16

# Double for loop: MLP iterations [15, 20, 30, 40], Attention iterations [15, 20, 30, 40]
for mlp_iters in 15 20 30 40; do
    for attn_iters in 15 20 30 40; do
        run_count=$((run_count + 1))
        echo "[$run_count/$total_runs] Running MLP_iters=$mlp_iters, ATTN_iters=$attn_iters"
        
        # Run the training with current hyperparameters
        torchrun --standalone --nproc_per_node=8 -m code.research.zeropower_testing \
            --mlp-method classic_newton_schulz \
            --mlp-hyperparams "{\"num_iters\": $mlp_iters}" \
            --attn-method classic_newton_schulz \
            --attn-hyperparams "{\"num_iters\": $attn_iters}"
        
        # Check if the run failed
        if [ $? -ne 0 ]; then
            echo "ERROR: Run failed for MLP_iters=$mlp_iters, ATTN_iters=$attn_iters"
            echo "Continuing with next configuration..."
        else
            echo "SUCCESS: Completed MLP_iters=$mlp_iters, ATTN_iters=$attn_iters"
        fi
        
        echo ""
    done
done

echo "Sweep completed! Processed $run_count configurations."
echo "Check logs/ directory for results."