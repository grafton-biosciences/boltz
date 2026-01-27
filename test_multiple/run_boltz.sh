#!/bin/bash

# Set CUDA environment for Triton
export CUDA_HOME=/usr/local/cuda-13.0
export CPATH=$CUDA_HOME/targets/sbsa-linux/include:$CPATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

/home/vladygin/miniconda3/envs/boltz-env/bin/boltz predict yamls1 --model boltz2_pc --parallel_processes 1 --batch_size 1  --atomic_affinity --diffusion_samples 3 --recycling_steps 3 --max_parallel_samples 3  --out_dir ./ --output_format pdb
