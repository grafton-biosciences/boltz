#!/bin/bash



batch=100
#source /shared/miniconda3/bin/activate boltz_vlad
iter=$(( ${system} / ${batch} ))


start_ns=$(date +%s%N)   # GNU date; macOS needs coreutils `gdate`
/global/common/software/m2129/conda/boltz-env/bin/boltz predict 6VJA.yaml --use_msa_server --model boltz2_pc --atomic_affinity --process_yaml --max_parallel_samples 1  --out_dir ./ --diffusion_samples 1 --cache=/pscratch/sd/v/vladygin/side_projects/ML_coding_series/Boltz-tests/.boltz --output_format pdb
end_ns=$(date +%s%N)
echo "Elapsed s: $(( (end_ns - start_ns)/1000000000 ))"
