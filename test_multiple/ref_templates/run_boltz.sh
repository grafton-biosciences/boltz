#!/bin/bash
#SBATCH --job-name=run_boltz_protes
#SBATCH -p g6xl-ubuntu
#SBATCH --gres=gpu:1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --time=7-00:00:00



system=$1
#vis_device=$(( $system % 8 ))
#export CUDA_VISIBLE_DEVICES=${vis_device}


cwd=$( pwd )
export HOME=/home/ubuntu
source ~/.bashrc

#source /shared/softwares/boltz_nucleotic/boltz_n-env/bin/activate
cd ${cwd}



start_ns=$(date +%s%N)   # GNU date; macOS needs coreutils `gdate`
/shared/softwares/boltz_nucleotic/boltz_n-env_prod/bin/boltz predict target_0.yaml --use_msa_server --model boltz2_ensemble --max_ensemble_size 20 --atomic_affinity --max_parallel_samples 1  --out_dir ./ --recycling_steps 3 --diffusion_samples 10 --cache=/shared/.boltz --output_format pdb
end_ns=$(date +%s%N)
echo "Elapsed s: $(( (end_ns - start_ns)/1000000000 ))"

best_ind=$( grep best_iptm_idx boltz_results_target_0/predictions/target_0/confidence_target_0_model_0.json | grep -o '[0-9]\+' )

cp boltz_results_target_0/predictions/target_0/confidence_target_0_model_${best_ind}.json confidence_target_0_model_0.json
cp boltz_results_target_0/predictions/target_0/target_0_model_${best_ind}.pdb target_0.pdb
