#!/bin/bash

cwd=$( pwd )
export HOME=/home/ubuntu
source ~/.bashrc

cd ${cwd}

/shared/softwares/boltz_nucleotic/boltz_n-env/bin/boltz predict antibody.yaml --model boltz2_pc --output_format pdb --atomic_affinity --cache=/shared/.boltz --override
