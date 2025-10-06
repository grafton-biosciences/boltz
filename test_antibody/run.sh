#!/bin/bash

boltz predict antibody.yaml --model boltz2_pc --output_format pdb --atomic_affinity --cache=~/.boltz --override --no_kernels
