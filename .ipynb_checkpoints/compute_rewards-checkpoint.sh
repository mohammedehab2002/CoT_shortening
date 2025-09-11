#!/bin/bash

# Job Flags
#SBATCH -p mit_normal_gpu --gres=gpu:h200:1 -t 6:00:00

module load miniforge/24.3.0-0

python label_rewards.py
