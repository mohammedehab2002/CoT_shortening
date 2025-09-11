#!/bin/bash

# Job Flags
#SBATCH -p sched_mit_sloan_gpu_r8 --gres=gpu:a100:4 -t 24:00:00

module load miniforge/24.3.0-0

python distributed_label_rewards_random.py
