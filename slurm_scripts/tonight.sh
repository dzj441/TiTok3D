#!/bin/bash

# Parameters

#SBATCH --job-name=train
#SBATCH -n 1                # Number of cores
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH --cpus-per-task=32
#SBATCH -t 48:00:00
#SBATCH -p fvl
#SBATCH --qos high
#SBATCH --gres=gpu:8
#SBATCH --mem=256G
#SBATCH -o slurm_output/%j.out
#SBATCH -e slurm_output/%j.err
#SBATCH -w fvl10

nvidia-smi

echo "start training"

WANDB_MODE=online accelerate launch --num_machines=1 --num_processes=8 --machine_rank=0 scripts/train_titok3D.py config=configs/training/titok3D_ll32_vae.yaml \
    experiment.project="titok3D_ll32_vae_new_128" \
    experiment.name="titok3D_ll32_vae_new_128_run1" \
    experiment.output_dir="titok3D_ll32_vae_new_128_run1" \


