#!/bin/bash

# Parameters

#SBATCH --job-name=wholesettraining
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
#SBATCH -w fvl14

nvidia-smi

echo "start training"

WANDB_MODE=online accelerate launch --num_machines=1 --num_processes=1 --machine_rank=0 scripts/train_titok3D.py config=configs/training/titok3D_bl128_vae.yaml \
    experiment.project="titok3D_bl128_vae_UCF_100k" \
    experiment.name="titok3D_bl128_vae_UCF_load_100k_1e-4" \
    experiment.output_dir="/root/autodl-tmp/titok3D_bl128_UCF_load_100k_1e-4" \


