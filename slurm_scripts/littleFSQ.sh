# rm -rf ./wandb/
rm -rf ./titok3D_ll32_vae_toy_littleFSQ
WANDB_MODE=offline accelerate launch --num_machines=1 --num_processes=1 --machine_rank=0 scripts/train_titok3D.py config=configs/training/titok3D_ll32_fsq.yaml \
    experiment.project="titok3D_ll32_vae_toy_littleFSQ" \
    experiment.name="titok3D_ll32_vae_toy_littleFSQ" \
    experiment.output_dir="titok3D_ll32_vae_toy_littleFSQ" \
