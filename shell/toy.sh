# rm -rf ./wandb/
rm -rf ./titok3D_ll32_vae_toy_run1/
WANDB_MODE=offline accelerate launch --num_machines=1 --num_processes=1 --machine_rank=0 scripts/train_titok3D.py config=configs/training/titok3D_ll32_vae_toy.yaml \
    experiment.project="titok3D_ll32_vae_toy_larger_latent_size2" \
    experiment.name="titok3D_ll32_vae_toy_larger_latent_size2" \
    experiment.output_dir="titok3D_ll32_vae_toy_larger_latent_size2" \