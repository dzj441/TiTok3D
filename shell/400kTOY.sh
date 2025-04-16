WANDB_MODE=online accelerate launch --num_machines=1 --num_processes=2 --machine_rank=0 scripts/train_titok3D.py config=configs/training/titok3D_ll32_vae_toy.yaml \
    experiment.project="titok3D_ll32_vae_toyUCF100VID_32_400k" \
    experiment.name="titok3D_ll32_vae_toyUCF100VID_32_400k_run1" \
    experiment.output_dir="titok3D_ll32_vae_toyUCF100VID_32_400k_run1" \

