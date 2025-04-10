WANDB_MODE=online accelerate launch --num_machines=1 --num_processes=2 --machine_rank=0 scripts/train_titok3D.py config=configs/training/titok3D_ll32_fsq.yaml \
    experiment.project="aaa_titok3D_ll32_vae_toy_littleFSQ" \
    experiment.name="aaa_titok3D_ll32_vae_toy_littleFSQ" \
    experiment.output_dir="aaa_titok3D_ll32_vae_toy_littleFSQ" \

nvidia-smi

WANDB_MODE=online accelerate launch --num_machines=1 --num_processes=2 --machine_rank=0 scripts/train_titok3D.py config=configs/training/titok3D_ll32_vae_little_128tokens.yaml \
    experiment.project="aaa_titok3D_ll32_vae_toy_little_128token" \
    experiment.name="aaa_titok3D_ll32_vae_toy_little_128token" \
    experiment.output_dir="aaa_titok3D_ll32_vae_toy_little_128token" \

nvidia-smi

WANDB_MODE=online accelerate launch --num_machines=1 --num_processes=2 --machine_rank=0 scripts/train_titok3D.py config=configs/training/titok3D_ll32_vae_little_largerTokenDim.yaml \
    experiment.project="aaa_titok3D_ll32_vae_toy_little_largerTokenDim" \
    experiment.name="aaa_titok3D_ll32_vae_toy_little_largerTokenDim" \
    experiment.output_dir="aaa_itok3D_ll32_vae_toy_little_largerTokenDim" \