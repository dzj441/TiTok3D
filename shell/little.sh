# rm -rf ./wandb/
rm -rf ./titok3D_ll32_vae_toy_little1
WANDB_MODE=offline accelerate launch --num_machines=1 --num_processes=2 --machine_rank=0 scripts/train_titok3D.py config=configs/training/titok3D_ll32_vae_little.yaml \
    experiment.project="ddditok3D_ll32_vae_toy_little_testDicrim" \
    experiment.name="ddditok3D_ll32_vae_toy_little_testDicrim1" \
    experiment.output_dir="ddditok3D_ll32_vae_toy_little_testDicrim1" \
