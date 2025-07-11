from utils.video_utils import load_and_preprocess_video,save_video_imageio
from modeling.titok import TiTok3D
import torch
from omegaconf import OmegaConf
videofrom = "datasets/UCF-101/leastOverfit_train/ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.avi"
tensorVid = load_and_preprocess_video(videofrom)
tensorVid = tensorVid.to("cuda")
print(tensorVid.shape)
# avi_path = "results/temp/origin/avi"
# mp4_path = "results/temp/origin/mp4"
# save_video_imageio(tensor=tensorVid,avi_dir=None,mp4_dir=mp4_path,fps=4)    
configPath = "configs/training/titok3D_bl128_vae_overfit.yaml"
config = OmegaConf.load(configPath)
# 1. 初始化模型架构
model = TiTok3D(config=config)
model = model.to("cuda")
device = "cuda"
from safetensors.torch import load_file
state_dict = load_file("titok3D_bl128_noST_UCF_mseOnly_mse_20k/checkpoint-20000/model.safetensors",device=device)
model.load_state_dict(state_dict,strict=False)
# 3. 移动到 GPU
model.eval()

encoded,posterior = model.encode(tensorVid)
# print(encoded.shape)

# encoded_tokens_cpu = encoded.cpu()
# torch.save(encoded_tokens_cpu,"vae32LatentArchery"+".pt")
# print(f"[Saved] encoded token")

output = model.decode(encoded)
avi_path = "results/temp/out/avi"
mp4_path = "results/temp/out/mp4"
print(output.shape)
save_video_imageio(output,None,mp4_path,fps=4)