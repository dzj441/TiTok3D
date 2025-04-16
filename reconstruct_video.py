from utils.video_utils import load_and_preprocess_video,save_video_imageio
from modeling.titok import TiTok3D
import torch
from omegaconf import OmegaConf
videofrom = "/home2/jinluo/projects/TiTok3D/datasets/TOY/train/Archery/v_Archery_g18_c05.avi"
tensorVid = load_and_preprocess_video(videofrom)
tensorVid = tensorVid.to("cuda")
print(tensorVid.shape)
avi_path = "results/temp/origin/avi"
mp4_path = "results/temp/origin/mp4"
save_video_imageio(tensor=tensorVid,avi_dir=None,mp4_dir=mp4_path,fps=4)    
configPath = "/home2/jinluo/projects/TiTok3D/configs/infer/titok3D_ll32_vae_c16.yaml"
config = OmegaConf.load(configPath)
# 1. 初始化模型架构
model = TiTok3D.from_pretrained("/home2/jinluo/projects/TiTok3D/temp_weights")
# 你需要一个 config，这一步必须有

# 2. 加载权重

# 3. 移动到 GPU
model = model.to("cuda")
model.eval()

encoded,posterior = model.encode(tensorVid)
print(encoded.shape)

encoded_tokens_cpu = encoded.cpu()
torch.save(encoded_tokens_cpu,"vae32LatentArchery"+".pt")
print(f"[Saved] encoded token")

output = model.decode(encoded)
avi_path = "results/temp/out/avi"
mp4_path = "results/temp/out/mp4"
print(output.shape)
save_video_imageio(output,None,mp4_path,fps=4)