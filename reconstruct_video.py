from utils.video_utils import load_and_preprocess_video,save_video_imageio
from modeling.titok import TiTok3D
import torch
from omegaconf import OmegaConf
videofrom = "/home2/jinluo/projects/TiTok3D/datasets/little/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g13_c04.avi"
tensorVid = load_and_preprocess_video(videofrom)
tensorVid = tensorVid.to("cuda:1")
print(tensorVid.shape)
avi_path = "results/temp/origin/avi"
mp4_path = "results/temp/origin/mp4"
save_video_imageio(tensor=tensorVid,avi_dir=avi_path,mp4_dir=mp4_path)    
configPath = "/home2/jinluo/projects/TiTok3D/configs/infer/titok3D_ll32_vae_c16.yaml"
config = OmegaConf.load(configPath)
# 1. 初始化模型架构
model = TiTok3D.from_pretrained("/home2/jinluo/projects/TiTok3D/temp_weights")
# 你需要一个 config，这一步必须有

# 2. 加载权重

# 3. 移动到 GPU
model = model.to("cuda:1")
model.eval()

output,_ = model(tensorVid)
avi_path = "results/temp/out/avi"
mp4_path = "results/temp/out/mp4"
save_video_imageio(output,avi_path,mp4_path)