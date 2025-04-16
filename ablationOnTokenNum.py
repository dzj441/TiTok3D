import os
import torch
from PIL import Image
import numpy as np
from omegaconf import OmegaConf
import imageio
from modeling.titok import TiTok3D

# ============================================
# 1. 定义全局变量：指定 latent 文件、checkpoint 文件和结果保存目录
# ============================================
VQ = False

X_PATH = "vqlatent.pt" if VQ else "vae32LatentArchery.pt"  # latent tensor 保存路径
CKPT_PATH = 'ckpt/tokenizer_titok_l32_imagenet' if VQ else "/home2/jinluo/projects/TiTok3D/temp_weights"  # 模型/ VAE 的 checkpoint 路径
SAVE_DIR = 'results/ablationOnTokenNumVQ' if VQ else 'results/trash'         # 结果保存总目录
os.makedirs(SAVE_DIR, exist_ok=True)

# 为不同实验创建子目录
exp1_dir = os.path.join(SAVE_DIR, "exp_single_token")
exp2_dir = os.path.join(SAVE_DIR, "exp_prefix")
exp3_dir = os.path.join(SAVE_DIR, "exp_suffix")
os.makedirs(exp1_dir, exist_ok=True)
os.makedirs(exp2_dir, exist_ok=True)
os.makedirs(exp3_dir, exist_ok=True)

# ============================================
# 2. 加载模型及 latent 张量
# ============================================
# 加载配置和模型（TiTok3D），用于解码 latent 得到视频 tensor
configPath = "/home2/jinluo/projects/TiTok3D/configs/infer/titok3D_ll32_vae_c16.yaml"
config = OmegaConf.load(configPath)
model = TiTok3D.from_pretrained("/home2/jinluo/projects/TiTok3D/temp_weights")
model = model.to("cuda")
model.eval()
model.requires_grad_(False)

# 加载 latent 张量，这里的 latent 经过 rearrange 后为 [1,16,1,32]，
# 其中 32 表示 token 数量（最后一个维度），16 是 token 的维度
latent = torch.load(X_PATH, map_location=torch.device('cuda'))
# latent = torch.randn(1,16,1,32,device="cuda")

# ============================================
# 3. 定义保存视频的函数 save_video_imageio
# ============================================
def save_video_imageio(
    tensor,
    avi_dir,
    mp4_dir,
    fps=16,
    mean=[0.5, 0.5, 0.5],
    std=[1.0, 1.0, 1.0],
    prefix="video"
):
    """
    将标准化后的视频 tensor 保存为 avi 和 mp4 格式。

    Args:
        tensor: torch.Tensor, shape [B, C, T, H, W]
        avi_dir: str, 保存 avi 视频的文件夹路径。如果为 None则不保存 avi
        mp4_dir: str, 保存 mp4 视频的文件夹路径
        fps: int, 每秒帧数
        mean, std: 用于反标准化
        prefix: 保存视频的命名前缀（如 video_0.mp4）

    Returns:
        None
    """
    if avi_dir is not None:
        os.makedirs(avi_dir, exist_ok=True)
    os.makedirs(mp4_dir, exist_ok=True)

    # 反标准化函数
    def denormalize(tensor, mean, std):
        mean = torch.tensor(mean, device=tensor.device).view(1, 3, 1, 1, 1)
        std = torch.tensor(std, device=tensor.device).view(1, 3, 1, 1, 1)
        return tensor * std + mean

    tensor = denormalize(tensor, mean, std).clamp(0, 1)  # [B, C, T, H, W]
    tensor = (tensor * 255).to(torch.uint8)
    B, C, T, H, W = tensor.shape

    for i in range(B):
        vid = tensor[i]  # [C, T, H, W]
        # 调整为 [T, H, W, C]，imageio 要求帧为最后一维
        vid = vid.permute(1, 2, 3, 0).cpu().numpy()
        # 生成 avi 和 mp4 文件（这里 avi 可选，如不需要可传 None）
        if avi_dir is not None:
            avi_path = os.path.join(avi_dir, f"{prefix}_{i}.avi")
            imageio.mimwrite(avi_path, vid, fps=fps, format="avi")
            print(f"[Saved] {avi_path}")
        mp4_path = os.path.join(mp4_dir, f"{prefix}_{i}.mp4")
        imageio.mimwrite(mp4_path, vid, fps=fps, format="ffmpeg")
        print(f"[Saved] {mp4_path}")

# 获取 token 数量，这里 token 数量为 32，位于 latent 的最后一个维度
num_tokens = latent.shape[-1]

# ============================================
# 4. 实验一：单个 token 实验（视频）
#     对于 32 个 token，每次只保留其中一个 token，其余位置置 0
# ============================================
print("开始实验1：单个 token 保留实验 (视频)")
for token_index in range(num_tokens):
    masked_latent = torch.zeros_like(latent)
    # 只保留对应位置的 token，其余位置为 0
    masked_latent[..., token_index] = latent[..., token_index]
    
    # 解码得到视频 tensor，形状为 [1, 3, 16, 128, 128]
    video = model.decode_tokens(masked_latent)
    
    # 定义当前实验的视频保存子目录（avi 与 mp4 分别保存在子文件夹中）
    avi_dir = os.path.join(exp1_dir, "avi")
    mp4_dir = os.path.join(exp1_dir, "mp4")
    prefix = f"only_token_{token_index:02d}"
    save_video_imageio(video, None, mp4_dir, prefix=prefix)
    print(f"保存实验1结果：{prefix}")

# ============================================
# 5. 实验二：前缀保留实验（视频）
#     从保留前 0 个到保留前 32 个 token，共 33 种情况：
#     若保留前 0 个，则所有 token 置 0；若保留前 i 个，则从索引 i 开始置 0
# ============================================
print("开始实验2：前缀保留实验 (视频)")
for num_keep in range(0, num_tokens + 1):
    masked_latent = latent.clone()
    # 保留前 num_keep 个 token，后续 token 置 0
    masked_latent[..., num_keep:] = 0
    video = model.decode_tokens(masked_latent)
    
    avi_dir = os.path.join(exp2_dir, "avi")
    mp4_dir = os.path.join(exp2_dir, "mp4")
    prefix = f"keep_first_{num_keep:02d}"
    save_video_imageio(video, None, mp4_dir, prefix=prefix)
    print(f"保存实验2结果（保留前 {num_keep} 个 token）：{prefix}")

# ============================================
# 6. 实验三：后缀保留实验（视频）
#     从保留后 0 个到保留后 32 个 token，共 33 种情况：
#     若保留后 0 个，则所有 token 置 0；若保留后 i 个，则将前面 token 置 0，仅保留最后 i 个
# ============================================
print("开始实验3：后缀保留实验 (视频)")
for num_keep in range(0, num_tokens + 1):
    masked_latent = latent.clone()
    # 将前面的 token 置 0，只保留最后 num_keep 个 token
    masked_latent[..., :num_tokens - num_keep] = 0
    video = model.decode_tokens(masked_latent)
    
    avi_dir = os.path.join(exp3_dir, "avi")
    mp4_dir = os.path.join(exp3_dir, "mp4")
    prefix = f"keep_last_{num_keep:02d}"
    save_video_imageio(video, None, mp4_dir, prefix=prefix)
    print(f"保存实验3结果（保留后 {num_keep} 个 token）：{prefix}")

print("所有实验结果均已保存！")
