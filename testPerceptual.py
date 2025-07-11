import torch
import torch.nn.functional as F
from pytorchvideo.models.hub import i3d_r50
from torchvision.models.feature_extraction import create_feature_extractor
from modeling.modules.perceptual_loss import PerceptualLoss
from utils.video_utils import load_and_preprocess_video,save_video_imageio

def resize_video_spatial(video: torch.Tensor, new_size: tuple) -> torch.Tensor:
    B, C, T, H, W = video.shape
    flat = video.reshape(B * T, C, H, W)
    flat_resized = F.interpolate(flat, size=new_size, mode="bilinear", align_corners=False)
    new_H, new_W = new_size
    return flat_resized.reshape(B, T, C, new_H, new_W).permute(0, 2, 1, 3, 4)

def refined_3d_perceptual_loss(
    model3d,
    vid_pred: torch.Tensor,
    vid_gt: torch.Tensor,
    return_nodes: dict = None,
    weights: dict = None,
) -> torch.Tensor:
    """
    基于 I3D 中间层特征的 3D Perceptual Loss。

    Args:
        model3d: 已加载并冻结参数的 i3d_r50 模型
        vid_pred / vid_gt: (B, C, T, H, W) 已 resize 到 224x224
        return_nodes: 要提取的中间层映射，默认取 "blocks.0","blocks.2","blocks.4"
        weights: 对应特征层的加权系数字典

    Returns:
        一个标量 Tensor，refined 3D perceptual loss
    """
    # 默认要提取的中间层
    if return_nodes is None:
        return_nodes = {
            "blocks.0": "stem_feat",   # 低层纹理
            "blocks.2": "mid_feat",    # 中层运动
            "blocks.4": "deep_feat",   # 高层语义
        }
    # 默认各层权重
    if weights is None:
        weights = {
            "stem_feat": 0.5,
            "mid_feat": 1.0,
            "deep_feat": 1.5,
        }

    # 创建特征提取器
    feat_extractor = create_feature_extractor(model3d, return_nodes=return_nodes)
    # 前向提取
    feats_p = feat_extractor(vid_pred)
    feats_g = feat_extractor(vid_gt)

    # 加权 MSE 叠加
    loss = 0.0
    total_w = 0.0
    for name, feat_p in feats_p.items():
        feat_g = feats_g[name]
        w = weights.get(name, 1.0)
        loss += w * F.mse_loss(feat_p, feat_g, reduction="mean")
        total_w += w

    # 归一化
    return loss / total_w


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    video_path = "datasets/UCF-101/leastOverfit_test/ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c02.avi"

    # (1) 加载并预处理真实视频
    # 假设你希望序列长度 T=8，空间分辨率 H=W=64
    video = load_and_preprocess_video(
        video_path,
        sequence_length=16,
        resolution=128,
        fps=16,
        strategy="center"
    ).to(device)  # 得到 shape [1, C, T, H, W]
    B, C, T, H, W = video.shape

    # (2) 构造“静止”视频：取首帧重复
    first_frame  = video[:, :, 0:1, :, :]           # [1, C, 1, H, W]
    static_video = first_frame.repeat(1, 1, T, 1, 1)  # [1, C, T, H, W]


    # (3) 计算 MSE Loss
    mse = F.mse_loss(video, static_video, reduction='mean').item()

    # (4) 计算 2D Perceptual Loss
    flat_gt     = video.reshape(B*T, C, H, W)
    flat_static = static_video.reshape(B*T, C, H, W)
    PL = PerceptualLoss("lpips-convnext_s-1.0-0.1").eval().to(device)

    p2d = PL(flat_gt, flat_static).mean().item()

    # (5) 计算“粗糙版”3D Perceptual Loss：使用 logits
    vid224    = resize_video_spatial(video,     (224, 224)).to(device)
    static224 = resize_video_spatial(static_video, (224, 224)).to(device)
    model3d = i3d_r50(pretrained=True).eval().to(device)

    for p in model3d.parameters():
        p.requires_grad = False
    out_p = model3d(vid224)
    out_g = model3d(static224)

    p3d = F.mse_loss(out_p, out_g, reduction='mean').item()

    # (6) 计算 refined 3D Perceptual Loss：提取中间层特征


    refined_p3d = refined_3d_perceptual_loss(model3d, vid224, static224).item()

    # 打印所有结果
    print(f"MSE loss                = {mse:.4f}")
    print(f"2D Perceptual loss      = {p2d:.4f}")
    print(f"3D Perceptual loss      = {p3d:.4f}")
    print(f"Refined 3D Perceptual loss = {refined_p3d:.4f}")

    # Save videos
    # save_video_imageio(
    #     video, 
    #     avi_dir=None, 
    #     mp4_dir="./temp", 
    #     fps=16, 
    #     mean=[0.0,0.0,0.0], 
    #     std=[1.0,1.0,1.0], 
    #     prefix="orig"
    # )
    # save_video_imageio(
    #     static_video, 
    #     avi_dir=None, 
    #     mp4_dir="./temp", 
    #     fps=16, 
    #     mean=[0.0,0.0,0.0], 
    #     std=[1.0,1.0,1.0], 
    #     prefix="static"
    # )

# i3d
'''
Net(
  (blocks): ModuleList(
    (0): ResNetBasicStem(
      (conv): Conv3d(3, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)
      (norm): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (activation): ReLU()
      (pool): MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=[0, 1, 1], dilation=1, ceil_mode=False)
    )
    (1): ResStage(
      (res_blocks): ModuleList(
        (0): ResBlock(
          (branch1_conv): Conv3d(64, 256, kernel_size=(1, 1, 1), stride=(np.int64(1), np.int64(1), np.int64(1)), bias=False)
          (branch1_norm): BatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (branch2): BottleneckBlock(
            (conv_a): Conv3d(64, 64, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False)
            (norm_a): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_a): ReLU()
            (conv_b): Conv3d(64, 64, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False)
            (norm_b): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_b): ReLU()
            (conv_c): Conv3d(64, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            (norm_c): BatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (activation): ReLU()
        )
        (1-2): 2 x ResBlock(
          (branch2): BottleneckBlock(
            (conv_a): Conv3d(256, 64, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False)
            (norm_a): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_a): ReLU()
            (conv_b): Conv3d(64, 64, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False)
            (norm_b): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_b): ReLU()
            (conv_c): Conv3d(64, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            (norm_c): BatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (activation): ReLU()
        )
      )
    )
    (2): MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0), dilation=1, ceil_mode=False)
    (3): ResStage(
      (res_blocks): ModuleList(
        (0): ResBlock(
          (branch1_conv): Conv3d(256, 512, kernel_size=(1, 1, 1), stride=(np.int64(1), np.int64(2), np.int64(2)), bias=False)
          (branch1_norm): BatchNorm3d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (branch2): BottleneckBlock(
            (conv_a): Conv3d(256, 128, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False)
            (norm_a): BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_a): ReLU()
            (conv_b): Conv3d(128, 128, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False)
            (norm_b): BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_b): ReLU()
            (conv_c): Conv3d(128, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            (norm_c): BatchNorm3d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (activation): ReLU()
        )
        (1): ResBlock(
          (branch2): BottleneckBlock(
            (conv_a): Conv3d(512, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            (norm_a): BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_a): ReLU()
            (conv_b): Conv3d(128, 128, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False)
            (norm_b): BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_b): ReLU()
            (conv_c): Conv3d(128, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            (norm_c): BatchNorm3d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (activation): ReLU()
        )
        (2): ResBlock(
          (branch2): BottleneckBlock(
            (conv_a): Conv3d(512, 128, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False)
            (norm_a): BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_a): ReLU()
            (conv_b): Conv3d(128, 128, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False)
            (norm_b): BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_b): ReLU()
            (conv_c): Conv3d(128, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            (norm_c): BatchNorm3d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (activation): ReLU()
        )
        (3): ResBlock(
          (branch2): BottleneckBlock(
            (conv_a): Conv3d(512, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            (norm_a): BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_a): ReLU()
            (conv_b): Conv3d(128, 128, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False)
            (norm_b): BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_b): ReLU()
            (conv_c): Conv3d(128, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            (norm_c): BatchNorm3d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (activation): ReLU()
        )
      )
    )
    (4): ResStage(
      (res_blocks): ModuleList(
        (0): ResBlock(
          (branch1_conv): Conv3d(512, 1024, kernel_size=(1, 1, 1), stride=(np.int64(1), np.int64(2), np.int64(2)), bias=False)
          (branch1_norm): BatchNorm3d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (branch2): BottleneckBlock(
            (conv_a): Conv3d(512, 256, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False)
            (norm_a): BatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_a): ReLU()
            (conv_b): Conv3d(256, 256, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False)
            (norm_b): BatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_b): ReLU()
            (conv_c): Conv3d(256, 1024, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            (norm_c): BatchNorm3d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (activation): ReLU()
        )
        (1): ResBlock(
          (branch2): BottleneckBlock(
            (conv_a): Conv3d(1024, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            (norm_a): BatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_a): ReLU()
            (conv_b): Conv3d(256, 256, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False)
            (norm_b): BatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_b): ReLU()
            (conv_c): Conv3d(256, 1024, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            (norm_c): BatchNorm3d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (activation): ReLU()
        )
        (2): ResBlock(
          (branch2): BottleneckBlock(
            (conv_a): Conv3d(1024, 256, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False)
            (norm_a): BatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_a): ReLU()
            (conv_b): Conv3d(256, 256, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False)
            (norm_b): BatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_b): ReLU()
            (conv_c): Conv3d(256, 1024, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            (norm_c): BatchNorm3d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (activation): ReLU()
        )
        (3): ResBlock(
          (branch2): BottleneckBlock(
            (conv_a): Conv3d(1024, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            (norm_a): BatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_a): ReLU()
            (conv_b): Conv3d(256, 256, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False)
            (norm_b): BatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_b): ReLU()
            (conv_c): Conv3d(256, 1024, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            (norm_c): BatchNorm3d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (activation): ReLU()
        )
        (4): ResBlock(
          (branch2): BottleneckBlock(
            (conv_a): Conv3d(1024, 256, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False)
            (norm_a): BatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_a): ReLU()
            (conv_b): Conv3d(256, 256, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False)
            (norm_b): BatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_b): ReLU()
            (conv_c): Conv3d(256, 1024, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            (norm_c): BatchNorm3d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (activation): ReLU()
        )
        (5): ResBlock(
          (branch2): BottleneckBlock(
            (conv_a): Conv3d(1024, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            (norm_a): BatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_a): ReLU()
            (conv_b): Conv3d(256, 256, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False)
            (norm_b): BatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_b): ReLU()
            (conv_c): Conv3d(256, 1024, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            (norm_c): BatchNorm3d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (activation): ReLU()
        )
      )
    )
    (5): ResStage(
      (res_blocks): ModuleList(
        (0): ResBlock(
          (branch1_conv): Conv3d(1024, 2048, kernel_size=(1, 1, 1), stride=(np.int64(1), np.int64(2), np.int64(2)), bias=False)
          (branch1_norm): BatchNorm3d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (branch2): BottleneckBlock(
            (conv_a): Conv3d(1024, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            (norm_a): BatchNorm3d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_a): ReLU()
            (conv_b): Conv3d(512, 512, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False)
            (norm_b): BatchNorm3d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_b): ReLU()
            (conv_c): Conv3d(512, 2048, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            (norm_c): BatchNorm3d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (activation): ReLU()
        )
        (1): ResBlock(
          (branch2): BottleneckBlock(
            (conv_a): Conv3d(2048, 512, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False)
            (norm_a): BatchNorm3d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_a): ReLU()
            (conv_b): Conv3d(512, 512, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False)
            (norm_b): BatchNorm3d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_b): ReLU()
            (conv_c): Conv3d(512, 2048, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            (norm_c): BatchNorm3d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (activation): ReLU()
        )
        (2): ResBlock(
          (branch2): BottleneckBlock(
            (conv_a): Conv3d(2048, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            (norm_a): BatchNorm3d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_a): ReLU()
            (conv_b): Conv3d(512, 512, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False)
            (norm_b): BatchNorm3d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act_b): ReLU()
            (conv_c): Conv3d(512, 2048, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            (norm_c): BatchNorm3d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (activation): ReLU()
        )
      )
    )
    (6): ResNetBasicHead(
      (pool): AvgPool3d(kernel_size=(4, 7, 7), stride=(1, 1, 1), padding=(0, 0, 0))
      (dropout): Dropout(p=0.5, inplace=False)
      (proj): Linear(in_features=2048, out_features=400, bias=True)
      (output_pool): AdaptiveAvgPool3d(output_size=1)
    )
  )
)
'''