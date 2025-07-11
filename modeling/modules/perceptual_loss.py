"""This file contains perceptual loss module using LPIPS and ConvNeXt-S.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 
"""

import torch
import torch.nn.functional as F

from torchvision import models
from .lpips import LPIPS

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

 
class PerceptualLoss(torch.nn.Module):
    def __init__(self, model_name: str = "convnext_s"):
        """Initializes the PerceptualLoss class.

        Args:
            model_name: A string, the name of the perceptual loss model to use.

        Raise:
            ValueError: If the model_name does not contain "lpips" or "convnext_s".
        """
        super().__init__()
        if ("lpips" not in model_name) and (
            "convnext_s" not in model_name):
            raise ValueError(f"Unsupported Perceptual Loss model name {model_name}")
        self.lpips = None
        self.convnext = None
        self.loss_weight_lpips = None
        self.loss_weight_convnext = None

        # Parsing the model name. We support name formatted in
        # "lpips-convnext_s-{float_number}-{float_number}", where the 
        # {float_number} refers to the loss weight for each component.
        # E.g., lpips-convnext_s-1.0-2.0 refers to compute the perceptual loss
        # using both the convnext_s and lpips, and average the final loss with
        # (1.0 * loss(lpips) + 2.0 * loss(convnext_s)) / (1.0 + 2.0).
        if "lpips" in model_name:
            self.lpips = LPIPS().eval()

        if "convnext_s" in model_name:
            self.convnext = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1).eval()

        if "lpips" in model_name and "convnext_s" in model_name:
            loss_config = model_name.split('-')[-2:]
            self.loss_weight_lpips, self.loss_weight_convnext = float(loss_config[0]), float(loss_config[1])
            print(f"self.loss_weight_lpips, self.loss_weight_convnext: {self.loss_weight_lpips}, {self.loss_weight_convnext}")

        self.register_buffer("imagenet_mean", torch.Tensor(_IMAGENET_MEAN)[None, :, None, None])
        self.register_buffer("imagenet_std", torch.Tensor(_IMAGENET_STD)[None, :, None, None])

        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """Computes the perceptual loss.

        Args:
            input: A tensor of shape (B, C, H, W), the input image. Normalized to [0, 1].
            target: A tensor of shape (B, C, H, W), the target image. Normalized to [0, 1].

        Returns:
            A scalar tensor, the perceptual loss.
        """
        # Always in eval mode.
        self.eval()
        loss = 0.
        num_losses = 0.
        lpips_loss = 0.
        convnext_loss = 0.
        # Computes LPIPS loss, if available.
        if self.lpips is not None:
            lpips_loss = self.lpips(input, target)
            if self.loss_weight_lpips is None:
                loss += lpips_loss
                num_losses += 1
            else:
                num_losses += self.loss_weight_lpips
                loss += self.loss_weight_lpips * lpips_loss

        if self.convnext is not None:
            # Computes ConvNeXt-s loss, if available.
            input = torch.nn.functional.interpolate(input, size=224, mode="bilinear", align_corners=False, antialias=True)
            target = torch.nn.functional.interpolate(target, size=224, mode="bilinear", align_corners=False, antialias=True)
            pred_input = self.convnext((input - self.imagenet_mean) / self.imagenet_std)
            pred_target = self.convnext((target - self.imagenet_mean) / self.imagenet_std)
            convnext_loss = torch.nn.functional.mse_loss(
                pred_input,
                pred_target,
                reduction="mean")
                
            if self.loss_weight_convnext is None:
                num_losses += 1
                loss += convnext_loss
            else:
                num_losses += self.loss_weight_convnext
                loss += self.loss_weight_convnext * convnext_loss
        
        # weighted avg.
        loss = loss / num_losses
        return loss

from torch import nn
from pytorchvideo.models.hub import i3d_r50
from torchvision.models.feature_extraction import create_feature_extractor

class PerceptualLoss3D(nn.Module):
    """
    3D 感知损失，基于 I3D 中间层时空特征。
    输入和目标均为 shape (B, C, T, H, W)，像素归一化至 [0,1]。
    """

    def __init__(
        self,
        return_nodes: dict = None,
        layer_weights: dict = None,
        resize_shape: tuple = (224, 224),
    ):
        """
        Args:
            return_nodes: 指定 I3D 要挂载的中间层，形如
                          {"blocks.0": "stem", "blocks.2": "mid", "blocks.4": "deep"}
            layer_weights: 各层的加权系数，key 与 return_nodes 的 value 对齐
            resize_shape: 空间插值到 (H, W)，I3D 预训练时为 224×224
        """
        super().__init__()

        # 默认要提取的中间层
        if return_nodes is None:
            return_nodes = {
                "blocks.0": "stem_feat",
                "blocks.2": "mid_feat",
                "blocks.4": "deep_feat",
            }
        # 默认各层权重
        if layer_weights is None:
            layer_weights = {
                "stem_feat": 0.5,
                "mid_feat": 1.0,
                "deep_feat": 1.5,
            }

        self.return_nodes = return_nodes
        self.layer_weights = layer_weights
        self.resize_shape = resize_shape  # (H, W)

        # 加载并冻结 I3D R50
        model3d = i3d_r50(pretrained=True).eval()
        for p in model3d.parameters():
            p.requires_grad = False

        # 挂载中间层特征提取器
        self.feat_extractor = create_feature_extractor(
            model3d, return_nodes=self.return_nodes
        )

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input, target: Tensor, shape (B, C, T, H, W), values in [0,1]
        Returns:
            标量 Tensor：加权后的 3D perceptual loss
        """
        # 1) 空间 resize 到模型要求的 resolution
        B, C, T, H, W = input.shape
        flat_in  = input.reshape(B * T, C, H, W)
        flat_tg  = target.reshape(B * T, C, H, W)
        flat_in  = F.interpolate(
            flat_in, size=self.resize_shape,
            mode="bilinear", align_corners=False, antialias=True
        )
        flat_tg  = F.interpolate(
            flat_tg, size=self.resize_shape,
            mode="bilinear", align_corners=False, antialias=True
        )
        Hn, Wn = self.resize_shape
        vid_in  = flat_in.reshape(B, T, C, Hn, Wn).permute(0, 2, 1, 3, 4)
        vid_tg  = flat_tg.reshape(B, T, C, Hn, Wn).permute(0, 2, 1, 3, 4)

        # 2) 提取时空特征
        feats_in = self.feat_extractor(vid_in)
        feats_tg = self.feat_extractor(vid_tg)

        # 3) 加权 MSE 叠加
        loss = 0.0
        total_w = 0.0
        for name, feat_in in feats_in.items():
            feat_tgt = feats_tg[name]
            w = self.layer_weights.get(name, 1.0)
            loss += w * F.mse_loss(feat_in, feat_tgt, reduction="mean")
            total_w += w

        # 4) 归一化
        return loss / total_w
