"""Building blocks for TiTok.

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

Reference: 
    https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/transformer.py
    https://github.com/baofff/U-ViT/blob/main/libs/timm.py
"""

import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from collections import OrderedDict
import einops
from einops.layers.torch import Rearrange
from einops import rearrange

import re
from typing import Dict

def modulate(x, shift, scale):
    return x * (1 + scale) + shift



class BaseResidualAttnBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        act_layer = nn.GELU,
        norm_layer = nn.LayerNorm, 
        trainable: bool = True
    ):
        super().__init__()
        # Attention 
        self.ln1  = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)  # batch_first=False

        # optional MLP
        self.mlp_ratio = mlp_ratio
        if mlp_ratio > 0:
            self.ln2 = norm_layer(d_model)
            hidden = int(d_model * mlp_ratio)
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc",   nn.Linear(d_model, hidden)),
                ("act",    act_layer()),
                ("c_proj", nn.Linear(hidden, d_model))
            ]))

        # DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # trainable
        self.set_trainable(trainable)

    def set_trainable(self, trainable: bool):
        for p in self.parameters():
            p.requires_grad = trainable

    def load_weights(self, state_dict: dict, prefix: str):
        # ln1
        self.ln1.weight.data.copy_( state_dict[f"{prefix}.ln_1.weight"] )
        self.ln1.bias.data  .copy_( state_dict[f"{prefix}.ln_1.bias"]   )
        # attn
        self.attn.in_proj_weight.data.copy_( state_dict[f"{prefix}.attn.in_proj_weight"] )
        self.attn.in_proj_bias.data  .copy_( state_dict[f"{prefix}.attn.in_proj_bias"]   )
        self.attn.out_proj.weight.data.copy_( state_dict[f"{prefix}.attn.out_proj.weight"] )
        self.attn.out_proj.bias.data  .copy_( state_dict[f"{prefix}.attn.out_proj.bias"]   )
        # ln2 + mlp
        if self.mlp_ratio > 0:
            self.ln2.weight.data.copy_(     state_dict[f"{prefix}.ln_2.weight"]      )
            self.ln2.bias.data.copy_(       state_dict[f"{prefix}.ln_2.bias"]        )
            self.mlp.c_fc.weight.data.copy_(   state_dict[f"{prefix}.mlp.c_fc.weight"]  )
            self.mlp.c_fc.bias.data.copy_(     state_dict[f"{prefix}.mlp.c_fc.bias"]    )
            self.mlp.c_proj.weight.data.copy_( state_dict[f"{prefix}.mlp.c_proj.weight"])
            self.mlp.c_proj.bias.data.copy_(   state_dict[f"{prefix}.mlp.c_proj.bias"]  )

    def forward(self, x: torch.Tensor, T: int, H: int, W: int) -> torch.Tensor:
        # 通用残差 + MLP
        res = self._attention_forward(x, T, H, W)   # implemented differently in spatial and temporal
        x   = x + self.drop_path(res)
        if self.mlp_ratio > 0:
            x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x

    def _attention_forward(self, x, T, H, W):
        raise NotImplementedError

class ResidualAttentionBlock(BaseResidualAttnBlock):
    """
    modified ResidualAttentionBlock by inheriting from BaseAttn
    """
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        trainable: bool = True
    ):
        super().__init__(
            d_model=d_model,
            n_head=n_head,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            trainable=trainable
        )

    def _attention_forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        """
        x: [N, B, C]
        返回 attention 产生的残差 [N, B, C]
        """
        # 1. LayerNorm
        x_norm = self.ln1(x)   # [N, B, C]

        # 2. 原生 MultiheadAttention 接受 (seq_len, batch, embed)
        attn_out = self.attn(
            x_norm,   # q
            x_norm,   # k
            x_norm,   # v
            need_weights=False
        )[0]         # [N, B, C]

        return attn_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N, B, C]
        """
        # 1) attention + 残差
        res = self._attention_forward(x)
        x = x + self.drop_path(res)

        # 2) 可选的 MLP + 残差
        if self.mlp_ratio > 0:
            x = x + self.drop_path(self.mlp(self.ln2(x)))

        return x

class ResidualSpatialAttnBlock(BaseResidualAttnBlock):
    """
    Spatial Attention:为每帧 HW tokens 做 attention,
    cls/reg tokens repeat→attention→mean 回聚。
    """
    def _attention_forward(self, x, T, H, W):
        B, C = x.shape[1], x.shape[2]
        cls  = x[0:1]
        toks = x[1:1+H*W*T]
        regs = x[1+H*W*T:]

        # 重排到 [HW, B*T, C]
        toks_s   = rearrange(toks, "(t h w) b c -> (h w) (b t) c",
                             b=B, h=H, w=W, t=T)
        cls_rep  = cls.repeat(1, T, 1)
        regs_rep = regs.repeat(1, T, 1)

        seq = torch.cat([cls_rep, toks_s, regs_rep], dim=0)  # [1+HW+R, B*T, C]
        ln1_seq = self.ln1(seq)
        out = self.attn(ln1_seq, ln1_seq, ln1_seq,
                        need_weights=False)[0]

        # 拆分
        cls_o  = out[0:1]
        toks_o = out[1:1+H*W]
        regs_o = out[1+H*W:]

        # fold 回
        toks_rec = rearrange(toks_o, "(h w) (b t) c -> (t h w) b c",
                             b=B, h=H, w=W, t=T)
        cls_rec  = rearrange(
            rearrange(cls_o, "1 (b t) c -> b t c", b=B, t=T).mean(1, keepdim=True),
            "b 1 c -> 1 b c"
        )
        regs_rec = rearrange(regs_o, "r (b t) c -> r b t c", b=B, t=T).mean(2)
        return torch.cat([cls_rec, toks_rec, regs_rec], dim=0)

class ResidualTemporalAttnBlock(BaseResidualAttnBlock):
    """
    Temporal Attention:为每个空间位置跨 T 帧做 attention,
    cls/reg tokens 保持原值(残差为 0)。
    """
    def _attention_forward(self, x, T, H, W):
        B, C = x.shape[1], x.shape[2]
        cls  = x[0:1]
        toks = x[1:1+H*W*T]
        regs = x[1+H*W*T:]

        # 重排到 [T, B*H*W, C]
        toks_t = rearrange(toks, "(t h w) b c -> t (b h w) c",
                           b=B, h=H, w=W, t=T)
        
        ln1_tok_t = self.ln1(toks_t)
        # 直接 attention → out_proj
        out_t = self.attn(ln1_tok_t, ln1_tok_t, ln1_tok_t,
                          need_weights=False)[0]  # [T, BHW, C]

        # fold 回
        toks_rec = rearrange(out_t, "t (b h w) c -> (t h w) b c",
                             b=B, h=H, w=W, t=T)

        # cls/reg 一律残差为 0
        zero_cls  = torch.zeros_like(cls)
        zero_regs = torch.zeros_like(regs)

        return torch.cat([zero_cls, toks_rec, zero_regs], dim=0)


class ResidualCausalTemporalAttnBlock(BaseResidualAttnBlock):
    """
    Causal Temporal Attention：
    """
    def _attention_forward(self, x, T, H, W):
        B, C = x.shape[1], x.shape[2]
        cls  = x[0:1]                          # [1, B, C]
        toks = x[1:1+H*W*T]                   # [T*H*W, B, C]
        regs = x[1+H*W*T:]                    # [N_reg, B, C]

        # reshape 到 [T, B*H*W, C]
        toks_t = rearrange(toks, "(t h w) b c -> t (b h w) c",
                           b=B, h=H, w=W, t=T)
        # 归一化
        ln1_tok_t = self.ln1(toks_t)

        # 构造 causal attn_mask: shape [T, T], True 表示要屏蔽的位置
        # 上三角(不含主对角线)为 True，即 j>i 时被 mask
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        # 带 mask 的 attention
        # 返回值是 tuple: (out, attn_weights)，这里只要 out
        out_t, _ = self.attn(
            ln1_tok_t,            # query:  [T, BHW, C]
            ln1_tok_t,            # key:    [T, BHW, C]
            ln1_tok_t,            # value:  [T, BHW, C]
            attn_mask=causal_mask,
            need_weights=False
        )

        # fold 回原维度 [(T*H*W), B, C]
        toks_rec = rearrange(out_t, "t (b h w) c -> (t h w) b c",
                             b=B, h=H, w=W, t=T)

        # cls/reg 位置不做 residual（全零）
        zero_cls  = torch.zeros_like(cls)
        zero_regs = torch.zeros_like(regs)

        # 拼回去： [1 + T*H*W + N_reg, B, C]
        return torch.cat([zero_cls, toks_rec, zero_regs], dim=0)

if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    ATTENTION_MODE = 'flash'
else:
    try:
        import xformers
        import xformers.ops
        ATTENTION_MODE = 'xformers'
    except:
        ATTENTION_MODE = 'math'
print(f'attention mode is {ATTENTION_MODE}')


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, L, C = x.shape

        qkv = self.qkv(x)
        if ATTENTION_MODE == 'flash':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads).float()
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            x = einops.rearrange(x, 'B H L D -> B L (H D)')
        elif ATTENTION_MODE == 'xformers':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B L H D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B L H D
            x = xformers.ops.memory_efficient_attention(q, k, v)
            x = einops.rearrange(x, 'B L H D -> B L (H D)', H=self.num_heads)
        elif ATTENTION_MODE == 'math':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class UViTBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False, use_checkpoint=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x, skip=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, skip)
        else:
            return self._forward(x, skip)

    def _forward(self, x, skip=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    

def _expand_token(token, batch_size: int):
    return token.unsqueeze(0).expand(batch_size, -1, -1)

class TiTokEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config.dataset.preprocessing.crop_size # 256
        self.patch_size = config.model.vq_model.vit_enc_patch_size # 16
        self.grid_size = self.image_size // self.patch_size # 16
        self.model_size = config.model.vq_model.vit_enc_model_size # large
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens # 32
        self.token_size = config.model.vq_model.token_size # 12 code-book entry dim

        if config.model.vq_model.get("quantize_mode", "vq") == "vae":
            self.token_size = self.token_size * 2 # needs to split into mean and std for VAE

        self.is_legacy = config.model.vq_model.get("is_legacy", True) # default is legacy

        self.width = {
                "small": 512,
                "base": 768,
                "large": 1024,
            }[self.model_size] #1024
        self.num_layers = {
                "small": 8,
                "base": 12,
                "large": 24,
            }[self.model_size] # 24
        self.num_heads = {
                "small": 8,
                "base": 12,
                "large": 16,
            }[self.model_size] # 16 
        
        self.patch_embed = nn.Conv2d(
            in_channels=3, out_channels=self.width,
              kernel_size=self.patch_size, stride=self.patch_size, bias=True) # (b,width,grid,grid) one grid is a patch
        
        scale = self.width ** -0.5 # 1/32
        self.class_embedding = nn.Parameter(scale * torch.randn(1, self.width)) # [1,embedding_dim]
        self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size ** 2 + 1, self.width)) # [16*16+1,embedding_dim]  for the encoded image latent additional 1 for cls embed
        self.latent_token_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.width)) # [latent_dim,embedding_dim]
        self.ln_pre = nn.LayerNorm(self.width)
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers): # 24 Attn
            self.transformer.append(ResidualAttentionBlock(
                self.width, self.num_heads, mlp_ratio=4.0
            ))
        self.ln_post = nn.LayerNorm(self.width) # 1024
        self.conv_out = nn.Conv2d(self.width, self.token_size, kernel_size=1, bias=True) # 12 possibly token dim?

    def forward(self, pixel_values, latent_tokens):
        batch_size = pixel_values.shape[0]
        x = pixel_values
        x = self.patch_embed(x) # [b,1024,16,16]
        x = x.reshape(x.shape[0], x.shape[1], -1) # [b,1024,256]
        x = x.permute(0, 2, 1) # shape = [*, grid ** 2, width]
        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1) # add cls embedding expand on batch dim
        x = x + self.positional_embedding.to(x.dtype) # shape = [*, grid ** 2 + 1, width]
        # add pe for original <cls>+<img>

        latent_tokens = _expand_token(latent_tokens, x.shape[0]).to(x.dtype) # expand on batch dim
        latent_tokens = latent_tokens + self.latent_token_positional_embedding.to(x.dtype)
        x = torch.cat([x, latent_tokens], dim=1) # <cls> <img> <latent>
        
        x = self.ln_pre(x) # batch,length,dim
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.num_layers):
            x = self.transformer[i](x) # for attn
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        latent_tokens = x[:, 1+self.grid_size**2:] # extract the latent parts
        latent_tokens = self.ln_post(latent_tokens)
        # fake 2D shape
        if self.is_legacy:
            latent_tokens = latent_tokens.reshape(batch_size, self.width, self.num_latent_tokens, 1) # [b,1024,32,1]
        else:
            # Fix legacy problem.
            latent_tokens = latent_tokens.reshape(batch_size, self.num_latent_tokens, self.width, 1).permute(0, 2, 1, 3)
        latent_tokens = self.conv_out(latent_tokens) # [b,12,32,1]
        latent_tokens = latent_tokens.reshape(batch_size, self.token_size, 1, self.num_latent_tokens) # [b,12,1,32]
        return latent_tokens
    

class TiTokDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config.dataset.preprocessing.crop_size # 256
        self.patch_size = config.model.vq_model.vit_dec_patch_size # 16
        self.grid_size = self.image_size // self.patch_size # 16
        self.model_size = config.model.vq_model.vit_dec_model_size # large
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens # 32
        self.token_size = config.model.vq_model.token_size
        self.is_legacy = config.model.vq_model.get("is_legacy", True)
        self.width = {
                "small": 512,
                "base": 768,
                "large": 1024,
            }[self.model_size]
        self.num_layers = {
                "small": 8,
                "base": 12,
                "large": 24,
            }[self.model_size]
        self.num_heads = {
                "small": 8,
                "base": 12,
                "large": 16,
            }[self.model_size]

        self.decoder_embed = nn.Linear(
            self.token_size, self.width, bias=True) # from token_size  12 to width 1024
        scale = self.width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(1, self.width))
        self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size ** 2 + 1, self.width))
        # add mask token and query pos embed
        self.mask_token = nn.Parameter(scale * torch.randn(1, 1, self.width)) # [1,1,1024]
        self.latent_token_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.width)) # [32,1024]
        self.ln_pre = nn.LayerNorm(self.width)
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(ResidualAttentionBlock(
                self.width, self.num_heads, mlp_ratio=4.0
            ))
        self.ln_post = nn.LayerNorm(self.width)

        if self.is_legacy:
            self.ffn = nn.Sequential(
                nn.Conv2d(self.width, 2 * self.width, 1, padding=0, bias=True),
                nn.Tanh(),
                nn.Conv2d(2 * self.width, 1024, 1, padding=0, bias=True),
            )
            self.conv_out = nn.Identity()
        else:
            # Directly predicting RGB pixels
            self.ffn = nn.Sequential(
                nn.Conv2d(self.width, self.patch_size * self.patch_size * 3, 1, padding=0, bias=True), # directly conv towards img pixels num
                Rearrange('b (p1 p2 c) h w -> b c (h p1) (w p2)',
                    p1 = self.patch_size, p2 = self.patch_size),)
            self.conv_out = nn.Conv2d(3, 3, 3, padding=1, bias=True)
    
    def forward(self, z_quantized):
        N, C, H, W = z_quantized.shape # 1,12,1,32
        assert H == 1 and W == self.num_latent_tokens, f"{H}, {W}, {self.num_latent_tokens}"
        x = z_quantized.reshape(N, C*H, W).permute(0, 2, 1) # NLD [batch,ls,ld]
        x = self.decoder_embed(x) # linear to self.width(1024)

        batchsize, seq_len, _ = x.shape

        mask_tokens = self.mask_token.repeat(batchsize, self.grid_size**2, 1).to(x.dtype) #[b,grid^2,mask_dim] <MASK>
        mask_tokens = torch.cat([_expand_token(self.class_embedding, mask_tokens.shape[0]).to(mask_tokens.dtype),
                                    mask_tokens], dim=1) # [cls, <Mask>*256]
        mask_tokens = mask_tokens + self.positional_embedding.to(mask_tokens.dtype) # PE
        x = x + self.latent_token_positional_embedding[:seq_len]
        x = torch.cat([mask_tokens, x], dim=1) # [b,1+ grid**2 + latent size]
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.num_layers):
            x = self.transformer[i](x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x[:, 1:1+self.grid_size**2] # remove cls embed and also latent part
        x = self.ln_post(x)
        # N L D -> N D H W
        x = x.permute(0, 2, 1).reshape(batchsize, self.width, self.grid_size, self.grid_size) # [b,1024,16,16]
        x = self.ffn(x.contiguous()) # no shape change b,w,grid,grid
        x = self.conv_out(x)
        return x # [b,w,grid,grid]


class BaseTiTok3DEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # —— 基础配置 —— 
        self.video_size   = config.dataset.preprocessing.spatial_size
        self.video_length = config.dataset.preprocessing.temporal_size
        sp = config.model.vq_model.vit_enc_spatial_patch_size
        tp = config.model.vq_model.vit_enc_temporal_patch_size
        self.spatial_grid_size  = self.video_size   // sp
        self.temporal_grid_size = self.video_length // tp

        size = config.model.vq_model.vit_enc_model_size
        self.width      = {"small":512,"base":768,"large":1024}[size]
        self.num_layers = {"small":8,  "base":12,  "large":24}[size]
        self.num_heads  = {"small":8,  "base":12,  "large":16}[size]

        # —— Patch Embed, Embeddings, Norms, Conv Out —— 
        self.patch_embed = nn.Conv3d(3, self.width,
                                     kernel_size=(tp,sp,sp),
                                     stride=(tp,sp,sp), bias=True)
        scale = self.width ** -0.5
        self.class_embedding = nn.Parameter(scale*torch.randn(1,self.width))
        self.positional_embedding = nn.Parameter(
            scale*torch.randn(self.temporal_grid_size*self.spatial_grid_size**2 + 1, self.width)
        )
        self.latent_token_positional_embedding = nn.Parameter(
            scale*torch.randn(config.model.vq_model.num_latent_tokens, self.width)
        )
        self.ln_pre  = nn.LayerNorm(self.width)
        self.ln_post = nn.LayerNorm(self.width)

        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        self.token_size = config.model.vq_model.token_size
        if config.model.vq_model.get("quantize_mode","vq")=="vae":
            self.token_size *= 2
        self.conv_out = nn.Conv2d(self.width, self.token_size, kernel_size=1, bias=True)


    def _build_transformer(self):
        """
        子类必须实现这个方法：
        把 self.transformer 设成 nn.ModuleList([...])。
        """
        raise NotImplementedError
    
    def _apply_transformer(self, x):
        """
        子类重写这一段，x:[L,b,W]→返回同形状张量。
        """
        raise NotImplementedError    
    
    def forward(self, pixel_values, latent_tokens):
        batch_size = pixel_values.shape[0]

        # 1) patch_embed -> [b, W, Tgrid, Hgrid, Wgrid]
        x = self.patch_embed(pixel_values)

        # 2) flatten & reshape -> [b, W, L_img] -> permute -> [b, L_img, W]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        # 3) cls emb + pos emb
        x = torch.cat([
            _expand_token(self.class_embedding, batch_size).to(x.dtype),
            x
        ], dim=1)                                            # [b, 1+L_vid, W]
        x = x + self.positional_embedding.to(x.dtype)      # [1+L_vid, W] broadcase -> [b,1+L_vid,W]

        # 4) latent tokens + pos emb
        lt = _expand_token(latent_tokens, batch_size).to(x.dtype)  # [b, L_lat, C]
        lt = lt + self.latent_token_positional_embedding.to(x.dtype)
        x = torch.cat([x, lt], dim=1)                         # [b, 1+L_vid+L_lat, W]

        # 5) ln_pre, NLD->LND
        x = self.ln_pre(x)                                     # [b, L, W]
        x = x.permute(1, 0, 2)                                 # [L, b, W]

        # 6) Transformer blocks
        x = self._apply_transformer(x)                         # attention

        # 7) LND->NLD
        x = x.permute(1, 0, 2)                                 # [b, L, W]

        # 8) extract latent, ln_post
        split = 1 + self.temporal_grid_size*self.spatial_grid_size**2
        lat = x[:, split:]                                     # [b, L_lat, W]
        lat = self.ln_post(lat)                                # [b, L_lat, W]

        # 9) fake 2D + conv_out + final reshape
        # legacy vs non-legacy 形状保持一致
        lat = lat.reshape(batch_size, self.width, self.num_latent_tokens, 1)
        lat = self.conv_out(lat)                               # [b, token_size, L_lat, 1]
        lat = lat.reshape(batch_size, self.token_size, 1, self.num_latent_tokens)
        return lat
    
    def _load_encoder_common_weights(self, state: Dict[str, torch.Tensor], prefix: str = "encoder."):
        """
        LOAD from 2Dtitok
          - class_embedding
          - latent_token_positional_embedding
          - ln_pre, ln_post  weight & bias
          - conv_out  weight & bias
        """
        # 1) class embedding
        self.class_embedding.data.copy_(state[f"{prefix}class_embedding"])
        # 2) latent token pos embedding
        self.latent_token_positional_embedding.data.copy_(
            state[f"{prefix}latent_token_positional_embedding"]
        )
        # 3) ln_pre
        self.ln_pre.weight.data.copy_(state[f"{prefix}ln_pre.weight"])
        self.ln_pre.bias.  data.copy_(state[f"{prefix}ln_pre.bias"])
        # 4) ln_post
        self.ln_post.weight.data.copy_(state[f"{prefix}ln_post.weight"])
        self.ln_post.bias. data.copy_(state[f"{prefix}ln_post.bias"])
        # 5) conv_out
        self.conv_out.weight.data.copy_(state[f"{prefix}conv_out.weight"])
        self.conv_out.bias.  data.copy_(state[f"{prefix}conv_out.bias"])

class TiTok3DEncoder(BaseTiTok3DEncoder):
    def __init__(self, config):
        super().__init__(config)
        self._build_transformer()

    def _build_transformer(self):
        self.transformer = nn.ModuleList([
            ResidualAttentionBlock(self.width, self.num_heads, mlp_ratio=4.0)
            for _ in range(self.num_layers)
        ])

    def _apply_transformer(self, x):
        # 纯空间 attention，每个 block 只接收 x
        for blk in self.transformer:
            x = blk(x)
        return x
    
    def load_pretrained(self,state,
                 load_common: bool = False,
                 load_attn: bool = False):

            if load_common:
                self._load_encoder_common_weights(state, prefix="encoder.")
                print("common weights have been loaded in TiTok3DEncoder")

            if load_attn:
                idx = 0
                for blk in self.transformer:
                    if isinstance(blk, ResidualAttentionBlock):
                        prefix = f"encoder.transformer.{idx}"
                        blk.load_weights(state, prefix)
                        idx += 1
                print(f"Titok3DEncoder has loaded spatial attention parameters ,totally {idx} attn blocks!")
    
    def freeze_attn(self):
        for blk in self.transformer:
            if isinstance(blk, ResidualAttentionBlock):
                blk.set_trainable(False)
        print("Titok3DEncoder's spatial attention is frozen:!")

class TiTok3DSTEncoder(BaseTiTok3DEncoder):

    def __init__(self,config):
        super().__init__(config)
        model_size = config.model.vq_model.vit_enc_model_size
        self.pattern_map = {
            "small": config.model.encoder.attention_pattern * 2,   # T1S4 + T1S4 -> 10 层
            "base":  config.model.encoder.attention_pattern * 3,   # T1S4 ×3 -> 15 层
            "large": config.model.encoder.attention_pattern * 6,   # T1S4 ×6 -> 30 层
        }
        if model_size not in self.pattern_map:
            raise ValueError(f"No default pattern for model_size='{model_size}'")
        self.block_pattern = self.pattern_map[model_size]
        counts = re.findall(r'[st](\d+)', self.block_pattern)
        self.num_layers = sum(int(cnt) for cnt in counts) # update numlayers
        
        self._build_transformer()

        self.num_spatial_blocks  = sum(isinstance(b, ResidualSpatialAttnBlock) for b in self.transformer)
        self.num_temporal_blocks = sum(isinstance(b, ResidualTemporalAttnBlock) for b in self.transformer)
    
    def load_pretrained(self,state,
                 load_common: bool = False,
                 load_spatial_attn: bool = False):
        # --- 可选：加载预训练权重 ---

            if load_common:
                self._load_encoder_common_weights(state, prefix="encoder.")
                print("common weights have been loaded in TiTok3DSTEncoder")

            if load_spatial_attn:
                spatial_idx = 0
                for blk in self.transformer:
                    if isinstance(blk, ResidualSpatialAttnBlock):
                        prefix = f"encoder.transformer.{spatial_idx}"
                        blk.load_weights(state, prefix)
                        spatial_idx += 1
                assert spatial_idx == self.num_spatial_blocks, (
                    f"Titok3DSTEncoder has loaded {spatial_idx} spatial blocks but expected {self.num_spatial_blocks}"
                )
                print(f"Titok3DSTEncoder has loaded spatial attention parameters ,totally {spatial_idx} spatialattn blocks!")
    
    def freeze_spatial(self):
        for blk in self.transformer:
            if isinstance(blk, ResidualSpatialAttnBlock):
                blk.set_trainable(False)
        print("Titok3DSTEncoder's spatial attention is frozen:!")

    def _build_transformer(self):
        '''
        build tfs based on given patterns
        '''
        parts = re.findall(r'([st])(\d+)', self.block_pattern)
        sequence = []
        for typ, cnt in parts:
            sequence += [typ] * int(cnt)
        if len(sequence) != self.num_layers:
            raise RuntimeError("Expanded pattern length mismatch.")

        blocks = []
        for typ in sequence:
            if typ == 's':
                blk = ResidualSpatialAttnBlock(
                    d_model=self.width,
                    n_head=self.num_heads,
                    mlp_ratio=4.0,
                    drop_path=0.0,
                    act_layer=nn.GELU,
                    norm_layer=nn.LayerNorm,
                    trainable=True,
                )
            else:  # 't'
                blk = ResidualTemporalAttnBlock(
                    d_model=self.width,
                    n_head=self.num_heads,
                    mlp_ratio=4.0,
                    drop_path=0.0,
                    act_layer=nn.GELU,
                    norm_layer=nn.LayerNorm,
                    trainable=True,
                )
            blocks.append(blk)

        self.transformer = nn.ModuleList(blocks)

    def _apply_transformer(self, x: torch.Tensor) -> torch.Tensor:
        T = self.temporal_grid_size
        H = self.spatial_grid_size
        W = self.spatial_grid_size
        for blk in self.transformer:
            x = blk(x, T=T, H=H, W=W)
        return x


class BaseTiTok3DDecoder(nn.Module):
    """
    Abstract Base decoder
    """
    def __init__(self, config):
        super().__init__()
        # —— 1) configs —— 
        self.config = config
        self.video_size   = config.dataset.preprocessing.spatial_size
        self.video_length = config.dataset.preprocessing.temporal_size
        self.spatial_patch_size = config.model.vq_model.vit_enc_spatial_patch_size
        self.temporal_patch_size = config.model.vq_model.vit_enc_temporal_patch_size

        
        self.spatial_grid_size  = self.video_size   // self.spatial_patch_size
        self.temporal_grid_size = self.video_length // self.temporal_patch_size

        self.model_size = config.model.vq_model.vit_dec_model_size
        self.width = {"small":512, "base":768, "large":1024}[self.model_size]
        self.num_heads       = {"small":8,   "base":12,   "large":16}[self.model_size]
        self.num_layers      = {"small":8,   "base":12,   "large":24}[self.model_size]
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens

        self.token_size = config.model.vq_model.token_size

        if self.config.model.vq_model.get("use_fsq",False):
            self.token_size = len(self.config.model.vq_model.fsq_level)

        self.is_legacy  = config.model.vq_model.get("is_legacy", True)

        # —— 2) modules —— 
        # (a) 从 latent_dim 映射到 model_dim
        self.decoder_embed = nn.Linear(self.token_size, self.width, bias=True)

        # (b) cls、mask、pos emb
        scale = self.width ** -0.5
        self.class_embedding = nn.Parameter(scale*torch.randn(1, self.width))
        self.mask_token      = nn.Parameter(scale*torch.randn(1, 1, self.width))
        self.positional_embedding = nn.Parameter(
            scale*torch.randn(self.temporal_grid_size*self.spatial_grid_size**2 + 1, self.width)
        )
        self.latent_token_positional_embedding = nn.Parameter(
            scale*torch.randn(self.num_latent_tokens, self.width)
        )

        # (c) LayerNorms
        self.ln_pre  = nn.LayerNorm(self.width)
        self.ln_post = nn.LayerNorm(self.width)

        # (d) FFN & conv_out

        if self.is_legacy:
            self.ffn = nn.Sequential(
                nn.Conv2d(self.width, 2 * self.width, 1, padding=0, bias=True),
                nn.Tanh(),
                nn.Conv2d(2 * self.width, 1024, 1, padding=0, bias=True),
            )
            self.conv_out = nn.Identity()
        else:
            # Directly predicting RGB pixels
            self.ffn = nn.Sequential(
                nn.Conv3d(self.width, self.temporal_patch_size*self.spatial_patch_size * self.spatial_patch_size * 3, 1, padding=0, bias=True), # directly conv towards img pixels num
                Rearrange('b (tp p1 p2 c) t h w -> b c (t tp) (h p1) (w p2)',
                    tp = self.temporal_patch_size ,p1 = self.spatial_patch_size, p2 = self.spatial_patch_size),)
            self.conv_out = nn.Conv3d(3, 3, 3, padding=1, bias=True)            


    def forward(self, z_quantized: torch.Tensor):
        """
        z_quantized: [B, token_size, 1, num_latent_tokens]
        返回：如果 legacy，shape=[B, width, tgrid, sgrid, sgrid]；否则 [B,3,T,H,W]
        """
        B, C, H, W = z_quantized.shape
        assert H == 1 and W == self.num_latent_tokens

        # 1) reshape to [B, L, D]
        x = z_quantized.reshape(B, C*H, W).permute(0, 2, 1) 
        # 2) embed → [B, L, width]
        x = self.decoder_embed(x) # 16 -> 1024
        L = x.shape[1]

        # 3) mask + cls + pos emb
        mask_seq = self.mask_token.repeat(
            B, self.temporal_grid_size*self.spatial_grid_size**2, 1
        )
        cls_seq  = _expand_token(self.class_embedding, B)
        tokens   = torch.cat([cls_seq, mask_seq], dim=1)
        tokens   = tokens + self.positional_embedding

        # 4) 加上 latent token pos emb
        x = x + self.latent_token_positional_embedding[:L]
        x = torch.cat([tokens, x], dim=1)  # [B, 1 + L + R, width]

        # 5) ln_pre & transpose → [1+L+R, B, width]
        x = self.ln_pre(x).permute(1, 0, 2)

        # 6) attention
        x = self._apply_transformer(x)

        # 7) back to [B, L', width]
        x = x.permute(1, 0, 2)

        # 8) 抽出 mask→vid 部分
        vid_seq = x[:, 1:1+self.temporal_grid_size*self.spatial_grid_size**2]
        vid_seq = self.ln_post(vid_seq)

        # 9) 恢复到 [B, width, tgrid, sgrid, sgrid]
        t, s = self.temporal_grid_size, self.spatial_grid_size
        x = vid_seq.permute(0,2,1).reshape(B, self.width, t, s, s)

        # 10) FFN + conv_out
        x = self.ffn(x.contiguous())
        x = self.conv_out(x)
        return x
    
    def _build_transformer(self):
        """
        子类必须实现这个方法：
        把 self.transformer 设成 nn.ModuleList([...])。
        """
        raise NotImplementedError
    
    def _apply_transformer(self, x):
        """
        子类重写这一段，x:[L,b,W]→返回同形状张量。
        """
        raise NotImplementedError    
    def _load_decoder_common_weights(
        self,
        state: Dict[str, torch.Tensor],
        prefix: str = "decoder."
    ):
        """
        LOAD from 2D TiTok Decoder checkpoint:
          - class_embedding
          - mask_token
          - latent_token_positional_embedding
          - decoder_embed weight & bias
          - ln_pre, ln_post weight & bias
        """
        # 1) class embedding
        self.class_embedding.data.copy_(state[f"{prefix}class_embedding"])
        # 2) mask token
        self.mask_token.data.copy_(state[f"{prefix}mask_token"])
        # 3) latent token pos embedding
        self.latent_token_positional_embedding.data.copy_(state[f"{prefix}latent_token_positional_embedding"])
        # 4) decoder_embed
        self.decoder_embed.weight.data.copy_(state[f"{prefix}decoder_embed.weight"])
        self.decoder_embed.bias.data.copy_(state[f"{prefix}decoder_embed.bias"])
        # 5) ln_pre
        self.ln_pre.weight.data.copy_(state[f"{prefix}ln_pre.weight"])
        self.ln_pre.bias.data.copy_(state[f"{prefix}ln_pre.bias"])
        # 6) ln_post
        self.ln_post.weight.data.copy_(state[f"{prefix}ln_post.weight"])
        self.ln_post.bias.data.copy_(state[f"{prefix}ln_post.bias"])

class TiTok3DDecoder(BaseTiTok3DDecoder):
    def __init__(self, config):
        super().__init__(config)
        self._build_transformer()

    def _build_transformer(self):
        # 纯空间 attention，每层都是 ResidualAttentionBlock
        self.transformer = nn.ModuleList([
            ResidualAttentionBlock(
                d_model=self.width,
                n_head=self.num_heads,
                mlp_ratio=4.0
            )
            for _ in range(self.num_layers)
        ])

    def _apply_transformer(self, x: torch.Tensor) -> torch.Tensor:
        # x: [L, B, width]  ->  out: [L, B, width]
        for blk in self.transformer:
            x = blk(x)
        return x

    def load_pretrained(self,state,
                 load_common: bool = False,
                 load_attn: bool = False):

            if load_common:
                self._load_decoder_common_weights(state, prefix="decoder.")
                print("common weights have been loaded in TiTok3DDecoder")

            if load_attn:
                idx = 0
                for blk in self.transformer:
                    if isinstance(blk, ResidualAttentionBlock):
                        prefix = f"decoder.transformer.{idx}"
                        blk.load_weights(state, prefix)
                        idx += 1

                print(f"Titok3DDecoder has loaded spatial attention parameters ,totally {idx} attn blocks!")
    
    def freeze_attn(self):
        for blk in self.transformer:
            if isinstance(blk, ResidualAttentionBlock):
                blk.set_trainable(False)
        print("Titok3Decoder's spatial attention is frozen:!")
    

class TiTok3DSTDecoder(BaseTiTok3DDecoder):
    """
    titok3DSTDecoder
    """

    def __init__(self,config):
        # 1) 选择 pattern，计算 num_layers
        self.pattern_map = {
            "small": config.model.decoder.attention_pattern * 2,   # T1S4 + T1S4 -> 10 layers
            "base":  config.model.decoder.attention_pattern * 3,   # T1S4 ×3 -> 15 layers
            "large": config.model.decoder.attention_pattern * 6,   # T1S4 ×6 -> 30 layers
        }
        model_size = config.model.vq_model.vit_dec_model_size
        if model_size not in self.pattern_map:
            raise ValueError(f"No pattern for model_size='{model_size}'")
        self.block_pattern = self.pattern_map[model_size]


        # 2) init from base  
        super().__init__(config)
        counts = re.findall(r'[st](\d+)', self.block_pattern)
        self.num_layers = sum(int(c) for c in counts)

        # 3) build tfs
        self._build_transformer()

        self.num_spatial_blocks  = sum(isinstance(b, ResidualSpatialAttnBlock) for b in self.transformer)
        self.num_temporal_blocks = sum(isinstance(b, ResidualTemporalAttnBlock) for b in self.transformer)

    def load_pretrained(self,state,
                 load_common: bool = False,
                 load_spatial_attn: bool = False):
            
            if load_common:
                self._load_decoder_common_weights(state,prefix="decoder.")
                print("common weights have been loaded in TiTok3DSTDecoder")
            if load_spatial_attn:
                spatial_idx = 0
                for blk in self.transformer:
                    if isinstance(blk, ResidualSpatialAttnBlock):
                        prefix = f"decoder.transformer.{spatial_idx}"
                        blk.load_weights(state, prefix)
                        spatial_idx += 1
                assert spatial_idx == self.num_spatial_blocks, (
                    f"Titok3DSTDecoder has Loaded {spatial_idx} spatial blocks but expected {self.num_spatial_blocks}"
                )
                print(f"Titok3DSTDecoder has loaded spatial attention parameters ,totally {spatial_idx} spatialattn blocks!")



    def freeze_spatial(self):
        for blk in self.transformer:
            if isinstance(blk, ResidualSpatialAttnBlock): # spatial attention can be frozen
                blk.set_trainable(False)
        print(" Titok3DSTDecoder's spatial attention is frozen:! ")

    def _build_transformer(self):
        # Expand pattern like "s4t1s4t1" → ['s','s','s','s','t', ...]
        parts = re.findall(r'([st])(\d+)', self.block_pattern)
        seq = []
        for typ, cnt in parts:
            seq += [typ] * int(cnt)
        if len(seq) != self.num_layers:
            raise RuntimeError("Expanded pattern length mismatch.")

        blocks = []
        for typ in seq:
            if typ == 's':
                blocks.append(
                    ResidualSpatialAttnBlock(
                        d_model=self.width,
                        n_head=self.num_heads,
                        mlp_ratio=4.0,
                        drop_path=0.0,
                        act_layer=nn.GELU,
                        norm_layer=nn.LayerNorm,
                        trainable=True
                    )
                )
            else:  # 't'
                blocks.append(
                    ResidualTemporalAttnBlock(
                        d_model=self.width,
                        n_head=self.num_heads,
                        mlp_ratio=4.0,
                        drop_path=0.0,
                        act_layer=nn.GELU,
                        norm_layer=nn.LayerNorm,
                        trainable=True
                    )
                )
        self.transformer = nn.ModuleList(blocks)

    def _apply_transformer(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [L, B, width]
        T, H, W = self.temporal_grid_size, self.spatial_grid_size, self.spatial_grid_size
        for blk in self.transformer:
            x = blk(x, T=T, H=H, W=W)
        return x


class TATiTokDecoder(TiTokDecoder):
    def __init__(self, config):
        super().__init__(config)
        scale = self.width ** -0.5
        self.text_context_length = config.model.vq_model.get("text_context_length", 77)
        self.text_embed_dim = config.model.vq_model.get("text_embed_dim", 768)
        self.text_guidance_proj = nn.Linear(self.text_embed_dim, self.width)
        self.text_guidance_positional_embedding = nn.Parameter(scale * torch.randn(self.text_context_length, self.width))

    def forward(self, z_quantized, text_guidance):
        N, C, H, W = z_quantized.shape
        assert H == 1 and W == self.num_latent_tokens, f"{H}, {W}, {self.num_latent_tokens}"
        x = z_quantized.reshape(N, C*H, W).permute(0, 2, 1) # NLD
        x = self.decoder_embed(x)

        batchsize, seq_len, _ = x.shape

        mask_tokens = self.mask_token.repeat(batchsize, self.grid_size**2, 1).to(x.dtype)
        mask_tokens = torch.cat([_expand_token(self.class_embedding, mask_tokens.shape[0]).to(mask_tokens.dtype),
                                    mask_tokens], dim=1)
        mask_tokens = mask_tokens + self.positional_embedding.to(mask_tokens.dtype)
        x = x + self.latent_token_positional_embedding[:seq_len]
        x = torch.cat([mask_tokens, x], dim=1)

        text_guidance = self.text_guidance_proj(text_guidance)
        text_guidance = text_guidance + self.text_guidance_positional_embedding
        x = torch.cat([x, text_guidance], dim=1)
        
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.num_layers):
            x = self.transformer[i](x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x[:, 1:1+self.grid_size**2] # remove cls embed
        x = self.ln_post(x)
        # N L D -> N D H W
        x = x.permute(0, 2, 1).reshape(batchsize, self.width, self.grid_size, self.grid_size)
        x = self.ffn(x.contiguous())
        x = self.conv_out(x)
        return x
    

class WeightTiedLMHead(nn.Module):
    def __init__(self, embeddings, target_codebook_size):
        super().__init__()
        self.weight = embeddings.weight
        self.target_codebook_size = target_codebook_size

    def forward(self, x):
        # x shape: [batch_size, seq_len, embed_dim]
        # Get the weights for the target codebook size
        weight = self.weight[:self.target_codebook_size]  # Shape: [target_codebook_size, embed_dim]
        # Compute the logits by matrix multiplication
        logits = torch.matmul(x, weight.t())  # Shape: [batch_size, seq_len, target_codebook_size]
        return logits


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(
        self,
        channels
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SimpleMLPAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        grad_checkpointing=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(
                model_channels,
            ))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, c):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """
        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(c)

        y = t + c

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = checkpoint(block, x, y)
        else:
            for block in self.res_blocks:
                x = block(x, y)

        return self.final_layer(x, y)

    def forward_with_cfg(self, x, t, c, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, c)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)