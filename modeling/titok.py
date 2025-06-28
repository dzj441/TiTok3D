"""This file contains the model definition of TiTok.

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
import torch.nn as nn
from einops import rearrange

from modeling.modules.base_model import BaseModel
from modeling.modules.blocks import TiTok3DEncoder, TiTok3DDecoder,TiTokEncoder,TiTokDecoder,TiTok3DSTEncoder,TiTok3DSTDecoder
from modeling.quantizer.quantizer import VectorQuantizer, DiagonalGaussianDistribution
from modeling.quantizer.fsq import FSQ
from modeling.modules.maskgit_vqgan import Encoder as Pixel_Eecoder
from modeling.modules.maskgit_vqgan import Decoder as Pixel_Decoder
from modeling.modules.maskgit_vqgan import VectorQuantizer as Pixel_Quantizer
import json
from safetensors.torch import load_file as load_safetensors
from omegaconf import OmegaConf
from pathlib import Path

from huggingface_hub import PyTorchModelHubMixin


class PretrainedTokenizer(nn.Module):
    def __init__(self, pretrained_weight):
        super().__init__()
        conf = OmegaConf.create(
            {"channel_mult": [1, 1, 2, 2, 4],
            "num_resolutions": 5,
            "dropout": 0.0,
            "hidden_channels": 128,
            "num_channels": 3,
            "num_res_blocks": 2,
            "resolution": 256,
            "z_channels": 256})
        self.encoder = Pixel_Eecoder(conf)
        self.decoder = Pixel_Decoder(conf)
        self.quantize = Pixel_Quantizer( # maskGIT VQ
            num_embeddings=1024, embedding_dim=256, commitment_cost=0.25)
        # Load pretrained weights
        self.load_state_dict(torch.load(pretrained_weight, map_location=torch.device("cpu")), strict=True)
        
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def encode(self, x):
        hidden_states = self.encoder(x)
        quantized_states, codebook_indices, codebook_loss = self.quantize(hidden_states)
        return codebook_indices.detach()
    
    @torch.no_grad()
    def decode(self, codes):
        quantized_states = self.quantize.get_codebook_entry(codes)
        rec_images = self.decoder(quantized_states)
        rec_images = torch.clamp(rec_images, 0.0, 1.0)
        return rec_images.detach()
    
    @torch.no_grad()
    def decode_tokens(self, codes):
        return self.decode(codes)


class TiTok(BaseModel, PyTorchModelHubMixin, tags=["arxiv:2406.07550", "image-tokenization"], repo_url="https://github.com/bytedance/1d-tokenizer", license="apache-2.0"):
    def __init__(self, config):

        if isinstance(config, dict):
            config = OmegaConf.create(config)

        super().__init__()
        self.config = config
        # This should be False for stage1 and True for stage2.
        self.finetune_decoder = config.model.vq_model.get("finetune_decoder", True)

        self.quantize_mode = config.model.vq_model.get("quantize_mode", "vq")
        if self.quantize_mode not in ["vq", "vae"]:
            raise ValueError(f"Unsupported quantize mode {self.quantize_mode}.")
        
        if self.finetune_decoder and self.quantize_mode not in ["vq"]:
            raise ValueError("Only supprot finetune_decoder with vq quantization for now.")

        self.encoder = TiTokEncoder(config) # important
        self.decoder = TiTokDecoder(config)
        
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        scale = self.encoder.width ** -0.5
        self.latent_tokens = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.encoder.width))
        
        self.apply(self._init_weights)

        if self.quantize_mode == "vq":
            self.quantize = VectorQuantizer(
                codebook_size=config.model.vq_model.codebook_size, # 4096
                token_size=config.model.vq_model.token_size, #  12
                commitment_cost=config.model.vq_model.commitment_cost, # 0.25
                use_l2_norm=config.model.vq_model.use_l2_norm,) # True
        elif self.quantize_mode == "vae":
            self.quantize = DiagonalGaussianDistribution
        else:
            raise NotImplementedError
        
        if self.finetune_decoder:
            # Freeze encoder/quantizer/latent tokens
            self.latent_tokens.requires_grad_(False) # 不训tokens
            self.encoder.eval()
            self.encoder.requires_grad_(False)
            self.quantize.eval()
            self.quantize.requires_grad_(False)
            # 只训decoder
            # Include MaskGiT-VQGAN's quantizer and decoder
            self.pixel_quantize = Pixel_Quantizer(
                num_embeddings=1024, embedding_dim=256, commitment_cost=0.25)
            self.pixel_decoder = Pixel_Decoder(OmegaConf.create(
                {"channel_mult": [1, 1, 2, 2, 4],
                "num_resolutions": 5,
                "dropout": 0.0,
                "hidden_channels": 128,
                "num_channels": 3,
                "num_res_blocks": 2,
                "resolution": 256,
                "z_channels": 256}))
        
    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights and config to a local directory."""
        # Assume 'self.config' is your DictConfig object
        # Convert to a regular dictionary
        dict_config = OmegaConf.to_container(self.config)
        # Save as JSON
        file_path = Path(save_directory) / "config.json"
        with open(file_path, 'w') as json_file:
            json.dump(dict_config, json_file, indent=4)
        super()._save_pretrained(save_directory)

    def _init_weights(self, module):
        """ Initialize the weights.
            :param:
                module -> torch.nn.Module: module to initialize
        """
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def encode(self, x):
        if self.finetune_decoder:
            with torch.no_grad():
                self.encoder.eval()
                self.quantize.eval()
                z = self.encoder(pixel_values=x, latent_tokens=self.latent_tokens)
                z_quantized, result_dict = self.quantize(z)
                result_dict["quantizer_loss"] *= 0
                result_dict["commitment_loss"] *= 0
                result_dict["codebook_loss"] *= 0
        else:
            z = self.encoder(pixel_values=x, latent_tokens=self.latent_tokens) # [b,12,1,32] VQ  [b,32,1,32] VAE
            if self.quantize_mode == "vq":
                z_quantized, result_dict = self.quantize(z)
            elif self.quantize_mode == "vae":
                posteriors = self.quantize(z)
                z_quantized = posteriors.sample()
                result_dict = posteriors

        return z_quantized, result_dict
    
    def decode(self, z_quantized):

        decoded = self.decoder(z_quantized) # [1,1024,16,16] for vq
        if self.finetune_decoder:
            quantized_states = torch.einsum(
                'nchw,cd->ndhw', decoded.softmax(1), # softmax on 1024 MASKGIT entries(each dimmed 256) # 1024个 entry的 distribution
                self.pixel_quantize.embedding.weight) # [1,256,16,16]
            decoded = self.pixel_decoder(quantized_states) # 对 1024 个 entry的加权和
        return decoded
    
    def decode_tokens(self, tokens):
        if self.quantize_mode == "vq":
            tokens = tokens.squeeze(1) # [b,latent_size] 
            batch, seq_len = tokens.shape # B x N
            z_quantized = self.quantize.get_codebook_entry(
                tokens.reshape(-1)).reshape(batch, 1, seq_len, -1) # flatten then get entry then reshape to [batch,1,latent_size,latent_dim]
            z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous() # [1,12,1,32]
        elif self.quantize_mode == "vae":
            z_quantized = tokens
        decoded = self.decode(z_quantized) # [1,ld,1,ls] latent dim | latent size
        return decoded
    
    def forward(self, x):
        z_quantized, result_dict = self.encode(x)
        decoded = self.decode(z_quantized)
        return decoded, result_dict

class TiTok3D_FSQ(BaseModel,PyTorchModelHubMixin):
    def __init__(self, config):

        if isinstance(config, dict):
            config = OmegaConf.create(config)

        super().__init__()
        self.config = config
        # This should be False for stage1 and True for stage2.
        self.finetune_decoder = config.model.vq_model.get("finetune_decoder", True)

        self.quantize_mode = config.model.vq_model.get("quantize_mode", "vq")
        if self.quantize_mode not in ["vq", "vae"]:
            raise ValueError(f"Unsupported quantize mode {self.quantize_mode}.")
        self.use_fsq = False
        self.fsq_levels = []
        if config.model.vq_model.get("use_fsq",False):
            self.use_fsq = config.model.vq_model.get("use_fsq",False)
        
        if self.use_fsq:
            self.fsq_levels = config.model.vq_model.fsq_level
        
        if self.finetune_decoder and self.quantize_mode not in ["vq"]:
            raise ValueError("Only supprot finetune_decoder with vq quantization for now.")

        self.encoder = TiTok3DEncoder(config)
        self.decoder = TiTok3DDecoder(config)
        
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        scale = self.encoder.width ** -0.5
        
        self.latent_tokens = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.encoder.width))
        
        self.apply(self._init_weights)

        if self.quantize_mode == "vq":
            if self.use_fsq:
                self.quantize = FSQ(
                    levels = self.fsq_levels
                )
            else:
                self.quantize = VectorQuantizer(
                    codebook_size=config.model.vq_model.codebook_size, # 4096
                    token_size=config.model.vq_model.token_size, #  12
                    commitment_cost=config.model.vq_model.commitment_cost, # 0.25
                    use_l2_norm=config.model.vq_model.use_l2_norm,) # True
        elif self.quantize_mode == "vae":
            self.quantize = DiagonalGaussianDistribution
        else:
            raise NotImplementedError

        

        
    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights and config to a local directory."""
        # Assume 'self.config' is your DictConfig object
        # Convert to a regular dictionary
        dict_config = OmegaConf.to_container(self.config)
        # Save as JSON
        file_path = Path(save_directory) / "config.json"
        with open(file_path, 'w') as json_file:
            json.dump(dict_config, json_file, indent=4)
        super()._save_pretrained(save_directory)

    def _init_weights(self, module):
        """ Initialize the weights.
            :param:
                module -> torch.nn.Module: module to initialize
        """
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def encode(self, x):
        z = self.encoder(x,self.latent_tokens) # [b,fsq_level,1,latent_tokens]
        z = z.squeeze(2)
        z = z.transpose(1,2) # [b,latent_size,fsq_level]
        z_hat,extra_dict = self.quantize(z.contiguous())
        return z_hat,extra_dict # [b,ls,fsq_levelNum]

    
    def decode(self, z_quantized:torch.Tensor):
        z_quantized = z_quantized.transpose(1, 2).unsqueeze(2) # [b,ls,fsq_levelNum] - > [b,fsq_levelNum,1,ls]
        if z_quantized.dtype != torch.float32:
            z_quantized = z_quantized.float()
        decoded = self.decoder(z_quantized) # [1,1024,16,16] for vq
        return decoded
    
    def forward(self, x):
        z_quantized, result_dict = self.encode(x)
        decoded = self.decode(z_quantized)
        return decoded, result_dict


class TiTok3D(BaseModel,PyTorchModelHubMixin):
    def __init__(self, config):

        if isinstance(config, dict):
            config = OmegaConf.create(config)

        super().__init__()
        self.config = config
        # This should be False for stage1 and True for stage2.
        self.finetune_decoder = config.model.vq_model.get("finetune_decoder", True)

        self.quantize_mode = config.model.vq_model.get("quantize_mode", "vq")
        if self.quantize_mode not in ["vq", "vae"]:
            raise ValueError(f"Unsupported quantize mode {self.quantize_mode}.")
        
        if self.finetune_decoder and self.quantize_mode not in ["vq"]:
            raise ValueError("Only supprot finetune_decoder with vq quantization for now.")

        self.encoder = TiTok3DEncoder(config)
        self.decoder = TiTok3DDecoder(config)
        
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        scale = self.encoder.width ** -0.5
        self.latent_tokens = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.encoder.width))
        
        self.apply(self._init_weights)

        if self.quantize_mode == "vq":
            self.quantize = VectorQuantizer(
                codebook_size=config.model.vq_model.codebook_size, # 4096
                token_size=config.model.vq_model.token_size, #  12
                commitment_cost=config.model.vq_model.commitment_cost, # 0.25
                use_l2_norm=config.model.vq_model.use_l2_norm,) # True
        elif self.quantize_mode == "vae":
            self.quantize = DiagonalGaussianDistribution
        else:
            raise NotImplementedError
        
        if self.finetune_decoder:
            # Freeze encoder/quantizer/latent tokens
            self.latent_tokens.requires_grad_(False) # 不训tokens
            self.encoder.eval()
            self.encoder.requires_grad_(False)
            self.quantize.eval()
            self.quantize.requires_grad_(False)
            # 只训decoder
            # Include MaskGiT-VQGAN's quantizer and decoder
            self.pixel_quantize = Pixel_Quantizer(
                num_embeddings=1024, embedding_dim=256, commitment_cost=0.25)
            self.pixel_decoder = Pixel_Decoder(OmegaConf.create(
                {"channel_mult": [1, 1, 2, 2, 4],
                "num_resolutions": 5,
                "dropout": 0.0,
                "hidden_channels": 128,
                "num_channels": 3,
                "num_res_blocks": 2,
                "resolution": 256,
                "z_channels": 256}))
        
    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights and config to a local directory."""
        # Assume 'self.config' is your DictConfig object
        # Convert to a regular dictionary
        dict_config = OmegaConf.to_container(self.config)
        # Save as JSON
        file_path = Path(save_directory) / "config.json"
        with open(file_path, 'w') as json_file:
            json.dump(dict_config, json_file, indent=4)
        super()._save_pretrained(save_directory)

    def _init_weights(self, module):
        """ Initialize the weights.
            :param:
                module -> torch.nn.Module: module to initialize
        """
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def encode(self, x):
        if self.finetune_decoder:
            with torch.no_grad():
                self.encoder.eval()
                self.quantize.eval()
                z = self.encoder(pixel_values=x, latent_tokens=self.latent_tokens)
                z_quantized, result_dict = self.quantize(z)
                result_dict["quantizer_loss"] *= 0
                result_dict["commitment_loss"] *= 0
                result_dict["codebook_loss"] *= 0
        else:
            z = self.encoder(pixel_values=x, latent_tokens=self.latent_tokens) # [b,12,1,32] VQ  [b,32,1,32] VAE
            if self.quantize_mode == "vq":
                z_quantized, result_dict = self.quantize(z)
            elif self.quantize_mode == "vae":
                posteriors = self.quantize(z)
                z_quantized = posteriors.sample()
                result_dict = posteriors

        return z_quantized, result_dict
    
    def decode(self, z_quantized):
        decoded = self.decoder(z_quantized) # [1,1024,16,16] for vq
        if self.finetune_decoder:
            quantized_states = torch.einsum(
                'nchw,cd->ndhw', decoded.softmax(1), # softmax on 1024 MASKGIT entries(each dimmed 256) # 1024个 entry的 distribution
                self.pixel_quantize.embedding.weight) # [1,256,16,16]
            decoded = self.pixel_decoder(quantized_states) # 对 1024 个 entry的加权和
        return decoded
    
    def decode_tokens(self, tokens):
        if self.quantize_mode == "vq":
            tokens = tokens.squeeze(1) # [b,latent_size] 
            batch, seq_len = tokens.shape # B x N
            z_quantized = self.quantize.get_codebook_entry(
                tokens.reshape(-1)).reshape(batch, 1, seq_len, -1) # flatten then get entry then reshape to [batch,1,latent_size,latent_dim]
            z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous() # [1,12,1,32]
        elif self.quantize_mode == "vae":
            z_quantized = tokens
        decoded = self.decode(z_quantized) # [1,ld,1,ls] latent dim | latent size
        return decoded
    
    def forward(self, x):
        z_quantized, result_dict = self.encode(x)
        decoded = self.decode(z_quantized)
        return decoded, result_dict
    
class TiTok3DST(TiTok3D):
    """
    时空一体化版本，复用 TiTok3D 的大部分逻辑，只替换 encoder/decoder
    """
    def __init__(self, config):
        # 先让父类完成大部分初始化
        super().__init__(config)

        self.encoder =  TiTok3DSTEncoder(config)
        self.decoder = TiTok3DSTDecoder(config)

        self.apply(self._init_weights)
        pretrained_path = config.model.get("pretrained_path",None)
        if pretrained_path:
            if pretrained_path.endswith(".safetensors"):
                state = dict(load_safetensors(pretrained_path))
            else:
                state = torch.load(pretrained_path, map_location="cpu")
            if "latent_tokens" in state and config.model.load_latent:
                self.latent_tokens.data.copy_( state["latent_tokens"] )
            
            self.encoder.load_pretrained(state=state,
                                        load_common=config.model.encoder.load_common,
                                        load_spatial_attn=config.model.encoder.load_spatial)
            
            self.decoder.load_pretrained(state=state,
                                        load_common=config.model.decoder.load_common,
                                        load_spatial_attn=config.model.decoder.load_spatial)
            if config.model.encoder.freeze_spatial:
                self.encoder.freeze_spatial()
            if config.model.decoder.freeze_spatial:
                self.decoder.freeze_spatial()
        

def main():
    # —— 直接给定一个字符串路径 —— 
    yaml_path = r"configs\training\titok3D_bl128_vae_UCF_Times.yaml"

    vid_path = r"1.avi"
    config = OmegaConf.load(yaml_path)



    from dataset.data import load_video_from_path_decord  
    vid,_ = load_video_from_path_decord(vid_path,frm_sampling_strategy="center",height=128,width=128,fps=16,num_frm=16)
    vid_frm_array = torch.from_numpy(vid).float().permute(3,0,1,2).unsqueeze(0)
    
    model = TiTok3DST(config)
    out,_ = model.forward(vid_frm_array)

    from utils.video_utils import save_video_imageio
    save_video_imageio(out,avi_dir=None,mp4_dir="./temp",mean=[0,0,0],std=[1,1,1])



if __name__ == "__main__":
    main()