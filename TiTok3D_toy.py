import yaml
from omegaconf import OmegaConf
import os
import sys
sys.path.append(os.getcwd())

import torch
from modeling.titok import TiTok3D
device = 'cuda' if torch.cuda.is_available() else 'cpu'
CONFIG_PATH = "config/infer/titok3D_ll32_vae_c16.yaml"
config = OmegaConf.load(CONFIG_PATH)
toyModel = TiTok3D(config=config)
toyModel.to(device)

ranIn = torch.randn((2,3,16,128,128),device=device)
# tokenization
if config.model.vq_model.quantize_mode == "vq": # examine encode
    # encoded_tokens = titok_tokenizer.encode(image.to(device))[1]["min_encoding_indices"] # [1,1,32] the selected indices
    pass
elif config.model.vq_model.quantize_mode == "vae":
    posteriors = toyModel.encode(ranIn.to(device))[1]
    encoded_tokens = posteriors.sample()
else:
    raise NotImplementedError
# image assets/ILSVRC2012_val_00010240.png is encoded into tokens tensor([[[ 887, 3979,  349,  720, 2809, 2743, 2101,  603, 2205, 1508, 1891, 4015, 1317, 2956, 3774, 2296,  484, 2612, 3472, 2330, 3140, 3113, 1056, 3779,  654, 2360, 1901, 2908, 2169,  953, 1326, 2598]]], device='cuda:0'), with shape torch.Size([1, 1, 32])
print(f"input is encoded into tokens {encoded_tokens}, with shape {encoded_tokens.shape}")
# de-tokenization
reconstructed_image = toyModel.decode_tokens(encoded_tokens) # examine decoder

print(reconstructed_image.shape)