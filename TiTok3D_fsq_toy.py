import yaml
from omegaconf import OmegaConf
import os
import sys
sys.path.append(os.getcwd())

import torch
from modeling.titok import TiTok3D_FSQ
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
CONFIG_PATH = "configs/training/titok3D_ll32_fsq.yaml"
config = OmegaConf.load(CONFIG_PATH)
toyModel = TiTok3D_FSQ(config=config)
toyModel.to(device)

ranIn = torch.randn((2,3,16,128,128),device=device)
# tokenization
encoded_tokens,extra = toyModel.encode(ranIn.to(device))
print("hi")
# # de-tokenization
reconstructed_image = toyModel.decode(encoded_tokens) # examine decoder

print(reconstructed_image.shape)