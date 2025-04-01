import torch
from PIL import Image
import numpy as np
import demo_util
from huggingface_hub import hf_hub_download
from modeling.maskgit import ImageBert
from modeling.titok import TiTok

# Choose one from ["tokenizer_titok_l32_imagenet", "tokenizer_titok_b64_imagenet",
#  "tokenizer_titok_s128_imagenet", "tokenizer_titok_bl128_vae_c16_imagenet", tokenizer_titok_bl64_vae_c16_imagenet",
# "tokenizer_titok_ll32_vae_c16_imagenet", "tokenizer_titok_sl256_vq8k_imagenet", "tokenizer_titok_bl128_vq8k_imagenet",
# "tokenizer_titok_bl64_vq8k_imagenet",]
print("debug")
titok_tokenizer = TiTok.from_pretrained("ckpt/tokenizer_titok_ll32_vae_c16_imagenet")
titok_tokenizer.eval()
titok_tokenizer.requires_grad_(False)
titok_generator = ImageBert.from_pretrained("ckpt/generator_titok_l32_imagenet")
titok_generator.eval()
titok_generator.requires_grad_(False)

# or alternatively, downloads from hf
# hf_hub_download(repo_id="fun-research/TiTok", filename="tokenizer_titok_l32.bin", local_dir="./")
# hf_hub_download(repo_id="fun-research/TiTok", filename="generator_titok_l32.bin", local_dir="./")

# load config
# config = demo_util.get_config("configs/infer/TiTok/titok_l32.yaml")
# titok_tokenizer = demo_util.get_titok_tokenizer(config)
# titok_generator = demo_util.get_titok_generator(config)

device = "cuda"
titok_tokenizer = titok_tokenizer.to(device)
titok_generator = titok_generator.to(device)

# reconstruct an image. I.e., image -> 32 tokens -> image
# img_path = "assets/ILSVRC2012_val_00008636.png"
img_path = "assets/generated_48.png"
img_name = img_path.split('/')[-1]

image = torch.from_numpy(np.array(Image.open(img_path)).astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
# tokenization
if titok_tokenizer.quantize_mode == "vq": # examine encode
    encoded_tokens = titok_tokenizer.encode(image.to(device))[1]["min_encoding_indices"] # [1,1,32] the selected indices
elif titok_tokenizer.quantize_mode == "vae":
    posteriors = titok_tokenizer.encode(image.to(device))[1]
    encoded_tokens = posteriors.sample()
else:
    raise NotImplementedError
# image assets/ILSVRC2012_val_00010240.png is encoded into tokens tensor([[[ 887, 3979,  349,  720, 2809, 2743, 2101,  603, 2205, 1508, 1891, 4015, 1317, 2956, 3774, 2296,  484, 2612, 3472, 2330, 3140, 3113, 1056, 3779,  654, 2360, 1901, 2908, 2169,  953, 1326, 2598]]], device='cuda:0'), with shape torch.Size([1, 1, 32])
print(f"image {img_path} is encoded into tokens {encoded_tokens}, with shape {encoded_tokens.shape}")
# de-tokenization
reconstructed_image = titok_tokenizer.decode_tokens(encoded_tokens) # examine decoder
reconstructed_image = torch.clamp(reconstructed_image, 0.0, 1.0)
reconstructed_image = (reconstructed_image * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()[0]
reconstructed_image = Image.fromarray(reconstructed_image).save( f"results/{img_name}" )
if titok_tokenizer.quantize_mode == 'vq':
    # generate an image
    sample_labels = [torch.randint(0, 999, size=(1,)).item()] # random IN-1k class
    generated_image = demo_util.sample_fn(
        generator=titok_generator,
        tokenizer=titok_tokenizer,
        labels=sample_labels,
        guidance_scale=4.5,
        randomize_temperature=1.0,
        num_sample_steps=8,
        device=device
    )
    Image.fromarray(generated_image[0]).save(f"results/generated_{sample_labels[0]}.png")
    print(f"results/generated_{sample_labels[0]}.png is saved")