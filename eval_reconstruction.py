#!/usr/bin/env python3
"""
Batch reconstruct videos listed in a text file using TiTok3D.
Reads relative AVI paths from a TXT, joins with a base directory, processes each video by encoding/decoding,
and saves both the original (as MP4) and reconstructed videos to specified output directories.
"""
import os
import argparse
import torch
from omegaconf import OmegaConf
from utils.video_utils import load_and_preprocess_video, save_video_imageio
from modeling.titok import TiTok3D
device = "cuda"
def reconstruct_video(input_path: str,
                      model: TiTok3D,
                      base_dir: str,
                      orig_out_dir: str,
                      recon_out_dir: str,
                      fps: int = 16):
    """
    Load, reconstruct, and save both original and reconstructed videos.
    """
    avi_path = os.path.join(base_dir, input_path)
    if not os.path.isfile(avi_path):
        print(f"[Warning] File not found: {avi_path}")
        return

    # Load and preprocess
    tensor_vid = load_and_preprocess_video(avi_path).to(device)
    print(f"Loaded {input_path}, tensor shape: {tensor_vid.shape}")

    # Prepare filenames
    name = os.path.splitext(os.path.basename(input_path))[0]



    # Save original as MP4
    save_video_imageio(tensor=tensor_vid, avi_dir=None, mp4_dir=orig_out_dir, fps=fps,prefix=name)
    print(f"Saved original MP4 to {orig_out_dir}")

    # Encode and decode
    with torch.no_grad():
        encoded, _ = model.encode(tensor_vid)
        output = model.decode(encoded)
    print(f"Reconstructed video shape: {output.shape}")

    # Save reconstructed as MP4
    save_video_imageio(tensor=output, avi_dir=None, mp4_dir=recon_out_dir, fps=fps,prefix=name)
    print(f"Saved reconstructed MP4 to {recon_out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch reconstruct videos from a list using TiTok3D"
    )
    parser.add_argument(
        "--base_dir", default="datasets/UCF-101/test",
        help="Base directory for the listed relative AVI paths"
    )
    parser.add_argument(
        "--list_txt", default="datasets/UCF-101/testlist.txt",
        help="Text file listing one relative AVI path per line"
    )
    parser.add_argument(
        "--orig_out_dir", default="results/original",
        help="Directory to save original videos (MP4 format)"
    )
    parser.add_argument(
        "--recon_out_dir", default="results/recon",
        help="Directory to save reconstructed videos (MP4 format)"
    )
    parser.add_argument(
        "--config", default="configs/infer/titok3D_ll32_vae_c16.yaml",
        help="Path to the TiTok3D config YAML"
    )
    parser.add_argument(
        "--weights", default="temp_weights",
        help="Path to pretrained weights directory"
    )
    parser.add_argument(
        "--fps", type=int, default=16,
        help="Frames per second for saving videos"
    )
    args = parser.parse_args()

    # Load config and model
    _ = OmegaConf.load(args.config)
    print(f"Loading model from {args.weights}...")
    model = TiTok3D.from_pretrained(args.weights).to('cuda').eval()

    # Ensure output directories exist
    os.makedirs(args.orig_out_dir, exist_ok=True)
    os.makedirs(args.recon_out_dir, exist_ok=True)

    # Read list of videos
    with open(args.list_txt, 'r') as f:
        video_list = [line.strip() for line in f if line.strip()]

    # Process each video
    for rel_path in video_list:
        reconstruct_video(
            input_path=rel_path,
            model=model,
            base_dir=args.base_dir,
            orig_out_dir=args.orig_out_dir,
            recon_out_dir=args.recon_out_dir,
            fps=args.fps
        )

if __name__ == '__main__':
    main()
