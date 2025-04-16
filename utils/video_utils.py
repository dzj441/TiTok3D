import os
import torch
import imageio
import numpy as np
from decord import VideoReader
from torch.nn.functional import interpolate as img_tensor_resize
import random
from pathlib import Path

class VideoNorm(object):
    """Apply Normalization to Image Pixels on GPU"""

    def __init__(
        self,
        mean=[0.5, 0.5, 0.5],
        std=[1.0, 1.0, 1.0],
        #mean=[0.48145466, 0.4578275, 0.40821073],
        #std=[0.26862954, 0.26130258, 0.27577711],
    ):
        # self.mean = torch.tensor(mean).cuda().view(1, 3, 1, 1)
        # self.std = torch.tensor(std).cuda().view(1, 3, 1, 1)
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def __call__(self, img):
        """
        Args:
            img: float image tensors, (N, 3, H, W)
        Returns:
            img: normalized float image tensors
        """
        if torch.max(img) > 1 and self.mean.max() <= 1:
            img.div_(255.0)
        re = img.sub_(self.mean).div_(self.std)
        return re



class VideoRandomSquareCrop(object):
    def __init__(self, crop_size, p=0.5):
        assert isinstance(crop_size, int)
        self.crop_size = crop_size
        self.p = p

    def __call__(self, video):
        """
        Args:
            img (torch.tensor): video to be cropped.

        Returns:
            torch.tensor: cropped video.
        """
        if isinstance(video, torch.Tensor):
            if len(video.shape) == 4:
                b, t, h, w = video.shape
            else:
                raise RuntimeError(
                    "Expecting 4-dimensional tensor of shape (b,t,h,w), got {}".format(
                        video.shape
                    )
                )

            # if random.uniform(0, 1) < self.p:
            #     video = torch.flip(video, (3,))

            x = random.randint(0, h - self.crop_size)
            y = random.randint(0, w - self.crop_size)

            return video[:, :, x : x + self.crop_size, y : y + self.crop_size]

        else:
            t, h, w, c = video.shape
            x = random.randint(0, h - self.crop_size)
            y = random.randint(0, w - self.crop_size)

            return video[:, x : x + self.crop_size, y : y + self.crop_size, :]



def load_video_from_path_decord(
    video_path,
    frm_sampling_strategy,
    height=None,
    width=None,
    start_time=None,
    end_time=None,
    fps=-1,
    num_frm=None,
):
    specified_num_frm = num_frm
    if not height or not width:
        vr = VideoReader(rf"{video_path}")
    else:
        vr = VideoReader(video_path, width=width, height=height)
    
    default_fps = vr.get_avg_fps()
    if default_fps <= fps:
        fps = -1

    if fps != -1:
        # resample the video to the specified fps
        duration = len(vr) / default_fps
        num_frames_to_sample = int(duration * fps)
        resample_indices = np.linspace(
            0, len(vr) - 1, num_frames_to_sample
        ).astype(int)
        
        # print(default_fps, fps, resample_indices)
        sampled_frms = vr.get_batch(resample_indices).asnumpy().astype(np.uint8)
        default_fps = fps
        

    else:
        sampled_frms = vr.get_batch(np.arange(0, len(vr), 1, dtype=int)).asnumpy().astype(np.uint8)

    vlen = sampled_frms.shape[0]

    if num_frm is None:
        num_frm = vlen

    num_frm = min(num_frm, vlen)

    if start_time or end_time:
        assert (
            fps > 0
        ), "must provide video fps if specifying start and end time."
        start_idx = min(int(start_time * fps), vlen)
        end_idx = min(int(end_time * fps), vlen)

    else:
        start_idx, end_idx = 0, vlen

    if frm_sampling_strategy == "uniform":
        frame_indices = np.linspace(0, vlen - 1, num_frm).astype(int)

    elif frm_sampling_strategy == "nlvl_uniform":
        frame_indices = np.arange(
            start_idx, end_idx, vlen / num_frm
        ).astype(int)

    elif frm_sampling_strategy == "nlvl_rand":
        frame_indices = np.arange(
            start_idx, end_idx, vlen / num_frm
        ).astype(int)

        strides = [
            frame_indices[i] - frame_indices[i - 1]
            for i in range(1, len(frame_indices))
        ] + [vlen - frame_indices[-1]]
        pertube = np.array(
            [np.random.randint(0, stride) for stride in strides]
        )

        frame_indices = frame_indices + pertube

    elif frm_sampling_strategy == "rand":
        # frame_indices = sorted(random.sample(range(vlen), num_frm))
        rand_start = random.randint(0, vlen - num_frm)
        frame_indices = np.array(range(rand_start, rand_start + num_frm)).astype(int)
    
    elif frm_sampling_strategy == "center":
        center = vlen // 2
        if num_frm % 2 ==0:
            frame_indices = np.array(range(center - num_frm // 2, center + num_frm // 2)).astype(int)
        else:
            frame_indices = np.array(range(center - num_frm // 2, center + num_frm // 2 + 1)).astype(int)
    
    elif frm_sampling_strategy == "headtail":
        frame_indices_head = sorted(
            random.sample(range(vlen // 2), num_frm // 2)
        )
        frame_indices_tail = sorted(
            random.sample(range(vlen // 2, vlen), num_frm // 2)
        )
        frame_indices = frame_indices_head + frame_indices_tail

    elif frm_sampling_strategy == "all":
        frame_indices = np.arange(0, vlen).astype(int)

    else:
        raise NotImplementedError(
            "Invalid sampling strategy {} ".format(frm_sampling_strategy)
        )

    raw_sample_frms = sampled_frms[
        frame_indices
    ]

    if specified_num_frm is None:
        masks = np.ones(len(raw_sample_frms), dtype=np.uint8)

    # pad the video if the number of frames is less than specified
    elif len(raw_sample_frms) < specified_num_frm:
        prev_length = len(raw_sample_frms)
        zeros = np.zeros(
            (specified_num_frm - prev_length, height, width, 3),
            dtype=np.uint8,
        )
        raw_sample_frms = np.concatenate((raw_sample_frms, zeros), axis=0)
        masks = np.zeros(specified_num_frm, dtype=np.uint8)
        masks[:prev_length] = 1

    else:
        masks = np.ones(specified_num_frm, dtype=np.uint8)


    return raw_sample_frms, masks

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
        avi_dir: str, 保存 avi 视频的文件夹路径
        mp4_dir: str, 保存 mp4 视频的文件夹路径
        fps: int, 每秒帧数
        mean, std: 用于反标准化
        prefix: 保存视频的命名前缀（如 video_0.mp4）

    Returns:
        None
    """
    if avi_dir:
        os.makedirs(avi_dir, exist_ok=True)
    os.makedirs(mp4_dir, exist_ok=True)

    # 反标准化
    def denormalize(tensor, mean, std):
        mean = torch.tensor(mean, device=tensor.device).view(1, 3, 1, 1, 1)
        std = torch.tensor(std, device=tensor.device).view(1, 3, 1, 1, 1)
        return tensor * std + mean
    
    tensor = denormalize(tensor, mean, std).clamp(0, 1)  # shape: [B, C, T, H, W]
    tensor = (tensor * 255).to(torch.uint8)  # [0, 255]
    B, C, T, H, W = tensor.shape

    for i in range(B):
        video = tensor[i]  # [C, T, H, W]
        video = video.permute(1, 2, 3, 0).cpu().numpy()  # -> [T, H, W, C] in RGB

        # avi_path = os.path.join(avi_dir, f"{prefix}_{i}.avi")
        mp4_path = os.path.join(mp4_dir, f"{prefix}_{i}.mp4")

        # imageio.mimwrite(avi_path, video, fps=fps, format="avi")
        imageio.mimwrite(mp4_path, video, fps=fps, format="ffmpeg")

        # print(f"[Saved] {avi_path}")
        print(f"[Saved] {mp4_path}")


def load_and_preprocess_video(
    video_path,
    sequence_length=16,
    resolution=128,
    resizecrop=False,
    fps=16,
    strategy="center",  # or "rand"
):
    """
    加载并标准化单个视频，输出 tensor: [1, C, T, H, W]
    """

    read_height = resolution if not resizecrop else int(resolution * 1.5)
    read_width = read_height

    if video_path.endswith("webm"):
        raise NotImplementedError("WebM not supported here.")
    else:
        frames, _ = load_video_from_path_decord(
            video_path,
            frm_sampling_strategy=strategy,
            fps=fps if fps is not None else -1,
            num_frm=sequence_length,
            height=read_height,
            width=read_width,
        )  # output: [T, H, W, 3]

    # 可选裁剪
    cropper = VideoRandomSquareCrop(resolution)
    frames = cropper(frames)  # still [T, H, W, 3]

    # 转换为 Tensor, 并规范顺序
    frames = torch.from_numpy(frames).float().permute(0, 3, 1, 2)  # [T, 3, H, W]

    # 标准化
    norm = VideoNorm()  # 默认 mean=[0.5]*3, std=[1.0]*3
    video = norm(frames)  # 仍是 [3, T, H, W]
    video = video.permute(1, 0, 2, 3)  # -> [3, T, H, W]

    return video.unsqueeze(0)  # 加上 batch 维度 -> [1, C, T, H, W]

