"""This file contains a class to evalute the reconstruction results.

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

import warnings

from typing import Sequence, Optional, Mapping, Text
import numpy as np
from scipy import linalg
import torch
import torch.nn.functional as F
from .evaluation.common_metrics_on_video_quality import calculate_fvd_given_timestamp,calculate_is

class VQGANEvaluator():
    def __init__(
        self,
        device,
        enable_rfid: bool = True,
        enable_inception_score: bool = True,
        enable_codebook_usage_measure: bool = False,
        enable_codebook_entropy_measure: bool = False,
        num_codebook_entries: int = 1024
    ):
        """Initializes VQGAN Evaluator.

        Args:
            device: The device to use for evaluation.
            enable_rfid: A boolean, whether enabling rFID score.
            enable_inception_score: A boolean, whether enabling Inception Score.
            enable_codebook_usage_measure: A boolean, whether enabling codebook usage measure.
            enable_codebook_entropy_measure: A boolean, whether enabling codebook entropy measure.
            num_codebook_entries: An integer, the number of codebook entries.
        """
        self._device = device

        self._enable_rfid = enable_rfid
        self._enable_inception_score = enable_inception_score
        self._enable_codebook_usage_measure = enable_codebook_usage_measure
        self._enable_codebook_entropy_measure = enable_codebook_entropy_measure
        self._num_codebook_entries = num_codebook_entries

        # Variables related to Inception score and rFID.
        self._inception_model = None
        self._is_num_features = 0
        self._rfid_num_features = 0
        if self._enable_inception_score or self._enable_rfid:
            self._rfid_num_features = 2048
            self._is_num_features = 1008
            self._inception_model = get_inception_model().to(self._device)
            self._inception_model.eval()
        self._is_eps = 1e-16
        self._rfid_eps = 1e-6

        self.reset_metrics()

    def reset_metrics(self):
        """Resets all metrics."""
        self._num_examples = 0
        self._num_updates = 0

        self._is_prob_total = torch.zeros(
            self._is_num_features, dtype=torch.float64, device=self._device
        )
        self._is_total_kl_d = torch.zeros(
            self._is_num_features, dtype=torch.float64, device=self._device
        )
        self._rfid_real_sigma = torch.zeros(
            (self._rfid_num_features, self._rfid_num_features),
            dtype=torch.float64, device=self._device
        )
        self._rfid_real_total = torch.zeros(
            self._rfid_num_features, dtype=torch.float64, device=self._device
        )
        self._rfid_fake_sigma = torch.zeros(
            (self._rfid_num_features, self._rfid_num_features),
            dtype=torch.float64, device=self._device
        )
        self._rfid_fake_total = torch.zeros(
            self._rfid_num_features, dtype=torch.float64, device=self._device
        )

        self._set_of_codebook_indices = set()
        self._codebook_frequencies = torch.zeros((self._num_codebook_entries), dtype=torch.float64, device=self._device)

    def update(
        self,
        real_videos: torch.Tensor,
        fake_videos: torch.Tensor,
        codebook_indices: Optional[torch.Tensor] = None
    ):
        """Updates the metrics with the given video,will call the functions in evaluation dir.
        

        Args:
            real_images: A torch.Tensor, the real videos.
            fake_images: A torch.Tensor, the fake videos.
            codebook_indices: A torch.Tensor, the indices of the codebooks for each video.

        Raises:
            ValueError: If the fake images is not in RGB (3 channel).
            ValueError: If the fake and real images have different shape.
        """

        batch_size = real_videos.shape[0]
        dim = tuple(range(1, real_videos.ndim))
        self._num_examples += batch_size
        self._num_updates += 1

        if self._enable_inception_score or self._enable_rfid:
            # Quantize to uint8 as a real image.
            fake_inception_images = (fake_videos * 255).to(torch.uint8)
            features_fake = self._inception_model(fake_inception_images)
            inception_logits_fake = features_fake["logits_unbiased"]
            inception_probabilities_fake = F.softmax(inception_logits_fake, dim=-1)
        
        if self._enable_inception_score:
            probabiliies_sum = torch.sum(inception_probabilities_fake, 0, dtype=torch.float64)

            log_prob = torch.log(inception_probabilities_fake + self._is_eps)
            if log_prob.dtype != inception_probabilities_fake.dtype:
                log_prob = log_prob.to(inception_probabilities_fake)
            kl_sum = torch.sum(inception_probabilities_fake * log_prob, 0, dtype=torch.float64)

            self._is_prob_total += probabiliies_sum
            self._is_total_kl_d += kl_sum

        if self._enable_rfid:
            real_inception_images = (real_videos * 255).to(torch.uint8)
            features_real = self._inception_model(real_inception_images)
            if (features_real['2048'].shape[0] != features_fake['2048'].shape[0] or
                features_real['2048'].shape[1] != features_fake['2048'].shape[1]):
                raise ValueError(f"Number of features should be equal for real and fake.")

            for f_real, f_fake in zip(features_real['2048'], features_fake['2048']):
                self._rfid_real_total += f_real
                self._rfid_fake_total += f_fake

                self._rfid_real_sigma += torch.outer(f_real, f_real)
                self._rfid_fake_sigma += torch.outer(f_fake, f_fake)

        if self._enable_codebook_usage_measure:
            self._set_of_codebook_indices |= set(torch.unique(codebook_indices, sorted=False).tolist())

        if self._enable_codebook_entropy_measure:
            entries, counts = torch.unique(codebook_indices, sorted=False, return_counts=True)
            self._codebook_frequencies.index_add_(0, entries.int(), counts.double())


    def result(self) -> Mapping[Text, torch.Tensor]:
        """Returns the evaluation result."""
        eval_score = {}

        if self._num_examples < 1:
            raise ValueError("No examples to evaluate.")
        
        if self._enable_inception_score:
            mean_probs = self._is_prob_total / self._num_examples
            log_mean_probs = torch.log(mean_probs + self._is_eps)
            if log_mean_probs.dtype != self._is_prob_total.dtype:
                log_mean_probs = log_mean_probs.to(self._is_prob_total)
            excess_entropy = self._is_prob_total * log_mean_probs
            avg_kl_d = torch.sum(self._is_total_kl_d - excess_entropy) / self._num_examples

            inception_score = torch.exp(avg_kl_d).item()
            eval_score["InceptionScore"] = inception_score

        if self._enable_rfid:
            mu_real = self._rfid_real_total / self._num_examples
            mu_fake = self._rfid_fake_total / self._num_examples
            sigma_real = get_covariance(self._rfid_real_sigma, self._rfid_real_total, self._num_examples)
            sigma_fake = get_covariance(self._rfid_fake_sigma, self._rfid_fake_total, self._num_examples)

            mu_real, mu_fake = mu_real.cpu(), mu_fake.cpu()
            sigma_real, sigma_fake = sigma_real.cpu(), sigma_fake.cpu()

            diff = mu_real - mu_fake

            # Product might be almost singular.
            covmean, _ = linalg.sqrtm(sigma_real.mm(sigma_fake).numpy(), disp=False)
            # Numerical error might give slight imaginary component.
            if np.iscomplexobj(covmean):
                if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                    m = np.max(np.abs(covmean.imag))
                    raise ValueError("Imaginary component {}".format(m))
                covmean = covmean.real

            tr_covmean = np.trace(covmean)

            if not np.isfinite(covmean).all():
                tr_covmean = np.sum(np.sqrt((
                    (np.diag(sigma_real) * self._rfid_eps) * (np.diag(sigma_fake) * self._rfid_eps))
                    / (self._rfid_eps * self._rfid_eps)
                ))

            rfid = float(diff.dot(diff).item() + torch.trace(sigma_real) + torch.trace(sigma_fake) 
                - 2 * tr_covmean
            )
            if torch.isnan(torch.tensor(rfid)) or torch.isinf(torch.tensor(rfid)):
                warnings.warn("The product of covariance of train and test features is out of bounds.")

            eval_score["rFID"] = rfid

        if self._enable_codebook_usage_measure:
            usage = float(len(self._set_of_codebook_indices)) / self._num_codebook_entries
            eval_score["CodebookUsage"] = usage

        if self._enable_codebook_entropy_measure:
            probs = self._codebook_frequencies / self._codebook_frequencies.sum()
            entropy = (-torch.log2(probs + 1e-8) * probs).sum()
            eval_score["CodebookEntropy"] = entropy

        return eval_score


class VideoEvaluator():
    def __init__(
        self,
        device,
        enable_fvd: bool = True,
        enable_inception_score: bool = True,
        clip_time_stamp: int = 16,
        fvd_method: str = 'styleganv',
        need_channel_adjustment_fvd: bool = False,
        need_channel_adjustment_is:bool = True,
        is_splits: int = 1
    ):
        """
        init VideoEvaluator

        Args:
            device:。
            enable_fvd: whether to calculate FVD 。
            enable_inception_score: bool whether to calculate Inception Score。
            clip_time_stamp: clip len。
            fvd_method: FVD calculation method, support 'styleganv' or 'videogpt'。
            need_channel_adjustment_fvd: video needs channel adjustment。
            is_splits: splits when calculating IS
        """
        self._device = device
        self._enable_fvd = enable_fvd
        self._enable_inception_score = enable_inception_score
        self._clip_time_stamp = clip_time_stamp
        self._fvd_method = fvd_method
        self._need_channel_adjustment_fvd = need_channel_adjustment_fvd
        self._need_channel_adjustment_is = need_channel_adjustment_is
        self._is_splits = is_splits

        self.reset_metrics()

    def reset_metrics(self):
        self._all_real_videos = None  # [N, C, T, H, W]
        self._all_fake_videos = None  # [N, C, T, H, W]

    def update(
        self,
        real_videos: torch.Tensor,  
        fake_videos: torch.Tensor   
    ):
        """
        store each video

        real_videos  [B, C, T, H, W]。
        fake_videos [B, C, T, H, W]。
        """
        real_videos = real_videos.detach().cpu()
        fake_videos = fake_videos.detach().cpu()
        
        if self._all_real_videos is None:
            self._all_real_videos = real_videos
        else:
            self._all_real_videos = torch.cat([self._all_real_videos, real_videos], dim=0)

        if self._all_fake_videos is None:
            self._all_fake_videos = fake_videos
        else:
            self._all_fake_videos = torch.cat([self._all_fake_videos, fake_videos], dim=0)

    def result(self) -> Mapping[Text, torch.Tensor]:
        
        eval_score = {}
        if self._all_real_videos is None or self._all_fake_videos is None:
            raise ValueError("No examples to evaluate.")


        real_videos = self._all_real_videos.to(self._device)
        fake_videos = self._all_fake_videos.to(self._device)
        

        real_videos_fvd = real_videos.contiguous()
        fake_videos_fvd = fake_videos.contiguous()

        if self._enable_fvd:
            fvd_result = calculate_fvd_given_timestamp(
                real_videos_fvd,
                fake_videos_fvd,
                device=self._device,
                clip_time_stamp=self._clip_time_stamp,
                method=self._fvd_method,
                need_channel_adjustment=self._need_channel_adjustment_fvd
            )
            eval_score["FVD"] = fvd_result["value"]
            eval_score["FVD_clip_time_stamp"] = fvd_result["clip_time_stamp"]
            eval_score["FVD_video_setting"] = fvd_result["video_setting"]
            eval_score["FVD_video_setting_name"] = fvd_result["video_setting_name"]

        if self._enable_inception_score:
            is_mean, is_std = calculate_is(
                fake_videos_fvd,
                device=self._device,
                splits=self._is_splits,
                needs_channel_adjustment=self._need_channel_adjustment_is
            )
            eval_score["InceptionScore_mean"] = is_mean
            eval_score["InceptionScore_std"] = is_std

        return eval_score


# class VideoEvaluator():
#     def __init__(
#         self,
#         device,
#         buffer_size: int = 256,
#         enable_fvd: bool = True,
#         enable_inception_score: bool = True,
#         clip_time_stamp: int = 16,
#         fvd_method: str = 'styleganv',
#         need_channel_adjustment_fvd: bool = False,
#         need_channel_adjustment_is: bool = True,
#         is_splits: int = 1
#     ):
#         self._device = device
#         self._enable_fvd = enable_fvd
#         self._enable_inception_score = enable_inception_score
#         self._clip_time_stamp = clip_time_stamp
#         self._fvd_method = fvd_method
#         self._need_channel_adjustment_fvd = need_channel_adjustment_fvd
#         self._need_channel_adjustment_is = need_channel_adjustment_is
#         self._is_splits = is_splits

#         self.buffer_size = buffer_size
#         self._real_buffer = []
#         self._fake_buffer = []
#         self._fvd_scores = []
#         self._is_means = []
#         self._is_stds = []

#     def reset_metrics(self):
#         self._real_buffer.clear()
#         self._fake_buffer.clear()
#         self._fvd_scores.clear()
#         self._is_means.clear()
#         self._is_stds.clear()

#     def update(self, real_videos: torch.Tensor, fake_videos: torch.Tensor):
#         # 1) 先像原来一样拉到 CPU 并缓存
#         real_videos = real_videos.detach().cpu()
#         fake_videos = fake_videos.detach().cpu()
#         self._real_buffer.append(real_videos)
#         self._fake_buffer.append(fake_videos)

#         # 2) 如果达到了 buffer_size，flush 一次
#         total = sum(b.shape[0] for b in self._real_buffer)
#         if total >= self.buffer_size:
#             self._flush_batch()

#     def _flush_batch(self):
#         real_batch = torch.cat(self._real_buffer, dim=0).to(self._device).contiguous()
#         fake_batch = torch.cat(self._fake_buffer, dim=0).to(self._device).contiguous()
#         # 调用你已有的 FVD 计算
#         if self._enable_fvd:
#             fvd_res = calculate_fvd_given_timestamp(
#                 real_batch,
#                 fake_batch,
#                 device=self._device,
#                 clip_time_stamp=self._clip_time_stamp,
#                 method=self._fvd_method,
#                 need_channel_adjustment=self._need_channel_adjustment_fvd
#             )
#             self._fvd_scores.append(fvd_res["value"])
#         # 调用你已有的 IS 计算
#         if self._enable_inception_score:
#             is_mean, is_std = calculate_is(
#                 fake_batch,
#                 device=self._device,
#                 splits=self._is_splits,
#                 needs_channel_adjustment=self._need_channel_adjustment_is
#             )
#             self._is_means.append(is_mean)
#             self._is_stds.append(is_std)

#         # 清空缓存，释放内存
#         self._real_buffer.clear()
#         self._fake_buffer.clear()

#     def result(self) -> Mapping[Text, torch.Tensor]:
#         # 把剩下不到 buffer_size 的也算一次
#         if len(self._real_buffer) > 0:
#             self._flush_batch()

#         if not self._fvd_scores and not self._is_means:
#             raise ValueError("No examples to evaluate.")

#         eval_score = {}
#         if self._enable_fvd:
#             # 简单算术平均
#             eval_score["FVD"] = sum(self._fvd_scores) / len(self._fvd_scores)
#         if self._enable_inception_score:
#             eval_score["InceptionScore_mean"] = sum(self._is_means) / len(self._is_means)
#             eval_score["InceptionScore_std"]  = sum(self._is_stds)  / len(self._is_stds)

#         return eval_score

