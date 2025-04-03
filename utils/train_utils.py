"""Training utils for TiTok.

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
import json
import os
import time
import math
from pathlib import Path
import pprint
import glob
from collections import defaultdict
import open_clip

from dataset.data import DecordVideoDataset
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from torch.optim import AdamW
from utils.lr_schedulers import get_scheduler
from modeling.modules import EMAModel, ReconstructionLoss_Stage1, ReconstructionLoss_Stage2, ReconstructionLoss_Single_Stage
from modeling.titok import TiTok, PretrainedTokenizer
# from modeling.tatitok import TATiTok
from modeling.titok import TiTok3D
from modeling.maskgit import ImageBert, UViTBert
from evaluator.evaluator import VideoEvaluator
from demo_util import get_titok_tokenizer, sample_fn

# from imagenet_classes import imagenet_idx2classname
from utils.viz_utils import make_viz_from_samples, make_viz_from_samples_generation, make_viz_from_samples_t2i_generation
from utils.video_utils import save_video_imageio
from torchinfo import summary


def get_config():
    """Reads configs from a yaml file and terminal."""
    cli_conf = OmegaConf.from_cli()

    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)

    return conf


class AverageMeter(object):
    """Computes and stores the average and current value.
    
    This class is borrowed from
    https://github.com/pytorch/examples/blob/main/imagenet/main.py#L423
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_pretrained_tokenizer(config, accelerator=None):
    if config.model.vq_model.finetune_decoder:
        # No need of pretrained tokenizer at stage2
        pretrianed_tokenizer = None
    else:
        pretrianed_tokenizer = PretrainedTokenizer(config.model.vq_model.pretrained_tokenizer_weight)
        if accelerator is not None:
            pretrianed_tokenizer.to(accelerator.device)
    return pretrianed_tokenizer


def create_clip_model():
    clip, _, _ = open_clip.create_model_and_transforms('ViT-L-14-336', pretrained='openai')
    del clip.visual
    tokenizer = open_clip.get_tokenizer('ViT-L-14-336')
    clip.transformer.batch_first = False
    clip.eval()
    clip.requires_grad_(False)
    return clip, tokenizer


def create_model_and_loss_module(config, logger, accelerator,
                                 model_type="titok3D",show_summary = False):
    """Creates TiTok model and loss module."""
    logger.info("Creating model and loss module.")
    if model_type == "titok":
        model_cls = TiTok
        loss_cls = ReconstructionLoss_Stage2 if config.model.vq_model.finetune_decoder else ReconstructionLoss_Stage1
    elif model_type == "titok3D":
        model_cls = TiTok3D
        loss_cls = ReconstructionLoss_Single_Stage 
    else:
        raise ValueError(f"Unsupported model_type {model_type}")
    model = model_cls(config)

    if config.experiment.get("init_weight", ""):
        # If loading a pretrained weight
        model_weight = torch.load(config.experiment.init_weight, map_location="cpu")
        if config.model.vq_model.finetune_decoder:
            # Add the MaskGIT-VQGAN's quantizer/decoder weight as well
            pretrained_tokenizer_weight = torch.load(
                config.model.vq_model.pretrained_tokenizer_weight, map_location="cpu"
            )
            # Only keep the quantize and decoder part
            pretrained_tokenizer_weight = {"pixel_" + k:v for k,v in pretrained_tokenizer_weight.items() if not "encoder." in k}
            model_weight.update(pretrained_tokenizer_weight)
        
        msg = model.load_state_dict(model_weight, strict=False)
        logger.info(f"loading weight from {config.experiment.init_weight}, msg: {msg}")

    # Create the EMA model.
    ema_model = None
    if config.training.use_ema:
        ema_model = EMAModel(model.parameters(), decay=0.999,
                            model_cls=model_cls, config=config)
        # Create custom saving and loading hooks so that `accelerator.save_state(...)` serializes in a nice format.
        def load_model_hook(models, input_dir):
            load_model = EMAModel.from_pretrained(os.path.join(input_dir, "ema_model"),
                                                  model_cls=model_cls, config=config)
            ema_model.load_state_dict(load_model.state_dict())
            ema_model.to(accelerator.device)
            del load_model

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                ema_model.save_pretrained(os.path.join(output_dir, "ema_model"))

        accelerator.register_load_state_pre_hook(load_model_hook)
        accelerator.register_save_state_pre_hook(save_model_hook)

    # Create loss module along with discrminator.
    loss_module = loss_cls(config=config) if loss_cls is not None else None

    # Print Model for sanity check.
    if accelerator.is_main_process:
        if model_type in ["titok"]:
            input_size = (1, 3, config.dataset.preprocessing.crop_size, config.dataset.preprocessing.crop_size)
            if show_summary:
                model_summary_str = summary(model, input_size=input_size, depth=5,
                col_names=("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds"))
                logger.info(model_summary_str)
        elif model_type in ["titok3D"]:
            input_video_size  = (1, 3,config.dataset.preprocessing.temporal_size ,config.dataset.preprocessing.spatial_size, config.dataset.preprocessing.spatial_size)
            input_size = input_video_size
            if show_summary:
                model_summary_str = summary(model, input_size=input_size, depth=5,
                col_names=("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds"))
                logger.info(model_summary_str)
        else:
            raise NotImplementedError

    return model, ema_model, loss_module


def create_optimizer(config, logger, model, loss_module,
                     model_type="titok", need_discrminator=True):
    """Creates optimizer for TiTok and discrminator."""
    logger.info("Creating optimizers.")
    optimizer_config = config.optimizer.params
    learning_rate = optimizer_config.learning_rate

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer_cls = AdamW
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    # Exclude terms we may not want to apply weight decay.
    exclude = (lambda n, p: p.ndim < 2 or "ln" in n or "bias" in n or 'latent_tokens' in n 
               or 'mask_token' in n or 'embedding' in n or 'norm' in n or 'gamma' in n or 'embed' in n)
    include = lambda n, p: not exclude(n, p)
    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
    optimizer = optimizer_cls(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.},
            {"params": rest_params, "weight_decay": optimizer_config.weight_decay},
        ],
        lr=learning_rate,
        betas=(optimizer_config.beta1, optimizer_config.beta2)
    )

    if (config.model.vq_model.finetune_decoder or model_type == "tatitok") and need_discrminator:
        discriminator_learning_rate = optimizer_config.discriminator_learning_rate
        discriminator_named_parameters = list(loss_module.named_parameters())
        discriminator_gain_or_bias_params = [p for n, p in discriminator_named_parameters if exclude(n, p) and p.requires_grad]
        discriminator_rest_params = [p for n, p in discriminator_named_parameters if include(n, p) and p.requires_grad]

        discriminator_optimizer = optimizer_cls(
            [
                {"params": discriminator_gain_or_bias_params, "weight_decay": 0.},
                {"params": discriminator_rest_params, "weight_decay": optimizer_config.weight_decay},
            ],
            lr=discriminator_learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2)
        )
    else:
        discriminator_optimizer = None

    return optimizer, discriminator_optimizer


def create_lr_scheduler(config, logger, accelerator, optimizer, discriminator_optimizer=None):
    """Creates learning rate scheduler for TiTok and discrminator."""
    logger.info("Creating lr_schedulers.")
    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps * accelerator.num_processes,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps * accelerator.num_processes,
        base_lr=config.lr_scheduler.params.learning_rate,
        end_lr=config.lr_scheduler.params.end_lr,
    )
    if discriminator_optimizer is not None:
        discriminator_lr_scheduler = get_scheduler(
            config.lr_scheduler.scheduler,
            optimizer=discriminator_optimizer,
            num_training_steps=config.training.max_train_steps * accelerator.num_processes - config.losses.discriminator_start,
            num_warmup_steps=config.lr_scheduler.params.warmup_steps * accelerator.num_processes,
            base_lr=config.lr_scheduler.params.learning_rate,
            end_lr=config.lr_scheduler.params.end_lr,
        )
    else:
        discriminator_lr_scheduler = None
    return lr_scheduler, discriminator_lr_scheduler


def create_dataloader(config, logger, accelerator):
    """Creates data loader for training and testing."""
    logger.info("Creating dataloaders.")
    # total_batch_size_without_accum = config.training.per_gpu_batch_size * accelerator.num_processes
    # total_batch_size = (
    #     config.training.per_gpu_batch_size * accelerator.num_processes * config.training.gradient_accumulation_steps
    # )

    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params
    loader_config = config.dataset.loader

    train_dataset = DecordVideoDataset(
        data_folder= dataset_config.train_path,
        data_list= dataset_config.train_list,
        fps= dataset_config.fps ,
        sequence_length= preproc_config.temporal_size,
        train = True,
        resolution = preproc_config.spatial_size,
        resizecrop= preproc_config.get("resizecrop",False) ,       
    )
    eval_dataset = DecordVideoDataset(
        data_folder= dataset_config.eval_path,
        data_list= dataset_config.eval_list,
        fps= dataset_config.fps, 
        sequence_length= preproc_config.temporal_size,
        train = False,
        resolution = preproc_config.spatial_size,
        resizecrop= preproc_config.get("resizecrop",False),
    )
    
    train_dataloader = DataLoader(
        dataset= train_dataset,
        batch_size= config.training.per_gpu_batch_size,
        shuffle= loader_config.shuffle,
        drop_last= loader_config.drop_last,
        pin_memory= loader_config.pin_memory,
        num_workers= loader_config.num_workers,
    )

    eval_dataloader = DataLoader(
        dataset= eval_dataset,
        batch_size= 1,
        shuffle= False,
        drop_last= loader_config.drop_last,
        pin_memory= loader_config.pin_memory,
        num_workers= loader_config.num_workers,
    )    

    return train_dataloader, eval_dataloader


def create_evaluator(config, logger, accelerator):
    """Creates evaluator."""
    logger.info("Creating evaluator.")
    if config.model.vq_model.get("quantize_mode", "vq") == "vq":
        raise NotImplementedError
    elif config.model.vq_model.get("quantize_mode", "vq") == "vae":
        evaluator = VideoEvaluator(
            device=accelerator.device,
            enable_fvd=True,
            enable_inception_score=True,
            clip_time_stamp= config.dataset.preprocessing.temporal_size,
            fvd_method=config.eval.method,
            need_channel_adjustment_fvd=config.eval.need_channel_adjustment_fvd,
            need_channel_adjustment_is=config.eval.need_channel_adjustment_is,
            is_splits=config.eval.is_splits,
        )
        
    else:
        raise NotImplementedError
    return evaluator


def auto_resume(config, logger, accelerator, ema_model,
                num_update_steps_per_epoch, strict=True):
    """Auto resuming the training."""
    global_step = 0
    first_epoch = 0
    # If resuming training.
    if config.experiment.resume:            
        accelerator.wait_for_everyone()
        local_ckpt_list = list(glob.glob(os.path.join(
            config.experiment.output_dir, "checkpoint*")))
        logger.info(f"All globbed checkpoints are: {local_ckpt_list}")
        if len(local_ckpt_list) >= 1:
            if len(local_ckpt_list) > 1:
                fn = lambda x: int(x.split('/')[-1].split('-')[-1])
                checkpoint_paths = sorted(local_ckpt_list, key=fn, reverse=True)
            else:
                checkpoint_paths = local_ckpt_list
            global_step = load_checkpoint(
                Path(checkpoint_paths[0]),
                accelerator,
                logger=logger,
                strict=strict
            )
            if config.training.use_ema:
                ema_model.set_step(global_step)
            first_epoch = global_step // num_update_steps_per_epoch
        else:
            logger.info("Training from scratch.")
    return global_step, first_epoch


def train_one_epoch(config, logger, accelerator,
                    model, ema_model, loss_module,
                    optimizer, discriminator_optimizer,
                    lr_scheduler, discriminator_lr_scheduler,
                    train_dataloader, eval_dataloader,
                    evaluator,
                    global_step,
                    model_type="titok3D",
                    clip_tokenizer=None,
                    clip_encoder=None,
                    pretrained_tokenizer=None):
    """One epoch training."""
    batch_time_meter = AverageMeter()
    data_time_meter = AverageMeter()
    end = time.time()

    model.train()

    autoencoder_logs = defaultdict(float)
    discriminator_logs = defaultdict(float)
    for i, batch in enumerate(train_dataloader):
        model.train()
        if "video" in batch:
            videos = batch["video"].to(
                accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
            )
        # if "text" in batch and model_type == "tatitok":
        #     text = batch["text"]
        #     with torch.no_grad():
        #         text_guidance = clip_tokenizer(text).to(accelerator.device)
        #         cast_dtype = clip_encoder.transformer.get_cast_dtype()
        #         text_guidance = clip_encoder.token_embedding(text_guidance).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        #         text_guidance = text_guidance + clip_encoder.positional_embedding.to(cast_dtype)
        #         text_guidance = text_guidance.permute(1, 0, 2)  # NLD -> LND
        #         text_guidance = clip_encoder.transformer(text_guidance, attn_mask=clip_encoder.attn_mask)
        #         text_guidance = text_guidance.permute(1, 0, 2)  # LND -> NLD
        #         text_guidance = clip_encoder.ln_final(text_guidance)  # [batch_size, n_ctx, transformer.width]


        data_time_meter.update(time.time() - end)

        # Obtain proxy codes
        if pretrained_tokenizer is not None:
            pretrained_tokenizer.eval()
            proxy_codes = pretrained_tokenizer.encode(videos)
        else:
            proxy_codes = None

        with accelerator.accumulate([model, loss_module]):
            loss_module.to(accelerator.device)
            if model_type == "titok3D":
                reconstructed_images, extra_results_dict = model(videos)
                if proxy_codes is None:
                    autoencoder_loss, loss_dict = loss_module(
                        videos,
                        reconstructed_images,
                        extra_results_dict,
                        global_step,
                        mode="generator",
                    )
                else:
                    
                    autoencoder_loss, loss_dict = loss_module(
                        proxy_codes,
                        reconstructed_images,
                        extra_results_dict
                    )    
            # elif model_type == "tatitok":
            #     reconstructed_images, extra_results_dict = model(images, text_guidance)
            #     autoencoder_loss, loss_dict = loss_module(
            #         images,
            #         reconstructed_images,
            #         extra_results_dict,
            #         global_step,
            #         mode="generator",
            #     )
            else:
                raise NotImplementedError

            # Gather the losses across all processes for logging.
            autoencoder_logs = {}
            for k, v in loss_dict.items():
                if k in ["discriminator_factor", "d_weight"]:
                    if type(v) == torch.Tensor:
                        autoencoder_logs["train/" + k] = v.cpu().item()
                    else:
                        autoencoder_logs["train/" + k] = v
                else:
                    autoencoder_logs["train/" + k] = accelerator.gather(v).mean().item()

            accelerator.backward(autoencoder_loss)

            if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()

            # Log gradient norm before zeroing it.
            if (
                accelerator.sync_gradients
                and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                and accelerator.is_main_process
            ):
                log_grad_norm(model, accelerator, global_step + 1)

            optimizer.zero_grad(set_to_none=True)

            # Train discriminator.
            discriminator_logs = defaultdict(float)
            if (config.model.vq_model.finetune_decoder or model_type == "tatitok") and accelerator.unwrap_model(loss_module).should_discriminator_be_trained(global_step):
                discriminator_logs = defaultdict(float)
                discriminator_loss, loss_dict_discriminator = loss_module(
                    videos,
                    reconstructed_images,
                    extra_results_dict,
                    global_step=global_step,
                    mode="discriminator",
                )

                # Gather the losses across all processes for logging.
                for k, v in loss_dict_discriminator.items():
                    if k in ["logits_real", "logits_fake"]:
                        if type(v) == torch.Tensor:
                            discriminator_logs["train/" + k] = v.cpu().item()
                        else:
                            discriminator_logs["train/" + k] = v
                    else:
                        discriminator_logs["train/" + k] = accelerator.gather(v).mean().item()

                accelerator.backward(discriminator_loss)

                if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(loss_module.parameters(), config.training.max_grad_norm)

                discriminator_optimizer.step()
                discriminator_lr_scheduler.step()
        
                # Log gradient norm before zeroing it.
                if (
                    accelerator.sync_gradients
                    and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                    and accelerator.is_main_process
                ):
                    log_grad_norm(loss_module, accelerator, global_step + 1)
                
                discriminator_optimizer.zero_grad(set_to_none=True)

        if accelerator.sync_gradients:
            if config.training.use_ema:
                ema_model.step(model.parameters())
            batch_time_meter.update(time.time() - end)
            end = time.time()

            if (global_step + 1) % config.experiment.log_every == 0:
                samples_per_second_per_gpu = (
                    config.training.gradient_accumulation_steps * config.training.per_gpu_batch_size / batch_time_meter.val
                )

                lr = lr_scheduler.get_last_lr()[0]
                logger.info(
                    f"Data (t): {data_time_meter.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                    f"Batch (t): {batch_time_meter.val:0.4f} "
                    f"LR: {lr:0.6f} "
                    f"Step: {global_step + 1} "
                    f"Total Loss: {autoencoder_logs['train/total_loss']:0.4f} "
                    f"Recon Loss: {autoencoder_logs['train/reconstruction_loss']:0.4f} "
                    f"perceptual_loss: {autoencoder_logs['train/perceptual_loss']:0.4f} "
                    f"kl Loss: {autoencoder_logs['train/kl_loss']:0.4f} "
                    f"weighted_gan_loss Loss: {autoencoder_logs['train/weighted_gan_loss']:0.4f} "
                    f"discriminator_factor: {autoencoder_logs['train/discriminator_factor']:0.4f} "
                    f"d_weight: {autoencoder_logs['train/d_weight']:0.4f} "
                    f"gan_loss: {autoencoder_logs['train/gan_loss']:0.4f} "
                )
                logs = {
                    "lr": lr,
                    "lr/generator": lr,
                    "samples/sec/gpu": samples_per_second_per_gpu,
                    "time/data_time": data_time_meter.val,
                    "time/batch_time": batch_time_meter.val,
                }
                logs.update(autoencoder_logs)
                logs.update(discriminator_logs)
                accelerator.log(logs, step=global_step + 1)

                # Reset batch / data time meters per log window.
                batch_time_meter.reset()
                data_time_meter.reset()

            # Save model checkpoint.
            if (global_step + 1) % config.experiment.save_every == 0:
                save_path = save_checkpoint(
                    model, config.experiment.output_dir, accelerator, global_step + 1, logger=logger)
                # Wait for everyone to save their checkpoint.
                accelerator.wait_for_everyone()

            # Generate images.
            if (global_step + 1) % config.experiment.generate_every == 0 and accelerator.is_main_process:
                # Store the model parameters temporarily and load the EMA parameters to perform inference.
                if config.training.get("use_ema", False):
                    ema_model.store(model.parameters())
                    ema_model.copy_to(model.parameters())

                reconstruct_videos(
                    model,
                    videos[:config.training.num_generated_videos],
                    accelerator,
                    global_step + 1,
                    config.experiment.output_dir,
                    logger=logger,
                    config=config,
                    model_type=model_type,
                    text_guidance= None,
                    pretrained_tokenizer=None
                )

                if config.training.get("use_ema", False):
                    # Switch back to the original model parameters for training.
                    ema_model.restore(model.parameters())


            # Evaluate reconstruction.
            if eval_dataloader is not None and (global_step + 1) % config.experiment.eval_every == 0 and accelerator.is_main_process:
                logger.info(f"Computing metrics on the validation set.")
                if config.training.get("use_ema", False):
                    ema_model.store(model.parameters())
                    ema_model.copy_to(model.parameters())
                    # Eval for EMA.
                    eval_scores = eval_reconstruction(
                        model,
                        eval_dataloader,
                        accelerator,
                        evaluator,
                        model_type=model_type,
                        clip_tokenizer=clip_tokenizer,
                        clip_encoder=clip_encoder,
                        pretrained_tokenizer=pretrained_tokenizer
                    )
                    logger.info(
                        f"EMA EVALUATION "
                        f"Step: {global_step + 1} "
                    )
                    logger.info(pprint.pformat(eval_scores))
                    if accelerator.is_main_process:
                        eval_log = {f'ema_eval/'+k: v for k, v in eval_scores.items()}
                        accelerator.log(eval_log, step=global_step + 1)
                    if config.training.get("use_ema", False):
                        # Switch back to the original model parameters for training.
                        ema_model.restore(model.parameters())
                else:
                    # Eval for non-EMA.
                    eval_scores = eval_reconstruction(
                        model,
                        eval_dataloader,
                        accelerator,
                        evaluator,
                        model_type=model_type,
                        clip_tokenizer=clip_tokenizer,
                        clip_encoder=clip_encoder,
                        pretrained_tokenizer=pretrained_tokenizer
                    )

                    logger.info(
                        f"Non-EMA EVALUATION "
                        f"Step: {global_step + 1} "
                    )
                    logger.info(pprint.pformat(eval_scores))
                    if accelerator.is_main_process:
                        eval_log = {f'eval/'+k: v for k, v in eval_scores.items()}
                        accelerator.log(eval_log, step=global_step + 1)

                accelerator.wait_for_everyone()

            global_step += 1

            if global_step >= config.training.max_train_steps:
                accelerator.print(
                    f"Finishing training: Global step is >= Max train steps: {global_step} >= {config.training.max_train_steps}"
                )
                break


    return global_step


def get_rar_random_ratio(config, cur_step):
    randomness_anneal_start = config.model.generator.randomness_anneal_start
    randomness_anneal_end = config.model.generator.randomness_anneal_end
    if cur_step < randomness_anneal_start:
        return 1.0
    elif cur_step > randomness_anneal_end:
        return 0.0
    else:
        return 1.0 - (cur_step - randomness_anneal_start) / (randomness_anneal_end - randomness_anneal_start)


def train_one_epoch_generator(
                    config, logger, accelerator,
                    model, ema_model, loss_module,
                    optimizer,
                    lr_scheduler,
                    train_dataloader,
                    tokenizer,
                    global_step,
                    model_type="maskgit"):
    """One epoch training."""
    batch_time_meter = AverageMeter()
    data_time_meter = AverageMeter()
    end = time.time()

    model.train()

    for i, batch in enumerate(train_dataloader):
        model.train()
        if config.dataset.params.get("pretokenization", ""):
            # the data is already pre-tokenized
            conditions, input_tokens = batch
            input_tokens = input_tokens.to(
                accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
            )
            conditions = conditions.to(
                accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
            )
        else:
            # tokenize on the fly
            if "image" in batch:
                images = batch["image"].to(
                    accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
                )
                conditions = batch["class_id"].to(
                    accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
                )

                # Encode images on the flight.
                with torch.no_grad():
                    tokenizer.eval()
                    input_tokens = tokenizer.encode(images)[1]["min_encoding_indices"].reshape(images.shape[0], -1)
            else:
                raise ValueError(f"Not found valid keys: {batch.keys()}")

        data_time_meter.update(time.time() - end)

        unwrap_model = accelerator.unwrap_model(model)


        if model_type == "maskgit":
            # Randomly masking out input tokens.
            masked_tokens, masks = unwrap_model.masking_input_tokens(
                input_tokens)
        elif model_type == "rar":
            unwrap_model.set_random_ratio(get_rar_random_ratio(config, global_step))
        else:
            raise NotImplementedError
            

        with accelerator.accumulate([model]):

            if model_type == "maskgit":
                logits = model(masked_tokens, conditions,
                            cond_drop_prob=config.model.generator.class_label_dropout)
                loss, loss_dict= loss_module(logits, input_tokens, weights=masks)
            elif model_type == "rar":
                condition = unwrap_model.preprocess_condition(
                    conditions, cond_drop_prob=config.model.generator.class_label_dropout
                )
                logits, labels = model(input_tokens, condition, return_labels=True)
                loss, loss_dict = loss_module(logits, labels)
            # Gather the losses across all processes for logging.
            gen_logs = {}
            for k, v in loss_dict.items():
                gen_logs["train/" + k] = accelerator.gather(v).mean().item()
            accelerator.backward(loss)

            if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()

            # Log gradient norm before zeroing it.
            if (
                accelerator.sync_gradients
                and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                and accelerator.is_main_process
            ):
                log_grad_norm(model, accelerator, global_step + 1)

            optimizer.zero_grad(set_to_none=True)

        if accelerator.sync_gradients:
            if config.training.use_ema:
                ema_model.step(model.parameters())
            batch_time_meter.update(time.time() - end)
            end = time.time()

            if (global_step + 1) % config.experiment.log_every == 0:
                samples_per_second_per_gpu = (
                    config.training.gradient_accumulation_steps * config.training.per_gpu_batch_size / batch_time_meter.val
                )

                lr = lr_scheduler.get_last_lr()[0]
                logger.info(
                    f"Data (t): {data_time_meter.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                    f"Batch (t): {batch_time_meter.val:0.4f} "
                    f"LR: {lr:0.6f} "
                    f"Step: {global_step + 1} "
                    f"Loss: {gen_logs['train/loss']:0.4f} "
                    f"Accuracy: {gen_logs['train/correct_tokens']:0.4f} "
                )
                logs = {
                    "lr": lr,
                    "lr/generator": lr,
                    "samples/sec/gpu": samples_per_second_per_gpu,
                    "time/data_time": data_time_meter.val,
                    "time/batch_time": batch_time_meter.val,
                }
                logs.update(gen_logs)
                accelerator.log(logs, step=global_step + 1)

                # Reset batch / data time meters per log window.
                batch_time_meter.reset()
                data_time_meter.reset()

            # Save model checkpoint.
            if (global_step + 1) % config.experiment.save_every == 0:
                save_path = save_checkpoint(
                    model, config.experiment.output_dir, accelerator, global_step + 1, logger=logger)
                # Wait for everyone to save their checkpoint.
                accelerator.wait_for_everyone()

            # Generate images.
            if (global_step + 1) % config.experiment.generate_every == 0 and accelerator.is_main_process:
                # Store the model parameters temporarily and load the EMA parameters to perform inference.
                if config.training.get("use_ema", False):
                    ema_model.store(model.parameters())
                    ema_model.copy_to(model.parameters())

                generate_images(
                    model,
                    tokenizer,
                    accelerator,
                    global_step + 1,
                    config.experiment.output_dir,
                    logger=logger,
                    config=config
                )

                if config.training.get("use_ema", False):
                    # Switch back to the original model parameters for training.
                    ema_model.restore(model.parameters())

            global_step += 1

            if global_step >= config.training.max_train_steps:
                accelerator.print(
                    f"Finishing training: Global step is >= Max train steps: {global_step} >= {config.training.max_train_steps}"
                )
                break


    return global_step


def train_one_epoch_t2i_generator(
                    config, logger, accelerator,
                    model, ema_model, loss_module,
                    optimizer,
                    lr_scheduler,
                    train_dataloader,
                    tokenizer,
                    clip_tokenizer,
                    clip_encoder,
                    global_step,
                    model_type="maskgen_vq"):
    """One epoch training."""
    batch_time_meter = AverageMeter()
    data_time_meter = AverageMeter()
    end = time.time()

    model.train()

    for i, batch in enumerate(train_dataloader):
        model.train()
        
        input_tokens = batch["tokens"].to(accelerator.device, memory_format=torch.contiguous_format, non_blocking=True)
        captions = batch["text"]
        if config.model.maskgen.micro_condition:
            aes_scores = batch["aes_score"].to(
                accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
            )
        else:
            aes_scores = None

        data_time_meter.update(time.time() - end)

        unwrap_model = accelerator.unwrap_model(model)

        condition, condition_pooled = unwrap_model.preprocess_condition(
            captions, clip_tokenizer, clip_encoder,
        )

        if model_type == "maskgen_vq":
            with accelerator.accumulate([model]):
                logits, masks = model(input_tokens, condition, condition_pooled, aes_scores)
                t2i_gen_loss, loss_dict = loss_module(logits, input_tokens, weights=masks)
        elif model_type == "maskgen_kl":
            with accelerator.accumulate([model]):
                t2i_gen_loss, loss_dict = model(input_tokens, condition, condition_pooled, aes_scores)
        else:
            raise NotImplementedError

        with accelerator.accumulate([model]):
            # Gather the losses across all processes for logging.
            t2i_gen_logs = {}
            for k, v in loss_dict.items():
                t2i_gen_logs["train/" + k] = accelerator.gather(v).mean().item()
            accelerator.backward(t2i_gen_loss)

            if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()

            # Log gradient norm before zeroing it.
            if (
                accelerator.sync_gradients
                and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                and accelerator.is_main_process
            ):
                log_grad_norm(model, accelerator, global_step + 1)

            optimizer.zero_grad(set_to_none=True)

        if accelerator.sync_gradients:
            if config.training.use_ema:
                ema_model.step(model.parameters())
            batch_time_meter.update(time.time() - end)
            end = time.time()

            if (global_step + 1) % config.experiment.log_every == 0:
                samples_per_second_per_gpu = (
                    config.training.gradient_accumulation_steps * config.training.per_gpu_batch_size / batch_time_meter.val
                )

                lr = lr_scheduler.get_last_lr()[0]
                logger.info(
                    f"Data (t): {data_time_meter.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                    f"Batch (t): {batch_time_meter.val:0.4f} "
                    f"LR: {lr:0.6f} "
                    f"Step: {global_step + 1} "
                    f"Loss: {t2i_gen_logs['train/loss']:0.4f} "
                )
                logs = {
                    "lr": lr,
                    "lr/generator": lr,
                    "samples/sec/gpu": samples_per_second_per_gpu,
                    "time/data_time": data_time_meter.val,
                    "time/batch_time": batch_time_meter.val,
                }
                logs.update(t2i_gen_logs)
                accelerator.log(logs, step=global_step + 1)

                # Reset batch / data time meters per log window.
                batch_time_meter.reset()
                data_time_meter.reset()

            # Save model checkpoint.
            if (global_step + 1) % config.experiment.save_every == 0:
                save_path = save_checkpoint(
                    model, config.experiment.output_dir, accelerator, global_step + 1, logger=logger)
                # Wait for everyone to save their checkpoint.
                accelerator.wait_for_everyone()

            # Generate images.
            if (global_step + 1) % config.experiment.generate_every == 0 and accelerator.is_main_process:
                # Store the model parameters temporarily and load the EMA parameters to perform inference.
                if config.training.get("use_ema", False):
                    ema_model.store(model.parameters())
                    ema_model.copy_to(model.parameters())

                t2i_generate_images(
                    model,
                    tokenizer,
                    captions,
                    aes_scores,
                    clip_tokenizer,
                    clip_encoder,
                    accelerator,
                    global_step + 1,
                    config.experiment.output_dir,
                    logger=logger,
                    config=config,
                    model_type=model_type,
                )

                if config.training.get("use_ema", False):
                    # Switch back to the original model parameters for training.
                    ema_model.restore(model.parameters())

            global_step += 1

            if global_step >= config.training.max_train_steps:
                accelerator.print(
                    f"Finishing training: Global step is >= Max train steps: {global_step} >= {config.training.max_train_steps}"
                )
                break


    return global_step


@torch.no_grad()
def eval_reconstruction(
    model,
    eval_loader,
    accelerator,
    evaluator,
    model_type="titok3D",
    clip_tokenizer=None,
    clip_encoder=None,
    pretrained_tokenizer=None
):
    model.eval()
    evaluator.reset_metrics()
    local_model = accelerator.unwrap_model(model)

    for batch in eval_loader: # batch should be a [B,C,T,H,W] VIDEO TENSOR
        videos = batch["video"].to(
            accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
        )
        
        # if model_type == "tatitok":
        #     conditions = batch["class_id"]
        #     text = [f"A photo of a {imagenet_idx2classname[condition.item()]}." for condition in conditions]
        #     text_guidance = clip_tokenizer(text).to(accelerator.device)
        #     cast_dtype = clip_encoder.transformer.get_cast_dtype()
        #     text_guidance = clip_encoder.token_embedding(text_guidance).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        #     text_guidance = text_guidance + clip_encoder.positional_embedding.to(cast_dtype)
        #     text_guidance = text_guidance.permute(1, 0, 2)  # NLD -> LND
        #     text_guidance = clip_encoder.transformer(text_guidance, attn_mask=clip_encoder.attn_mask)
        #     text_guidance = text_guidance.permute(1, 0, 2)  # LND -> NLD
        #     text_guidance = clip_encoder.ln_final(text_guidance)  # [batch_size, n_ctx, transformer.width]

        original_videos = torch.clone(videos)
        if model_type == "titok3D":
            reconstructed_videos, model_dict = local_model(videos)
        # elif model_type == "tatitok":
        #     reconstructed_videos, model_dict = local_model(videos, text_guidance)
        else:
            raise NotImplementedError

        if pretrained_tokenizer is not None:
            reconstructed_videos = pretrained_tokenizer.decode(reconstructed_videos.argmax(1))
            
        reconstructed_videos = torch.clamp(reconstructed_videos, 0.0, 1.0)
        # Quantize to uint8
        reconstructed_videos = torch.round(reconstructed_videos * 255.0) / 255.0
        original_videos = torch.clamp(original_videos, 0.0, 1.0)
        reconstructed_videos = torch.round(reconstructed_videos * 255.0) / 255.0

        # For VQ model.
        # evaluator.update(original_videos, reconstructed_videos)
        # For VAE model. will fall in here
        evaluator.update(original_videos, reconstructed_videos)
            
    model.train()
    return evaluator.result()


# @torch.no_grad()
# def reconstruct_images(model, original_images, fnames, accelerator, 
#                     global_step, output_dir, logger, config=None,
#                     model_type="titok", text_guidance=None, 
#                     pretrained_tokenizer=None):
#     logger.info("Reconstructing images...")
#     original_images = torch.clone(original_images)
#     model.eval()
#     dtype = torch.float32
#     if accelerator.mixed_precision == "fp16":
#         dtype = torch.float16
#     elif accelerator.mixed_precision == "bf16":
#         dtype = torch.bfloat16

#     with torch.autocast("cuda", dtype=dtype, enabled=accelerator.mixed_precision != "no"):
#         enc_tokens, encoder_dict = accelerator.unwrap_model(model).encode(original_images)
    
#     if model_type == "titok":
#         reconstructed_images = accelerator.unwrap_model(model).decode(enc_tokens)
#     elif model_type == "tatitok":
#         reconstructed_images = accelerator.unwrap_model(model).decode(enc_tokens, text_guidance)
#     if pretrained_tokenizer is not None:
#         reconstructed_images = pretrained_tokenizer.decode(reconstructed_images.argmax(1))
#     images_for_saving, images_for_logging = make_viz_from_samples(
#         original_images,
#         reconstructed_images
#     )
#     # Log images.
#     if config.training.enable_wandb:
#         accelerator.get_tracker("wandb").log_images(
#             {f"Train Reconstruction": images_for_saving},
#             step=global_step
#         )
#     else:
#         accelerator.get_tracker("tensorboard").log_images(
#             {"Train Reconstruction": images_for_logging}, step=global_step
#         )
#     # Log locally.
#     root = Path(output_dir) / "train_images"
#     os.makedirs(root, exist_ok=True)
#     for i,img in enumerate(images_for_saving):
#         filename = f"{global_step:08}_s-{i:03}-{fnames[i]}.png"
#         path = os.path.join(root, filename)
#         img.save(path)

#     model.train()

@torch.no_grad()
def reconstruct_videos(model, original_videos, accelerator, 
                    global_step, output_dir, logger, config=None,
                    model_type="titok3D", text_guidance=None, 
                    pretrained_tokenizer=None):
    logger.info("Reconstructing videos...")
    original_videos = torch.clone(original_videos)
    model.eval()
    dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype = torch.bfloat16

    with torch.autocast("cuda", dtype=dtype, enabled=accelerator.mixed_precision != "no"):
        enc_tokens, encoder_dict = accelerator.unwrap_model(model).encode(original_videos)
    
    if model_type == "titok3D":
        reconstructed_videos = accelerator.unwrap_model(model).decode(enc_tokens)
    # elif model_type == "tatitok":
    #     reconstructed_images = accelerator.unwrap_model(model).decode(enc_tokens, text_guidance)

    # we don't do video visualization here. We only save them to pathes
    
    # if pretrained_tokenizer is not None:
    #     reconstructed_images = pretrained_tokenizer.decode(reconstructed_images.argmax(1))
    # images_for_saving, images_for_logging = make_viz_from_samples(
    #     original_images,
    #     reconstructed_images
    # )
    # Log images.
    # if config.training.enable_wandb:
    #     accelerator.get_tracker("wandb").log_images(
    #         {f"Train Reconstruction": images_for_saving},
    #         step=global_step
    #     )
    # else:
    #     accelerator.get_tracker("tensorboard").log_images(
    #         {"Train Reconstruction": images_for_logging}, step=global_step
    #     )
    # Log locally.
    
    root = Path(output_dir) / "train_vids"
    os.makedirs(root, exist_ok=True)
    mp4root = root / "mp4"
    aviroot = root / "avi"
    
    for i,vid in enumerate(reconstructed_videos):
        save_video_imageio(vid,aviroot,mp4root)
    model.train()


@torch.no_grad()
def generate_images(model, tokenizer, accelerator, 
                    global_step, output_dir, logger, config=None):
    model.eval()
    tokenizer.eval()
    logger.info("Generating images...")
    generated_image = sample_fn(
        accelerator.unwrap_model(model),
        tokenizer,
        guidance_scale=config.model.generator.get("guidance_scale", 3.0),
        guidance_decay=config.model.generator.get("guidance_decay", "constant"),
        guidance_scale_pow=config.model.generator.get("guidance_scale_pow", 3.0),
        randomize_temperature=config.model.generator.get("randomize_temperature", 2.0),
        softmax_temperature_annealing=config.model.generator.get("softmax_temperature_annealing", False),
        num_sample_steps=config.model.generator.get("num_steps", 8),
        device=accelerator.device,
        return_tensor=True
    )
    images_for_saving, images_for_logging = make_viz_from_samples_generation(
        generated_image)

    # Log images.
    if config.training.enable_wandb:
        accelerator.get_tracker("wandb").log_images(
            {"Train Generated": [images_for_saving]}, step=global_step
        )
    else:
        accelerator.get_tracker("tensorboard").log_images(
            {"Train Generated": images_for_logging}, step=global_step
        )
    # Log locally.
    root = Path(output_dir) / "train_generated_images"
    os.makedirs(root, exist_ok=True)
    filename = f"{global_step:08}_s-generated.png"
    path = os.path.join(root, filename)
    images_for_saving.save(path)

    model.train()
    return


@torch.no_grad()
def t2i_generate_images(model, tokenizer, captions, aes_scores, clip_tokenizer, clip_encoder, accelerator, 
                    global_step, output_dir, logger, config=None, model_type="maskgen_kl"):
    model.eval()
    tokenizer.eval()
    local_model = accelerator.unwrap_model(model)
    logger.info("Generating images...")

    if model_type == "maskgen_vq":
        tokens = local_model.generate(
            captions=captions[:config.training.num_generated_images],
            sample_aesthetic_score=aes_scores[:config.training.num_generated_images] if config.model.maskgen.micro_condition else None,
            num_steps=config.model.maskgen.get("num_iter", 16),
            guidance_scale=config.model.maskgen.cfg,
            guidance_decay=config.model.maskgen.cfg_schedule,
            clip_tokenizer=clip_tokenizer,
            clip_encoder=clip_encoder,
            guidance_decay_scale_pow=config.model.maskgen.cfg_decay_scale_pow,
            randomize_temperature=config.model.maskgen.randomize_temperature,
            softmax_temperature_annealing=config.model.maskgen.get("softmax_temperature_annealing", True),
            prob_sorting=config.model.maskgen.get("prob_sorting", True)
        )
    elif model_type == "maskgen_kl":
        tokens = local_model.sample_tokens(config.training.num_generated_images, 
            clip_tokenizer=clip_tokenizer, clip_encoder=clip_encoder, 
            captions=captions[:config.training.num_generated_images], 
            aes_scores=aes_scores[:config.training.num_generated_images] if config.model.maskgen.micro_condition else None,
            num_iter=config.model.maskgen.num_iter, cfg_schedule=config.model.maskgen.cfg_schedule,
            cfg=config.model.maskgen.cfg, temperature=config.model.maskgen.temperature
        )
    else:
        raise NotImplementedError

    text_guidance = clip_tokenizer(captions[:config.training.num_generated_images]).to(accelerator.device)
    cast_dtype = clip_encoder.transformer.get_cast_dtype()
    text_guidance = clip_encoder.token_embedding(text_guidance).to(cast_dtype)  # [batch_size, n_ctx, d_model]
    text_guidance = text_guidance + clip_encoder.positional_embedding.to(cast_dtype)
    text_guidance = text_guidance.permute(1, 0, 2)  # NLD -> LND
    text_guidance = clip_encoder.transformer(text_guidance, attn_mask=clip_encoder.attn_mask)
    text_guidance = text_guidance.permute(1, 0, 2)  # LND -> NLD
    text_guidance = clip_encoder.ln_final(text_guidance)  # [batch_size, n_ctx, transformer.width]
    
    generated_image = tokenizer.decode_tokens(tokens, text_guidance=text_guidance)

    images_for_saving, images_for_logging = make_viz_from_samples_t2i_generation(generated_image, captions[:config.training.num_generated_images])

    # Log images.
    if config.training.enable_wandb:
        accelerator.get_tracker("wandb").log_images(
            {"Train Generated": [images_for_saving]}, step=global_step
        )
    else:
        accelerator.get_tracker("tensorboard").log_images(
            {"Train Generated": images_for_logging}, step=global_step
        )
    # Log locally.
    root = Path(output_dir) / "train_generated_images"
    os.makedirs(root, exist_ok=True)
    filename = f"{global_step:08}_s-generated.png"
    path = os.path.join(root, filename)
    images_for_saving.save(path)

    model.train()
    return


def save_checkpoint(model, output_dir, accelerator, global_step, logger) -> Path:
    save_path = Path(output_dir) / f"checkpoint-{global_step}"

    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained_weight(
            save_path / "unwrapped_model",
            save_function=accelerator.save,
            state_dict=state_dict,
        )
        json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))
        logger.info(f"Saved state to {save_path}")

    accelerator.save_state(save_path)
    return save_path


def load_checkpoint(checkpoint_path: Path, accelerator, logger, strict=True):
    logger.info(f"Load checkpoint from {checkpoint_path}")

    accelerator.load_state(checkpoint_path, strict=strict)
    
    with open(checkpoint_path / "metadata.json", "r") as f:
        global_step = int(json.load(f)["global_step"])

    logger.info(f"Resuming at global_step {global_step}")
    return global_step


def log_grad_norm(model, accelerator, global_step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)