"""Training script for TiTok.

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
    https://github.com/huggingface/open-muse
"""
import math
import os
import sys
sys.path.append(os.getcwd())
from pathlib import Path
import torch.distributed as dist 
from accelerate.utils import set_seed
from accelerate import Accelerator

import torch
from omegaconf import OmegaConf
from utils.logger import setup_logger

from utils.train_utils import (
    get_config, create_pretrained_tokenizer, 
    create_model_and_loss_module,
    create_optimizer, create_lr_scheduler, create_dataloader,
    create_evaluator, auto_resume, save_checkpoint, 
    train_one_epoch)

    
def call_code_exit():
    cleanup()
    print(":)")
    exit(0)    

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def main():
    workspace = os.environ.get('WORKSPACE', '')
    if workspace:
        torch.hub.set_dir(workspace + "/models/hub")

    config = get_config()
    # Enable TF32 on Ampere GPUs.
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    output_dir = config.experiment.output_dir
    os.makedirs(output_dir, exist_ok=True)
    config.experiment.logging_dir = os.path.join(output_dir, "logs")

    # Whether logging to Wandb or Tensorboard.
    tracker = "tensorboard"
    if config.training.enable_wandb:
        tracker = "wandb"
    
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with=tracker,
        project_dir=config.experiment.logging_dir,
        split_batches=False,
    )

    logger = setup_logger(name="TiTok", log_level="INFO",
    output_file=f"{output_dir}/log{accelerator.process_index}.txt")

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(config.experiment.name)
        config_path = Path(output_dir) / "config.yaml"
        logger.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)
        logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed, device_specific=True)

    # if config.model.vq_model.quantize_mode == 'vq' and accelerator.local_process_index == 0: # VAE don't need this ; also FSQ don't need
    #     # download the maskgit-vq tokenizer weight
    #     from huggingface_hub import hf_hub_download
    #     hf_hub_download(repo_id="fun-research/TiTok", filename=f"{config.model.vq_model.pretrained_tokenizer_weight}", local_dir="./")
        
    accelerator.wait_for_everyone()
    
    pretrained_tokenizer = None
    # if config.model.vq_model.quantize_mode == 'vq':
    #     pretrained_tokenizer = create_pretrained_tokenizer(config,
    #                                                     accelerator)
    model_type = config.model.vq_model.get("model_type","titok3D")
    model, ema_model, loss_module = create_model_and_loss_module(
        config, logger, accelerator, model_type=model_type) 
    
    optimizer, discriminator_optimizer = create_optimizer(config, logger, model, loss_module)

    lr_scheduler, discriminator_lr_scheduler = create_lr_scheduler(
        config, logger, accelerator, optimizer, discriminator_optimizer)

    train_dataloader, eval_dataloader = create_dataloader(config, logger, accelerator)


    # Set up evaluator.
    evaluator = create_evaluator(config, logger, accelerator)

    # Prepare everything with accelerator.
    logger.info("Preparing model, optimizer and dataloaders") 
    
    if config.model.vq_model.get("need_discriminator",False): # for vq
        model, loss_module, optimizer, discriminator_optimizer, lr_scheduler, discriminator_lr_scheduler = accelerator.prepare(
            model, loss_module, optimizer, discriminator_optimizer, lr_scheduler, discriminator_lr_scheduler
        )
    else:
        model, optimizer, lr_scheduler,train_dataloader,eval_dataloader = accelerator.prepare(
            model, optimizer, lr_scheduler,train_dataloader,eval_dataloader
        )
    if config.training.use_ema:
        ema_model.to(accelerator.device)

    total_batch_size_without_accum = config.training.per_gpu_batch_size * accelerator.num_processes # 8
    num_batches = math.ceil(
        config.experiment.max_train_examples / total_batch_size_without_accum) # 13000/8=1625 batch per epoch
    num_update_steps_per_epoch = math.ceil(num_batches / config.training.gradient_accumulation_steps) # 1625 updates per epoch
    num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch) # 650000/ 342 400 000

    # Start training.
    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")
    logger.info(f"  Instantaneous batch size per gpu = { config.training.per_gpu_batch_size}")
    logger.info(f"""  Total train batch size (w. parallel, distributed & accumulation) = {(
        config.training.per_gpu_batch_size *
        accelerator.num_processes *
        config.training.gradient_accumulation_steps)}""")
    logger.info(f"    totally {num_train_epochs} epochs")
    global_step = 0
    first_epoch = 0


    
    global_step, first_epoch = auto_resume(
        config, logger, accelerator, ema_model, num_update_steps_per_epoch,
        strict=True)
    logger.info(f"global_step:{global_step} ,first_epoch: {first_epoch}")
    if config.training.get("epoch",0) != 0:
        num_train_epochs =  config.training.get("epoch")
    logger.info(f"global_step:{global_step} ,first_epoch: {first_epoch},num_train_epochs:{num_train_epochs}")
    first_epoch = math.ceil(first_epoch/accelerator.num_processes)
    for current_epoch in range(first_epoch, num_train_epochs):
        accelerator.print(f"Epoch {current_epoch}/{num_train_epochs-1} started.")
        global_step = train_one_epoch(config, logger, accelerator,
                            model, ema_model, loss_module,
                            optimizer, discriminator_optimizer,
                            lr_scheduler, discriminator_lr_scheduler,
                            train_dataloader, eval_dataloader,
                            evaluator,
                            global_step,
                            model_type=model_type,
                            pretrained_tokenizer=pretrained_tokenizer)
        # Stop training if max steps is reached.
        if global_step >= config.training.max_train_steps:
            accelerator.print(
                f"Finishing training: Global step is >= Max train steps: {global_step} >= {config.training.max_train_steps}"
            )
            break


    accelerator.wait_for_everyone()
    # Save checkpoint at the end of training.
    save_checkpoint(model, output_dir, accelerator, global_step, logger=logger)
    # Save the final trained checkpoint
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        if config.training.use_ema:
            ema_model.copy_to(model.parameters())
        model.save_pretrained_weight(output_dir)
    accelerator.end_training()


if __name__ == "__main__":
    main()
