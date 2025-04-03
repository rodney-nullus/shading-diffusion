import math, logging, os, random, shutil
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as tvf

import datasets, transformers, diffusers
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from safetensors.torch import load_model

from diffusers.training_utils import EMAModel, compute_snr

from models.vlm import VLM
from models.projector import Projector

from dataloader.celeba_pbr import get_dataloader

from utils.writer import Writer

class Trainer:
    def __init__(self, diff_model, configs):
        
        self.logger = get_logger(__name__)
        self.configs = configs
        
        if isinstance(configs.resolution, set):
            self.width, self.height = configs.resolution
        else:
            self.width, self.height = configs.resolution, configs.resolution
        
        self.project_dir = f"{configs.output_dir}/{configs.exp_name}/{configs.train_model}"
        
        # Initialize accelerator and logger for training
        project_config = ProjectConfiguration(project_dir=self.project_dir)
        self.accelerator = Accelerator(
            mixed_precision=configs.mixed_precision,
            gradient_accumulation_steps=configs.gradient_accumulation_steps,
            project_config=project_config
        )
        self.device = self.accelerator.device
        Writer.set_writer(results_dir=self.logs_dir)
        
        # Disable AMP for MPS.
        if torch.backends.mps.is_available():
            self.accelerator.native_amp = False
        
        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        self.logger.info(self.accelerator.state, main_process_only=False)
        if self.accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()
        
        # If passed along, set the training seed now.
        if configs.random_seed is not None:
            set_seed(configs.random_seed)
        
        # Handle the repository creation
        if self.accelerator.is_local_main_process:
            self.checkpoints_dir = os.path.join(self.project_dir, "checkpoints")
            if not os.path.exists(self.checkpoints_dir):
                os.makedirs(self.checkpoints_dir, exist_ok=True)
            
            # Create logs dir
            self.logs_dir = os.path.join(self.project_dir, "logs", configs.train_phase)
            if not os.path.exists(self.logs_dir):
                os.makedirs(self.logs_dir, exist_ok=True)
            
        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if configs.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        if configs.scale_lr:
            configs.learning_rate = (
                configs.learning_rate * configs.gradient_accumulation_steps * configs.train_batch_size * self.accelerator.num_processes
            )
        
        # For mixed precision training we cast all non-trainable weights to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
            configs.mixed_precision = self.accelerator.mixed_precision
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16
            configs.mixed_precision = self.accelerator.mixed_precision
        
        if torch.backends.mps.is_available() and configs.mixed_precision == "bf16":
            # due to pytorch#99272, MPS does not yet support bfloat16.
            raise ValueError(
                "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
            )
        
        # Load training components
        # Load dataloader
        train_loader, eval_loader = get_dataloader(configs)
        
        if configs.train_phase == "vae":
            
            vae = diff_model.vae
            
            if configs.gradient_checkpointing:
                vae.enable_gradient_checkpointing()
            
            optimizer = torch.optim.AdamW(vae.parameters(), lr=configs.learning_rate)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=configs.total_train_epochs_vae)
            
            # Prepare everything
            # There is no specific order to remember, you just need to unpack the
            # objects in the same order you gave them to the prepare method.
            self.vae, self.optimizer, self.lr_scheduler, self.train_loader, self.eval_loader = self.accelerator.prepare(
                vae, optimizer, lr_scheduler, train_loader, eval_loader, 
            )
            
        elif configs.train_phase == "unet":
            
            # Get the most recent checkpoint of vae checkpoints
            dirs = os.listdir(self.checkpoints_dir)
            dirs = [d for d in dirs if d.startswith(f"checkpoint-{configs.train_model[:3]}-vae")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[3]))
            path = dirs[-1] if len(dirs) > 0 else None
            
            if path is not None:
                weight_path = f"{self.checkpoints_dir}/{path}/model.safetensors"
                load_model(diff_model, weight_path)
            
            # Load vae model
            vae = diff_model.vae.requires_grad_(False)
            self.vae = vae.to(self.device, self.weight_dtype)
            
            # Loda text encoder
            text_encoder = diff_model.text_encoder.require_grad_(False)
            self.text_encoders = text_encoder.to(self.device, self.weight_dtype)
            
            # Load unet model
            unet = diff_model.unet
            
            if configs.gradient_checkpointing:
                unet.enable_gradient_checkpointing()
            
            # Create EAM for the unet
            if configs.use_ema:
                
                from diffusers import UNet2DConditionModel
                
                ema_unet = EMAModel(
                    unet.parameters(),
                    model_cls=UNet2DConditionModel,
                    model_config=unet.config,
                    foreach=configs.foreach_ema,
                )
                
                if configs.offload_ema:
                    self.ema_unet = ema_unet.pin_memory()
                else:
                    self.ema_unet = ema_unet.to(self.device)
            
            # Load deepseek vl2
            vlm = VLM(configs=configs)
            vlm.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
            
            # Projector for contrastive learning
            #cl_projector = Projector(in_dims=diff_model.vae.encoder.conv_in.out_channels, hidden_dims=128)
            
            # Projector for adding rotation 6D
            r_projector = Projector(in_dims=1286, hidden_dims=1280, out_dims=1280)
            
            optimizer = torch.optim.AdamW(list(unet.parameters()) + list(r_projector.parameters()), lr=configs.learning_rate)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=configs.total_train_epochs_unet)
        
            # Prepare everything
            # There is no specific order to remember, you just need to unpack the
            # objects in the same order you gave them to the prepare method.
            self.unet, self.vlm, self.r_projector, self.optimizer, self.lr_scheduler, self.train_loader, self.eval_loader = self.accelerator.prepare(
                unet, vlm, r_projector, optimizer, lr_scheduler, train_loader, eval_loader
            )
        
        self.train_resize = transforms.Resize((self.height, self.width), interpolation=transforms.InterpolationMode.BILINEAR)
        self.train_crop = transforms.CenterCrop((self.height, self.width)) if configs.center_crop else transforms.RandomCrop(configs.resolution)
        self.train_flip = transforms.RandomHorizontalFlip(p=1.0)
        self.train_transforms = transforms.Compose([transforms.Normalize([0.5], [0.5])])
    
    def train(self):
        
        # Initial log
        total_batch_size = self.configs.train_batch_size * self.accelerator.num_processes * self.configs.gradient_accumulation_steps

        self.logger.info("***** Running training *****")
        self.logger.info(f"  Num examples = {len(self.train_loader)}")
        
        if self.configs.train_phase == "vae":
            total_train_epochs = self.configs.total_train_epochs_vae
        elif self.configs.train_phase == "unet":
            total_train_epochs = self.configs.total_train_epochs_unet
        
        self.logger.info(f"  Num Epochs = {total_train_epochs}")
        self.logger.info(f"  Instantaneous batch size per device = {self.configs.train_batch_size}")
        self.logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        self.logger.info(f"  Gradient Accumulation steps = {self.configs.gradient_accumulation_steps}")
        self.logger.info(f"  Total optimization steps = {self.configs.max_train_steps}")
        
        # Potentially load in the weights and states from a previous save
        num_update_steps_per_epoch = math.ceil(len(self.train_loader) / self.configs.gradient_accumulation_steps)
        if self.configs.resume_from_checkpoint:
            if self.configs.resume_from_checkpoint != "latest":
                path = os.path.basename(self.configs.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(self.checkpoints_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[3]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                self.accelerator.print(
                    f"Checkpoint '{self.configs.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                self.configs.resume_from_checkpoint = None
                self.global_step = 0
                current_epoch = 0
            else:
                self.accelerator.print(f"Resuming from checkpoint {path}")
                self.accelerator.load_state( os.path.join(self.checkpoints_dir, path))
                initial_global_step = int(path.split("-")[3])

                self.global_step = initial_global_step
                current_epoch = initial_global_step // num_update_steps_per_epoch

        else:
            self.global_step = 0
            current_epoch = 0
        
        # Epoch loop
        while True:
            
            with self.accelerator.autocast():
                # Train for one epoch
                if self.configs.train_phase == "vae":
                    print(f"Train Phase: {self.configs.train_phase}, Epoch: {current_epoch}")
                    self.train_vae_epoch()
                elif self.configs.train_phase == "unet":
                    print(f"Train Phase: {self.configs.train_phase}, Epoch: {current_epoch}")
                    self.train_unet_epoch()
                
                # Evaluate the model
                if self.accelerator.is_main_process:
                    print(f"Evaluation Phase: {self.configs.train_phase}, Epoch: {current_epoch}")
                    with torch.no_grad():
                        self.eval_epoch()
                
                if current_epoch == total_train_epochs:
                    self.save_model()
                    break
                
                current_epoch += 1
        
        self.accelerator.end_training()
    
    def train_vae_epoch(self):
        
        # Create train data iterator
        train_iter = iter(self.train_loader)
        
        # Train loop for vae
        self.diff_model.vae.train()
        train_loss = 0.0
        progress_bar = tqdm(range(len(self.train_loader)), ncols=90, disable=not self.accelerator.is_local_main_process)
        for step in progress_bar:
            with self.accelerator.accumulate(self.diff_model.unet):
                # Load data
                train_data = next(train_iter)

                if self.configs.train_model == "geo-diff":
                    # Inverse view coordinates for normalization
                    v_coords_norm = 1 / train_data["v_coords"]
                    model_input = torch.cat([v_coords_norm, train_data["normal"]], dim=-1).permute(0,3,1,2)
                elif self.configs.train_model == "tex-diff":
                    model_input = torch.cat([train_data["albedo"], \
                        train_data["roughness"], train_data["specular"]], dim=-1).permute(0,3,1,2)
                
                latents = self.vae.encode(model_input.to(self.weight_dtype)).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
                model_output = self.vae.decode(latents).sample

                # Compute loss
                loss = nn.functional.mse_loss(model_output, model_input)
                
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = self.accelerator.gather(loss.repeat(self.configs.train_batch_size)).mean()
                train_loss += avg_loss.item() / self.configs.gradient_accumulation_steps
                
                # Logs
                logs = {"loss": train_loss}
                progress_bar.set_postfix(**logs)
                Writer.add_scalar(f"train_{self.configs.train_phase}/loss", loss.item(), step=self.global_step)
                Writer.add_scalar(f"train_{self.configs.train_phase}/lr", self.lr_scheduler.get_last_lr()[0], step=self.global_step)
                
                # Backpropagate
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    params_to_clip = self.diff_model.vae.parameters()
                    self.accelerator.clip_grad_norm_(params_to_clip, self.configs.max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                
                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    self.global_step += 1
                    self.accelerator.log({"train_loss": train_loss}, step=self.global_step)
                    train_loss = 0.0
                    self.save_model()
    
    def train_unet_epoch(self):
        
        # Create train data iterator
        train_iter = iter(self.train_loader)
        
        # Train loop for unet
        self.diff_model.vae.eval()
        self.diff_model.unet.train()
        train_loss = 0.0
        progress_bar = tqdm(range(len(self.train_loader)), ncols=90, disable=not self.accelerator.is_local_main_process)
        for step in progress_bar:
            with self.accelerator.accumulate(self.diff_model.unet):
                # Load data
                train_data = next(train_iter)
                
                # Preprocess
                train_data = self.preprocess_train(train_data)
            
                # Prompting
                prompt_list = train_data["prompt"]
                answer_list = []
                with torch.no_grad():
                    for index, prompts in enumerate(zip(prompt_list[0], prompt_list[1], prompt_list[2], prompt_list[3], prompt_list[4])):
                        
                        self.vlm.system_prompt = "你是一个有帮助的AI助手, 用来为文生图模型生成提示词, 请只生成提示词部分"
                        
                        # Get prompts
                        if self.configs.train_model == "geo-diff":
                            instruction = [
                                {
                                    "role": "<|User|>",
                                    "content": "图中人物的性别是什么",
                                    "images": []
                                },
                                {
                                    "role": "<|Assistant|>",
                                    "content": f"{prompts[0]}"
                                },
                                {
                                    "role": "<|User|>",
                                    "content": "请描述图中人物的年龄",
                                    "images": []
                                },
                                {
                                    "role": "<|Assistant|>",
                                    "content": f"{prompts[2]}"
                                },
                                {
                                    "role": "<|User|>",
                                    "content": "请描述图中人物的面部特征",
                                    "images": []
                                },
                                {
                                    "role": "<|Assistant|>",
                                    "content": f"{prompts[3]}"
                                },
                                {
                                    "role": "<|User|>",
                                    "content": """
                                        根据以上得到的性别、年龄、面部特征\n
                                        编写一句话描述一个人的外貌\n
                                        需要包含性别、年龄、面部特征的全部信息\n
                                        并翻译为英语输出
                                    """,
                                    "images": []
                                },
                                {
                                    "role": "<|Assistant|>",
                                    "content": ""
                                }
                            ]
                        elif self.configs.train_model == "tex-diff":
                            instruction = [
                                {
                                    "role": "<|User|>",
                                    "content": "图中人物的性别是什么",
                                    "images": []
                                },
                                {
                                    "role": "<|Assistant|>",
                                    "content": f"{prompts[0]}"
                                },
                                {
                                    "role": "<|User|>",
                                    "content": "请描述图中人物的年龄",
                                    "images": []
                                },
                                {
                                    "role": "<|Assistant|>",
                                    "content": f"{prompts[2]}"
                                },
                                {
                                    "role": "<|User|>",
                                    "content": "请描述图中人物的面部特征",
                                    "images": []
                                },
                                {
                                    "role": "<|Assistant|>",
                                    "content": f"{prompts[3]}"
                                },
                                {
                                    "role": "<|User|>",
                                    "content": """
                                        根据以上得到的性别、年龄、面部特征\n
                                        编写一句话描述一个人的外貌\n
                                        需要包含性别、年龄、面部特征的全部信息\n
                                        并翻译为英语输出
                                    """,
                                    "images": []
                                },
                                {
                                    "role": "<|Assistant|>",
                                    "content": ""
                                }
                            ]
                        
                        processed_input = self.vlm.tokenize(instruction, [])
                        answer_list.append(self.vlm(processed_input.to(self.device_1)))
                    
                    # Prompt embedding
                    prompt_embeds_list = []
                    for tokenizer, text_encoder in zip(self.tokenizers, self.text_encoders):
                        text_inputs = tokenizer(
                            answer_list,
                            padding="max_length",
                            max_length=tokenizer.model_max_length,
                            truncation=True,
                            return_tensors="pt",
                        )
                        text_input_ids = text_inputs.input_ids
                        prompt_embeds = text_encoder(
                            text_input_ids.to(text_encoder.device),
                            output_hidden_states=True,
                            return_dict=False,
                        )

                        # We are only ALWAYS interested in the pooled output of the final text encoder
                        pooled_prompt_embeds = prompt_embeds[0]
                        prompt_embeds = prompt_embeds[-1][-2]
                        bs_embed, seq_len, _ = prompt_embeds.shape
                        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
                        prompt_embeds_list.append(prompt_embeds)
                
                    prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)
                    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)

                    # Finetune unet model
                    # 1. Use pretrained vae to get the encoded latents of samples
                    if self.configs.train_model == "geo-diff":
                        # Inverse view coordinates for normalization
                        v_coords_norm = 1 / train_data["v_coords"]  
                        model_input = torch.cat([v_coords_norm, train_data["normal"]], dim=-1).permute(0,3,1,2)
                    elif self.configs.train_model == "tex-diff":
                        model_input = torch.cat([train_data["albedo"], \
                            train_data["roughness"], train_data["specular"]], dim=-1).permute(0,3,1,2)
                
                
                    image_latents = self.diff_model.vae.encode(model_input).latent_dist.sample()
                
                    B, C, H, W = image_latents.shape
                
                    # 2. Sample noise that we'll add to the latents
                    noise = torch.randn_like(image_latents)
                    if self.configs.noise_offset:
                        # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                        noise += self.configs.noise_offset * torch.randn((B, C, 1, 1), device=self.device_1)
                    
                    if self.configs.timestep_bias_strategy == "none":
                        # Sample a random timestep for each image without bias.
                        timesteps = torch.randint(
                            0, self.diff_model.noise_scheduler.config.num_train_timesteps, (B,), device=self.device_1
                        )
                    else:
                        # Sample a random timestep for each image, potentially biased by the timestep weights.
                        # Biasing the timestep weights allows us to spend less time training irrelevant timesteps.
                        weights = self.generate_timestep_weights(self.configs, 
                            self.diff_model.noise_scheduler.config.num_train_timesteps).to(self.device_1)
                        timesteps = torch.multinomial(weights, B, replacement=True).long()
                    
                    # Add noise to the model input according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_model_input = self.diff_model.noise_scheduler.add_noise(image_latents, noise, timesteps).to(dtype=self.weight_dtype)
                    
                    # time ids
                    add_time_ids = torch.cat(
                        [self.compute_time_ids(s, c, self.device_1) for s, c in zip(train_data["original_sizes"], train_data["crop_top_lefts"])]
                    )
                
                # Adding rotation condition to text embeds
                condition_embeds = self.r_projector(torch.cat([pooled_prompt_embeds, train_data["rotation"]], dim=-1))
                
                # Predict the noise residual
                unet_added_conditions = {"time_ids": add_time_ids.to(self.device_2), "text_embeds": condition_embeds.to(self.device_2)}
                model_pred = self.unet(
                    noisy_model_input.to(self.device_2),
                    timesteps.to(self.device_2),
                    prompt_embeds.to(self.device_2),
                    added_cond_kwargs=unet_added_conditions,
                    return_dict=False,
                )[0]
                
                # Get the target for loss depending on the prediction type
                if self.configs.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    self.diff_model.noise_scheduler.register_to_config(prediction_type=self.configs.prediction_type)

                if self.diff_model.noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif self.diff_model.noise_scheduler.config.prediction_type == "v_prediction":
                    target = self.diff_model.noise_scheduler.get_velocity(model_input, noise, timesteps)
                elif self.diff_model.noise_scheduler.config.prediction_type == "sample":
                    # We set the target to latents here, but the model_pred will return the noise sample prediction.
                    target = model_input
                    # We will have to subtract the noise residual from the prediction to get the target sample.
                    model_pred = model_pred - noise
                else:
                    raise ValueError(f"Unknown prediction type {self.diff_model.noise_scheduler.config.prediction_type}")

                if self.configs.snr_gamma is None:
                    loss = nn.functional.mse_loss(model_pred.float(), target.to(self.device_2).float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(self.diff_model.noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, self.configs.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if self.diff_model.noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif self.diff_model.noise_scheduler.config.prediction_type.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)
                    
                    loss = nn.functional.mse_loss(model_pred.float(), target.to(self.device_2).float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights.to(self.device_2)
                    loss = loss.mean()
                
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = self.accelerator.gather(loss.repeat(self.configs.train_batch_size)).mean()
                train_loss += avg_loss.item() / self.configs.gradient_accumulation_steps
                
                # Backpropagate
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    params_to_clip = self.diff_model.unet.parameters()
                    self.accelerator.clip_grad_norm_(params_to_clip, self.configs.max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                
                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    # if self.configs.use_ema:
                    #     self.ema_unet.step(unet.parameters())
                    self.global_step += 1
                    self.accelerator.log({"train_loss": train_loss}, step=self.global_step)
                    train_loss = 0.0
                    self.save_model()
                
                # Logs
                logs = {"step_loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                Writer.add_scalar(f"{self.configs.train_phase}/loss", loss.item(), step=self.global_step)
        
    def eval_epoch(self):
        
        # Create train data iterator
        eval_iter = iter(self.eval_loader)
        
        # Train loop for unet
        self.diff_model.vae.eval()
        self.diff_model.unet.eval()
        eval_loss = 0.0
        progress_bar = tqdm(range(len(self.eval_loader))[:100], ncols=90, disable=not self.accelerator.is_local_main_process)
        for step in progress_bar:
            # Load data
            eval_data = next(eval_iter)
            
            if self.configs.train_phase == "vae":
                if self.configs.train_model == "geo-diff":
                    # Inverse view coordinates for normalization
                    v_coords_norm = 1 / eval_data["v_coords"]
                    model_input = torch.cat([v_coords_norm, eval_data["normal"]], dim=-1).permute(0,3,1,2)
                elif self.configs.train_model == "tex-diff":
                    model_input = torch.cat([eval_data["albedo"], \
                        eval_data["roughness"], eval_data["specular"]], dim=-1).permute(0,3,1,2)
                
                model_output, _ = self.diff_model.vae_forward(model_input, return_latent=True)
                
                # Compute loss
                eval_loss += nn.functional.mse_loss(model_output, model_input)
            
            elif self.configs.train_phase == "unet":
                
                # Preprocess
                eval_data = self.preprocess_train(eval_data)

                # Prompting
                prompt_list = eval_data["prompt"]
                answer_list = []
                for index, prompts in enumerate(zip(prompt_list[0], prompt_list[1], prompt_list[2], prompt_list[3], prompt_list[4])):
                    
                    self.vlm.system_prompt = "你是一个有帮助的AI助手, 用来为文生图模型生成提示词, 请只生成提示词部分"
                    
                    # Get prompts
                    if self.configs.train_model == "geo-diff":
                        instruction = [
                            {
                                "role": "<|User|>",
                                "content": "图中人物的性别是什么",
                                "images": []
                            },
                            {
                                "role": "<|Assistant|>",
                                "content": f"{prompts[0]}"
                            },
                            {
                                "role": "<|User|>",
                                "content": "请描述图中人物的年龄",
                                "images": []
                            },
                            {
                                "role": "<|Assistant|>",
                                "content": f"{prompts[2]}"
                            },
                            {
                                "role": "<|User|>",
                                "content": "请描述图中人物的面部特征",
                                "images": []
                            },
                            {
                                "role": "<|Assistant|>",
                                "content": f"{prompts[3]}"
                            },
                            {
                                "role": "<|User|>",
                                "content": """
                                    根据以上得到的性别、年龄、面部特征\n
                                    编写一句话描述一个人的外貌\n
                                    需要包含性别、年龄、面部特征的全部信息\n
                                    并翻译为英语输出
                                """,
                                "images": []
                            },
                            {
                                "role": "<|Assistant|>",
                                "content": ""
                            }
                        ]
                    elif self.configs.train_model == "tex-diff":
                        instruction = [
                            {
                                "role": "<|User|>",
                                "content": "图中人物的性别是什么",
                                "images": []
                            },
                            {
                                "role": "<|Assistant|>",
                                "content": f"{prompts[0]}"
                            },
                            {
                                "role": "<|User|>",
                                "content": "请描述图中人物的年龄",
                                "images": []
                            },
                            {
                                "role": "<|Assistant|>",
                                "content": f"{prompts[2]}"
                            },
                            {
                                "role": "<|User|>",
                                "content": "请描述图中人物的面部特征",
                                "images": []
                            },
                            {
                                "role": "<|Assistant|>",
                                "content": f"{prompts[3]}"
                            },
                            {
                                "role": "<|User|>",
                                "content": """
                                    根据以上得到的性别、年龄、面部特征\n
                                    编写一句话描述一个人的外貌\n
                                    需要包含性别、年龄、面部特征的全部信息\n
                                    并翻译为英语输出
                                """,
                                "images": []
                            },
                            {
                                "role": "<|Assistant|>",
                                "content": ""
                            }
                        ]
                    
                    with torch.no_grad():
                        processed_input = self.vlm.tokenize(instruction, [])
                        answer_list.append(self.vlm(processed_input.to(self.device_2)))
                
                # Prompt embedding
                prompt_embeds_list = []
                with torch.no_grad():
                    for tokenizer, text_encoder in zip(self.tokenizers, self.text_encoders):
                        text_inputs = tokenizer(
                            answer_list,
                            padding="max_length",
                            max_length=tokenizer.model_max_length,
                            truncation=True,
                            return_tensors="pt",
                        )
                        text_input_ids = text_inputs.input_ids
                        prompt_embeds = text_encoder(
                            text_input_ids.to(text_encoder.device_1),
                            output_hidden_states=True,
                            return_dict=False,
                        )

                        # We are only ALWAYS interested in the pooled output of the final text encoder
                        pooled_prompt_embeds = prompt_embeds[0]
                        prompt_embeds = prompt_embeds[-1][-2]
                        bs_embed, seq_len, _ = prompt_embeds.shape
                        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
                        prompt_embeds_list.append(prompt_embeds)
                
                prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
                pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)

                if self.configs.train_model == "geo-diff":
                    # Inverse view coordinates for normalization
                    v_coords_norm = 1 / eval_data["v_coords"]  
                    model_input = torch.cat([v_coords_norm, eval_data["normal"]], dim=-1).permute(0,3,1,2)
                elif self.configs.train_model == "tex-diff":
                    model_input = torch.cat([eval_data["albedo"], \
                        eval_data["roughness"], eval_data["specular"]], dim=-1).permute(0,3,1,2)
                
                with torch.no_grad():
                    image_latents = self.diff_model.vae.encode(model_input).latent_dist.sample()
            
                B, C, H, W = image_latents.shape
                
                # 2. Sample noise that we'll add to the latents
                noise = torch.randn_like(image_latents)
                if self.configs.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += self.configs.noise_offset * torch.randn((B, C, 1, 1), device=self.device_1)
                
                if self.configs.timestep_bias_strategy == "none":
                    # Sample a random timestep for each image without bias.
                    timesteps = torch.randint(
                        0, self.diff_model.noise_scheduler.config.num_train_timesteps, (B,), device=self.device_1
                    )
                else:
                    # Sample a random timestep for each image, potentially biased by the timestep weights.
                    # Biasing the timestep weights allows us to spend less time training irrelevant timesteps.
                    weights = self.generate_timestep_weights(self.configs, 
                        self.diff_model.noise_scheduler.config.num_train_timesteps).to(self.device_1)
                    timesteps = torch.multinomial(weights, B, replacement=True).long()
                
                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_model_input = self.diff_model.noise_scheduler.add_noise(image_latents, noise, timesteps).to(dtype=self.weight_dtype)
                
                # time ids
                add_time_ids = torch.cat(
                    [self.compute_time_ids(s, c, self.device_1) for s, c in zip(eval_data["original_sizes"], eval_data["crop_top_lefts"])]
                )
                
                # Adding rotation condition to text embeds
                condition_embeds = self.r_projector(torch.cat([pooled_prompt_embeds, eval_data["rotation"]], dim=-1))
                
                # Predict the noise residual
                unet_added_conditions = {"time_ids": add_time_ids, "text_embeds": condition_embeds}
                model_pred = self.diff_model.unet(
                    noisy_model_input,
                    timesteps,
                    prompt_embeds,
                    added_cond_kwargs=unet_added_conditions,
                    return_dict=False,
                )[0]
                
                # Get the target for loss depending on the prediction type
                if self.configs.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    self.diff_model.noise_scheduler.register_to_config(prediction_type=self.configs.prediction_type)

                if self.diff_model.noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif self.diff_model.noise_scheduler.config.prediction_type == "v_prediction":
                    target = self.diff_model.noise_scheduler.get_velocity(model_input, noise, timesteps)
                elif self.diff_model.noise_scheduler.config.prediction_type == "sample":
                    # We set the target to latents here, but the model_pred will return the noise sample prediction.
                    target = model_input
                    # We will have to subtract the noise residual from the prediction to get the target sample.
                    model_pred = model_pred - noise
                else:
                    raise ValueError(f"Unknown prediction type {self.diff_model.noise_scheduler.config.prediction_type}")

                if self.configs.snr_gamma is None:
                    loss = nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(self.diff_model.noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, self.configs.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if self.diff_model.noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif self.diff_model.noise_scheduler.config.prediction_type.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)
                    
                    loss = nn.functional.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()
                
                eval_loss += loss
            
            # Compute metrics
            avg_loss = eval_loss / len(self.eval_loader)
            Writer.add_scalar(f"{self.configs.train_phase}/loss", avg_loss.item(), step=self.global_step)
    
    def save_model(self):
        
        # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues.
        if self.accelerator.distributed_type == DistributedType.DEEPSPEED or self.accelerator.is_main_process:
            if self.global_step % self.configs.checkpointing_steps == 0:
                # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                if self.configs.checkpoints_total_limit is not None:
                    checkpoints = os.listdir(self.checkpoints_dir)
                    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[3]))

                    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                    if len(checkpoints) >= self.configs.checkpoints_total_limit:
                        num_to_remove = len(checkpoints) - self.configs.checkpoints_total_limit + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]

                        self.logger.info(
                            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                        )
                        self.logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                        for removing_checkpoint in removing_checkpoints:
                            removing_checkpoint = os.path.join(self.checkpoints_dir, removing_checkpoint)
                            shutil.rmtree(removing_checkpoint)

                save_path = os.path.join(self.checkpoints_dir, f"checkpoint-{self.configs.train_model[:3]}-{self.configs.train_phase}-{self.global_step}")
                self.accelerator.save_state(save_path)
                self.logger.info(f"Saved state to {save_path}")
    
    def preprocess_train(self, batch):   
        # Adapted from train_text_to_image_sdxl.preprocess_train
        images = batch["rgb"]
        # image aug
        samples = []
        original_sizes = []
        all_images = []
        crop_top_lefts = []
        for image in images:
            original_sizes.append((image.shape[0], image.shape[1]))
            image = self.train_resize(image)
            if self.configs.random_flip and random.random() < 0.5:
                # flip
                image = self.train_flip(image)
            if self.configs.center_crop:
                y1 = max(0, int(round((image.shape[0] - self.height) / 2.0)))
                x1 = max(0, int(round((image.shape[0] - self.width) / 2.0)))
                image = self.train_crop(image)
            else:
                y1, x1, h, w = self.train_crop.get_params(image, (self.height, self.width))
                image = tvf.crop(image, y1, x1, h, w)
            crop_top_left = (y1, x1)
            crop_top_lefts.append(crop_top_left)
            image = self.train_transforms(image)
            all_images.append(image)

        batch["original_sizes"] = original_sizes
        batch["crop_top_lefts"] = crop_top_lefts
        return batch
    
    def generate_timestep_weights(self, args, num_timesteps):
        
        weights = torch.ones(num_timesteps)

        # Determine the indices to bias
        num_to_bias = int(args.timestep_bias_portion * num_timesteps)

        if args.timestep_bias_strategy == "later":
            bias_indices = slice(-num_to_bias, None)
        elif args.timestep_bias_strategy == "earlier":
            bias_indices = slice(0, num_to_bias)
        elif args.timestep_bias_strategy == "range":
            # Out of the possible 1000 timesteps, we might want to focus on eg. 200-500.
            range_begin = args.timestep_bias_begin
            range_end = args.timestep_bias_end
            if range_begin < 0:
                raise ValueError(
                    "When using the range strategy for timestep bias, you must provide a beginning timestep greater or equal to zero."
                )
            if range_end > num_timesteps:
                raise ValueError(
                    "When using the range strategy for timestep bias, you must provide an ending timestep smaller than the number of timesteps."
                )
            bias_indices = slice(range_begin, range_end)
        else:  # 'none' or any other string
            return weights
        if args.timestep_bias_multiplier <= 0:
            return ValueError(
                "The parameter --timestep_bias_multiplier is not intended to be used to disable the training of specific timesteps."
                " If it was intended to disable timestep bias, use `--timestep_bias_strategy none` instead."
                " A timestep bias multiplier less than or equal to 0 is not allowed."
            )
    
    def compute_time_ids(self, original_size, crops_coords_top_left, device):
        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        target_size = (self.width, self.height)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], device=device, dtype=self.weight_dtype)
        return add_time_ids
