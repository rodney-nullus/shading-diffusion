import math, logging, os, random, shutil
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

from tqdm.auto import tqdm
from packaging import version

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
from compel import Compel
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, ControlNetModel
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.optimization import get_scheduler

from peft import LoraConfig, TaskType, get_peft_model
from peft.utils import get_peft_model_state_dict

from configs.training_configs_unet import Configs

from dataloader.celeba_pbr import get_dataloader

from models.vae_shader import VAE
from models.neural_renderer import NeuralRenderer

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

class Trainer:
    def __init__(self, configs: Configs):
        
        self.logger = get_logger(__name__)
        self.configs = configs
        # self.scaling_factor = 1.0076718898718957
        # self.shift_mean = torch.tensor([-0.0042, -0.0069,  0.0035, -0.0018])
        
        self.scaling_factor = torch.load("scaling_factor.pth")
        self.shift_factor = torch.load("shifting_factor.pth")
        
        if isinstance(configs.resolution, set):
            self.width, self.height = configs.resolution
        else:
            self.width, self.height = configs.resolution, configs.resolution
        
        # Handle the repository creation
        self.project_dir = f"{configs.output_dir}/{configs.exp_name}"
        
        # Create checkpoint dir
        self.checkpoints_dir = os.path.join(self.project_dir, "checkpoints")
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir, exist_ok=True)
        
        # Create logs dir
        self.logs_dir = os.path.join(self.project_dir, "logs")
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir, exist_ok=True)
        
        # Create sample dir
        self.sample_dir = os.path.join(self.project_dir, "samples")
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir, exist_ok=True)
        
        # Initialize accelerator and logger for training
        project_config = ProjectConfiguration(project_dir=self.project_dir, logging_dir=self.logs_dir)
        self.accelerator = Accelerator(
            mixed_precision=configs.mixed_precision,
            gradient_accumulation_steps=configs.gradient_accumulation_steps,
            project_config=project_config,
            log_with="tensorboard"
        )
        self.device = self.accelerator.device
        self.accelerator.init_trackers("unet_run", config={})
        
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
        
        # Load dataloader
        train_loader, eval_loader = get_dataloader(configs)
        
        self.total_train_epochs = configs.total_train_epochs
        self.num_update_steps_per_epoch = math.ceil(len(train_loader) / configs.gradient_accumulation_steps)
        self.total_train_steps = self.total_train_epochs * self.num_update_steps_per_epoch
        self.total_batch_size = configs.train_batch_size * self.accelerator.num_processes * configs.gradient_accumulation_steps
        
        vae_shader: VAE = VAE(configs=configs)
        neural_renderer: NeuralRenderer = NeuralRenderer()
        
        # Pose encoder
        pose_encoder = nn.Sequential(
            nn.Linear(4 * (1 + 256), 768),
            nn.SiLU(),
            nn.Linear(768, 768)
        )
        
        # Intrinsics head: predicts focal length (f) and principal point offsets (cx, cy)
        # We global-pool features and regress 3 values
        # intrinsic_net = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),   # [B,C,1,1]
        #     nn.Flatten(),              # [B,C]
        #     nn.Linear(256, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 1)           # [f, cx, cy]
        # )
        
        # Load tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(
            configs.pretrained_model_name_or_path,
            subfolder="tokenizer",
            use_fast=False,
        )
        
        # Add new token to tokenizer
        new_tokens = ["[POSE]"]
        self.tokenizer.add_tokens(new_tokens)
        self.pose_token_id = self.tokenizer.convert_tokens_to_ids("[POSE]")
        
        # Load text_encoder
        text_encoder = CLIPTextModel.from_pretrained(
            configs.pretrained_model_name_or_path, 
            subfolder="text_encoder"
        )
        text_encoder.resize_token_embeddings(len(self.tokenizer))
        self.token_embedding = text_encoder.get_input_embeddings()
        
        # Load noise scheduler
        self.noise_scheduler: DDPMScheduler = DDPMScheduler.from_pretrained(configs.pretrained_model_name_or_path, subfolder="scheduler")
        
        # Load unet model
        unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(configs.pretrained_model_name_or_path, subfolder="unet")
        unet.enable_xformers_memory_efficient_attention()
        
        if configs.gradient_checkpointing:
            unet.enable_gradient_checkpointing()
        
        # Create EAM for the unet
        if configs.use_ema:
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
        
        # Freeze parameters of models to save more memory
        unet.requires_grad_(False)
        text_encoder.requires_grad_(False)
        
        # Set attention module trainable
        for name, module in unet.named_modules():
            # CrossAttention / AttentionBlock
            if isinstance(module, torch.nn.MultiheadAttention) \
                or "attn" in name.lower() \
                or "attention" in name.lower():
                for param in module.parameters():
                    param.requires_grad = True
        
        # Fix vae, ns parameters
        for name, params in vae_shader.named_parameters():
            params.requires_grad = False
        
        for name, params in neural_renderer.named_parameters():
            params.requires_grad = False
        
        # Move unet, vae and text_encoder to device and cast to weight_dtype
        unet.to(self.accelerator.device, dtype=self.weight_dtype)
        vae_shader.to(self.accelerator.device, dtype=self.weight_dtype)
        text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        pose_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        
        if configs.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers
                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    self.logger.warning(
                        """xFormers 0.0.16 cannot be used for training in some GPUs. \\
                           If you observe problems during training, please update xFormers \\
                           to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."""
                    )
                unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")
        
        # Create pipeline
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.configs.pretrained_model_name_or_path,
            text_encoder=text_encoder,
            vae=vae_shader.vae,
            unet=unet
        ).to(self.device)
        self.pipeline.torch_dtype = self.weight_dtype
        self.pipeline.set_progress_bar_config(disable=True)
        
        if self.configs.random_seed is None:
            self.generator = None
        else:
            self.generator = torch.Generator(device=self.device).manual_seed(self.configs.random_seed)
        
        # Initialize the optimizer
        self.trainable_params = list(filter(lambda p: p.requires_grad, unet.parameters())) + list(pose_encoder.parameters())

        optimizer = torch.optim.AdamW(
            params=self.trainable_params,
            lr=configs.learning_rate,
            betas=(configs.adam_beta1, configs.adam_beta2),
            weight_decay=configs.adam_weight_decay,
            eps=configs.adam_epsilon,
        )
        
        # Scheduler and math around the number of training steps.
        # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
        num_warmup_steps_for_scheduler = configs.lr_warmup_steps * self.accelerator.num_processes
        if configs.max_train_steps is None:
            len_train_dataloader_after_sharding = math.ceil(len(train_loader) / self.accelerator.num_processes)
            num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / configs.gradient_accumulation_steps)
            num_training_steps_for_scheduler = (
                configs.total_train_epochs * num_update_steps_per_epoch * self.accelerator.num_processes
            )
        else:
            num_training_steps_for_scheduler = configs.max_train_steps * self.accelerator.num_processes

        lr_scheduler = get_scheduler(
            configs.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps_for_scheduler,
            num_training_steps=num_training_steps_for_scheduler,
        )
    
        # Prepare everything
        # There is no specific order to remember, you just need to unpack the
        # objects in the same order you gave them to the prepare method.
        (self.vae_shader,
         self.neural_renderer,
         self.pose_encoder,
         self.unet, 
         self.text_encoder,
         self.optimizer, 
         self.lr_scheduler, 
         self.train_loader, 
         self.eval_loader) = self.accelerator.prepare(
            vae_shader, neural_renderer, pose_encoder, unet, text_encoder, optimizer, lr_scheduler, train_loader, eval_loader)
        
        # Load pretrained weights from vae phase
        dirs = os.listdir(self.project_dir)
        vae_dirs = [d for d in dirs if d.startswith(f"vae")]
        vae_path = vae_dirs[-1] if len(vae_dirs) > 0 else None
        ns_dirs = [d for d in dirs if d.startswith(f"ns")]
        ns_path = ns_dirs[-1] if len(ns_dirs) > 0 else None
        
        if vae_path is None or ns_path is None:
            raise RuntimeError("Pretrained weights does not exist.")
        else:
            vae_weights = torch.load(f"{self.project_dir}/vae.pth")
            self.vae_shader.load_state_dict(vae_weights)
            # ns_weights = torch.load(f"{self.project_dir}/ns.pth")
            # self.neural_renderer.load_state_dict(ns_weights)
        
        # Initialize Compel
        self.compel = Compel(tokenizer=self.tokenizer, text_encoder=self.text_encoder)
        
        self.train_resize = transforms.Resize((self.height, self.width), interpolation=transforms.InterpolationMode.BILINEAR)
        self.train_crop = transforms.CenterCrop((self.height, self.width)) if configs.center_crop else transforms.RandomCrop(configs.resolution)
        self.train_flip = transforms.RandomHorizontalFlip(p=1.0)
        self.train_transforms = transforms.Compose([transforms.Normalize([0.5], [0.5])])
        
        # Metrics
        self.psnr = PeakSignalNoiseRatio().to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(self.device)
    
    def train(self):
        
        # Initial log
        self.logger.info("***** Running training *****")
        self.logger.info(f"  Num examples = {len(self.train_loader)}")
        
        self.logger.info(f"  Num Epochs = {self.total_train_epochs}")
        self.logger.info(f"  Instantaneous batch size per device = {self.configs.train_batch_size}")
        self.logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {self.total_batch_size}")
        self.logger.info(f"  Gradient Accumulation steps = {self.configs.gradient_accumulation_steps}")
        self.logger.info(f"  Total optimization steps = {self.total_train_steps}")
        
        # Potentially load in the weights and states from a previous save
        if self.configs.resume_from_checkpoint:
            if self.configs.resume_from_checkpoint != "latest":
                path = os.path.basename(self.configs.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(self.checkpoints_dir)
                dirs = [d for d in dirs if d.startswith(f"checkpoint-{self.configs.train_model[:3]}-{self.configs.train_phase}")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[3]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                self.accelerator.print(
                    f"Checkpoint '{self.configs.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                self.configs.resume_from_checkpoint = None
                self.initial_step = 0
                self.global_step = 0
                current_epoch = 0
            else:
                self.accelerator.print(f"Resuming from checkpoint {path}")
                self.accelerator.load_state( os.path.join(self.checkpoints_dir, path))
                global_step = int(path.split("-")[3])

                self.initial_step = global_step % self.num_update_steps_per_epoch
                self.global_step = global_step
                current_epoch = global_step // self.num_update_steps_per_epoch

        else:
            self.initial_step = 0
            self.global_step = 0
            current_epoch = 0
        
        # Epoch loop
        while True:
            
            with self.accelerator.autocast():
                # Train for one epoch
                print(f"Train Phase: {self.configs.train_phase}, Epoch: {current_epoch}")
                self.train_epoch()
                
                # Evaluate the model
                if self.accelerator.is_main_process:
                    print(f"Evaluation Phase: {self.configs.train_phase}, Epoch: {current_epoch}")
                    with torch.no_grad():
                        self.eval_epoch()
                
                assert self.configs.total_train_epochs is not None or self.configs.max_train_steps is not None
                
                if self.configs.total_train_epochs is not None:
                    if current_epoch == self.total_train_epochs:
                        break
                elif self.configs.max_train_steps is not None:
                    current_step = self.num_update_steps_per_epoch * current_epoch
                    if current_step == self.configs.max_train_steps:
                        break
                
                current_epoch += 1
        
        self.accelerator.end_training()
    
    def train_epoch(self):
        
        # Create train data iterator
        train_iter = iter(self.train_loader)
        
        # Train loop for unet
        self.unet.train()
        self.pose_encoder.train()
        train_loss = 0.0
        progress_bar = tqdm(
            range(self.initial_step, len(self.train_loader)),
            total=len(self.train_loader),
            initial=self.initial_step,
            ncols=90, 
            disable=not self.accelerator.is_local_main_process
        )
        for step in progress_bar:
            with self.accelerator.accumulate(self.unet):
                # Load data
                train_data = next(train_iter)
                
                # Skip data without prompt records
                if train_data["prompt"] == "0":
                    continue
                    
                # Prompt embedding + pose embedding
                prompt_embeds = self.compel(train_data["prompt"])
                last_idx = prompt_embeds.shape[1] - 1
                
                # Pose encoding
                fourier_feats = self.fourier_encode(x=train_data["rotation"], num_bands=128)
                pose_embeds = self.pose_encoder(torch.cat([train_data["rotation"],fourier_feats], dim=-1))
                prompt_embeds[:, last_idx, :] = prompt_embeds[:, last_idx, :] + pose_embeds

                # Finetune unet model
                # 1. Use pretrained vae to get the encoded latents of samples
                with torch.no_grad():
                    input_list = [
                        train_data["rgb"],
                        train_data["depth"], 
                        train_data["normal"],
                        train_data["albedo"],
                        train_data["roughness"],
                        train_data["specular"],
                        train_data["mask"]
                    ]
                    model_input = torch.cat(input_list, dim=-1).permute(0,3,1,2)
                    
                    posterior = self.vae_shader.vae.encode(model_input).latent_dist
                    latents = posterior.sample()
                    latents = (latents - self.shift_factor[None,:,None,None].to(latents.device)) * \
                        self.scaling_factor[None,:,None,None].to(latents.device)
                
                # 2. Sample noise that we'll add to the latents
                B, C, H, W = latents.shape
                noise = torch.randn_like(latents)
                if self.configs.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += self.configs.noise_offset * torch.randn((B, C, 1, 1), device=self.device)
                
                if self.configs.timestep_bias_strategy == "none":
                    # Sample a random timestep for each image without bias.
                    timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=self.device)
                else:
                    # Sample a random timestep for each image, potentially biased by the timestep weights.
                    # Biasing the timestep weights allows us to spend less time training irrelevant timesteps.
                    weights = self.generate_timestep_weights(self.configs, 
                        self.noise_scheduler.config.num_train_timesteps).to(self.device)
                    timesteps = torch.multinomial(weights, B, replacement=True).long()
                
                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps).to(dtype=self.weight_dtype)
                
                # Predict the noise residual
                model_pred = self.unet(
                    sample=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]
                
                # Get the target for loss depending on the prediction type
                if self.configs.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    self.noise_scheduler.register_to_config(prediction_type=self.configs.prediction_type)

                if self.noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif self.noise_scheduler.config.prediction_type == "v_prediction":
                    target = self.noise_scheduler.get_velocity(model_input, noise, timesteps)
                elif self.noise_scheduler.config.prediction_type == "sample":
                    # We set the target to latents here, but the model_pred will return the noise sample prediction.
                    target = model_input
                    # We will have to subtract the noise residual from the prediction to get the target sample.
                    model_pred = model_pred - noise
                else:
                    raise ValueError(f"Unknown prediction type {self.diff_model.noise_scheduler.config.prediction_type}")

                if self.configs.snr_gamma is None:
                    loss = nn.functional.mse_loss(model_pred.float(), target.to(self.device).float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = self.compute_snr(self.noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, self.configs.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
                    if self.noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif self.noise_scheduler.config.prediction_type.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)
                    
                    loss = nn.functional.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()
                
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = self.accelerator.gather(loss.repeat(self.configs.train_batch_size)).mean()
                train_loss += avg_loss.item() / self.configs.gradient_accumulation_steps
                
                # Backpropagate
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    params_to_clip = self.trainable_params
                    self.accelerator.clip_grad_norm_(params_to_clip, self.configs.max_grad_norm)
                self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    
                    # Logs
                    self.global_step += 1
                    self.accelerator.log({f"train_{self.configs.train_phase}/loss": loss.item()}, step=self.global_step)
                    train_loss = 0.0
                    
                    if self.configs.use_ema:
                        if self.configs.offload_ema:
                            self.ema_unet.to(device="cuda", non_blocking=True)
                            self.ema_unet.step(self.unet.parameters())
                        if self.configs.offload_ema:
                            self.ema_unet.to(device="cpu", non_blocking=True)
                    
                    self.save_checkpoint()
                
                logs = {"step_loss": loss.detach().item()}
                progress_bar.set_postfix(**logs)
        
        self.lr_scheduler.step()
        
    def eval_epoch(self):
        
        # Create train data iterator
        eval_iter = iter(self.eval_loader)
        
        # Train loop for unet
        self.unet.eval()
        self.vae_shader.eval()
        self.pose_encoder.eval()
        total_eval_loss = 0.
        total_psnr = 0.
        total_ssim = 0.
        total_lpips = 0.
        save_out_row_list = []
        eval_num = 2
        progress_bar = tqdm(range(eval_num), ncols=90, disable=not self.accelerator.is_local_main_process)
        for step in progress_bar:
            # Load data
            eval_data = next(eval_iter)
                
            # Skip data without prompt records
            if eval_data["prompt"] == "0":
                continue
            
            # Prompt embedding + pose embedding
            prompt_embeds = self.compel(eval_data["prompt"])
            last_idx = prompt_embeds.shape[1] - 1
            
            # Pose encoding
            fourier_feats = self.fourier_encode(x=eval_data["rotation"], num_bands=128)
            pose_embeds = self.pose_encoder(torch.cat([eval_data["rotation"],fourier_feats], dim=-1))
            prompt_embeds[:, last_idx, :] = prompt_embeds[:, last_idx, :] + pose_embeds

            # Finetune unet model
            # 1. Use pretrained vae to get the encoded latents of samples
            input_list = [
                eval_data["rgb"],
                eval_data["depth"], 
                eval_data["normal"],
                eval_data["albedo"],
                eval_data["roughness"],
                eval_data["specular"],
                eval_data["mask"]
            ]
            model_input = torch.cat(input_list, dim=-1).permute(0,3,1,2)
            
            posterior = self.vae_shader.vae.encode(model_input).latent_dist
            latents = posterior.sample()
            latents = (latents - self.shift_factor[None,:,None,None].to(latents.device)) * \
                self.scaling_factor[None,:,None,None].to(latents.device)
            
            # 2. Sample noise that we'll add to the latents
            B, C, H, W = latents.shape
            noise = torch.randn_like(latents)
            if self.configs.noise_offset:
                # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                noise += self.configs.noise_offset * torch.randn((B, C, 1, 1), device=self.device)
            
            if self.configs.timestep_bias_strategy == "none":
                # Sample a random timestep for each image without bias.
                timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=self.device)
            else:
                # Sample a random timestep for each image, potentially biased by the timestep weights.
                # Biasing the timestep weights allows us to spend less time training irrelevant timesteps.
                weights = self.generate_timestep_weights(self.configs, 
                    self.noise_scheduler.config.num_train_timesteps).to(self.device)
                timesteps = torch.multinomial(weights, B, replacement=True).long()
            
            # Add noise to the model input according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps).to(dtype=self.weight_dtype)
            
            # Predict the noise residual
            model_pred = self.unet(
                sample=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]
            
            # Get the target for loss depending on the prediction type
            if self.configs.prediction_type is not None:
                # set prediction_type of scheduler if defined
                self.noise_scheduler.register_to_config(prediction_type=self.configs.prediction_type)

            if self.noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                target = self.noise_scheduler.get_velocity(model_input, noise, timesteps)
            elif self.noise_scheduler.config.prediction_type == "sample":
                # We set the target to latents here, but the model_pred will return the noise sample prediction.
                target = model_input
                # We will have to subtract the noise residual from the prediction to get the target sample.
                model_pred = model_pred - noise
            else:
                raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

            if self.configs.snr_gamma is None:
                loss = nn.functional.mse_loss(model_pred.float(), target.to(self.device).float(), reduction="mean")
            else:
                # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                # This is discussed in Section 4.2 of the same paper.
                snr = self.compute_snr(self.noise_scheduler, timesteps)
                mse_loss_weights = torch.stack([snr, self.configs.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                    dim=1
                )[0]
                if self.noise_scheduler.config.prediction_type == "epsilon":
                    mse_loss_weights = mse_loss_weights / snr
                elif self.noise_scheduler.config.prediction_type.config.prediction_type == "v_prediction":
                    mse_loss_weights = mse_loss_weights / (snr + 1)
                
                loss = nn.functional.mse_loss(model_pred.float(), target.to(self.device).float(), reduction="none")
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights.to(self.device)
                loss = loss.mean()
            
            total_eval_loss += loss
            
            # Run inference to test synthesis.
            if self.configs.enable_xformers_memory_efficient_attention:
                self.pipeline.enable_xformers_memory_efficient_attention()
            
            # Sample test images
            # Prompt embedding + pose embedding
            prompt_embeds = self.compel(eval_data["prompt"])
            last_idx = prompt_embeds.shape[1] - 1
            
            # Pose encoding
            fourier_feats = self.fourier_encode(x=eval_data["rotation"], num_bands=128)
            pose_embeds = self.pose_encoder(torch.cat([eval_data["rotation"],fourier_feats], dim=-1))
            prompt_embeds[:, last_idx, :] = prompt_embeds[:, last_idx, :] + pose_embeds
            
            latents = self.pipeline(prompt_embeds=prompt_embeds, 
                                    height=self.height, width=self.width,
                                    num_inference_steps=20, 
                                    generator=self.generator, 
                                    output_type="latent").images
            latents = latents / self.scaling_factor[None,:,None,None].to(latents.device) + \
                self.shift_factor[None,:,None,None].to(latents.device)
            
            geo_output, mat_output, _ = self.vae_shader.vae.decode(latents)
            
            rgb_pred = geo_output[:,:3]
            depth_pred = geo_output[:,3]
            normal_pred = geo_output[:,4:7]
            mask_pred = geo_output[:,7]
            albedo_pred = mat_output[:,:3]
            roughness_pred = mat_output[:,3]
            metallic_pred = mat_output[:,4]
            
            save_out_row_list.append([rgb_pred, 
                                      depth_pred[:,None].repeat(1,3,1,1), 
                                      normal_pred,
                                      albedo_pred,
                                      roughness_pred[:,None].repeat(1,3,1,1), 
                                      metallic_pred[:,None].repeat(1,3,1,1), 
                                      mask_pred[:,None].repeat(1,3,1,1)])
            
            # # Render image
            # denorm_depth = (depth_pred * (self.configs.z_far - self.configs.z_near) + \
            #     self.configs.z_near) * mask_pred
            # fov = torch.tensor(10., device=depth_pred.device)[None].repeat(depth_pred.shape[0])
            # cam_pos_pred = self.get_cam_coords(denorm_depth.unsqueeze(1), 
            #                                 width=depth_pred.shape[2], height=depth_pred.shape[1], fov=fov)
            # mask = mask_pred.bool()
            
            # render_buffer = {
            #     "rgb_gt": eval_data["rgb"],
            #     "normal_gt": normal_pred.permute(0,2,3,1),
            #     "albedo_gt": mat_output[:,:3].permute(0,2,3,1),
            #     "roughness_gt": mat_output[:,3],
            #     "specular_gt": mat_output[:,4],
            #     "pos_in_cam_gt": cam_pos_pred,
            #     "hdri_gt": eval_data["hdri"],
            #     "mask": mask
            # }
            
            # rgb_shading = torch.ones(1,self.width,self.height,3,device=self.device)
            # shading_rgb, _ = self.neural_renderer(render_buffer=render_buffer, num_light_samples=128, inference=True)
            # rgb_shading[mask] = shading_rgb

            # # Compute metrics
            # total_psnr += self.psnr(rgb_shading, eval_data["rgb"])
            # total_ssim += self.ssim(rgb_shading.permute(0,3,1,2), eval_data["rgb"].permute(0,3,1,2))
            # total_lpips += self.lpips(rgb_shading.permute(0,3,1,2), eval_data["rgb"].permute(0,3,1,2))
            
            # save_out_row_list[-1].append(rgb_shading.permute(0,3,1,2))
            
            # Compute metrics
            avg_eval_loss = total_eval_loss / eval_num
            # avg_psnr = total_psnr / eval_num
            # avg_ssim = total_ssim / eval_num
            # avg_lpips = total_lpips / eval_num
            self.accelerator.log({
                f"eval_{self.configs.train_phase}/eval_loss": avg_eval_loss.item(),
                # f"eval_{self.configs.train_phase}/psnr": avg_psnr.item(),
                # f"eval_{self.configs.train_phase}/ssim": avg_ssim.item(),
                # f"eval_{self.configs.train_phase}/lpips": avg_lpips.item(),
            }, step=self.global_step)

            # Evaluate the visual result and save the model
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                
                # Save weights
                torch.save(self.unet.state_dict(), f"{self.project_dir}/unet.pth")
                torch.save(self.pose_encoder.state_dict(), f"{self.project_dir}/pose.pth")
                
                # Save samples
                save_out_col_list = []
                for save_out in save_out_row_list:
                    save_out_col_list.append(torch.cat(save_out,dim=2))
                
                save_out = torch.cat(save_out_col_list[:10], dim=3)[0]
                
                tvf.to_pil_image(save_out).save(f"{self.sample_dir}/sample_{self.configs.train_phase}_{self.global_step}.png")
    
    # Helper functions
    def unwrap_model(self, model):
        model = self.accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model
    
    def fourier_encode(self, x, num_bands=6):
        # x: (B, D) 张量
        freqs = 2.0 ** torch.arange(num_bands, device=x.device)
        xb = x.unsqueeze(-1) * freqs  # (B, D, K)
        return torch.cat([torch.sin(xb), torch.cos(xb)], dim=-1).view(x.shape[0], -1)
    
    def get_cam_coords(self, depth, width, height, fov):
        fovx = torch.deg2rad(fov)
        fovy = 2 * torch.atan(torch.tan(fovx / 2) / (width / height))
        cam_pos = torch.zeros(depth.shape[0], height, width, 3, device=depth.device)
        Y = 1 - (torch.arange(height, device=depth.device) + 0.5) / height
        Y = Y * 2 - 1
        X = (torch.arange(width, device=depth.device) + 0.5) / width
        X = X * 2 - 1
        Y, X = torch.meshgrid(Y, X, indexing="ij")
        cam_pos[..., 0] = depth.squeeze() * X[None,:,:] * torch.tan(fovx[:,None,None] / 2)
        cam_pos[..., 1] = depth.squeeze() * Y[None,:,:] * torch.tan(fovy[:,None,None] / 2)
        cam_pos[..., 2] = depth.squeeze()
        return cam_pos
    
    def save_output_sample(self, output_tensor):
        vc = output_tensor[:3]
        normal = output_tensor[3:6]
        albedo = output_tensor[6:9]
        roughness = output_tensor[9:10].repeat(3,1,1)
        specular = output_tensor[10:11].repeat(3,1,1)
        mat = torch.cat([vc, normal, albedo, roughness, specular], dim=1)
        tvf.to_pil_image(mat).save(f"{self.sample_dir}/{self.global_step}_{self.configs.train_phase}_sample.png")
    
    def save_checkpoint(self, with_lora=False):
        
        # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues.
        if self.accelerator.distributed_type == DistributedType.DEEPSPEED or self.accelerator.is_main_process:
            if self.global_step % self.configs.checkpointing_steps == 0:
                # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                if self.configs.checkpoints_total_limit is not None:
                    checkpoints = os.listdir(self.checkpoints_dir)
                    checkpoints = [d for d in checkpoints if d.startswith(f"checkpoint-{self.configs.train_model[:3]}-{self.configs.train_phase}")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[3]))

                    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                    if len(checkpoints) >= self.configs.checkpoints_total_limit:
                        num_to_remove = len(checkpoints) - self.configs.checkpoints_total_limit + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]

                        tqdm.write(
                            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                        )
                        tqdm.write(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                        for removing_checkpoint in removing_checkpoints:
                            removing_checkpoint = os.path.join(self.checkpoints_dir, removing_checkpoint)
                            shutil.rmtree(removing_checkpoint)

                save_path = os.path.join(self.checkpoints_dir, f"checkpoint-{self.configs.train_model[:3]}-{self.configs.train_phase}-{self.global_step}")
                self.accelerator.save_state(save_path)
                
                if with_lora:
                    unwrapped_unet = self.unwrap_model(self.unet)
                    unet_lora_state_dict = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(unwrapped_unet)
                    )

                    StableDiffusionPipeline.save_lora_weights(
                        save_directory=save_path,
                        unet_lora_layers=unet_lora_state_dict,
                        safe_serialization=True,
                    )
                
                tqdm.write(f"Saved state to {save_path}")
    
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
    
    def compute_snr(self, scheduler, timesteps):
        """
        计算 signal-to-noise ratio = α_cumprod/β_cumprod
        timesteps: Tensor 或 array of ints
        scheduler: diffusion scheduler，需有属性 alphas_cumprod
        """
        # scheduler.alphas_cumprod 是个 tensor 或 numpy array
        alphas = scheduler.alphas_cumprod[timesteps]
        betas = 1.0 - alphas
        return alphas / betas

    def encode_full_text(self, batch_text: str, pose_embeds: torch.Tensor, max_length=77, device="cuda"):
        
        batch_emb_list = []
        # 1. tokenize into sub‑texts of <=77 tokens
        for text in batch_text:
            words = text.split()
            chunks, cur = [], []
            for w in words:
                if len(self.tokenizer.encode(" ".join(cur + [w]))) <= max_length:
                    cur.append(w)
                else:
                    chunks.append(" ".join(cur))
                    cur = [w]
            if cur:
                chunks.append(" ".join(cur))

            # 2. encode each chunk
            embeddings = []
            for chunk in chunks[:-1]:
                chunk_input = self.tokenizer(chunk,
                        max_length=max_length,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt").to(device)
                
                with torch.no_grad():
                    chunk_hidden_states = self.text_encoder(**chunk_input).last_hidden_state  # shape [1,77,dim]
                embeddings.append(chunk_hidden_states)

            # 3. Process last chunk
            last_chunk_input = self.tokenizer(chunks[-1],
                                max_length=max_length,
                                truncation=True,
                                padding="max_length",
                                return_tensors="pt").to(device)
            
            B, N = last_chunk_input.input_ids.shape
            if N + 1 > self.tokenizer.model_max_length:
                input_ids = last_chunk_input.input_ids[:, :-1]
            
            pose_ids = torch.full((B, 1), self.pose_token_id, dtype=torch.long, device=input_ids.device)
            last_chunk_input.input_ids = torch.cat([input_ids, pose_ids], dim=1)
            
            # 3.1 Get token embeddings
            last_chunk_input_embeds = self.token_embedding(last_chunk_input.input_ids)
            
            # 3.2 Append pose_emb to last position
            # pose_embeds: [B, D]
            last_chunk_input_embeds[:,-1,:] = last_chunk_input_embeds[:,-1,:] + pose_embeds.unsqueeze(1)
            
            # 4. Input to CLIP text encoder (only Transformer)
            attention_mask = (last_chunk_input.input_ids != self.tokenizer.pad_token_id).long()
            embedding = self.text_encoder(inputs_embeds=last_chunk_input_embeds,
                                          attention_mask=attention_mask)  # shape [1,77,dim]
            
            embeddings.append(embedding.last_hidden_state)

            # 3. 聚合：mean pooling
            full_emb = torch.stack(embeddings, dim=0).mean(dim=0)  # [1,dim]
            batch_emb_list.append(full_emb)
        
        batch_emb = torch.cat(batch_emb_list, dim=0)
        
        return batch_emb