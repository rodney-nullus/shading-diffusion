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

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, ControlNetModel
from diffusers.training_utils import EMAModel
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler
from peft import LoraConfig, TaskType, get_peft_model
from peft.utils import get_peft_model_state_dict

from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from models.projector import Projector

from configs.configs_unet import Configs

from dataloader.celeba_pbr import get_dataloader

from loss.mask_loss import BCEDiceBoundaryLoss

class Trainer:
    def __init__(self, configs: Configs):
        
        self.logger = get_logger(__name__)
        self.configs = configs
        
        if isinstance(configs.resolution, set):
            self.width, self.height = configs.resolution
        else:
            self.width, self.height = configs.resolution, configs.resolution
        
        # Handle the repository creation
        self.project_dir = f"{configs.output_dir}/{configs.exp_name}/{configs.train_model}"
        
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
        if configs.train_phase == "vae":
            self.accelerator.init_trackers("vae_run", config={})
        elif configs.train_phase == "unet":
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
        
        # Load training components
        # Load dataloader
        train_loader, eval_loader = get_dataloader(configs)
        
        # Load deepseek vl2
        self.vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained("deepseek-ai/deepseek-vl2-tiny")
        self.vl_chat_processor.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        self.bos_token_id = self.vl_chat_processor.tokenizer.bos_token_id
        self.eos_token_id = self.vl_chat_processor.tokenizer.eos_token_id

        vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-vl2-tiny", trust_remote_code=True)
        self.vl_gpt = vl_gpt.to(torch.bfloat16).to("cuda").eval()
        
        # VLM settings
        self.max_new_tokens = configs.max_new_tokens
        self.do_sample = configs.do_sample
        self.use_cache = configs.use_cache
        
        # Load vae with pretrained weights
        vae: AutoencoderKL = AutoencoderKL.from_pretrained(configs.pretrained_model_name_or_path, subfolder="vae")
        
        # Change the number of input channels of vae encoder
        conv_in_out_chns = vae.encoder.conv_in.out_channels
        vae.encoder.conv_in = nn.Conv2d(configs.geo_diff_inchns, conv_in_out_chns, kernel_size=3, stride=1, padding=1)
        
        # Change the number of output channels of vae decoder
        conv_out_in_chns = vae.decoder.conv_out.in_channels
        vae.decoder.conv_out = nn.Conv2d(conv_out_in_chns, configs.geo_diff_outchns,kernel_size=3, stride=1, padding=1)
        
        vae.enable_xformers_memory_efficient_attention()
        
        dirs = os.listdir(self.checkpoints_dir)
        dirs = [d for d in dirs if d.startswith(f"checkpoint-{configs.train_model[:3]}-vae")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[3]))
        path = dirs[-1] if len(dirs) > 0 else None
        
        if path is not None:
            weight_path = f"{self.checkpoints_dir}/{path}/model.safetensors"
            load_model(vae, weight_path)
        
        # Load tokenizer
        self.tokeinzer = CLIPTokenizer.from_pretrained(
            configs.pretrained_model_name_or_path,
            subfolder="tokenizer",
            use_fast=False,
        )
        
        # Load text_encoder
        text_encoder = CLIPTextModel.from_pretrained(
            configs.pretrained_model_name_or_path, 
            subfolder="text_encoder"
        )
        
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
        
        # freeze parameters of models to save more memory
        unet.requires_grad_(True)
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        
        # Move unet, vae and text_encoder to device and cast to weight_dtype
        unet.to(self.accelerator.device, dtype=self.weight_dtype)
        vae.to(self.accelerator.device, dtype=self.weight_dtype)
        text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        
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
        
        # Initialize the optimizer
        # Projector for contrastive learning
        #cl_projector = Projector(in_dims=diff_model.vae.encoder.conv_in.out_channels, hidden_dims=128)
        
        # Projector for adding rotation 6D
        #r_projector = Projector(in_dims=774, hidden_dims=1280, out_dims=768)
        
        # self.trainable_param = list(lora_layers) + list(r_projector.parameters())
        trainable_params = list(filter(lambda p: p.requires_grad, unet.parameters()))

        optimizer = torch.optim.AdamW(
            params=trainable_params,
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
                configs.num_train_epochs * num_update_steps_per_epoch * self.accelerator.num_processes
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
        (self.vae,
         self.unet, 
         self.text_encoder,
         self.optimizer, 
         self.lr_scheduler, 
         self.train_loader, 
         self.eval_loader) = self.accelerator.prepare(
         vae, unet, text_encoder, optimizer, lr_scheduler, train_loader, eval_loader)
        
        self.train_resize = transforms.Resize((self.height, self.width), interpolation=transforms.InterpolationMode.BILINEAR)
        self.train_crop = transforms.CenterCrop((self.height, self.width)) if configs.center_crop else transforms.RandomCrop(configs.resolution)
        self.train_flip = transforms.RandomHorizontalFlip(p=1.0)
        self.train_transforms = transforms.Compose([transforms.Normalize([0.5], [0.5])])
    
    def train(self):
        
        # Initial log
        total_train_epochs = self.configs.num_train_epochs
        total_batch_size = self.configs.train_batch_size * self.accelerator.num_processes * self.configs.gradient_accumulation_steps
        num_update_steps_per_epoch = math.ceil(len(self.train_loader) / self.configs.gradient_accumulation_steps)

        self.logger.info("***** Running training *****")
        self.logger.info(f"  Num examples = {len(self.train_loader)}")
        
        self.logger.info(f"  Num Epochs = {total_train_epochs}")
        self.logger.info(f"  Instantaneous batch size per device = {self.configs.train_batch_size}")
        self.logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        self.logger.info(f"  Gradient Accumulation steps = {self.configs.gradient_accumulation_steps}")
        self.logger.info(f"  Total optimization steps = {total_train_epochs * num_update_steps_per_epoch}")
        
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

                self.initial_step = global_step % num_update_steps_per_epoch
                self.global_step = global_step
                current_epoch = global_step // num_update_steps_per_epoch

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
                
                if current_epoch == total_train_epochs:
                    break
                
                current_epoch += 1
        
        self.accelerator.end_training()
    
    def train_epoch(self):
        
        # Create train data iterator
        train_iter = iter(self.train_loader)
        
        # Train loop for unet
        self.vae.eval()
        self.unet.train()
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
                
                # Preprocess
                train_data = self.preprocess_train(train_data)
            
                # Prompting
                prompt_list = train_data["prompt"]
                ids_list = []
                with torch.no_grad():
                    for index, prompts in enumerate(zip(prompt_list[0], prompt_list[1], prompt_list[2], prompt_list[3], prompt_list[4])):
                        
                        system_prompt = "你是一个有帮助的AI助手, 用来为文生图模型生成提示词, 请只生成提示词部分"
                        
                        # Get prompts
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
                        
                        processed_input = self.vl_chat_processor(
                            conversations=instruction,
                            images=[],
                            force_batchify=True,
                            system_prompt=system_prompt
                        ).to(self.device)
                        
                        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**processed_input)
                        
                        # run model to get the response
                        outputs = self.vl_gpt.language.generate(
                            inputs_embeds=inputs_embeds,
                            attention_mask=processed_input.attention_mask,
                            pad_token_id=self.eos_token_id,
                            bos_token_id=self.bos_token_id,
                            eos_token_id=self.eos_token_id,
                            max_new_tokens=self.max_new_tokens,
                            do_sample=self.do_sample,
                            use_cache=self.use_cache,
                            temperature=self.configs.temperature,
                            top_p=self.configs.top_p
                        )
                        ids_list.append(outputs)
                    output_ids = torch.cat(ids_list, dim=0)
                    
                    # Prompt embedding
                    text_inputs = self.tokenizer(
                        ["111"],
                        padding="max_length",
                        max_length=self.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                    text_input_ids = text_inputs.input_ids
                    encoder_hidden_states = self.text_encoder(
                        text_input_ids.to(self.text_encoder.device),
                        output_hidden_states=True,
                        return_dict=False,
                    )[0]

                # Finetune unet model
                # 1. Use pretrained vae to get the encoded latents of samples
                with torch.no_grad():
                    if self.configs.train_model == "geo-diff":
                        # View coordinates normalization
                        v_coords_norm = train_data["v_coords"] / 800.
                        model_input = torch.cat([v_coords_norm, train_data["normal"]], dim=-1).permute(0,3,1,2)
                    
                    latents = self.vae.encode(model_input).latent_dist.sample()
                    latents = latents * self.vae.config.scaling_factor
                
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
                
                # Adding rotation condition to text embeds
                B, S, D = encoder_hidden_states.shape
                condition_embeds = self.r_projector(torch.cat([encoder_hidden_states, train_data["rotation"].unsqueeze(1).repeat(1,S,1)], dim=-1))
                
                # Predict the noise residual
                model_pred = self.unet(
                    noisy_latents,
                    timesteps,
                    condition_embeds,
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
                    loss = nn.functional.mse_loss(model_pred.float(), target.to(self.device_2).float(), reduction="mean")
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
                    params_to_clip = self.trainable_param
                    self.accelerator.clip_grad_norm_(params_to_clip, self.configs.max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()
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
        
    def eval_epoch(self):
        
        # Create train data iterator
        eval_iter = iter(self.eval_loader)
        
        # Train loop for unet
        if self.configs.train_phase == "vae":
            self.vae.eval()
        elif self.configs.train_phase == "unet":
            self.vae.eval()
            self.unet.eval()
        eval_loss = 0.0
        progress_bar = tqdm(range(100), ncols=90, disable=not self.accelerator.is_local_main_process)
        for step in progress_bar:
            # Load data
            eval_data = next(eval_iter)
            
            if self.configs.train_phase == "vae":
                
                g_buffer = [
                    eval_data["v_coords"], 
                    eval_data["normal"],
                    eval_data["albedo"],
                    eval_data["roughness"],
                    eval_data["specular"]
                ]
                model_input = torch.cat(g_buffer, dim=-1).permute(0,3,1,2).to(self.weight_dtype)
                
                posterior = self.vae.encode(model_input).latent_dist
                latents = posterior.sample()
                model_output = self.vae.decode(latents).sample.clamp(0,1)
                
                # Compute loss
                vc_rec = model_output[:,:3]
                normal_rec = model_output[:,3:6]
                tex_rec = model_output[:,6:]
                vc_loss = nn.functional.mse_loss(vc_rec, eval_data["v_coords"].permute(0,3,1,2))
                tex_loss = nn.functional.mse_loss(tex_rec, model_input[:,6:])
                normal_loss = 1 - nn.functional.cosine_similarity(normal_rec, eval_data["normal"].permute(0,3,1,2)).mean()
                #normal_loss = nn.functional.mse_loss(normal_rec, g_buffer_gt[:,3:6])
                kl_loss = posterior.kl().mean()
                eval_loss += vc_loss + tex_loss + normal_loss + 0.000001 * kl_loss
            
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
                        answer_list.append(self.vlm(processed_input.to(self.device)))
                
                # Prompt embedding
                text_inputs = self.tokenizer(
                    answer_list,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids
                encoder_hidden_states = self.text_encoder(
                    text_input_ids.to(self.text_encoder.device),
                    output_hidden_states=True,
                    return_dict=False,
                )[0]

                # 1. Use pretrained vae to get the encoded latents of samples
                if self.configs.train_model == "geo-diff":
                    model_input = torch.cat([eval_data["v_coords"], eval_data["normal"]], dim=-1).permute(0,3,1,2)
                
                latents = self.vae.encode(model_input).latent_dist.sample()
                
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
                
                # Adding rotation condition to text embeds
                B, S, D = encoder_hidden_states.shape
                condition_embeds = self.r_projector(torch.cat([encoder_hidden_states, eval_data["rotation"].unsqueeze(1).repeat(1,S,1)], dim=-1))
                
                # Predict the noise residual
                model_pred = self.unet(
                    noisy_latents,
                    timesteps,
                    condition_embeds,
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
                
                eval_loss += loss
        
        # Compute metrics
        avg_loss = eval_loss / len(self.eval_loader)
        self.accelerator.log({
            f"eval_{self.configs.train_phase}/mse_metirc": avg_loss.item()
        }, step=self.global_step)

        # Evaluate the visual result and save the model
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            
            # Save checkpoint firstly
            self.save_checkpoint()
            
            # VAE Test
            if self.configs.train_phase == "vae":
                unwarpped_vae = self.unwrap_model(self.vae)
                unwarpped_vae.save_pretrained(self.project_dir)
                
                output_list = []
                for i in range(10):
                    output_list.append(model_output[i])
                
                output_tensor = torch.cat(output_list, dim=2)
                self.save_output_sample(output_tensor)
            
            # UNet Test
            elif self.configs.train_phase == "unet":
                unwrapped_vae = self.unwrap_model(self.vae.to(torch.float32))
                unwrapped_unet = self.unwrap_model(self.unet.to(torch.float32))
                if self.configs.use_ema:
                    self.ema_unet.copy_to(unwrapped_unet.parameters())
                
                pipeline = StableDiffusionPipeline.from_pretrained(
                    self.configs.pretrained_model_name_or_path,
                    text_encoder=self.text_encoder,
                    vae=unwrapped_vae,
                    unet=unwrapped_unet
                ).to(self.device)
                
                pipeline.save_pretrained(self.project_dir)
                pipeline.torch_dtype = self.weight_dtype
                pipeline.set_progress_bar_config(disable=True)
                
                if self.configs.random_seed is None:
                    generator = None
                else:
                    generator = torch.Generator(device=self.device).manual_seed(self.configs.random_seed)
                
                unet_lora_state_dict = convert_state_dict_to_diffusers(
                    get_peft_model_state_dict(unwrapped_unet)
                )
                StableDiffusionPipeline.save_lora_weights(
                    save_directory=self.project_dir,
                    unet_lora_layers=unet_lora_state_dict,
                    safe_serialization=True
                )

                # Run a final round of inference.
                self.logger.info("Running inference for collecting generated images...")
                if self.configs.enable_xformers_memory_efficient_attention:
                    pipeline.enable_xformers_memory_efficient_attention()
                # Sample test images
                output_list = []
                for i in range(3):
                    with torch.autocast("cuda"):
                        # Prompt embedding
                        text_inputs = self.tokenizer(
                            answer_list,
                            padding="max_length",
                            max_length=self.tokenizer.model_max_length,
                            truncation=True,
                            return_tensors="pt",
                        )
                        text_input_ids = text_inputs.input_ids
                        encoder_hidden_states = self.text_encoder(
                            text_input_ids.to(self.text_encoder.device),
                            output_hidden_states=True,
                            return_dict=False,
                        )[0]
                        
                        B, S, D = encoder_hidden_states.shape
                        condition_embeds = self.r_projector(torch.cat([encoder_hidden_states, eval_data["rotation"].unsqueeze(1).repeat(1,S,1)], dim=-1))
                        latent = pipeline(prompt_embeds=condition_embeds, num_inference_steps=20, generator=generator, output_type="latent").images
                        output = self.vae.decode(latent * self.configs.scaling_factor, return_dict=False, generator=generator)[0]
                    output_list.append(output[0])
                
                output_tensor = torch.cat(output_list, dim=2)
                self.save_output_sample(output_tensor)
    
    # Helper functions
    def unwrap_model(self, model):
        model = self.accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model
    
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
    
    def compute_snr(self, timesteps, scheduler):
        """
        计算 signal-to-noise ratio = α_cumprod/β_cumprod
        timesteps: Tensor 或 array of ints
        scheduler: diffusion scheduler，需有属性 alphas_cumprod
        """
        # scheduler.alphas_cumprod 是个 tensor 或 numpy array
        alphas = scheduler.alphas_cumprod[timesteps]
        betas = 1.0 - alphas
        return alphas / betas
