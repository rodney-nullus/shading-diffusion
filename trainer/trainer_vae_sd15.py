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

from diffusers import StableDiffusionPipeline, DiffusionPipeline
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

from models.vlm import VLM
from models.projector import Projector

from dataloader.celeba_pbr import get_dataloader

from configs.configs_unet import Configs

from  loss.mask_loss import BCEDiceBoundaryLoss

class Trainer:
    def __init__(self, args, configs: Configs):
        
        self.logger = get_logger(__name__)
        self.configs: Configs = configs
        
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
            
        vae = diff_model.vae
            
        if configs.gradient_checkpointing:
            vae.enable_gradient_checkpointing()
        
        optimizer = torch.optim.AdamW(vae.parameters(), lr=configs.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=configs.total_train_epochs_vae)
        
        # Prepare everything
        # There is no specific order to remember, you just need to unpack the
        # objects in the same order you gave them to the prepare method.
        (self.vae, 
            self.optimizer, 
            self.lr_scheduler, 
            self.train_loader, 
            self.eval_loader) = self.accelerator.prepare(
            vae, optimizer, lr_scheduler, train_loader, eval_loader, 
        )
        
        self.train_resize = transforms.Resize((self.height, self.width), interpolation=transforms.InterpolationMode.BILINEAR)
        self.train_crop = transforms.CenterCrop((self.height, self.width)) if configs.center_crop else transforms.RandomCrop(configs.resolution)
        self.train_flip = transforms.RandomHorizontalFlip(p=1.0)
        self.train_transforms = transforms.Compose([transforms.Normalize([0.5], [0.5])])
    
    def train(self):
        
        # Initial log
        total_batch_size = self.configs.train_batch_size * self.accelerator.num_processes * self.configs.gradient_accumulation_steps
        num_update_steps_per_epoch = math.ceil(len(self.train_loader) / self.configs.gradient_accumulation_steps)

        self.logger.info("***** Running training *****")
        self.logger.info(f"  Num examples = {len(self.train_loader)}")
        
        total_train_epochs = self.configs.total_train_epochs_vae
        
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
        
        # Train loop for vae
        self.vae.train()
        train_loss = 0.0
        progress_bar = tqdm(
            range(self.initial_step, len(self.train_loader)),
            total=len(self.train_loader),
            initial=self.initial_step,
            ncols=90,
            disable=not self.accelerator.is_local_main_process
        )
        for step in progress_bar:
            with self.accelerator.accumulate(self.vae):
                # Load data
                train_data = next(train_iter)

                g_buffer = [
                    train_data["v_coords"], 
                    train_data["normal"],
                    train_data["albedo"],
                    train_data["roughness"],
                    train_data["specular"]
                ]
                model_input = torch.cat(g_buffer, dim=-1).permute(0,3,1,2)
                
                posterior = self.vae.encode(model_input.to(self.weight_dtype)).latent_dist
                latents = posterior.sample()
                model_output = self.vae.decode(latents).sample.clamp(0,1)

                # Compute loss
                vc_rec = model_output[:,:3]
                normal_rec = model_output[:,3:6]
                tex_rec = model_output[:,6:]
                vc_loss = nn.functional.mse_loss(vc_rec, model_input[:,:3])
                normal_loss = 1 - nn.functional.cosine_similarity(normal_rec, train_data["normal"].permute(0,3,1,2)).mean()
                #normal_loss = nn.functional.mse_loss(normal_rec, g_buffer_gt[:,3:6])
                tex_loss = nn.functional.mse_loss(tex_rec, model_input[:,6:])
                kl_loss = posterior.kl().mean()
                total_loss = vc_loss + tex_loss + normal_loss + 0.000001 * kl_loss
                
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = self.accelerator.gather(total_loss.repeat(self.configs.train_batch_size)).mean()
                train_loss += avg_loss.item() / self.configs.gradient_accumulation_steps
                
                # Backpropagate
                self.accelerator.backward(total_loss)
                if self.accelerator.sync_gradients:
                    params_to_clip = self.vae.parameters()
                    self.accelerator.clip_grad_norm_(params_to_clip, self.configs.max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                
                # Logs
                logs = {"loss": train_loss}
                progress_bar.set_postfix(**logs)
                self.accelerator.log({
                    f"train_{self.configs.train_phase}/total_loss": total_loss.item(),
                    f"train_{self.configs.train_phase}/rec_loss": vc_loss.item()+normal_loss.item()+tex_loss.item(),
                    f"train_{self.configs.train_phase}/kl_loss": 0.000001 * kl_loss.item()
                }, step=self.global_step)
                
                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    self.global_step += 1
                    train_loss = 0.0
                    self.save_checkpoint()
        
        self.initial_step = 0
        
    def eval_epoch(self):
        
        # Create train data iterator
        eval_iter = iter(self.eval_loader)
        
        # Train loop for unet
        self.vae.eval()
        eval_loss = 0.0
        progress_bar = tqdm(range(100), ncols=90, disable=not self.accelerator.is_local_main_process)
        for step in progress_bar:
            # Load data
            eval_data = next(eval_iter)
                
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
