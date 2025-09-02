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

from diffusers import AutoencoderKL
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

from dataloader.celeba_pbr import get_dataloader
from models.vae_shader import VAE
from models.neural_renderer import NeuralRenderer

from utils.io import load_hdr

from loss.mask_loss import BCEDiceBoundaryLoss

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

from configs.training_configs_vae import Configs

class Trainer:
    def __init__(self, configs: Configs):
        
        self.logger = get_logger(__name__)
        self.configs = configs
        
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
        self.accelerator.init_trackers(f"{configs.train_phase}_run", config={})
        
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
        
        vae_shader = VAE(configs=configs)
        vae_shader.vae.requires_grad_(False)
        vae_shader.vae.decoder.requires_grad_(True)
        
        # Intrinsics head: predicts focal length (f) and principal point offsets (cx, cy)
        # We global-pool features and regress 3 values
        intrinsic_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # [B,C,1,1]
            nn.Flatten(),              # [B,C]
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)           # [f]
        )
        
        neural_renderer = NeuralRenderer()
        
        if configs.train_phase == "vae":
            trainable_params = list(p for p in vae_shader.vae.parameters() if p.requires_grad) + list(intrinsic_net.parameters())
            optimizer = torch.optim.AdamW(trainable_params, lr=configs.learning_rate)
        elif configs.train_phase == "ns":
            optimizer = torch.optim.AdamW(neural_renderer.parameters(), lr=configs.learning_rate)
        else:
            raise RuntimeError("Train phase should be 'vae' or 'ns'.")
        
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.total_train_epochs)
        
        # Prepare everything
        # There is no specific order to remember, you just need to unpack the
        # objects in the same order you gave them to the prepare method.
        (self.vae_shader, 
         self.intrinsic_net,
         self.neural_renderer,
         self.optimizer, 
         self.lr_scheduler, 
         self.train_loader, 
         self.eval_loader) = self.accelerator.prepare(vae_shader, intrinsic_net, neural_renderer, optimizer, lr_scheduler, train_loader, eval_loader)
        
        if configs.train_phase == "ns":
            # Load pretrained weights from vae phase
            dirs = os.listdir(self.project_dir)
            dirs = [d for d in dirs if d.startswith(f"vae")]
            path = dirs[-1] if len(dirs) > 0 else None
            
            if path is None:
                raise RuntimeError("VAE weights does not exist.")
            else:
                # vae_weights = torch.load(f"{self.project_dir}/vae.pth")
                vae_weights = torch.load(f"experiments/exp_27/vae.pth")
                self.vae_shader.load_state_dict(vae_weights)
        
        self.train_resize = transforms.Resize((self.height, self.width), interpolation=transforms.InterpolationMode.BILINEAR)
        self.train_crop = transforms.CenterCrop((self.height, self.width)) if configs.center_crop else transforms.RandomCrop(configs.resolution)
        self.train_flip = transforms.RandomHorizontalFlip(p=1.0)
        self.train_transforms = transforms.Compose([transforms.Normalize([0.5], [0.5])])
        self.bce_loss = BCEDiceBoundaryLoss()
        
        # Metrics
        self.psnr = PeakSignalNoiseRatio().to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(self.device)
        
    def train(self):
        
        # Initial log
        self.logger.info("***** Running training *****")
        self.logger.info(f"  Training backbone: {self.configs.pretrained_model_name_or_path}")
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
                dirs = [d for d in dirs if d.startswith(f"checkpoint-{self.configs.train_phase}")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[2]))
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
                self.accelerator.load_state(os.path.join(self.checkpoints_dir, path))
                global_step = int(path.split("-")[2])

                self.initial_step = global_step % self.num_update_steps_per_epoch
                self.global_step = global_step
                current_epoch = global_step // self.num_update_steps_per_epoch

        else:
            self.initial_step = 0
            self.global_step = 0
            current_epoch = 0
        
        self.count = 0
        
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
                self.count += 1
        
        self.accelerator.end_training()
    
    def train_epoch(self):
        
        # Create train data iterator
        train_iter = iter(self.train_loader)
        
        # Train loop for vae
        self.vae_shader.train()
        train_loss = 0.0
        progress_bar = tqdm(
            range(self.initial_step, len(self.train_loader)),
            total=len(self.train_loader),
            initial=self.initial_step,
            ncols=90,
            disable=not self.accelerator.is_local_main_process
        )
        for step in progress_bar:
            with self.accelerator.accumulate(self.vae_shader):
                # Load data
                train_data = next(train_iter)
                
                rgb_gt = train_data["rgb"].permute(0,3,1,2)
                depth_gt = train_data["depth"].permute(0,3,1,2)
                normal_gt = train_data["normal"].permute(0,3,1,2)
                mask_gt = train_data["mask"].permute(0,3,1,2)
                albedo_gt = train_data["albedo"].permute(0,3,1,2)
                roughness_gt = train_data["roughness"].permute(0,3,1,2)
                specular_gt = train_data["specular"].permute(0,3,1,2)
                
                fov_gt = train_data["fov"]
                
                if self.configs.train_phase == "vae":
                
                    (rgb_pred, 
                     depth_pred, 
                     normal_pred, 
                     mask_pred,
                     albedo_pred,
                     roughness_pred,
                     specular_pred,
                     kl_loss,
                     mat_feature) = self.vae_shader((rgb_gt-0.5)/0.5)
                    
                    # rgb loss
                    rgb_loss = nn.functional.l1_loss(rgb_pred, rgb_gt)
                    
                    # depth loss
                    depth_loss = nn.functional.l1_loss(depth_pred, depth_gt)
                    
                    # normal loss
                    normal_loss = 1 - nn.functional.cosine_similarity(normal_pred, normal_gt).mean()
                    
                    # mask loss
                    mask_loss = self.bce_loss(mask_pred, mask_gt)
                    
                    mat_loss = 0
                    # albedo loss
                    mat_loss += nn.functional.mse_loss(albedo_pred, albedo_gt)
                    
                    # roughness loss
                    mat_loss += nn.functional.mse_loss(roughness_pred, roughness_gt)
                    
                    # specular loss
                    mat_loss += nn.functional.mse_loss(specular_pred, specular_gt)
                    
                    # fov loss
                    # Estimate fov
                    intrinsic_pred = self.intrinsic_net(mat_feature)
                    fov_norm = nn.functional.sigmoid(intrinsic_pred[:,0])
                    fov_pred = fov_norm * 180
                    
                    # Depth denormailze
                    denorm_depth = (depth_pred * (self.configs.z_far - self.configs.z_near) + self.configs.z_near) * mask_pred
                    
                    # Camera position reconstruction
                    cam_pos_pred = self.get_cam_coords(denorm_depth, width=self.width, height=self.height, fov=fov_pred)
                    cam_pos_pred[...,:2] = cam_pos_pred[...,:2] / 80.
                    cam_pos_pred[...,2] = cam_pos_pred[...,2] / 800.
                    
                    fov_loss = nn.functional.l1_loss(torch.log10(fov_pred), torch.log10(fov_gt.float()))
                    
                    cam_pos_loss = nn.functional.l1_loss(cam_pos_pred, train_data["cam_coords"])
                    
                    
                    
                    total_loss = rgb_loss + depth_loss + normal_loss + \
                        mask_loss + mat_loss + fov_loss
                    
                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = self.accelerator.gather(total_loss.repeat(self.configs.train_batch_size)).mean()
                    train_loss += avg_loss.item() / self.configs.gradient_accumulation_steps
                    
                    log_dict = {
                        f"train_{self.configs.train_phase}/total_loss": total_loss.item(),
                        f"train_{self.configs.train_phase}/rgb_loss": rgb_loss.item(),
                        f"train_{self.configs.train_phase}/depth_loss": depth_loss.item(),
                        f"train_{self.configs.train_phase}/normal_loss": normal_loss.item(),
                        f"train_{self.configs.train_phase}/mask_loss": mask_loss.item(),
                        f"train_{self.configs.train_phase}/mat_loss": mat_loss.item(),
                        f"train_{self.configs.train_phase}/fov_loss": fov_loss.item(),
                        f"train_{self.configs.train_phase}/cam_pos_loss": cam_pos_loss.item(),
                        # f"train_{self.configs.train_phase}/kl_loss": 0.000001 * kl_loss.item()
                    }
                    
                    # Backpropagate
                    self.accelerator.backward(total_loss)
                    if self.accelerator.sync_gradients:
                        params_to_clip = self.vae_shader.parameters()
                        self.accelerator.clip_grad_norm_(params_to_clip, self.configs.max_grad_norm)
                    self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                
                elif self.configs.train_phase == "ns":
                    
                    with torch.no_grad():
                        (rgb_pred, 
                         depth_pred, 
                         normal_pred, 
                         mask_pred,
                         albedo_pred,
                         roughness_pred,
                         specular_pred,
                         kl_loss,
                         mat_feature_list) = self.vae_shader(rgb_gt)
                    
                    denorm_depth = (depth_pred * (self.configs.z_far - self.configs.z_near) + \
                        self.configs.z_near) * mask_pred
                    
                    fov = torch.tensor(85., device=depth_pred.device)[None].repeat(depth_pred.shape[0])
                    cam_pos_pred = self.get_cam_coords(denorm_depth.unsqueeze(1), width=self.width, height=self.height, fov=fov)
                    mask = mask_pred.squeeze(1).bool()
                    normal_pred = (normal_pred * 2) - 1.
                    
                    render_buffer = {
                        "rgb_gt": rgb_gt.permute(0,2,3,1),
                        "normal_gt": normal_pred.permute(0,2,3,1),
                        "albedo_gt": albedo_pred.permute(0,2,3,1),
                        "roughness_gt": roughness_pred.permute(0,2,3,1),
                        "specular_gt": specular_pred.permute(0,2,3,1),
                        "pos_in_cam_gt": cam_pos_pred,
                        "hdri_gt": train_data["hdri"],
                        "mask": mask
                    }
                    
                    rgb_shading, rand_indices = self.neural_renderer(render_buffer=render_buffer, num_light_samples=128, inference=False)
                    
                    total_loss = nn.functional.l1_loss(train_data["rgb"][mask][rand_indices], rgb_shading)
                    # total_loss = nn.functional.l1_loss(train_data["rgb"][mask], rgb_shading)
                    
                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = self.accelerator.gather(total_loss.repeat(self.configs.train_batch_size)).mean()
                    train_loss += avg_loss.item() / self.configs.gradient_accumulation_steps
                    
                    log_dict = {
                        f"train_{self.configs.train_phase}/rendering_loss": total_loss.item()
                    }
                
                    # Backpropagate
                    self.accelerator.backward(total_loss)
                    if self.accelerator.sync_gradients:
                        params_to_clip = self.neural_renderer.parameters()
                        self.accelerator.clip_grad_norm_(params_to_clip, self.configs.max_grad_norm)
                    self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                
                # Logs
                logs = {"loss": train_loss}
                progress_bar.set_postfix(**logs)
                self.accelerator.log(log_dict, step=self.global_step)
                
                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    self.global_step += 1
                    train_loss = 0.0
                    self.save_checkpoint()
        
        self.lr_scheduler.step()
        self.initial_step = 0
        
    def eval_epoch(self):
        
        # Create train data iterator
        eval_iter = iter(self.eval_loader)
        
        # Train loop for unet
        self.vae_shader.eval()
        total_eval_loss = 0.
        total_rgb_loss = 0.
        total_depth_loss = 0.
        total_normal_loss = 0.
        total_fov_loss = 0.
        total_cam_pos_loss = 0.
        total_mat_loss = 0.
        total_mask_loss = 0.
        total_psnr = 0.
        total_ssim = 0.
        total_lpips = 0.
        save_out_row_list = []
        eval_num = 5
        progress_bar = tqdm(range(eval_num), ncols=90, disable=not self.accelerator.is_local_main_process)
        for step in progress_bar:
            # Load data
            eval_data = next(eval_iter)
            
            rgb_gt = eval_data["rgb"].permute(0,3,1,2)
            depth_gt = eval_data["depth"].permute(0,3,1,2)
            normal_gt = eval_data["normal"].permute(0,3,1,2)
            mask_gt = eval_data["mask"].permute(0,3,1,2)
            albedo_gt = eval_data["albedo"].permute(0,3,1,2)
            roughness_gt = eval_data["roughness"].permute(0,3,1,2)
            specular_gt = eval_data["specular"].permute(0,3,1,2)
            
            fov_gt = eval_data["fov"]
            
            (rgb_pred, 
             depth_pred, 
             normal_pred, 
             mask_pred,
             albedo_pred,
             roughness_pred,
             specular_pred,
             kl_loss,
             mat_feature) = self.vae_shader((rgb_gt-0.5)/0.5)
            
            save_out_row_list.append([rgb_pred, 
                                      depth_pred.repeat(1,3,1,1), 
                                      normal_pred,
                                      albedo_pred,
                                      roughness_pred.repeat(1,3,1,1), 
                                      specular_pred.repeat(1,3,1,1), 
                                      mask_pred.repeat(1,3,1,1)])
            
            if self.configs.train_phase == "vae":
                # rgb loss
                rgb_loss = nn.functional.l1_loss(rgb_pred, rgb_gt)
                total_rgb_loss += rgb_loss
                
                # depth loss
                depth_loss = nn.functional.l1_loss(depth_pred, depth_gt)
                total_depth_loss += depth_loss
                
                # normal loss
                normal_loss = 1 - nn.functional.cosine_similarity(normal_pred, normal_gt).mean()
                total_normal_loss += normal_loss
                
                # mask loss
                mask_loss = self.bce_loss(mask_pred.unsqueeze(1), mask_gt.unsqueeze(1))
                total_mask_loss += mask_loss
                
                mat_loss = 0
                # albedo loss
                mat_loss += nn.functional.mse_loss(albedo_pred, albedo_gt)
                
                # roughness loss
                mat_loss += nn.functional.mse_loss(roughness_pred, roughness_gt)
                
                # specular loss
                mat_loss += nn.functional.mse_loss(specular_pred, specular_gt)
                total_mat_loss += mat_loss
                
                # fov loss
                # Estimate fov
                intrinsic_pred = self.intrinsic_net(mat_feature)
                fov_norm = nn.functional.sigmoid(intrinsic_pred[:,0])
                fov_pred = fov_norm * 180
                
                # Depth denormailze
                denorm_depth = (depth_pred * (self.configs.z_far - self.configs.z_near) + self.configs.z_near) * mask_pred
                
                # Camera position reconstruction
                cam_pos_pred = self.get_cam_coords(denorm_depth.unsqueeze(1), width=self.width, height=self.height, fov=fov_pred)
                cam_pos_pred[...,:2] = cam_pos_pred[...,:2] / 80.
                cam_pos_pred[...,2] = cam_pos_pred[...,2] / 800.
                
                fov_loss = nn.functional.l1_loss(torch.log10(fov_pred), torch.log10(fov_gt.float()))
                total_fov_loss += fov_loss
                
                cam_pos_loss = nn.functional.l1_loss(cam_pos_pred, eval_data["cam_coords"])
                total_cam_pos_loss += cam_pos_loss
                
                total_eval_loss += rgb_loss + depth_loss + normal_loss + \
                        mask_loss + mat_loss + fov_loss + 0.000001 * kl_loss
            
            elif self.configs.train_phase == "ns":
                
                denorm_depth = (depth_pred * (self.configs.z_far - self.configs.z_near) + \
                        self.configs.z_near) * mask_pred
                    
                fov = torch.tensor(85., device=depth_pred.device)[None].repeat(depth_pred.shape[0])
                cam_pos_pred = self.get_cam_coords(denorm_depth.unsqueeze(1), width=self.width, height=self.height, fov=fov)
                mask = mask_pred.squeeze(1).bool()
                normal_pred = (normal_pred * 2) - 1.
                
                # Reconstruction rendering
                render_buffer = {
                    "rgb_gt": rgb_gt.permute(0,2,3,1),
                    "normal_gt": normal_pred.permute(0,2,3,1),
                    "albedo_gt": albedo_pred.permute(0,2,3,1),
                    "roughness_gt": roughness_pred.permute(0,2,3,1),
                    "specular_gt": specular_pred.permute(0,2,3,1),
                    "pos_in_cam_gt": cam_pos_pred,
                    "hdri_gt": eval_data["hdri"],
                    "mask": mask
                }
                
                rgb_shading = torch.ones(1,self.width,self.height,3,device=self.device)
                shading_result, _ = self.neural_renderer(render_buffer=render_buffer, num_light_samples=128, inference=True)
                rgb_shading[mask] = shading_result
                rgb_shading = rgb_shading.permute(0,3,1,2)
                
                save_out_row_list[-1].append(rgb_shading)
                
                # Compute metrics
                total_eval_loss += nn.functional.l1_loss(rgb_gt, rgb_shading)
                total_psnr += self.psnr(rgb_shading, rgb_gt)
                total_ssim += self.ssim(rgb_shading, rgb_gt)
                total_lpips += self.lpips(rgb_shading, rgb_gt)
                
                # Relighting rendering
                test_hdri = load_hdr("dataset/hdri/02.exr", resize=False).to(self.device)[None]
                
                render_buffer = {
                    "rgb_gt": rgb_gt.permute(0,2,3,1),
                    "normal_gt": normal_pred.permute(0,2,3,1),
                    "albedo_gt": albedo_pred.permute(0,2,3,1),
                    "roughness_gt": roughness_pred.permute(0,2,3,1),
                    "specular_gt": specular_pred.permute(0,2,3,1),
                    "pos_in_cam_gt": cam_pos_pred,
                    "hdri_gt": test_hdri,
                    "mask": mask
                }
                
                rgb_relighting = torch.ones(1,self.width,self.height,3,device=self.device)
                shading_result, _ = self.neural_renderer(render_buffer=render_buffer, num_light_samples=128, inference=True)
                rgb_relighting[mask] = shading_result
                rgb_relighting = rgb_relighting.permute(0,3,1,2)
                
                save_out_row_list[-1].append(rgb_relighting)
        
        # Compute average metrics
        if self.configs.train_phase == "vae":
            avg_eval_loss = total_eval_loss / eval_num
            avg_rgbd_loss = total_rgb_loss / eval_num
            avg_fov_loss = total_fov_loss / eval_num
            avg_cpos_loss = total_cam_pos_loss / eval_num
            avg_normal_loss = total_normal_loss / eval_num
            avg_mat_loss = total_mat_loss / eval_num
            avg_mask_loss = total_mask_loss / eval_num
            
            log_dict = {
                f"eval_{self.configs.train_phase}/eval_loss": avg_eval_loss.item(),
                f"eval_{self.configs.train_phase}/rgbd_loss": avg_rgbd_loss.item(),
                f"eval_{self.configs.train_phase}/fov_loss": avg_fov_loss.item(),
                f"eval_{self.configs.train_phase}/cam_pos_loss": avg_cpos_loss.item(),
                f"eval_{self.configs.train_phase}/normal_loss": avg_normal_loss.item(),
                f"eval_{self.configs.train_phase}/mat_loss": avg_mat_loss.item(),
                f"eval_{self.configs.train_phase}/mask_loss": avg_mask_loss.item()
            }
        elif self.configs.train_phase == "ns":
            avg_eval_loss = total_eval_loss / eval_num
            avg_psnr = total_psnr / eval_num
            avg_ssim = total_ssim / eval_num
            avg_lpips = total_lpips / eval_num
            
            log_dict = {
                f"eval_{self.configs.train_phase}/eval_loss": avg_eval_loss.item(),
                f"eval_{self.configs.train_phase}/psnr": avg_psnr.item(),
                f"eval_{self.configs.train_phase}/ssim": avg_ssim.item(),
                f"eval_{self.configs.train_phase}/lpips": avg_lpips.item()
            }
        
        self.accelerator.log(log_dict, step=self.global_step)

        # Evaluate the visual result and save the model
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            
            # Save weights
            if self.configs.train_phase == "vae":
                torch.save(self.vae_shader.state_dict(), f"{self.project_dir}/vae.pth")
                torch.save(self.intrinsic_net.state_dict(), f"{self.project_dir}/intrinsic.pth")
                
            elif self.configs.train_phase == "ns":
                torch.save(self.neural_renderer.state_dict(), f"{self.project_dir}/ns.pth")

            # Save samples
            save_out_col_list = []
            for save_out in save_out_row_list:
                save_out_col_list.append(torch.cat(save_out,dim=2))
            
            save_out = torch.cat(save_out_col_list[:3], dim=3)[0]
            
            tvf.to_pil_image(save_out).save(f"{self.sample_dir}/sample_{self.configs.train_phase}_{self.global_step}.png")
    
    # Helper functions
    def unwrap_model(self, model):
        model = self.accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model
    
    def save_checkpoint(self, with_lora=False):
        
        # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues.
        if self.accelerator.distributed_type == DistributedType.DEEPSPEED or self.accelerator.is_main_process:
            if self.global_step % self.configs.checkpointing_steps == 0:
                # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                if self.configs.checkpoints_total_limit is not None:
                    checkpoints = os.listdir(self.checkpoints_dir)
                    checkpoints = [d for d in checkpoints if d.startswith(f"checkpoint-{self.configs.train_phase}")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[2]))

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

                save_path = os.path.join(self.checkpoints_dir, f"checkpoint-{self.configs.train_phase}-{self.global_step}")
                self.accelerator.save_state(save_path)
                
                tqdm.write(f"Saved state to {save_path}")
    
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
        cam_pos[..., 2] = -depth.squeeze()
        return cam_pos
    
    def backproject(self, depth, intrinsics):
        """
        Given depth map [B,1,H,W] and intrinsics [B,3], compute per-pixel 3D positions.
        depth: metric depth in same units as f.
        intrinsics: (f, cx, cy), where cx, cy are offsets in pixel units from image center.
        Returns: positions [B,3,H,W]
        """
        B, _, H, W = depth.shape
        device = depth.device
        cx, cy = intrinsics[:,1], intrinsics[:,2]
        fov_norm = nn.functional.sigmoid(intrinsics[:,0])
        fov_pred = fov_norm * 180
        fov_radians = fov_pred * torch.pi / 180.0
        fx = W / (2.0 * torch.tan(fov_radians / 2.0))
        fy = H / (2.0 * torch.tan(fov_radians / 2.0))
        
        # Create meshgrid of pixel coords
        ys, xs = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        xs = xs.unsqueeze(0).expand(B,-1,-1).float()
        ys = ys.unsqueeze(0).expand(B,-1,-1).float()
        # principal point at center + offset
        c_x = (W - 1) / 2 + cx.unsqueeze(-1).unsqueeze(-1) * W/2
        c_y = (H - 1) / 2 + cy.unsqueeze(-1).unsqueeze(-1) * H/2
        # backproject
        X = (xs - c_x) * depth.squeeze(1) / fx.unsqueeze(-1).unsqueeze(-1)
        Y = (ys - c_y) * depth.squeeze(1) / fy.unsqueeze(-1).unsqueeze(-1)
        Z = depth.squeeze(1)
        return torch.stack([X, Y, Z], dim=1)  # [B,3,H,W]
    
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
