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
from models.vae_shader import VAE, NeuralShader

from loss.mask_loss import BCEDiceBoundaryLoss

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
        
        vae = VAE(configs=configs)
        
        # Intrinsics head: predicts focal length (f) and principal point offsets (cx, cy)
        # We global-pool features and regress 3 values
        intrinsic_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # [B,C,1,1]
            nn.Flatten(),              # [B,C]
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)           # [f, cx, cy]
        )
        
        trainable_params = list(vae.parameters()) + list(intrinsic_net.parameters())
        
        neural_shader = NeuralShader(width=256, height=256)
        
        if configs.train_phase == "ns":
            trainable_params = list(neural_shader.parameters())
        
        optimizer = torch.optim.AdamW(trainable_params, lr=configs.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.total_train_epochs)
        
        # Prepare everything
        # There is no specific order to remember, you just need to unpack the
        # objects in the same order you gave them to the prepare method.
        (self.vae, 
         self.intrinsic_net,
         self.neural_shader,
         self.optimizer, 
         self.lr_scheduler, 
         self.train_loader, 
         self.eval_loader) = self.accelerator.prepare(vae, intrinsic_net, neural_shader, optimizer, lr_scheduler, train_loader, eval_loader)
        
        if configs.train_phase == "ns":
            # Load pretrained weights from vae phase
            dirs = os.listdir(self.checkpoints_dir)
            dirs = [d for d in dirs if d.startswith(f"checkpoint-vae")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[2]))
            path = dirs[-1] if len(dirs) > 0 else None
            
            if path is None:
                self.accelerator.print(
                    f"Checkpoint '{self.configs.resume_from_checkpoint}' does not exist"
                )
                raise
            else:
                # vae_weights = torch.load(f"{self.project_dir}/vae.pth")
                vae_weights = torch.load("experiments/exp_19_vae/vae.pth")
                self.vae.load_state_dict(vae_weights)
        
        self.train_resize = transforms.Resize((self.height, self.width), interpolation=transforms.InterpolationMode.BILINEAR)
        self.train_crop = transforms.CenterCrop((self.height, self.width)) if configs.center_crop else transforms.RandomCrop(configs.resolution)
        self.train_flip = transforms.RandomHorizontalFlip(p=1.0)
        self.train_transforms = transforms.Compose([transforms.Normalize([0.5], [0.5])])
        self.bce_loss = BCEDiceBoundaryLoss()
        
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
                
                rgb_gt = train_data["rgb"].permute(0,3,1,2)
                
                geo_output, mat_output, mat_feature_list, kl_loss = self.vae(input=model_input.to(self.weight_dtype))
                
                # Estimate fov
                intrinsic_pred = self.intrinsic_net(mat_feature_list[-1])
                fov_norm = nn.functional.sigmoid(intrinsic_pred[:,0])
                fov_pred = fov_norm * 180
                
                # Depth denormailze
                depth_pred = geo_output[:,3]
                denorm_depth = (depth_pred * (self.configs.z_far - self.configs.z_near) + self.configs.z_near) * train_data["mask"].squeeze()
                
                # Camera position reconstruction
                cam_pos_pred = self.get_cam_coords(denorm_depth.unsqueeze(1), width=rgb_gt.shape[3], height=rgb_gt.shape[2], fov=fov_pred)
                cam_pos_pred[...,:2] = cam_pos_pred[...,:2] / 80.
                cam_pos_pred[...,2] = - cam_pos_pred[...,2] / 800.

                # Compute loss
                rgbd_gt = torch.cat([rgb_gt, train_data["depth"].permute(0,3,1,2)], dim=1)
                rgbd_loss = nn.functional.l1_loss(geo_output[:,:4], rgbd_gt)
                
                fov_loss = nn.functional.l1_loss(torch.log10(fov_pred), torch.log10(train_data["fov"].float()))
                
                normal_pred = geo_output[:,4:7]
                normal_loss = 1 - nn.functional.cosine_similarity(normal_pred, train_data["normal"].permute(0,3,1,2)).mean()
                
                mask_pred = geo_output[:,7]
                mask_loss = self.bce_loss(mask_pred.unsqueeze(1), train_data["mask"].permute(0,3,1,2))
                
                cam_pos_loss = nn.functional.l1_loss(cam_pos_pred, train_data["cam_coords"])
                
                mat_loss = nn.functional.mse_loss(mat_output, model_input[:,7:12])
                
                total_loss = rgbd_loss + normal_loss + fov_loss + mask_loss + mat_loss + 0.000001 * kl_loss
                
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = self.accelerator.gather(total_loss.repeat(self.configs.train_batch_size)).mean()
                train_loss += avg_loss.item() / self.configs.gradient_accumulation_steps
                
                log_dict = {
                    f"train_{self.configs.train_phase}/total_loss": total_loss.item(),
                    f"train_{self.configs.train_phase}/rgbd_loss": rgbd_loss.item(),
                    f"train_{self.configs.train_phase}/normal_loss": normal_loss.item(),
                    f"train_{self.configs.train_phase}/fov_loss": fov_loss.item(),
                    f"train_{self.configs.train_phase}/cam_pos_loss": cam_pos_loss.item(),
                    f"train_{self.configs.train_phase}/mat_loss": mat_loss.item(),
                    f"train_{self.configs.train_phase}/mask_loss": mask_loss.item(),
                    f"train_{self.configs.train_phase}/kl_loss": 0.000001 * kl_loss.item()
                }
                
                if self.configs.train_phase == "ns":
                    
                    shading_rgb = self.neural_shader(depth_map=depth_pred,
                                             fov=10.,
                                             mat_map=mat_output,
                                             normal_map=normal_pred,
                                             mask=mask_pred, env_map=train_data["hdri"])
                    
                    total_loss = nn.functional.l1_loss(train_data["rgb"], shading_rgb)
                    
                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = self.accelerator.gather(total_loss.repeat(self.configs.train_batch_size)).mean()
                    train_loss += avg_loss.item() / self.configs.gradient_accumulation_steps
                    
                    log_dict = {
                        f"train_{self.configs.train_phase}/total_loss": total_loss.item()
                    }
                
                # Backpropagate
                self.accelerator.backward(total_loss)
                if self.accelerator.sync_gradients:
                    params_to_clip = self.vae.parameters()
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
        self.vae.eval()
        eval_loss = 0.
        rgbd_loss = 0.
        normal_loss = 0.
        fov_loss = 0.
        cam_pos_loss = 0.
        mat_loss = 0.
        mask_loss = 0.
        eval_num = 1
        progress_bar = tqdm(range(eval_num), ncols=90, disable=not self.accelerator.is_local_main_process)
        for step in progress_bar:
            # Load data
            eval_data = next(eval_iter)
            
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
            
            rgb_gt = eval_data["rgb"].permute(0,3,1,2)
            
            geo_output, mat_output, mat_feature_list, kl_loss = self.vae(model_input.to(self.weight_dtype))
            
            rgb_pred = geo_output[:,:3]
            depth_pred = geo_output[:,3]
            normal_pred = geo_output[:,4:7]
            mask_pred = geo_output[:,7]
            
            # Estimate fov
            intrinsic_pred = self.intrinsic_net(mat_feature_list[-1])
            fov_norm = nn.functional.sigmoid(intrinsic_pred[:,0])
            fov_pred = fov_norm * 180
            
            # Depth denormailze
            denorm_depth = (depth_pred * (self.configs.z_far - self.configs.z_near) + self.configs.z_near) * eval_data["mask"].squeeze()
            
            # Camera position reconstruction
            cam_pos_pred = self.get_cam_coords(denorm_depth.unsqueeze(1), width=rgb_gt.shape[3], height=rgb_gt.shape[2], fov=fov_pred)
            cam_pos_pred[...,:2] = cam_pos_pred[...,:2] / 80.
            cam_pos_pred[...,2] = - cam_pos_pred[...,2] / 800.
            
            # Compute metrics
            rgbd_gt = torch.cat([rgb_gt, eval_data["depth"].permute(0,3,1,2)], dim=1)
            rgbd_loss += nn.functional.l1_loss(geo_output[:,:4], rgbd_gt)
            
            fov_loss += nn.functional.l1_loss(torch.log10(fov_pred), torch.log10(eval_data["fov"].float()))
            
            normal_loss += 1 - nn.functional.cosine_similarity(normal_pred, eval_data["normal"].permute(0,3,1,2)).mean()
            
            mask_loss += self.bce_loss(mask_pred.unsqueeze(1), eval_data["mask"].permute(0,3,1,2))
            
            cam_pos_loss += nn.functional.l1_loss(cam_pos_pred, eval_data["cam_coords"])
            
            mat_loss += nn.functional.mse_loss(mat_output, model_input[:,7:12])
            
            eval_loss += rgbd_loss + normal_loss + fov_loss + mask_loss + mat_loss + 0.000001 * kl_loss
        
        # Compute average metrics
        avg_eval_loss = eval_loss / eval_num
        avg_rgbd_loss = rgbd_loss / eval_num
        avg_fov_loss = fov_loss / eval_num
        avg_cpos_loss = cam_pos_loss / eval_num
        avg_normal_loss = normal_loss / eval_num
        avg_mat_loss = mat_loss / eval_num
        avg_mask_loss = mask_loss / eval_num
        self.accelerator.log({
            f"eval_{self.configs.train_phase}/eval_loss": avg_eval_loss.item(),
            f"eval_{self.configs.train_phase}/rgbd_loss": avg_rgbd_loss.item(),
            f"eval_{self.configs.train_phase}/fov_loss": avg_fov_loss.item(),
            f"eval_{self.configs.train_phase}/cam_pos_loss": avg_cpos_loss.item(),
            f"eval_{self.configs.train_phase}/normal_loss": avg_normal_loss.item(),
            f"eval_{self.configs.train_phase}/mat_loss": avg_mat_loss.item(),
            f"eval_{self.configs.train_phase}/mask_loss": avg_mask_loss.item()
        }, step=self.global_step)

        # Evaluate the visual result and save the model
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            
            # Save checkpoint firstly
            torch.save(self.vae.state_dict(), f"{self.project_dir}/vae.pth")
            torch.save(self.intrinsic_net.state_dict(), f"{self.project_dir}/intrinsic.pth")
            
            # Save samples
            self.save_output_sample(rgb_pred, depth_pred, normal_pred, mask_pred, mat_output)
    
    # Helper functions
    def unwrap_model(self, model):
        model = self.accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model
    
    def save_output_sample(self, rgb, depth, normal, mask, mat):
        
        rgb_list = []
        depth_list = []
        normal_list = []
        mask_list = []
        albedo_list = []
        roughness_list = []
        specular_list = []
        for i in range(10):
            rgb_list.append(rgb[i])
            depth_list.append(depth[i].repeat(3,1,1))
            normal_list.append(normal[i])
            mask_list.append(mask[i].repeat(3,1,1))
            albedo_list.append(mat[i,:3])
            roughness_list.append(mat[i,3].repeat(3,1,1))
            specular_list.append(mat[i,4].repeat(3,1,1))
        
        rgb_o = torch.cat(rgb_list, dim=2)
        depth_o = torch.cat(depth_list, dim=2)
        normal_o = torch.cat(normal_list, dim=2)
        mask_o = torch.cat(mask_list, dim=2)
        albedo_o = torch.cat(albedo_list, dim=2)
        roughness_o = torch.cat(roughness_list, dim=2)
        specular_o = torch.cat(specular_list, dim=2)
        
        output = torch.cat([rgb_o, 
                            depth_o, 
                            normal_o, 
                            albedo_o, 
                            roughness_o, 
                            specular_o, 
                            mask_o], dim=1)
        tvf.to_pil_image(output).save(f"{self.sample_dir}/sample_{self.configs.train_phase}_{self.global_step}.png")
    
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
        cam_pos[..., 2] = depth.squeeze()
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
