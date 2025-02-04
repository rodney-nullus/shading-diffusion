import os
from tqdm import tqdm
from dataclasses import asdict

import torch

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

from dataloader.celeba_pbr import get_dataloader

from utils.graphic_utils import *

from models.geometry_diffusion import GeometryDiffusion
from models.texture_diffusion import TextureDiffusion

class Trainer:
    def __init__(self, configs):

        self.configs = configs
        
        # Initialize accelerator and tensorboard logging
        project_config = ProjectConfiguration(project_dir=configs.output_dir)
        self.accelerator = Accelerator(
            mixed_precision=configs.mixed_precision,
            #gradient_accumulation_steps=configs.gradient_accumulation_steps,
            log_with="tensorboard",
            project_config=project_config
        )
        
        self.device = self.accelerator.device
        self.accelerator.init_trackers(project_name=configs.exp_name)
        
        self.weights_dir = f'{configs.output_dir}/{configs.exp_name}/weights'
        os.makedirs(self.weights_dir, exist_ok=True)
        
        # load models
        geo_diff = GeometryDiffusion(configs)
        
        # Load dataloader
        train_loader, eval_loader = get_dataloader(configs)
        
        # Load optimizer
        geo_optim = torch.optim.Adam(geo_diff.parameters(), lr=configs.learning_rate)
        
        # Load lr_scheduler
        geo_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(geo_optim, T_max=configs.total_epochs)
        
        # Loss function
        self.loss_fn = torch.nn.MSELoss()
        
        # Prepare everything
        # There is no specific order to remember, you just need to unpack the
        # objects in the same order you gave them to the prepare method.
        self.geo_diff, self.geo_optim, self.geo_lr_scheduler, self.train_loader, self.eval_loader = self.accelerator.prepare(
            geo_diff, geo_optim, geo_lr_scheduler, train_loader, eval_loader, 
        )
    
    def train(self):
        
        # Initilize training settings
        self.epoch_num = 0
        self.global_step = 0
        
        while True:
            
            # Train for one epoch
            self.train_epoch()
            
            # Evaluate the model
            self.eval_epoch()
            
            # Save model
            if self.epoch_num % self.configs.save_model_epochs == 0:
                self.accelerator.wait_for_everyone()
                self.accelerator.save_model(self.geo_diff, f"{self.weights_dir}/geo_diff_{self.epoch_num}.pth")
            
            if self.epoch_num == self.configs.total_epochs:
                break
            
            self.epoch_num += 1
        
        self.accelerator.end_training()
    
    def train_epoch(self):
        
        train_iter = iter(self.train_loader)
        
        self.geo_diff.train()
        progress_bar = tqdm(range(len(self.train_loader)), ncols=100, 
                            disable=not self.accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {self.epoch_num}")
        for step in progress_bar:
            
            # Load data
            train_data = next(train_iter)

            # Train Geometry Diffusion
            if self.configs.train_mode == "geo-vae":
                
                # Inverse view coordinates normalization
                v_coords_norm = 1 / train_data['v_coords'] * 10.
                
                input_data = torch.cat([v_coords_norm, train_data['normal']], dim=-1).permute(0,3,1,2)
                
                rec_data, latents = self.geo_diff.vae_forward(input_data, return_latent=True)
                
                rec_v_coords, rec_normal = rec_data.chunk(2, dim=1)

                geo_loss = self.loss_fn(rec_data, input_data)

                # Backward pass
                self.geo_optim.zero_grad()
                self.accelerator.backward(geo_loss)
                self.geo_optim.step()
                
                progress_bar.update(1)
                logs = {"loss": geo_loss.detach().item(), "lr": self.geo_lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                self.accelerator.log(logs, step=self.global_step)
                self.global_step += 1
            
            elif self.configs.train_mode == "tex-ddpm":
                pass
            
            # Train Texture Diffusion
            elif self.configs.train_mode == "tex-vae":
                pass
            
            else:
                raise ValueError("Invalid train mode")
        
        self.geo_lr_scheduler.step()
        
        # Log the loss
        self.accelerator.track(self.geo_loss.item(), "geo_loss", self.global_step)
        
        
        
    def eval_epoch(self):
        pass