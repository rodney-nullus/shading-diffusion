import os
from tqdm import tqdm

import torch

from accelerate import Accelerator

from dataloader.celeba_pbr import get_dataloader
from models.sd_models import ShadingDiffusion

class trainer:
    def __init__(self, configs):

        self.configs = configs
        
        # Initialize accelerator and tensorboard logging
        self.accelerator = Accelerator(
            mixed_precision=configs.mixed_precision,
            #gradient_accumulation_steps=configs.gradient_accumulation_steps,
            log_with="tensorboard",
            project_dir=os.path.join(configs.output_dir),
        )
        
        self.device = self.accelerator.device
        
        if self.accelerator.is_main_process:
            if configs.output_dir is not None:
                os.makedirs(configs.output_dir, exist_ok=True)
            self.accelerator.init_trackers()
        
        # load model
        model = ShadingDiffusion(configs)
        
        # Load dataloader
        train_loader, eval_loader = get_dataloader(configs)
        
        # Load optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=configs.learning_rate)
        
        # Load lr_scheduler
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=configs.total_epochs)
        
        # Prepare everything
        # There is no specific order to remember, you just need to unpack the
        # objects in the same order you gave them to the prepare method.
        self.model, self.optimizer, self.train_loader, self.eval_loader, self.lr_scheduler = self.accelerator.prepare(
            model, optimizer, train_loader, eval_loader, lr_scheduler
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
            
            if self.epoch_num == self.configs.total_epochs:
                break
            
            self.epoch_num += 1
        
        self.accelerator.end_training()
    
    def train_epoch(self):
        
        train_iter = iter(self.train_loader)
        
        progress_bar = tqdm(total=len(self.train_loader), disable=not self.accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {self.epoch_num}")
        for _ in progress_bar:
            
            # Load data
            data = next(train_iter)
            
            # Forward pass
            rec_data, latent = self.model.vae_forward(data, return_latent=True)
        
        
    
    def eval_epoch(self):
        pass