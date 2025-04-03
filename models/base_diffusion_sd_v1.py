import  torch.nn as nn

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, ControlNetModel

class BaseDiffusion(nn.Module):
    
    def __init__(self, configs):
        super().__init__()
        
        # Save configuration for everywhere
        self.configs = configs
        
        # Load componants of stable diffusion
        self.vae = AutoencoderKL.from_pretrained(configs.pretrained_model_name_or_path, subfolder="vae")
        self.vae.enable_xformers_memory_efficient_attention()
        
        self.unet = UNet2DConditionModel.from_pretrained(configs.pretrained_model_name_or_path, subfolder="unet")
        self.unet.enable_xformers_memory_efficient_attention()
        
        self.noise_scheduler = DDPMScheduler.from_pretrained(configs.pretrained_model_name_or_path, subfolder="scheduler")
        
        self.tokeinzer = CLIPTokenizer.from_pretrained(
            configs.pretrained_model_name_or_path,
            subfolder="tokenizer",
            use_fast=False,
        )
        
        self.text_encoder = CLIPTextModel.from_pretrained(
            configs.pretrained_model_name_or_path, 
            subfolder="text_encoder"
        )
        
        # Freeze text encoders.
        self.text_encoder.requires_grad_(False)
