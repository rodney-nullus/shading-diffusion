import random
import numpy as np

import torch
import  torch.nn as nn

from diffusers import AutoencoderKL, SD3Transformer2DModel, FlowMatchEulerDiscreteScheduler, StableDiffusion3Pipeline
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    SiglipImageProcessor,
    SiglipVisionModel,
    T5EncoderModel,
    T5TokenizerFast,
)

class BaseDiffusion(nn.Module):
    
    tokenizer_1 = AutoTokenizer.from_pretrained(
        "sd-legacy/stable-diffusion-v1-5",
        subfolder="tokenizer",
        use_fast=False,
    )
    
    tokenizer_2 = AutoTokenizer.from_pretrained(
        "sd-legacy/stable-diffusion-v1-5",
        subfolder="tokenizer_2",
        use_fast=False,
    )
    
    text_encoder_1 = CLIPTextModel.from_pretrained(
        "sd-legacy/stable-diffusion-v1-5", 
        subfolder="text_encoder"
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        "sd-legacy/stable-diffusion-v1-5", 
        subfolder="text_encoder_2"
    )
    
    # Freeze text encoders.
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    
    def __init__(self, configs):
        super().__init__()
        
        # Save configuration for everywhere
        self.configs = configs
        
        # Load componants of stable diffusion
        self.vae = AutoencoderKL.from_pretrained("sd-legacy/stable-diffusion-v1-5", subfolder="vae")
        self.vae.enable_xformers_memory_efficient_attention()
        
        self.unet = UNet2DConditionModel.from_pretrained("sd-legacy/stable-diffusion-v1-5", subfolder="unet")
        self.unet.enable_xformers_memory_efficient_attention()
        
        self.noise_scheduler = DDPMScheduler.from_pretrained("sd-legacy/stable-diffusion-v1-5", subfolder="scheduler")
