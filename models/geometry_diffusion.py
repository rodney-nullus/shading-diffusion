import torch
import  torch.nn as nn

from diffusers import AutoencoderKL, UNet2DConditionModel, EulerAncestralDiscreteScheduler, StableDiffusionXLPipeline
from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPTextModel, CLIPTextModelWithProjection

class GeometryDiffusion(nn.Module):
    def __init__(self, configs, load_loacl=False):
        super().__init__()
        
        self.configs = configs
        
        self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/sdxl-turbo", subfolder="text_encoder")
        self.text_encoder_2 = CLIPTextModel.from_pretrained("stabilityai/sdxl-turbo", subfolder="text_encoder_2")
        
        self.scheduler = EulerAncestralDiscreteScheduler.from_pretrained("stabilityai/sdxl-turbo", subfolder="scheduler")
        self.unet = UNet2DConditionModel.from_pretrained("stabilityai/sdxl-turbo", subfolder="unet")
        
        ## Load the VAE from pre-trained weights then customize it
        pretrained_vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-turbo", subfolder="vae")
        
        # Change the number of input channels of encoder
        pretrained_vae.encoder.conv_in = nn.Conv2d(configs.geo_diff_inchns, pretrained_vae.encoder.conv_in.out_channels, 
                                                   kernel_size=3, stride=1, padding=1)
        
        # Change the number of output channels of decoder
        pretrained_vae.decoder.conv_out = nn.Conv2d(pretrained_vae.decoder.conv_out.in_channels, configs.geo_diff_outchns,
                                                    kernel_size=3, stride=1, padding=1)
        
        # Freeze all parameters except customized layers
        for param in pretrained_vae.parameters():
            param.requires_grad = False
        
        for param in pretrained_vae.encoder.conv_in.parameters():
            param.requires_grad = True
        
        for param in pretrained_vae.decoder.conv_out.parameters():
            param.requires_grad = True
        
        self.vae = pretrained_vae
    
        # Create the pipeline
        # self.pipeline = StableDiffusionXLPipeline(vae=self.vae, 
        #                                           unet=self.unet, 
        #                                           scheduler=self.scheduler, 
        #                                           text_encoder=self.text_encoder, 
        #                                           text_encoder_2=self.text_encoder_2)

    def forward(self, x, text):
        pass
    
    def vae_forward(self, x, return_latent=False):
        
        latents = self.vae.encode(x).latent_dist
        samples = latents.sample()
        rec_x = self.vae.decode(samples).sample
        
        if return_latent:
            return rec_x, latents
        
        return rec_x
