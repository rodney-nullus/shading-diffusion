import torch.nn as nn
from .base_diffusion_sd_v1 import BaseDiffusion

class TextureDiffusion(BaseDiffusion):
    def __init__(self, configs):
        super().__init__(configs)

        # Change the number of input channels of vae encoder
        conv_in_out_chns = self.vae.encoder.conv_in.out_channels
        self.vae.encoder.conv_in = nn.Conv2d(configs.tex_diff_inchns, conv_in_out_chns, kernel_size=3, stride=1, padding=1)
        
        # Change the number of output channels of vae decoder
        conv_out_in_chns = self.vae.decoder.conv_out.in_channels
        self.vae.decoder.conv_out = nn.Conv2d(conv_out_in_chns, configs.tex_diff_outchns,kernel_size=3, stride=1, padding=1)
        
        # Freeze all parameters except customized layers
        for param in self.vae.parameters():
            param.requires_grad = False
        
        for param in self.vae.encoder.conv_in.parameters():
            param.requires_grad = True
        
        for param in self.vae.decoder.conv_out.parameters():
            param.requires_grad = True
        
        self.vae.enable_xformers_memory_efficient_attention()

    def forward(self, x, text):
        pass

    def vae_forward(self, x, return_latent=False):
        
        latents = self.vae.encode(x).latent_dist
        samples = latents.sample()
        rec_x = self.vae.decode(samples).sample
        
        if return_latent:
            return rec_x, latents
        
        return rec_x
    
    def unet_forward(self):
        pass