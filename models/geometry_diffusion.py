import  torch.nn as nn
from .base_diffusion_sd_v1 import BaseDiffusion

class GeometryDiffusion(BaseDiffusion):
    def __init__(self, configs):
        super().__init__(configs)
        
        # Change the number of input channels of vae encoder
        conv_in_out_chns = self.vae.encoder.conv_in.out_channels
        self.vae.encoder.conv_in = nn.Conv2d(configs.geo_diff_inchns, conv_in_out_chns, kernel_size=3, stride=1, padding=1)
        
        # Change the number of output channels of vae decoder
        conv_out_in_chns = self.vae.decoder.conv_out.in_channels
        self.vae.decoder.conv_out = nn.Conv2d(conv_out_in_chns, configs.geo_diff_outchns,kernel_size=3, stride=1, padding=1)
        
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
    
    def to(self, *args, **kwargs):
        
        # Move nn.Module components
        super().to(*args, **kwargs)
        
        # Manually call `to()` on custom submodule
        self.vae.to(*args, **kwargs)
        
        return self
