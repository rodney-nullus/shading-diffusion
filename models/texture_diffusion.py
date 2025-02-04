import torch.nn as nn


class TextureDiffusion(nn.Module):
    def __init__(self, configs, load_loacl=False):
        super().__init__()

        self.configs = configs

    def forward(self, x, text):
        pass

    def vae_forward(self, x, return_latent=False):
        pass