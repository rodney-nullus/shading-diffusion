from typing import Optional, Tuple
import copy

import torch
import torch.nn as nn
from torch.nn.functional import grid_sample

from diffusers import AutoencoderKL
from diffusers.models.attention_processor import SpatialNorm
from diffusers.models.unets.unet_2d_blocks import (
    UNetMidBlock2D,
    get_up_block
)

from .positional_embedder import get_embedder
from .gfft import GaussianFourierFeatureTransform
from .mlp_layers import FC
from .sampler import build_envmap_cdf, sample_envmap_direction

from configs.training_configs_vae import Configs

class NeuralShader(nn.Module):
    def __init__(self,
                 width,
                 height,
                 hidden_features_size=128,
                 hidden_features_layers=2,
                 activation="relu",
                 last_activation=None,
                 fourier_features="positional",
                 mapping_size=256,
                 fft_scale=8,
                 device="cuda"):
        
        super().__init__()
        
        self.width = width
        self.height = height
        self.device = device
        
        # self.fourier_feature_transform = None
        # if fourier_features == "positional":
        #     self.fourier_feature_transform, channels = get_embedder(fft_scale, 11)
        # elif fourier_features == "gfft":
        #     self.fourier_feature_transform = GaussianFourierFeatureTransform(11, mapping_size=mapping_size//2, scale=fft_scale, device=device)
        #     channels = mapping_size
        # elif fourier_features == "none":
        #     pass
        # else:
        #     raise "Invalid fourier features setting. Shoud be one of 'positional', 'gfft', 'none'."
                
        # Diffuse Network
        self.brdf_net = FC(in_features=8, 
                           out_features=3, 
                           hidden_features=[hidden_features_size] * hidden_features_layers, 
                           activation=activation, 
                           last_activation=None)
        
        cam_pos = torch.tensor([0., 0., 0.])[None,None,:]
        self.cam_pos = nn.Parameter(cam_pos, requires_grad=False)
    
    def get_cam_coords(self, depth, width, height, fov):
        fov = torch.tensor(fov, device=depth.device)[None].repeat(depth.shape[0])
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
    
    def forward(self, depth_map, fov, mat_map, normal_map, mask, env_map, spp=128):
        
        B, H, W = depth_map.shape
        
        with torch.no_grad():
            
            # 1) Importance sampling
            # Build CDFs
            cdf_marg, cdf_cond = build_envmap_cdf(env_map)
            
            # Draw inbound light samples
            w_i, uv = sample_envmap_direction(cdf_marg, cdf_cond, num_samples=spp)
            
            # Fetch radiance from envmap using uv coordinates
            # uv: [N, 2], values in [0, 1]
            B, ENV_H, ENV_W = env_map.shape[:3]
            u_idx = (uv[..., 0] * (ENV_W - 1)).long().clamp(0, ENV_W - 1)
            v_idx = (uv[..., 1] * (ENV_H - 1)).long().clamp(0, ENV_H - 1)
            
            # 构造 batch 维度的索引
            batch_idx = torch.arange(B, device=env_map.device)    # [B]
            batch_idx = batch_idx.view(B, 1).expand(-1, u_idx.size(1))  # [B, N]
            
            # 直接用高级索引，最后会保留 C 通道维度：
            # env_map[batch_idx, v_idx, u_idx] → [B, N, C]
            L_e = env_map[batch_idx, v_idx, u_idx]  # [B, N, 3]
            
            # # 2) Build screen uv grid
            # ys = torch.linspace(0.0, 1.0, steps=self.height, device=self.device)
            # xs = torch.linspace(0.0, 1.0, steps=self.width, device=self.device)
            # v, u = torch.meshgrid(ys, xs, indexing='ij')
            # uv_hw = torch.stack([u, v], dim=-1)  # [H, W, 2]
            # uv_batch = uv_hw.unsqueeze(0).expand(B, -1, -1, -1)  #[B,H,W,2]
            
            # # 3) Sample brdf feature
            # grid = uv_batch * 2.0 - 1.0  # Scale to [-1,1]
            # # grid = grid.reshape(1, 1, self.height*self.width, 2)        # [1,1,N,2]
            # sampled = grid_sample(latent, grid, align_corners=True)
            # masked_sample = sampled
            # brdf_codes = sampled.squeeze(0).squeeze(1).transpose(0,1)        # [N, C]
            
            # 4) Compute outbound direction
            cam_coords = self.get_cam_coords(depth=depth_map, width=self.width, height=self.height, fov=fov)
            w_o = self.cam_pos - cam_coords.unsqueeze(1)
            w_o = nn.functional.normalize(w_o, dim=-1)                                                                          # [B,1,H,W,3]
            
            half_dirs = w_i[:,:,None,None,:] + w_o                                                                              # [B,N,H,W,3]
            half_dirs = torch.nn.functional.normalize(half_dirs, dim=-1)                                                        # [B,N,H,W,3]
            
            n_d_i = (normal_map.permute(0,2,3,1).unsqueeze(1) * w_i[:,:,None,None,:]).sum(dim=-1, keepdim=True).clamp(min=0)    # [B,N,H,W,1]
            # n_d_o = (normal_map * w_o).sum(dim=-1, keepdim=True).clamp(min=0)                                                 # [S,N,1]
            # h_d_n = (normal_map * half_dirs).sum(dim=-1, keepdim=True).clamp(min=0)                                           # [S,N,1]
            # h_d_o = (half_dirs * w_o).sum(dim=-1, keepdim=True).clamp(min=0)                                                  # [S,N,1]
            
            L = (L_e[:,:,None,None,:] * n_d_i).sum(dim=1) / spp
            masked_L = L[mask.bool()]
            
            # if self.fourier_feature_transform is not None:
            #     diffuse_shading_input = self.fourier_feature_transform(diffuse_shading_input)
        
            masked_mat_sample = mat_map.permute(0,2,3,1)[mask.bool()]
            masked_w_o = w_o.squeeze(1)[mask.bool()]
            
            brdf_input = torch.cat([masked_mat_sample, masked_w_o], dim=-1)
        
        brdf = self.brdf_net(brdf_input)
        color = brdf * masked_L
        
        shading_rgb = torch.zeros(B, H, W, 3, device=self.device)
        shading_rgb[mask.bool()] = color.clamp(min=0.,max=1.)

        return shading_rgb

class ShadingDecoder(nn.Module):
    r"""
    The `Decoder` layer of a variational autoencoder that decodes its latent representation into an output sample.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        up_block_types (`Tuple[str, ...]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            The types of up blocks to use. See `~diffusers.models.unet_2d_blocks.get_up_block` for available options.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
        norm_type (`str`, *optional*, defaults to `"group"`):
            The normalization type to use. Can be either `"group"` or `"spatial"`.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        up_block_types: Tuple[str, ...] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        norm_type: str = "group",  # group, spatial
        mid_block_add_attention=True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.geo_up_blocks = nn.ModuleList([])
        self.mat_up_blocks = nn.ModuleList([])

        temb_channels = in_channels if norm_type == "spatial" else None

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default" if norm_type == "group" else norm_type,
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=temb_channels,
            add_attention=mid_block_add_attention,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1

            geo_up_block = get_up_block(
                up_block_type,
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=output_channel,
                temb_channels=temb_channels,
                resnet_time_scale_shift=norm_type,
            )
            
            mat_up_block = copy.deepcopy(geo_up_block)
            
            self.geo_up_blocks.append(geo_up_block)
            self.mat_up_blocks.append(mat_up_block)
            prev_output_channel = output_channel

        # out
        if norm_type == "spatial":
            self.conv_norm_out = SpatialNorm(block_out_channels[0], temb_channels)
        else:
            self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        
        self.geo_conv_out = nn.Conv2d(block_out_channels[0], 8, 3, padding=1)
        self.mat_conv_out = nn.Conv2d(block_out_channels[0], 5, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(
        self,
        sample: torch.Tensor,
        latent_embeds: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        r"""The forward method of the `Decoder` class."""

        sample = self.conv_in(sample)

        geo_upscale_dtype = next(iter(self.geo_up_blocks.parameters())).dtype
        mat_upscale_dtype = next(iter(self.mat_up_blocks.parameters())).dtype
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            # middle
            sample = self._gradient_checkpointing_func(self.mid_block, sample, latent_embeds)
            geo_sample = sample.to(geo_upscale_dtype)
            mat_sample = sample.to(mat_upscale_dtype)

            # up
            for up_block in self.geo_up_blocks:
                geo_sample = up_block(geo_sample, latent_embeds)
            
            up_mat_feature_list = []
            for up_block in self.mat_up_blocks:
                up_mat_feature_list.append(mat_sample)
                mat_sample = up_block(mat_sample, latent_embeds)
        else:
            # middle
            sample = self.mid_block(sample, latent_embeds)
            geo_sample = sample.to(geo_upscale_dtype)
            mat_sample = sample.to(mat_upscale_dtype)

            # up
            for up_block in self.geo_up_blocks:
                geo_sample = up_block(geo_sample, latent_embeds)
            
            up_mat_feature_list = []
            for up_block in self.mat_up_blocks:
                up_mat_feature_list.append(mat_sample)
                mat_sample = up_block(mat_sample, latent_embeds)
        
        # post-process
        if latent_embeds is None:
            geo_sample = self.conv_norm_out(geo_sample)
        else:
            geo_sample = self.conv_norm_out(geo_sample, latent_embeds)
        geo_sample = self.conv_act(geo_sample)
        geo_sample = self.geo_conv_out(geo_sample)
        
        if latent_embeds is None:
            mat_sample = self.conv_norm_out(mat_sample)
        else:
            mat_sample = self.conv_norm_out(mat_sample, latent_embeds)
        mat_sample = self.conv_act(mat_sample)
        mat_sample = self.mat_conv_out(mat_sample)
        
        return geo_sample, mat_sample, up_mat_feature_list

class VAE(nn.Module):
    def __init__(self, configs: Configs):
        
        super().__init__()
        
        # Save configuration for everywhere
        self.configs = configs
        
        # Load componants of stable diffusion
        vae = AutoencoderKL.from_pretrained(configs.pretrained_model_name_or_path, subfolder="vae")
        vae.enable_xformers_memory_efficient_attention()
        
        # Change the number of input channels of vae encoder
        conv_in_out_chns = vae.encoder.conv_in.out_channels
        vae.encoder.conv_in = nn.Conv2d(13, conv_in_out_chns, kernel_size=3, stride=1, padding=1)
        # vae.encoder.requires_grad_(False)
        
        pretrained_vae_config = vae.config
        
        shading_decoder: ShadingDecoder = ShadingDecoder(
            in_channels = pretrained_vae_config.latent_channels,
            out_channels = pretrained_vae_config.out_channels,
            up_block_types = pretrained_vae_config.up_block_types,
            block_out_channels = pretrained_vae_config.block_out_channels,
            layers_per_block = pretrained_vae_config.layers_per_block,
            norm_num_groups = pretrained_vae_config.norm_num_groups,
            act_fn = pretrained_vae_config.act_fn
        )
        
        # Copy pretained weights to shading decoder
        orig_decoder_sd = vae.decoder.state_dict()
        new_decoder_sd = shading_decoder.state_dict()
        shared_keys = orig_decoder_sd.keys() & new_decoder_sd.keys()
        filtered_sd = {k: orig_decoder_sd[k].clone() for k in shared_keys}
        shading_decoder.load_state_dict(filtered_sd, strict=False)
        
        up_blocks_copy = copy.deepcopy(vae.decoder.up_blocks)
        up_blocks_sd = up_blocks_copy.state_dict()
        shading_decoder.geo_up_blocks.load_state_dict(up_blocks_sd)
        shading_decoder.mat_up_blocks.load_state_dict(up_blocks_sd)
        
        # Replace pretrained 
        vae.decoder = shading_decoder
        
        # Change the number of output channels of vae decoder
        # conv_out_in_chns = vae.decoder.conv_out.in_channels
        # vae.decoder.conv_out = nn.Conv2d(conv_out_in_chns, 7, kernel_size=3, stride=1, padding=1)
            
        if configs.gradient_checkpointing:
            vae.enable_gradient_checkpointing()
        
        self.vae = vae
    
    def forward(self, input):
        
        posterior = self.vae.encode(input).latent_dist
        latents = posterior.sample()
        geo_output, mat_output, mat_feature_list = self.vae.decoder(latents)
        kl_loss = posterior.kl().mean()
            
        return geo_output.clamp(0,1), mat_output.clamp(0,1), mat_feature_list, kl_loss
