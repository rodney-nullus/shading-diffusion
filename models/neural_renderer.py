from typing import List
import numpy as np
import torch
import torch.nn as nn

from utils.sampler import Sampler
from .sampler import build_envmap_cdf, sample_envmap_direction

from .gfft import GaussianFourierFeatureTransform
from .positional_embedder import get_embedder

class Sine(nn.Module):
    r"""Applies the sine function with frequency scaling element-wise:

    :math:`\text{Sine}(x)= \sin(\omega * x)`

    Args:
        omega: factor used for scaling the frequency

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional dimensions
        - Output: :math:`(N, *)`, same shape as the input
    """

    def __init__(self, omega):
        super().__init__()
        self.omega = omega

    def forward(self, x):
        return torch.sin(self.omega * x)

def make_module(module):
    # Create a module instance if we don't already have one
    if isinstance(module, torch.nn.Module):
        return module
    else:
        return module()

class FullyConnectedBlock(torch.nn.Module):
    def __init__(self, dim_in, dim_out, bias=True, activation=torch.nn.ReLU):
        super().__init__()

        self.linear = torch.nn.Linear(dim_in, dim_out, bias=bias)
        self.activation = make_module(activation) if activation is not None else torch.nn.Identity()

    def forward(self, input):
        return self.activation(self.linear(input))

class FullyConnectedResidualBlock(torch.nn.Module):
    def __init__(self, dim_in, dims_hidden, dim_out, bias=True,
                 activation_hidden=torch.nn.ReLU, activation=torch.nn.ReLU):
        super().__init__()

        self.dimensions = [dim_in] + dims_hidden + [dim_out]
        self.num_layers = len(self.dimensions) - 1

        # The only reason why we add the residual layers explicitly to this module
        # instead of using nn.Sequential, is that the graph visualization looks better
        # (e. g. in Tensorboard)
        for i in range(self.num_layers):
            if i == self.num_layers - 1:
                layer = FullyConnectedBlock(self.dimensions[i], self.dimensions[i + 1], activation=None)
            else:
                layer = FullyConnectedBlock(self.dimensions[i], self.dimensions[i + 1], activation=make_module(activation_hidden))
            self.add_module(f'Residual{i:d}', layer)

        self.shortcut = torch.nn.Identity() if dim_in == dim_out else torch.nn.Linear(dim_in, dim_out) 

        self.activation = torch.nn.Identity() if activation is None else make_module(activation)

    def forward(self, input):
        Fx = input
        for i in range(self.num_layers):
            Fx = self.__getattr__(f'Residual{i:d}')(Fx)

        x = self.shortcut(input)

        return self.activation(Fx + x)

def siren_init_first(**kwargs):
    module = kwargs['module']
    n = kwargs['n']
    if isinstance(module, nn.Linear):
        module.weight.data.uniform_(-1 / n, 
                                     1 / n)

def siren_init(**kwargs):
    module = kwargs['module']
    n = kwargs['n']
    omega = kwargs['omega']
    if isinstance(module, nn.Linear):
        module.weight.data.uniform_(-np.sqrt(6 / n) / omega, 
                                     np.sqrt(6 / n) / omega)

def init_weights_normal(**kwargs):
    module = kwargs['module']
    if isinstance(module, nn.Linear):
        if hasattr(module, 'weight'):
            nn.init.kaiming_normal_(module.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        if hasattr(module, 'bias'):
            nn.init.zeros_(module.bias)

def init_weights_normal_last(**kwargs):
    module = kwargs['module']
    if isinstance(module, nn.Linear):
        if hasattr(module, 'weight'):
            nn.init.xavier_normal_(module.weight, gain=1)
            module.weight.data = -torch.abs(module.weight.data)
        if hasattr(module, 'bias'):
            nn.init.zeros_(module.bias)
            
class FC(nn.Module):
    def __init__(self, in_features, out_features, hidden_features: List[int], activation='relu', last_activation=None, bias=True, first_omega=30, hidden_omega=30.0):
        super().__init__()

        layers = []

        activations_and_inits = {
            'sine': (Sine(first_omega),
                     siren_init,
                     siren_init_first,
                     None),
            'relu': (nn.ReLU(inplace=True),
                     init_weights_normal,
                     init_weights_normal,
                     None),
            'relu2': (nn.ReLU(inplace=True),
                     init_weights_normal,
                     init_weights_normal,
                     init_weights_normal_last),
            'softplus': (nn.Softplus(),
                        init_weights_normal,
                        None)
        }

        activation_fn, weight_init, first_layer_init, last_layer_init = activations_and_inits[activation]


        # First layer
        layer = FullyConnectedBlock(in_features, hidden_features[0], bias=bias, activation=activation_fn)
        if first_layer_init is not None: 
            layer.apply(lambda module: first_layer_init(module=module, n=in_features))
        layers.append(layer)

        for i in range(len(hidden_features)):
            n = hidden_features[i]

            # Initialize the layer right away
            layer = FullyConnectedBlock(n, n, bias=bias, activation=activation_fn)
            layer.apply(lambda module: weight_init(module=module, n=n, omega=hidden_omega))
            layers.append(layer)

        # Last layer
        layer = FullyConnectedBlock(hidden_features[-1], out_features, bias=bias, activation=last_activation)
        layer.apply(lambda module: weight_init(module=module, n=hidden_features[-1], omega=hidden_omega))
        if last_layer_init is not None: 
            layer.apply(lambda module: last_layer_init(module=module, n=in_features))
        layers.append(layer)

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
# class NeilfLighting(NeILFModel):
#     def __init__(self, config_model):
#         super().__init__(config_model)
    
#     def sample_split_incident_lights(self, neilf_weights, inputs, is_training=False):
        
#         self.load_state_dict(neilf_weights)

#         incident_dirs, incident_areas =  self.neilf_pbr.sample_incident_rays(inputs['normals'], is_training=is_training)

#         incident_lights = self.neilf_pbr.sample_incident_lights(inputs['positions'], incident_dirs)

#         return incident_lights, incident_dirs

class NeuralShader(nn.Module):
    def __init__(self,
                 hidden_features_size=128,
                 hidden_features_layers=2,
                 activation="relu",
                 last_activation=None,
                 fourier_features="positional",
                 mapping_size=256,
                 fft_scale=8,
                 num_train_sample=4096,
                 device='cuda'):
        
        super().__init__()
        
        self.num_train_sample = num_train_sample
        
        self.fourier_feature_transform = None
        if fourier_features == "positional":
            self.fourier_feature_transform, channels = get_embedder(fft_scale, 11)
        elif fourier_features == "gfft":
            self.fourier_feature_transform = GaussianFourierFeatureTransform(11, mapping_size=mapping_size//2, scale=fft_scale, device=device)
            channels = mapping_size
        elif fourier_features == "none":
            pass
        else:
            raise "Invalid fourier features setting. Shoud be one of 'positional', 'gfft', 'none'."
                
        # Diffuse Network
        self.diffuse = FC(in_features=channels, 
                          out_features=hidden_features_size, 
                          hidden_features=[hidden_features_size] * hidden_features_layers, 
                          activation=activation, 
                          last_activation=None)
        
        # Specular Network
        self.specular = FC(in_features=hidden_features_size + 3, 
                           out_features=3, 
                           hidden_features=[hidden_features_size // 2] * hidden_features_layers, 
                           activation=activation, 
                           last_activation=last_activation)
        
        # # BRDF Network
        # self.brdf = FC(in_features=channels, 
        #                out_features=hidden_features_size, 
        #                hidden_features=[hidden_features_size] * hidden_features_layers, 
        #                activation=activation, 
        #                last_activation=None)
    
    def forward(self, normal, albedo, roughness, specular, mask, in_dirs, out_dirs, hdri_samples, inference):
        
        with torch.no_grad():
            
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
            
            if not inference:
                num_all_training_pixels = normal[mask].shape[0]
                if self.num_train_sample < num_all_training_pixels:
                    num_train_sample = self.num_train_sample
                else:
                    num_train_sample = num_all_training_pixels
                
                rand_indices = torch.randperm(num_all_training_pixels, device=albedo.device)[:num_train_sample]
                
                in_dirs = in_dirs[:,:,None,None,:].repeat(1,1,normal.shape[2],normal.shape[1],1).permute(0,2,3,1,4)[mask][rand_indices]
                hdri_samples = hdri_samples[:,:,None,None,:].repeat(1,1,normal.shape[2],normal.shape[1],1).permute(0,2,3,1,4)[mask][rand_indices]
                normal = normal[mask][rand_indices].unsqueeze(1).broadcast_to(*in_dirs.shape[:-1],3)
                albedo = albedo[mask][rand_indices].unsqueeze(1).broadcast_to(*in_dirs.shape[:-1],3)
                roughness = roughness[mask][rand_indices].unsqueeze(1).broadcast_to(*in_dirs.shape[:-1],1)
                specular = specular[mask][rand_indices].unsqueeze(1).broadcast_to(*in_dirs.shape[:-1],1)
                out_dirs = out_dirs[mask][rand_indices].unsqueeze(1).broadcast_to(*in_dirs.shape[:-1],3)
            else:
                in_dirs = in_dirs[:,:,None,None,:].repeat(1,1,normal.shape[2],normal.shape[1],1).permute(0,2,3,1,4)[mask]
                hdri_samples = hdri_samples[:,:,None,None,:].repeat(1,1,normal.shape[2],normal.shape[1],1).permute(0,2,3,1,4)[mask]
                normal = normal[mask].unsqueeze(1).broadcast_to(*in_dirs.shape[:-1],3)
                albedo = albedo[mask].unsqueeze(1).broadcast_to(*in_dirs.shape[:-1],3)
                roughness = roughness[mask].unsqueeze(1).broadcast_to(*in_dirs.shape[:-1],1)
                specular = specular[mask].unsqueeze(1).broadcast_to(*in_dirs.shape[:-1],1)
                out_dirs = out_dirs[mask].unsqueeze(1).broadcast_to(*in_dirs.shape[:-1],3)
                
                rand_indices = None
            
            g_buffer = torch.cat([albedo, roughness, specular, normal], dim=-1)                                             # [S,N,8]
            
            half_dirs = in_dirs + out_dirs                                                                                  # [S,N,3]
            half_dirs = torch.nn.functional.normalize(half_dirs, dim=-1)                                                    # [S,N,1]
            n_d_i = (normal * in_dirs).sum(dim=-1, keepdim=True).clamp(min=0)                                               # [S,N,1]
            n_d_o = (normal * out_dirs).sum(dim=-1, keepdim=True).clamp(min=0)                                              # [S,N,1]
            h_d_n = (normal * half_dirs).sum(dim=-1, keepdim=True).clamp(min=0)                                             # [S,N,1]
            h_d_o = (half_dirs * out_dirs).sum(dim=-1, keepdim=True).clamp(min=0)                                           # [S,N,1]
            
            diffuse_shading_input = torch.cat([hdri_samples * n_d_i, g_buffer], -1)                                         # [S,N,11]
            
            if self.fourier_feature_transform is not None:
                diffuse_shading_input = self.fourier_feature_transform(diffuse_shading_input)
        
        diffue_feature = self.diffuse(diffuse_shading_input)
        # color = self.specular(torch.cat([diffue_feature, n_d_i, n_d_o, h_d_n, h_d_o, out_dirs], dim=-1))
        color = self.specular(torch.cat([diffue_feature, out_dirs], dim=-1))
        
        # Calculate intergal for all incident directions
        color = color.mean(dim=1)

        return color.clamp(min=0.,max=1.), rand_indices

class NeuralRenderer(nn.Module):
    def __init__(self):
        super().__init__()
        self.shader = NeuralShader()
        env_width, env_height = 64, 32
        
        # Initialze local environment map 
        # Azimuth range (-pi - pi)
        Az = ((torch.arange(env_width) + 0.5) / env_width - 0.5) * 2 * torch.pi
        
        # Elevation range (0 - 0.5 pi)
        El = ((torch.arange(env_height) + 0.5) / env_height) * torch.pi * 0.5
        
        El, Az = torch.meshgrid(El, Az, indexing='ij')
        
        Az = Az[:, :, None]
        El = El[:, :, None]
        
        # X:left; Y: up; Z: out of screen.
        lx = torch.cos(Az) * torch.cos(El)
        ly = torch.sin(El)
        lz = torch.sin(Az) * torch.cos(El)
        
        ls = torch.cat([lx, ly, lz], dim=-1).reshape(-1, 3)
        self.ls = nn.Parameter(ls, requires_grad=False)
        
        cam_pos = torch.tensor([0., 0., 0.])[None, None, None, :]
        self.cam_pos = nn.Parameter(cam_pos, requires_grad=False)
        
        self.sampler = Sampler()

    def forward(self, render_buffer, num_light_samples, inference):
        
        pos_in_cam_gt = render_buffer['pos_in_cam_gt']                                                                                 # [B,H,W]
        env_map = render_buffer['hdri_gt']                                                                                     # [B,env_h,env_w,3]
        
        # Sampling the HDRi environment map, getting sampled light and inbound direction
        # sampled_hdri_map, sampled_direction = self.sampler.uniform_sampling(hdri_map=env_map, num_samples=num_light_samples)
        
        # 1) Importance sampling
        spp = num_light_samples
        # Build CDFs
        cdf_marg, cdf_cond, weighted = build_envmap_cdf(env_map)
        
        # Draw inbound light samples
        w_i, uv = sample_envmap_direction(cdf_marg, cdf_cond, num_samples=spp)
        sampled_direction = w_i
        
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
        sampled_hdri_map = L_e
        
        # Calculate outbound direction
        in_dirs = sampled_direction                                                              # [S,N,3]
        out_dirs = self.cam_pos - pos_in_cam_gt
        out_dirs = nn.functional.normalize(out_dirs, dim=-1)                                                                    # [S,N,3]
        
        shading_input = {
            'normal': render_buffer['normal_gt'],                                      # [S,N,3]
            'albedo': render_buffer['albedo_gt'],                                      # [S,N,3]
            'roughness': render_buffer['roughness_gt'],                        # [S,N,1]
            'specular': render_buffer['specular_gt'],                          # [S,N,1]
            'in_dirs': in_dirs,                                                                                                 # [S,N,3]
            'out_dirs': out_dirs,                                                                   # [S,N,3]
            'hdri_samples': sampled_hdri_map,                                                        # [S,N,3],
            'mask': render_buffer['mask'],
            'inference': inference
        }
        
        masked_rgb_pixels, rand_indices = self.shader(**shading_input)

        return masked_rgb_pixels, rand_indices
    
    # Utility function
    def split_model_inputs(self, input, total_pixels, split_size):
        '''
        Split the input to fit Cuda memory for large resolution.
        Can decrease the value of split_num in case of cuda out of memory error.
        '''
        split_size = split_size                                                                                                            # [S]
        split_input = []
        split_indexes = torch.split(torch.arange(total_pixels).cuda(), split_size, dim=0)
        for indexes in split_indexes:
            data = {}
            data['normal'] = torch.index_select(input['normal'], 1, indexes)
            data['albedo'] = torch.index_select(input['albedo'], 1, indexes)
            data['roughness'] = torch.index_select(input['roughness'], 1, indexes)
            data['specular'] = torch.index_select(input['specular'], 1, indexes)
            data['in_dirs'] = torch.index_select(input['in_dirs'], 1, indexes)
            data['out_dirs'] = torch.index_select(input['out_dirs'], 1, indexes)
            data['hdri_samples'] = torch.index_select(input['hdri_samples'], 1, indexes)
            split_input.append(data)
            
        return split_input
    
    def save_model(self, weights_dir, reason=""):
        torch.save(self.state_dict(), weights_dir + f"/{self.__class__.__name__ + reason}.pth")
