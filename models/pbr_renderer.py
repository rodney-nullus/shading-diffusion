import math
import torch
import torch.nn as nn

from utils.sampler import Sampler
from .sampler import build_envmap_cdf, sample_envmap_direction


# ---------- Neural Networks ----------
class BRDFNet(nn.Module):
    """
    Neural decoder g(z(x), n, v, l) predicting full BRDF value f(n,v,l).
    Input: surface normal n, view dir v, light dir l (all 3D vectors)
    Output: RGB BRDF value
    """
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),  # RGB reflection
            nn.ReLU()
        )
    def forward(self, mat, h):
        # mat: g-buffer material
        # h: half vector
        x = torch.cat([mat, h], dim=-1)
        return self.fc(x)

class ISNet(nn.Module):
    """
    Predicts analytic proxy sampling parameters for mixture PDF:
      p_env, w_d, mu_dx, mu_dy, alpha_x, alpha_y, rho, mu_sx, mu_sy
    Input: surface normal and view direction
    Output: dict of sampling parameters
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 9)
        )
    def forward(self, mat, view_dirs):
        x = torch.cat([mat, view_dirs], dim=-1)
        out = self.fc(x)
        return {
            'p_env': torch.sigmoid(out[...,0]),
            'w_d':     torch.sigmoid(out[...,1]),
            'mu_dx':    out[...,2],
            'mu_dy':    out[...,3],
            'alpha_x': torch.clamp(out[...,4], 0.01, 1.0),
            'alpha_y': torch.clamp(out[...,5], 0.01, 1.0),
            'rho':     torch.tanh(out[...,6]) * 0.99,
            'mu_sx':    out[...,7],
            'mu_sy':    out[...,8]
        }

class PBRRenderer(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.brdf_net = BRDFNet()
        self.is_net = ISNet()
        
        cam_pos = torch.tensor([0., 0., 0.])[None, None, None, :]
        self.cam_pos = nn.Parameter(cam_pos, requires_grad=False)

    def forward(self, render_buffer, spp):
        
        pos_map = render_buffer['pos_in_cam_gt']                                                   # [B,H,W]
        env_map = render_buffer['hdri_gt']                                                         # [B,env_h,env_w,3]
        
        B, H, W, _ = env_map.shape
        
        # Build env_map CDFs
        cdf_marg, cdf_cond, weight = build_envmap_cdf(env_map)
        total_weight = weight.sum()
        
        # Calculate view direction
        view_map = self.cam_pos - pos_map
        view_map = nn.functional.normalize(view_map, dim=-1)                                       # [S,N,3]
        
        # Rendering loop
        _,_,H,W = pos_map.shape
        out = torch.zeros((H, W, 3), device=pos_map.device)
        for y in range(H):
            for x in range(W):
                n = render_buffer["normal"][:,:,y,x]
                v = view_map[:,:,y,x]
                params = self.is_net(n, v)
                p_env = params['p_env']
                L_o = torch.zeros(3).to(v.device)
                for _ in range(spp):
                    if torch.rand(1).item() < p_env:
                        dirs, uv = sample_envmap_direction(cdf_marg, cdf_cond, num_samples=1)
                        l = dirs[0,0]
                        # true pdf_env from precomputed weight
                        row = min(int(uv[0,0,1]*H), H-1)
                        col = min(int(uv[0,0,0]*W), W-1)
                        pdf_env = weight[0,row,col] / total_weight
                        pdf_bsdf = 0.0
                    else:
                        u = torch.rand(3, device=pos_map.device)
                        l, pdf_bsdf = self.sample_analytic(u, v, params)
                        pdf_env = 0.0
                    cos_theta = l.dot(n).clamp(min=0.0)
                    if cos_theta > 0:
                        f = self.brdf_net(n.unsqueeze(0), v.unsqueeze(0), l.unsqueeze(0)).squeeze(0)
                        if pdf_env>0:
                            U = col; V = row
                            Li = env_map[V,U]
                        else:
                            Li = self.env_map_eval(env_map, l)
                        pdf_mix = p_env*pdf_env + (1-p_env)*pdf_bsdf + 1e-12
                        L_o += f * Li * cos_theta / pdf_mix

                out[y,x] = L_o / spp
        
        # Fetch radiance from envmap using uv coordinates
        # uv: [N, 2], values in [0, 1]
        B, ENV_H, ENV_W = env_map.shape[:3]
        device = env_map.device
        u_idx = (uv[..., 0] * (ENV_W - 1)).long().clamp(0, ENV_W - 1)
        v_idx = (uv[..., 1] * (ENV_H - 1)).long().clamp(0, ENV_H - 1)
        
        # 构造 batch 维度的索引
        batch_idx = torch.arange(B, device=env_map.device)    # [B]
        batch_idx = batch_idx.view(B, 1).expand(-1, u_idx.size(1))  # [B, N]
        
        # 直接用高级索引，最后会保留 C 通道维度：
        # env_map[batch_idx, v_idx, u_idx] → [B, N, C]
        L_e = env_map[batch_idx, v_idx, u_idx]  # [B, N, 3]
        sampled_hdri_map = L_e

        return masked_rgb_pixels, rand_indices
    
    # ---------- BSDF Sampling ----------
    def sample_cosine_tilt(self, u, n_d):
        r = torch.sqrt(u[0]); phi = 2*math.pi*u[1]
        x = r*math.cos(phi); y = r*math.sin(phi)
        z = torch.sqrt(torch.clamp(1-x*x-y*y,0))
        up = torch.tensor([0,0,1], device=u.device)
        t = torch.cross(up,n_d); t = t/t.norm() if t.norm()>1e-6 else torch.tensor([1,0,0],device=u.device)
        b = torch.cross(n_d, t)
        return x*t + y*b + z*n_d

    def sample_ggx_isotropic(self, u):
        cos_t = torch.sqrt(1-u[0]); sin_t = torch.sqrt(torch.clamp(1-cos_t*cos_t,0))
        phi = 2*math.pi*u[1]
        return torch.tensor([sin_t*math.cos(phi), sin_t*math.sin(phi), cos_t], device=u.device)

    def sample_analytic(self, u, wo_i, params):
        w_d = params['w_d']
        mu_dx = params['mu_dx']; mu_dy = params['mu_dy']
        n_d = torch.tensor([-mu_dx, -mu_dy, 1.0], device=u.device); n_d = n_d/n_d.norm()
        ax = params['alpha_x']; ay = params['alpha_y']; rho = params['rho']
        msx = params['mu_sx']; msy = params['mu_sy']
        if u[0] < w_d:
            wo = self.sample_cosine_tilt(u[1:], n_d)
            pdf = w_d * (wo.dot(n_d).clamp(min=0.0)/math.pi)
        else:
            h0 = self.sample_ggx_isotropic(u[1:])
            M = torch.tensor([[ax,0,-msx],[rho*ay/math.sqrt(1-rho*rho), ay, -msy],[0,0,1]], device=u.device)
            h = M @ h0; h = h / h.norm()
            wo = 2*(wo_i.dot(h))*h - wo_i
            Minv = torch.inverse(M); ht = Minv @ h; det = torch.det(Minv)
            D_std = 1.0/math.pi
            pdf_s = D_std * det / (ht.norm()**3 * 4*torch.abs(wo.dot(h)) + 1e-12)
            pdf = (1.0 - w_d) * pdf_s
        return wo, pdf
