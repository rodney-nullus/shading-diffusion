import torch

def build_envmap_cdf(envmap):
    """
    Build a 2D CDF for importance sampling an HDR environment map.

    Args:
        envmap (torch.Tensor): [H, W, 3] HDR image in linear RGB.

    Returns:
        marginal_cdf (torch.Tensor): [H+1] row cumulative distribution.
        conditional_cdf (torch.Tensor): [H, W+1] per-row cumulative distributions.
    """
    B, H, W, _ = envmap.shape

    # 1. Compute per-pixel luminance (Rec. 709)
    lum = 0.2126 * envmap[...,0] + 0.7152 * envmap[...,1] + 0.0722 * envmap[...,2]  # [H,W]

    # 2. Weight by sin(theta) for solid angle
    i = torch.arange(H, dtype=envmap.dtype, device=envmap.device)
    theta = (i + 0.5) * torch.pi / H
    sin_theta = torch.sin(theta)[:, None]   # [H,1]
    weighted = lum * sin_theta              # [H,W]

    # 3. Build conditional CDFs (per row)
    pdf_rows = weighted + 1e-8                  # avoid zeros
    cdf_cond = torch.cumsum(pdf_rows, dim=2)    # [B,H,W]
    row_sums = cdf_cond[:,:,-1:]                # [B,H,1]
    cdf_cond = torch.cat([
        torch.zeros((B,H,1), dtype=envmap.dtype, device=envmap.device), 
        cdf_cond / row_sums
    ], dim=2)                                   # [B,H,W+1]

    # 4. Build marginal CDF over rows
    pdf_marginal = row_sums.squeeze(2)          # [B,H]
    cdf_marg = torch.cumsum(pdf_marginal, dim=1)
    cdf_marg = torch.cat([
        torch.zeros((B,1), dtype=envmap.dtype, device=envmap.device), 
        cdf_marg / cdf_marg[:,-1].unsqueeze(-1)
    ], dim=1)                                   # [B,H+1]

    return cdf_marg, cdf_cond

def sample_envmap_direction(cdf_marg, cdf_cond, num_samples=1):
    """
    Sample spherical directions from the 2D CDF.

    Args:
        cdf_marg (torch.Tensor): [H+1] marginal CDF over rows.
        cdf_cond (torch.Tensor): [H, W+1] conditional CDFs over columns.
        num_samples (int): number of samples to draw.

    Returns:
        dirs (torch.Tensor): [num_samples, 3] sampled unit vectors.
        sample_uv (torch.Tensor): [num_samples, 2] (u,v) lat-long coordinates in [0,1].
    """
    B = cdf_marg.shape[0]
    H = cdf_marg.shape[1] - 1
    W = cdf_cond.shape[2] - 1

    # 1. Sample row indices via inverse transform
    u_m = torch.rand((B,num_samples), device=cdf_marg.device)  # uniform [0,1)
    rows = torch.searchsorted(cdf_marg, u_m, right=False).clamp(min=0,max=H) - 1  # [num_samples]
    cdf_m0 = cdf_marg.gather(1, rows)
    cdf_m1 = cdf_marg.gather(1, rows + 1)
    
    t_m = (u_m - cdf_m0) / (cdf_m1 - cdf_m0 + 1e-10)

    # 2. Sample column indices per selected row
    batch_idx = torch.arange(B, device=cdf_marg.device).unsqueeze(1)    # [B,1]
    cdf_rows = cdf_cond[batch_idx, rows]                                # [B, N, W+1]
    u_c = torch.rand((B,num_samples), device=cdf_marg.device)
    # Vectorized searchsorted for each row
    cols = torch.searchsorted(cdf_rows, u_c.unsqueeze(-1), right=False).clamp(min=0,max=W) - 1
    cols = cols.squeeze(-1)
    
    cdf_c0 = torch.gather(cdf_rows, 2, cols.unsqueeze(-1)).squeeze(-1)          # [B, N]
    cdf_c1 = torch.gather(cdf_rows, 2, (cols + 1).unsqueeze(-1)).squeeze(-1)    # [B, N]
    t_c   = (u_c - cdf_c0) / (cdf_c1 - cdf_c0 + 1e-10)                          # [B, N]

    # 3. Compute continuous (u,v) in [0,1]
    v = (rows.float() + t_m) / H
    u = (cols.float() + t_c) / W

    # 4. Convert (u,v) to spherical direction
    phi = u * 2 * torch.pi
    theta = v * torch.pi
    sin_theta = torch.sin(theta)
    dirs = torch.stack([
        sin_theta * torch.cos(phi),
        sin_theta * torch.sin(phi),
        torch.cos(theta)
    ], dim=-1)  # [num_samples,3]

    sample_uv = torch.stack([u, v], dim=-1)
    return dirs, sample_uv
