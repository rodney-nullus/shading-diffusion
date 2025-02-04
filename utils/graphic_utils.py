import math
import torch
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def get_world2view(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def get_world2view2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def get_view2world(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = C2W
    return np.float32(Rt)

def get_projection_matrix(znear, zfar, fovx, fovy):
    
    # Convert fov to radians
    fovx = fovx * torch.pi / 180.0
    fovy = fovy * torch.pi / 180.0
    
    tan_half_fovx = torch.tan((fovx / 2))
    tan_half_fovy = torch.tan((fovy / 2))

    top = tan_half_fovx * znear
    bottom = -top
    right = tan_half_fovy * znear
    left = -right
    
    P = torch.zeros(fovx.shape[0], 4, 4).to(fovx.device)
    P[:, 0, 0] = 2.0 * znear / (right - left)
    P[:, 0, 2] = (right + left) / (right - left)
    P[:, 1, 1] = 2.0 * znear / (top - bottom)
    P[:, 1, 2] = (top + bottom) / (top - bottom)
    P[:, 2, 2] = -1.0 * (zfar + znear) / (zfar - znear)
    P[:, 2, 3] = -2.0 * zfar * znear / (zfar - znear)
    P[:, 3, 2] = -1.0
    return P.unsqueeze(1)

def fov2focal(fov, pixels):
    return pixels / (2 * torch.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*torch.atan(pixels/(2*focal))

def coords2homo(coords):
    return torch.cat([coords, torch.ones_like(coords[..., :1])], dim=-1)
