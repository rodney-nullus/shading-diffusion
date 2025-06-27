import os, math, json, random
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from utils.io import load_sdr, load_hdr
from utils.graphic_utils import *
from utils.rotations import matrix_to_rotation_6d

class CELEBAPBR(Dataset):
    def __init__(self, configs, mode):
        super().__init__()

        self.configs = configs
        
        if isinstance(configs.resolution, set):
            self.width, self.height = configs.resolution
        else:
            self.width, self.height = configs.resolution, configs.resolution
        
        self.data_dir = configs.data_dir
        
        self.rgb_dir = os.path.join(self.data_dir, "masked_rgb")
        self.albedo_dir = os.path.join(self.data_dir, "albedo")
        self.roughness_dir = os.path.join(self.data_dir, "roughness")
        self.specular_dir = os.path.join(self.data_dir, "specular")
        self.depth_dir = os.path.join(self.data_dir, "depth")
        self.normal_dir = os.path.join(self.data_dir, "normal")
        self.mask_dir = os.path.join(self.data_dir, "mask")
        self.hdri_dir = os.path.join(self.data_dir, "hdri")
        
        self.gt_indices_list = self._get_meta_data_list()
        
        dataset_length = 20000
        train_num = round(dataset_length * (1 - 0.1))
        
        if mode == 'train':
            self.data_list = self.gt_indices_list[:train_num]
        elif mode == 'eval':
            self.data_list = self.gt_indices_list[train_num:]
        elif mode == 'all':
            self.data_list = self.gt_indices_list
        
        # Load predicted fov
        with open(configs.data_dir + "/pred_fov.json", "r") as f:
            self.pred_fov_dict = json.load(f)
        
        # Load view pos
        with open(configs.data_dir + "/CelebAMask-HQ-pose-anno.txt", "r") as f:
            pose_list = f.readlines()
        
        # Phase view pos
        pose_list.pop(0)
        pose_list.pop(0)
        self.pose_dict = {}
        for item in pose_list:
            item = item.replace("\n", "")
            self.pose_dict[item.split(" ")[0].split(".")[0].zfill(5)] = item.split(" ")[1:]

        # Load prompt data
        with open(configs.data_dir + "/data_prompt_en.json", "r") as f:
            self.prompt_dict = json.load(f)
    
    def _get_meta_data_list(self):
        
        rgb_file_list = sorted(os.listdir(self.rgb_dir))
        
        gt_indices_list = []
        for rgb_file_dir in rgb_file_list:
            gt_indices_list.append(rgb_file_dir[:5])
        
        return gt_indices_list
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        
        data_index = self.data_list[index]
        
        # Load texture data and do the preprocess (for the convenience of following operation, we set background of all data to 1.)
        anno_mask = load_sdr(os.path.join(self.mask_dir, f'{data_index}_mask.png'), resize=(self.width, self.height))[...,0].unsqueeze(-1)
        tex_mask_bg = (~(anno_mask.bool())).float()
        
        rgb = load_sdr(os.path.join(self.rgb_dir, f'{data_index}_masked_rgb.png'), resize=(self.width, self.height))
        rgb = rgb * anno_mask + tex_mask_bg
        
        albedo = load_sdr(os.path.join(self.albedo_dir, f'{data_index}_albedo.png'), resize=(self.width, self.height))
        albedo = albedo * anno_mask + tex_mask_bg
        
        roughness = load_sdr(os.path.join(self.roughness_dir, f'{data_index}_roughness.png'), resize=(self.width, self.height))
        roughness = roughness.unsqueeze(-1) * anno_mask + tex_mask_bg
        
        specular = load_sdr(os.path.join(self.specular_dir, f'{data_index}_specular.png'), resize=(self.width, self.height))
        specular = specular.unsqueeze(-1) * anno_mask + tex_mask_bg
        
        hdri = load_hdr(os.path.join(self.hdri_dir, f'{data_index}_hdri.exr'), resize=False)
        
        # Load geometry data
        depth = load_hdr(os.path.join(self.depth_dir, f'{data_index}_depth.exr'), resize=(self.width, self.height))[...,0]
        normal = load_sdr(os.path.join(self.normal_dir, f'{data_index}_normal.png'), resize=(self.width, self.height))
        normal = normal * anno_mask + tex_mask_bg
        
        # Get view coordinates from estimated fov
        fov = self.pred_fov_dict[str(data_index)]
        v_coords = self.get_view_coords(depth, self.width, self.height, fov)
        # View coordinates normalization
        v_coords[...,:2] = v_coords[...,:2] / 80.
        v_coords[...,2] = - v_coords[...,2] / 800.
        depth_mask = (~(depth == 0)).float().unsqueeze(-1)
        geo_mask_bg = (depth == 0).float().unsqueeze(-1)
        depth = depth.unsqueeze(-1) * depth_mask + geo_mask_bg
        v_coords = v_coords * depth_mask + geo_mask_bg
        
        # Load prompts
        prompt_gt = self.prompt_dict[str(data_index)]
        
        # Get clip coordinates
        # P = get_projection_matrix(self.configs.z_near, self.configs.z_far, fov_gt, fov_gt)
        # view_coords_homo = coords2homo(view_coords_gt)
        # clip_coords = torch.matmul(P, view_coords_homo[...,None]).squeeze(-1)
        
        # Get yaw, pitch, roll angles
        pose_gt = torch.tensor([float(item) for item in self.pose_dict[data_index]])
        
        # Compute rotation 6D
        yaw, pitch, roll = pose_gt
        # Convert degree to radian
        yaw = yaw / 180 * torch.pi
        pitch = pitch / 180 * torch.pi
        roll = roll / 180 * torch.pi
        rotation_matrix = torch.zeros(3, 3)
        rotation_matrix[0, 0] = torch.tensor(np.cos(yaw) * np.cos(pitch))
        rotation_matrix[0, 1] = torch.tensor(np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.sin(yaw) * np.cos(roll))
        rotation_matrix[0, 2] = torch.tensor(np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll))
        rotation_matrix[1, 0] = torch.tensor(np.sin(yaw) * np.cos(pitch))
        rotation_matrix[1, 1] = torch.tensor(np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(roll))
        rotation_matrix[1, 2] = torch.tensor(np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll))
        rotation_matrix[2, 0] = torch.tensor(-np.sin(pitch))
        rotation_matrix[2, 1] = torch.tensor(np.cos(pitch) * np.sin(roll))
        rotation_matrix[2, 2] = torch.tensor(np.cos(pitch) * np.cos(roll))
        
        rotation_6D = matrix_to_rotation_6d(rotation_matrix)
        
        data_buffer = {
            "rgb": rgb,
            "normal": normal,
            "albedo": albedo,
            "roughness": roughness,
            "specular": specular,
            "depth": depth,
            "v_coords": v_coords,
            "mask": anno_mask,
            "hdri": hdri,
            "rotation": rotation_6D,
            "prompt": prompt_gt,
            "file_index": str(data_index)
        }
        
        return data_buffer
    
    def get_view_coords(self, depth, width, height, fov):
        fovx = math.radians(fov)
        fovy = 2 * math.atan(math.tan(fovx / 2) / (width / height))
        vpos = torch.zeros(height, width, 3)
        Y = 1 - (torch.arange(height) + 0.5) / height
        Y = Y * 2 - 1
        X = (torch.arange(width) + 0.5) / width
        X = X * 2 - 1
        Y, X = torch.meshgrid(Y, X, indexing="ij")
        vpos[..., 0] = depth * X * math.tan(fovx / 2)
        vpos[..., 1] = depth * Y * math.tan(fovy / 2)
        vpos[..., 2] = -depth
        return vpos

def get_dataloader(configs):
    
    # Fix random seed
    random_seed = configs.random_seed
    
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    def worker_init_fn(worker_id):
        np.random.seed(random_seed + worker_id)
    
    train_dataset = CELEBAPBR(configs, mode="train")
    train_loader = DataLoader(train_dataset, 
                              batch_size=configs.train_batch_size, 
                              shuffle=True, 
                              num_workers=configs.num_workers,
                              worker_init_fn=worker_init_fn,
                              pin_memory=True)
    
    eval_dataset = CELEBAPBR(configs, mode="eval")
    eval_loader = DataLoader(eval_dataset, 
                             batch_size=configs.eval_batch_size, 
                             shuffle=True, 
                             num_workers=configs.num_workers,
                             worker_init_fn=worker_init_fn,
                             pin_memory=True)
    
    return train_loader, eval_loader
