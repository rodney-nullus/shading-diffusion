import os, math, json, random
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from utils.io import load_sdr, load_hdr
from utils.graphic_utils import *

class CELEBAPBR(Dataset):
    def __init__(self, configs, mode):
        super().__init__()

        self.configs = configs
        self.width = configs.image_width
        self.height = configs.image_height
        
        self.data_dir = configs.data_dir
        
        self.rgb_dir = os.path.join(self.data_dir, 'masked_rgb')
        self.albedo_dir = os.path.join(self.data_dir, 'albedo')
        self.roughness_dir = os.path.join(self.data_dir, 'roughness')
        self.specular_dir = os.path.join(self.data_dir, 'specular')
        self.depth_dir = os.path.join(self.data_dir, 'depth')
        self.normal_dir = os.path.join(self.data_dir, 'normal')
        self.mask_dir = os.path.join(self.data_dir, 'mask')
        self.hdri_dir = os.path.join(self.data_dir, 'hdri')
        
        self.gt_indices_list = self._get_meta_data_list()
        
        dataset_length = 20000
        train_num = round(dataset_length * (1 - 0.2))
        
        if mode == 'train':
            self.data_list = self.gt_indices_list[:train_num]
        elif mode == 'eval':
            self.data_list = self.gt_indices_list[train_num:]
        
        # Load predicted fov
        with open(configs.data_dir + '/pred_fov.json', 'r') as f:
            self.pred_fov_dict = json.load(f)
        
        # Load view pos
        with open(configs.data_dir + '/CelebAMask-HQ-pose-anno.txt', 'r') as f:
            pose_list = f.readlines()
        
        # Phase view pos
        pose_list.pop(0)
        pose_list.pop(0)
        self.pose_dict = {}
        for item in pose_list:
            item = item.replace('\n', '')
            self.pose_dict[item.split(' ')[0].split('.')[0].zfill(5)] = item.split(' ')[1:]
    
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
        
        rgb_gt = load_sdr(os.path.join(self.rgb_dir, f'{data_index}_masked_rgb.png'), resize=(self.width, self.height))
        rgb_gt = (rgb_gt * 2 - 1.).float()
        albedo_gt = load_sdr(os.path.join(self.albedo_dir, f'{data_index}_albedo.png'), resize=(self.width, self.height))
        albedo_gt = (albedo_gt * 2 - 1.).float()
        roughness_gt = load_sdr(os.path.join(self.roughness_dir, f'{data_index}_roughness.png'), resize=(self.width, self.height))
        roughness_gt = (roughness_gt * 2 - 1.).float()
        specular_gt = load_sdr(os.path.join(self.specular_dir, f'{data_index}_specular.png'), resize=(self.width, self.height))
        specular_gt = (specular_gt * 2 - 1.).float()
        normal_gt = load_sdr(os.path.join(self.normal_dir, f'{data_index}_normal.png'), resize=(self.width, self.height))
        normal_gt = (normal_gt * 2 - 1.).float()
        depth_gt = load_hdr(os.path.join(self.depth_dir, f'{data_index}_depth.exr'), resize=(self.width, self.height))[...,0]
        hdri_gt = load_hdr(os.path.join(self.hdri_dir, f'{data_index}_hdri.exr'), resize=False)
        mask_gt = load_sdr(os.path.join(self.mask_dir, f'{data_index}_mask.png'), resize=(self.width, self.height))[...,0].bool()
        
        # Get view coordinates from estimated fov
        fov_gt = self.pred_fov_dict[str(data_index)]
        v_coords_gt = self.get_view_coords(depth_gt, self.width, self.height, fov_gt)
        
        # Get clip coordinates
        # P = get_projection_matrix(self.configs.z_near, self.configs.z_far, fov_gt, fov_gt)
        # view_coords_homo = coords2homo(view_coords_gt)
        # clip_coords = torch.matmul(P, view_coords_homo[...,None]).squeeze(-1)
        
        # Get yaw, pitch, roll angles
        pose_gt = torch.tensor([float(item) for item in self.pose_dict[data_index]])
        
        # Set white background for each data
        inverted_mask = ~mask_gt
        masked_bg = inverted_mask.float().unsqueeze(-1)
        rgb_gt = rgb_gt + masked_bg
        albedo_gt = albedo_gt + masked_bg
        roughness_gt = roughness_gt.unsqueeze(-1) + masked_bg
        specular_gt = specular_gt.unsqueeze(-1) + masked_bg
        normal_gt = normal_gt + masked_bg
        depth_gt = depth_gt.unsqueeze(-1) + masked_bg
        v_coords_gt = v_coords_gt + masked_bg
        
        # data shape: [H, W, C]
        # data_buffer = {
        #     'rgb': rgb_gt.reshape(-1, 3),
        #     'normal': normal_gt.reshape(-1, 3),
        #     'albedo': albedo_gt.reshape(-1, 3),
        #     'roughness': roughness_gt.reshape(-1, 1),
        #     'specular': specular_gt.reshape(-1, 1),
        #     'depth': depth_gt.reshape(-1, 1),
        #     'v_coords': v_coords_gt.reshape(-1, 3),
        #     'mask': mask_gt.reshape(-1, 1).int(),
        #     'hdri': hdri_gt,
        #     'pose': pose_gt,
        #     #'fov': fov_gt,
        #     'file_index': str(data_index)
        # }
        
        data_buffer = {
            'rgb': rgb_gt,
            'normal': normal_gt,
            'albedo': albedo_gt,
            'roughness': roughness_gt,
            'specular': specular_gt,
            'depth': depth_gt,
            'v_coords': v_coords_gt,
            'mask': mask_gt.int(),
            'hdri': hdri_gt,
            'pose': pose_gt,
            #'fov': fov_gt,
            'file_index': str(data_index)
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
        Y, X = torch.meshgrid(Y, X, indexing='ij')
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
    
    train_dataset = CELEBAPBR(configs, mode='train')
    train_loader = DataLoader(train_dataset, 
                              batch_size=configs.train_batch_size, 
                              shuffle=True, 
                              num_workers=0,
                              worker_init_fn=worker_init_fn,
                              pin_memory=True)
    
    eval_dataset = CELEBAPBR(configs, mode='eval')
    eval_loader = DataLoader(eval_dataset, 
                             batch_size=configs.eval_batch_size, 
                             shuffle=True, 
                             num_workers=0,
                             worker_init_fn=worker_init_fn,
                             pin_memory=True)
    
    return train_loader, eval_loader
