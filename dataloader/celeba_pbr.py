import os, math, json, random
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from utils.io import load_sdr, load_hdr

class CELEBAPBR(Dataset):
    def __init__(self, configs, mode):
        super().__init__()

        self.width, self.height = configs.image_size
        
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
        
        # Load pred fov
        with open(configs.fov_file_dir, 'r') as f:
            self.pred_fov_dict = json.load(f)
    
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
        mask_gt = load_sdr(os.path.join(self.mask_dir, f'{data_index}_mask.png'), resize=(self.width, self.height))
        
        # Get view pos from estimated fov
        pred_fov = self.pred_fov_dict[str(data_index)]
        
        pos_in_cam_gt = self.get_cam_pos(depth=depth_gt, width=self.width, height=self.height, fov=pred_fov)
        
        data_buffer = {
            'rgb_gt': rgb_gt.permute(2, 0, 1),
            'normal_gt': normal_gt.permute(2, 0, 1),
            'albedo_gt': albedo_gt.permute(2, 0, 1),
            'roughness_gt': roughness_gt[...,None].permute(2, 0, 1),
            'specular_gt': specular_gt[...,None].permute(2, 0, 1),
            'depth_gt': depth_gt[...,None].permute(2, 0, 1),
            'pos_in_cam_gt': pos_in_cam_gt.permute(2, 0, 1),
            'mask_gt': mask_gt.permute(2, 0, 1),
            'hdri_gt': hdri_gt.permute(2, 0, 1),
            'file_index': str(data_index)
        }
        
        return data_buffer
    
    def get_cam_pos(self, depth, width, height, fov):
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
                              num_workers=4,
                              worker_init_fn=worker_init_fn,
                              pin_memory=True)
    
    eval_dataset = CELEBAPBR(configs, mode='eval')
    eval_loader = DataLoader(eval_dataset, 
                             batch_size=configs.eval_batch_size, 
                             shuffle=True, 
                             num_workers=4,
                             worker_init_fn=worker_init_fn,
                             pin_memory=True)
    
    return train_loader, eval_loader