import os, math, json, random
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from util_io import load_sdr

class TEST(Dataset):
    def __init__(self):
        super().__init__()

        self.width, self.height = 256, 256
        
        self.data_list = self._get_meta_data_list("data")

        # Load prompt data
        with open("data_prompt_en.json", "r") as f:
            self.prompt_dict = json.load(f)
    
    def _get_meta_data_list(self, data_dir):
        
        rgb_file_list = sorted(os.listdir(data_dir))
        
        gt_indices_list = []
        for rgb_file_dir in rgb_file_list:
            gt_indices_list.append(rgb_file_dir[:5])
        
        return gt_indices_list
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        
        data_index = self.data_list[index]
        
        # Load texture data and do the preprocess (for the convenience of following operation, we set background of all data to 1.)
        rgb = load_sdr(f'data/{data_index}_masked_rgb.png', resize=(self.width, self.height))
        
        # Load prompts
        prompt_gt = self.prompt_dict[str(data_index)]
        
        data_buffer = {
            "rgb": rgb,
            "prompt": prompt_gt,
            "file_index": str(data_index)
        }
        
        return data_buffer
