import os
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import _get_rank

import datasets
from typing import List, Optional, Union, Tuple
from models.model_utils.ray_utils import get_ray_directions


def create_spheric_poses(cam_h=1.0, cam_r=2.0, n_steps=120):
    """
    inputs:
        cam_h: camera height
        cam_r: radius of spheric trace
        n_steps: number of steps for a full circle
    """
    center = torch.as_tensor([0.,0.,0.])
    up = torch.as_tensor([0., 0., 1.])

    all_c2w = []
    for theta in torch.linspace(0, 2 * math.pi, n_steps):
        cam_pos = torch.stack([cam_r * theta.cos(), cam_r * theta.sin(), torch.tensor(cam_h)])
        l = F.normalize(center - cam_pos, p=2, dim=0)
        s = F.normalize(l.cross(up), p=2, dim=0)
        u = F.normalize(s.cross(l), p=2, dim=0)
        c2w = torch.cat([torch.stack([s, u, -l], dim=1), cam_pos[:,None]], axis=1)
        all_c2w.append(c2w)

    all_c2w = torch.stack(all_c2w, dim=0)
    
    return all_c2w

class SphericDatasetBase():

    def setup(self, config,split='pred'):
        """ create basic sphreic dataset with round camera poses
        Args:
            config (args): config with attributes img_wh, focal_length, cam_h, cam_r,n_steps
            split (str): only 'pred' is supported.
        """
        self.img_wh = config.img_wh
        self.f = config.focal_length
        self.config = config
        self.w, self.h = self.img_wh
        self.rank = _get_rank()
        
        
        self.directions = get_ray_directions(self.w, self.h, self.f, self.f, self.w/2, self.h/2)
        self.all_c2w = create_spheric_poses(config.cam_h, config.cam_r , n_steps=config.n_steps)
        self.all_images = torch.zeros((config.n_steps, self.h, self.w, 3), dtype=torch.float32)
        self.all_fg_masks = torch.ones((config.n_steps, self.h, self.w), dtype=torch.float32)

        self.directions, self.all_c2w, self.all_images, self.all_fg_masks = \
            self.directions.float().to(self.rank), \
            self.all_c2w.float().to(self.rank), \
            self.all_images.float().to(self.rank), \
            self.all_fg_masks.float().to(self.rank)
        
        print(f'Done loading {len(self.all_images)} images with res:{self.img_wh}')

    def pixels_to_rays(self,y,x,index=None):
        return self.directions[y,x]

class SphericDataset(Dataset, SphericDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, index):
        return {
            'index': index
        }


class SphericIterableDataset(IterableDataset, SphericDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@datasets.register('spheric')
class SphericDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def setup(self, stage=None):
        if stage in [None, 'predict']:
            self.predict_dataset = SphericDataset(self.config,'pred')            
        else:
            raise NotImplementedError(f'Sphere Dataset not implemented for stage:{stage}')

    def prepare_data(self):
        pass
    
    def general_loader(self, dataset, batch_size):
        sampler = None
        return DataLoader(
            dataset, 
            num_workers=os.cpu_count(), 
            batch_size=batch_size,
            pin_memory=True,
            sampler=sampler
        )
    
    def train_dataloader(self):
        return self.general_loader(self.train_dataset, batch_size=1)

    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1) 

    def predict_dataloader(self):
        return self.general_loader(self.predict_dataset, batch_size=1)       
