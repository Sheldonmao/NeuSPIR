import os
import json
import math
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import _get_rank

import datasets
from models.model_utils.ray_utils import get_ray_directions

LIGHT_PROBS=[
    'ehingen_hillside_1k',
    'lebombo_1k',
    'photo_studio_loft_hall_1k',
    'solitude_night_1k',
    'table_mountain_2_puresky_1k',
    'ulmer_muenster_1k',
    'urban_street_03_1k',
]

def create_spheric_poses(cameras, n_steps=120):
    center = torch.as_tensor([0.,0.,0.], dtype=cameras.dtype, device=cameras.device)
    mean_d = (cameras - center[None,:]).norm(p=2, dim=-1).mean()
    mean_h = cameras[:,2].mean()
    r = (mean_d**2 - mean_h**2).sqrt()
    up = torch.as_tensor([0., 0., 1.], dtype=center.dtype, device=center.device)

    all_c2w = []
    for theta in torch.linspace(0, 2 * math.pi, n_steps):
        cam_pos = torch.stack([r * theta.cos(), r * theta.sin(), mean_h])
        l = F.normalize(center - cam_pos, p=2, dim=0)
        s = F.normalize(l.cross(up), p=2, dim=0)
        u = F.normalize(s.cross(l), p=2, dim=0)
        c2w = torch.cat([torch.stack([s, u, -l], dim=1), cam_pos[:,None]], axis=1)
        all_c2w.append(c2w)

    all_c2w = torch.stack(all_c2w, dim=0)
    
    return all_c2w

class ShinySynDatasetBase():
    def setup(self, config, split):
        self.config = config
        self.split = split
        self.rank = _get_rank()
        self.scene_dir = os.path.join(self.config.root_dir,'synthesis-images',self.config.scene)

        self.use_mask = True

        with open(os.path.join(self.scene_dir, f"transforms_{self.split}.json"), 'r') as f:
            meta = json.load(f)

        if 'w' in meta and 'h' in meta:
            W, H = int(meta['w']), int(meta['h'])
        else:
            W, H = 800, 800

        w, h = self.config.img_wh
        assert round(W / w * h) == H
        
        self.w, self.h = w, h
        self.img_wh = [w,h]

        self.near, self.far = self.config.near_plane, self.config.far_plane

        self.focal = 0.5 * w / math.tan(0.5 * meta['camera_angle_x']) # scaled focal length

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(self.w, self.h, self.focal, self.focal, self.w//2, self.h//2, self.config.use_pixel_centers).to(self.rank) # (h, w, 3)           

        self.all_c2w, self.all_images, self.all_fg_masks = [], [], []

        
        if split=='train':
            for i, frame in enumerate(meta['frames']):
                c2w = torch.from_numpy(np.array(frame['transform_matrix'])[:3, :4])
                self.all_c2w.append(c2w)

                img_path = os.path.join(self.scene_dir, f"{frame['file_path']}.png")
                img = Image.open(img_path)
                img = img.resize(self.config.img_wh, Image.BICUBIC)
                img = TF.to_tensor(img).permute(1, 2, 0) # (4, h, w) => (h, w, 4)

                self.all_fg_masks.append(img[..., -1]) # (h, w)
                self.all_images.append(img[...,:3])

        elif split=='test':
            self.all_light_prob,self.all_albedo_images = [],[]
            for i, frame in enumerate(meta['frames'][17::50]): # align with NeRFactor test set
            # for i, frame in enumerate(meta['frames'][1::40]): # align with NeRFactor test set
                c2w = torch.from_numpy(np.array(frame['transform_matrix'])[:3, :4])
                for light_prob in LIGHT_PROBS:
                    self.all_c2w.append(c2w)
                    img_path = os.path.join(self.scene_dir, f"{frame['file_path']}_{light_prob}.png")
                    img = Image.open(img_path)
                    img = img.resize(self.config.img_wh, Image.BICUBIC)
                    img = TF.to_tensor(img).permute(1, 2, 0) # (4, h, w) => (h, w, 4)

                    self.all_fg_masks.append(img[..., -1]) # (h, w)
                    self.all_images.append(img[...,:3])
                    self.all_light_prob.append(os.path.join(self.config.root_dir,'light-probes','test',f"{light_prob}.hdr"))

                    ## load albedo
                    albdeo_path = os.path.join(self.scene_dir, f"{frame['file_path'][:-4]}albedo.png")
                    albedo_img = Image.open(albdeo_path)
                    albedo_img = albedo_img.resize(self.config.img_wh, Image.BICUBIC)
                    albedo_img = TF.to_tensor(albedo_img).permute(1, 2, 0) # (4, h, w) => (h, w, 4)
                    self.all_albedo_images.append(albedo_img[...,:3])
        
        elif split=='val':
            self.all_light_prob,self.all_albedo_images = [],[]
            for i, frame in enumerate(meta['frames']): # align with NeRFactor validation
                c2w = torch.from_numpy(np.array(frame['transform_matrix'])[:3, :4])
                for light_prob in [LIGHT_PROBS[0],]:
                    self.all_c2w.append(c2w)
                    img_path = os.path.join(self.scene_dir, f"{frame['file_path']}_{light_prob}.png")
                    img = Image.open(img_path)
                    img = img.resize(self.config.img_wh, Image.BICUBIC)
                    img = TF.to_tensor(img).permute(1, 2, 0) # (4, h, w) => (h, w, 4)

                    self.all_fg_masks.append(img[..., -1]) # (h, w)
                    self.all_images.append(img[...,:3])
                    self.all_light_prob.append(os.path.join(self.config.root_dir,'light-probes','test',f"{light_prob}.hdr"))

                    ## load albedo
                    albdeo_path = os.path.join(self.scene_dir, f"{frame['file_path'][:-4]}albedo.png")
                    albedo_img = Image.open(albdeo_path)
                    albedo_img = albedo_img.resize(self.config.img_wh, Image.BICUBIC)
                    albedo_img = TF.to_tensor(albedo_img).permute(1, 2, 0) # (4, h, w) => (h, w, 4)
                    self.all_albedo_images.append(albedo_img[...,:3])

        elif self.split == 'pred':
            self.all_c2w = create_spheric_poses(self.all_c2w[:,:,3], n_steps=100)
            self.all_images = torch.zeros((100, self.h, self.w, 3), dtype=torch.float32)
            self.all_fg_masks = torch.zeros((100, self.h, self.w), dtype=torch.float32)  

        self.all_c2w, self.all_images, self.all_fg_masks = \
            torch.stack(self.all_c2w, dim=0).float().to(self.rank), \
            torch.stack(self.all_images, dim=0).float().to(self.rank), \
            torch.stack(self.all_fg_masks, dim=0).float().to(self.rank)
        
        if hasattr(self, 'all_albedo_images'):
            self.all_albedo_images = torch.stack(self.all_albedo_images, dim=0).float().to(self.rank)

        # print(f'processing dataset:{split}, all_images.shape = {self.all_images.shape} ')
        
    def pixels_to_rays(self,y,x,index=None):
        return self.directions[y,x]

class ShinySynDataset(Dataset, ShinySynDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, index):
        return {
            'index': index
        }


class ShinySynIterableDataset(IterableDataset, ShinySynDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@datasets.register('shiny-syn')
class ShinySynDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = ShinySynIterableDataset(self.config, 'train')
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = ShinySynDataset(self.config, 'val')
        if stage in [None, 'test']:
            self.test_dataset = ShinySynDataset(self.config, 'test')
        if stage in [None, 'predict']:
            self.predict_dataset = ShinySynDataset(self.config, 'test')

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
