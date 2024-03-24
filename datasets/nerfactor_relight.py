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
import re

class NerfactorSynDatasetBase():

    def setup(self, config, split):
        self.config = config
        self.split = split
        self.rank = _get_rank()
        self.scene_dir = os.path.join(self.config.root_dir,"rendered-images",self.config.scene)

        self.use_mask = True
        
        # load meta file
        if self.split == 'predict':
            meta_file = os.path.join(self.scene_dir, f"transforms_test.json")
        else:
            meta_file = os.path.join(self.scene_dir, f"transforms_{self.split}.json")
        with open(meta_file, 'r') as f:
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

        self.all_c2w, self.all_images, self.all_fg_masks, self.all_frame_id = [], [], [], []

        
        if split=='train':
            if self.config.N_train_samples >0: # sparsely sample the input traning images
                sparsity = len(meta['frames'])//self.config.N_train_samples
            else: 
                sparsity = 1
            print('using sparsity:',sparsity)
            for i, frame in enumerate(meta['frames'][::sparsity]):
                c2w = torch.from_numpy(np.array(frame['transform_matrix'])[:3, :4])
                self.all_c2w.append(c2w)

                img_path = os.path.join(self.scene_dir, f"{frame['file_path']}.png")
                img = Image.open(img_path)
                img = img.resize(self.config.img_wh, Image.BICUBIC)
                img = TF.to_tensor(img).permute(1, 2, 0) # (4, h, w) => (h, w, 4)

                self.all_fg_masks.append(img[..., -1]) # (h, w)
                self.all_images.append(img[...,:3])
                
                frame_id = int(re.search(r"\d+", f"{frame['file_path']}.png").group())
                self.all_frame_id.append(frame_id) # save like 0

        else:
            light_probs = ['city','courtyard','forest','interior','night','studio','sunrise','sunset']
            self.all_light_prob,self.all_albedo_images =[],[]

            if split=='test':
                frame_list = meta['frames'][49::50]
            elif split == 'val':
                frame_list = meta['frames']
            elif split == 'predict':
                frame_list = meta['frames'][::2]
                light_probs = ['sunset']
            
            
            for i, frame in enumerate(frame_list): # align with NeRFactor test set
                c2w = torch.from_numpy(np.array(frame['transform_matrix'])[:3, :4])
                for light_prob in light_probs:
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

                    frame_id = int(re.search(r"\d+", f"{frame['file_path']}.png").group())
                    self.all_frame_id.append(frame_id) # save like 0

        self.all_c2w, self.all_images, self.all_fg_masks,self.all_frame_id = \
            torch.stack(self.all_c2w, dim=0).float().to(self.rank), \
            torch.stack(self.all_images, dim=0).float().to(self.rank), \
            torch.stack(self.all_fg_masks, dim=0).float().to(self.rank), \
            torch.tensor(self.all_frame_id).float() \
        
        if hasattr(self, 'all_albedo_images'):
            self.all_albedo_images = torch.stack(self.all_albedo_images, dim=0).float().to(self.rank)

        # print(f'processing dataset:{split}, all_images.shape = {self.all_images.shape} ')
        
    def pixels_to_rays(self,y,x,index=None):
        return self.directions[y,x]

class NerfactorSynDataset(Dataset, NerfactorSynDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, index):
        return {
            'index': index
        }


class NerfactorSynIterableDataset(IterableDataset, NerfactorSynDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@datasets.register('nerfactor-syn')
class NerfactorSynDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = NerfactorSynIterableDataset(self.config, 'train')
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = NerfactorSynDataset(self.config, 'val')
        if stage in [None, 'test']:
            self.test_dataset = NerfactorSynDataset(self.config, 'test')
        if stage in [None, 'predict']:
            self.predict_dataset = NerfactorSynDataset(self.config, 'predict')

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
