import glob
import os
from pathlib import Path

import albumentations as alb
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import pandas as pd
from pytorch_lightning import LightningDataModule
import skimage.io
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from torch.utils.data import DataLoader, \
    Dataset


class WMTileDataset(Dataset):
    
    def __init__(self, image_metadata, stage, dataset_cfg, transforms):
        super().__init__()
        self.augmentation_transforms = transforms
        self.dataset_config = dataset_cfg
        self.image_metadata = image_metadata
        self.stage = stage
        self._collect_tiles()

    def __len__(self):
        return len(self.all_tiles)

    def __getitem__(self, idx):
        img_path = self.all_tiles[idx]
        img = skimage.io.imread(img_path)
        augmented = self.augmentation_transforms(image=img)
        return augmented['image']

    def _collect_tiles(self):
        all_tiles = []
        for image_id, row in self.image_metadata.iterrows():
            subdir = row.image_subdir
            tiles = glob.glob(f'{self.dataset_config.base_dir}/{subdir}/{image_id}*.png')
            all_tiles.extend(tiles)
        self.all_tiles = all_tiles


class WMTileDataModule(LightningDataModule):

    NAME = 'WM'

    def __init__(self, dataset_cfg):
        super().__init__()

        self.dataset_cfg = dataset_cfg

        # extract parameters related to data loading:
        self.data_dir = dataset_cfg.base_dir
        self.batch_size = dataset_cfg.batch_size
        self.num_workers = dataset_cfg.num_workers
        self.seed = dataset_cfg.seed

        # image metadata contains information at the slide level 
        # which we can use later to make plots, etc.
        self.image_metadata = pd.read_csv(dataset_cfg.image_meta_path, 
                                          index_col='image_id')

    def setup(self, stage):

        self.train_df, self.val_df = train_test_split(self.image_metadata, 
                                            train_size=self.dataset_cfg.train_fraction, 
                                            stratify=self.image_metadata['Subtype'])
        
        self._setup_transforms()

        if stage == 'fit':
            self.train_dataset = WMTileDataset(self.train_df, 
                                               stage, 
                                               self.dataset_cfg, 
                                               self._train_transforms)
        elif stage == 'validate':
            self.val_dataset = WMTileDataset(self.val_df, 
                                             stage, 
                                             self.dataset_cfg, 
                                             self._validation_transforms)
        else:
            raise NotImplementedError('!!!')

    def _create_transforms(self, aug_cnf):
        '''
        Helper method which returns a list of augmentations based
        on the `aug_cnf` config
        '''
        # internal function to assist with creating the transforms
        def get_object(transform_spec):
            try:
                return getattr(alb, transform_spec.name)(**transform_spec.params)
            except Exception as ex:
                print(f'Error: could not find the prescribed transform: {transform_spec.name}')
                raise ex

        if aug_cnf is None:
            augs = []
        else:
            augs = [get_object(aug) for aug in aug_cnf]
        return augs

    def _get_transforms_for_phase(self, phase):
        key = f'{phase}_augmentations'
        if key in self.dataset_cfg:
            augmentations_cfg = self.dataset_cfg[key]
        else:
            augmentations_cfg = None
        transform_list = self._create_transforms(augmentations_cfg)
        transform_list.append(ToTensorV2())
        return alb.Compose(transform_list)

    def _setup_transforms(self):
        self._train_transforms = self._get_transforms_for_phase('fit')
        self._validation_transforms = self._get_transforms_for_phase('validate')

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)