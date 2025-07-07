import glob
import os

import albumentations as A
from albumentations.pytorch import ToTensorV2
from hydra.utils import get_original_cwd
import numpy as np
from PIL import Image
from pytorch_lightning import LightningDataModule
from skimage.io import imread
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, \
    Dataset


class FaceDataset(Dataset):

    def __init__(self, img_arr, transforms):
        '''
        Receives a numpy array of size (N,3,H,W) 
        and a composed albumentations transformer
        '''
        super().__init__()
        self.img_arr = img_arr
        self.transforms = transforms

    def __len__(self):
        return self.img_arr.shape[0]


    def __getitem__(self, idx):
        img = self.img_arr[idx]
        augmented = self.transforms(image=img)
        return augmented['image']


class FaceDataModule(LightningDataModule):

    NAME = 'faces-demo'

    CACHED_IMG_ARR = 'faces/faces'

    def __init__(self, dataset_cfg):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.batch_size = dataset_cfg.batch_size
        self.num_workers = dataset_cfg.num_workers
        self.cached_img_array_path = f'{get_original_cwd()}/{self.dataset_cfg.base_dir}/faces/faces.{self.dataset_cfg.final_image_size}.npy'

    def prepare_data(self):

        # a numpy object which has the images saved so 
        # we don't have to constantly load + crop one by one
        if not os.path.exists(self.cached_img_array_path):
            all_photos = self._compile_images()
            np.save(self.cached_img_array_path, all_photos)


    def _compile_images(self):
        photo_paths = glob.glob(f'{get_original_cwd()}/{self.dataset_cfg.base_dir}/faces/**/*.jpg', recursive=True)
        print(f'Found {len(photo_paths)} images')
        def prep_img(p, d, final_img_size):
            img = imread(p)
            img = img[d:-d, d:-d]
            return np.array(Image.fromarray(img).resize([final_img_size,final_img_size]))
        
        full_img_size = self.dataset_cfg.full_image_size
        final_img_size = self.dataset_cfg.final_image_size
        d = (full_img_size - final_img_size) // 2
        all_photos = [prep_img(x, d, final_img_size) for x in photo_paths]
        all_photos = np.stack(all_photos).astype('uint8')
        return all_photos


    def setup(self, stage):
        data = np.load(self.cached_img_array_path)
        data = data.astype(np.uint8)

        # get the image size (assuming squares) by looking at the first image and
        # grabbing the H dimension:
        self.img_size = data[0].shape[2]

        X_train, X_val = train_test_split(data, 
                                          train_size=self.dataset_cfg.train_fraction, 
                                          random_state=self.dataset_cfg.seed)
        self._setup_transforms()
        self.train_dataset = FaceDataset(X_train, self._train_transforms)
        self.val_dataset = FaceDataset(X_val, self._val_transforms)


    def _create_transforms(self, aug_cnf):
        '''
        Helper method which returns a list of augmentations based
        on the `aug_cnf` config
        '''
        # internal function to assist with creating the transforms
        def get_object(transform_spec):
            try:
                return getattr(A, transform_spec.name)(**transform_spec.params)
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
        return A.Compose(transform_list)


    def _setup_transforms(self):
        self._train_transforms = self._get_transforms_for_phase('fit')
        self._val_transforms = self._get_transforms_for_phase('validate')


    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.dataset_cfg.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True)


    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)
