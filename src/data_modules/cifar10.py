from pathlib import Path

import albumentations as alb
from albumentations.pytorch import ToTensorV2
from hydra.utils import get_original_cwd
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torch import Generator
from torchvision import transforms
from torchvision.datasets import CIFAR10


class TransformWrapper(object):
    '''
    Given that we are using a torchvision dataset, we cannot natively
    pass it the albumentations.Compose object as that requires keyword
    arguments. To that end, this wraps the alb.Compose object to permit
    usage
    https://github.com/albumentations-team/albumentations/issues/879
    '''
    def __init__(self, transforms: alb.Compose):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))


class CustomCIFAR10(CIFAR10):
    '''
    Since we use albumentations, the transformed
    images are returned in a dictionary. We use this
    class as a thin wrapper on the torchvision.datasets.CIFAR10 
    class to extract that
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        img_dict, class_id = super().__getitem__(idx)
        return img_dict['image'].float()/255.0, class_id


class CIFAR10DataModule(LightningDataModule):

    NAME = 'cifar10'

    def __init__(self, dataset_cfg):
        super().__init__()

        self.dataset_cfg = dataset_cfg
        # note that since we have hydra managing config,
        # the working directory *might* be different for
        # each run. Make the data directory outside the
        # run-specific directory so we that we are not
        # constantly downloading data
        try:
            self.data_dir = Path(get_original_cwd()) / \
                Path(dataset_cfg.base_dir) / Path(CIFAR10DataModule.NAME)
        except ValueError:
            self.data_dir = Path.cwd().parent / \
                Path(dataset_cfg.base_dir) / Path(CIFAR10DataModule.NAME)
        self.batch_size = dataset_cfg.batch_size
        self.num_workers = dataset_cfg.num_workers
        self.seed = dataset_cfg.seed
        self.train_fraction = dataset_cfg.train_fraction
        self.validation_fraction = 1 - self.train_fraction
        self._set_transforms()

    def _set_transforms(self):
        '''
        Apply transforms for training, validation, and testing
        '''
        self._train_transforms = self._get_transforms_for_phase('fit')
        self._validation_transforms = self._get_transforms_for_phase('validation')
        self._test_transforms = self._get_transforms_for_phase('test')

    def _get_transforms_for_phase(self, phase):

        # taken from https://github.com/Lightning-Universe/lightning-bolts/\
        # blob/master/pl_bolts/transforms/dataset_normalizations.py
        normalize = alb.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
        )

        key = f'{phase}_augmentations'
        if key in self.dataset_cfg:
            augmentations_cfg = self.dataset_cfg[key]
        else:
            augmentations_cfg = None
        transform_list = self._create_transforms(augmentations_cfg)
        #transform_list.append(normalize)
        transform_list.append(ToTensorV2())
        return TransformWrapper(alb.Compose(transform_list))

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

    def prepare_data(self):
        '''
        This method is used to download data and should NOT assign
        instance state per the docs. In the case of single-node training, etc.
        this doesn't matter, but is important for multi-node/distributed
        implementations
        '''
        CustomCIFAR10(self.data_dir, download=True, train=True)
        CustomCIFAR10(self.data_dir, download=True, train=False)

    def setup(self, stage: str):
        if stage == 'fit' or stage is None:
            # we first get the same datasets for train and validation, with
            # the only difference being the applied transforms.
            full_train_dataset = CustomCIFAR10(self.data_dir, train=True,
                                         transform=self._train_transforms)
            full_validation_dataset = CustomCIFAR10(self.data_dir, train=True,
                                              transform=self._validation_transforms)

            # we next split both "full" datasets to get our final train
            # and validation sets. Note that we use the same generator for
            # both so that we are not leaking training data into the
            # validation set
            split_fractions = [self.train_fraction, self.validation_fraction]
            self.train_dataset, _ = random_split(
                full_train_dataset, split_fractions,
                generator=Generator().manual_seed(self.seed))
            _, self.val_dataset = random_split(
                full_validation_dataset, split_fractions,
                generator=Generator().manual_seed(self.seed))

        if stage == 'test':
            self.test_dataset = CustomCIFAR10(self.data_dir, train=False,
                                        transform=self._test_transforms)

        if stage == 'predict':
            self.test_dataset = CustomCIFAR10(self.data_dir, train=False,
                                        transform=self._test_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
