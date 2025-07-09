# wm_autoencoder
PyTorch based autoencoder for learning latent representations of histopathology images for Waldenstrom's Macroglobulinemia. 

We make use of the Lightning project (https://lightning.ai/docs/pytorch/stable/) to wrap PyTorch and reduce some of the boilerplate.

For configuration, we use the Hydra project (https://hydra.cc/)

## Setup your environment

- Use the conda environment files in `conda/` to create Conda environments with the correct packages. Note that there are CPU and GPU versions depending on your hardware. For example, to create the GPU-compatible version (which works on Harvard's FASRC cluster):

```
mamba env create -f conda/env.gpu.yaml
```

## To run

#### A small demo dataset- faces in the wild
While the primary objective is to model the latent space of the histopathology images, a simple example can be constructed using the "labeled faces in the wild" dataset available here: https://www.kaggle.com/datasets/jessicali9530/lfw-dataset

Create a folder named `datasets` (in *this* directory- the root of the project). This name is intentional and is reflected in the default project configuration (see `conf/config.yaml`). You can certainly change that name, but for now just leave as-is. Download (and unzip) the archive to this directory. In the end, this directory listing should be successful:

```
ls datasets/faces/lfw-deepfunneled/
```
(this should list a bunch of names - each of those folders contains images of the corresponding person)

Now that the images are in the expected location, you can activate your Conda environment and run the following:
```
<path to conda env>/bin/python3 src/main.py \
    +dataset=faces_std \
    ++pl_module.name=resnet50_autoencoder_v2 \
    ++pl_module.latent_dim=256 \
    +optimizer=adam \
    +lr_scheduler=cosine_annealing \
    ++trainer.grad_acc=1 \
    ++trainer.max_epochs=250 \
    ++num_workers=2 \
    ++dataset.batch_size=24
```
Note that all these configuration parameters refer to YAML files in the `conf/` folder. For example, specifying `+optimizer=adam` tells the configuration manager Hydra to use `conf/optimizer/adam.yaml` file.

The options prefixed with two plus signs (`++`) means an explicit override. For example, in `conf/config.yaml` you will see `max_epochs: 10` in the `trainer` section. By specifying `++trainer.max_epochs=250` we override that to train for 250 epochs.

## Results

By default, Hydra will create independent, timestamped output folders for each execution. In that folder you will find a directory structure like:

```
outputs/YYYY-MM-DD/HH-MM-SS/
├── .hydra
│   ├── config.yaml
│   ├── hydra.yaml
│   └── overrides.yaml
├── lightning_logs
│   └── version_0
│       ├── checkpoints
│       │   ├── epoch=222-val_reconstruction_loss=0.06.ckpt
│       │   └── last.ckpt
│       ├── hparams.yaml
│       └── metrics.csv
└── main.log
```
Some highlights:
- `.hydra/config.yaml` has the final configuration based on your command line options once composed with the default options. This serves as the ground-truth for your configuration options when attempting to recreate results at a later date.
- The `metrics.csv` file has training metrics such as train and validation loss.
- The `checkpoints/` folder has lightning-compatible model checkpoints, which are fully compatible with pytorch.

## Using the models

To use the models for projecting or viewing results, you can do the following (from the `src/` directory) to view the result of passing an image through the autoencoder:

```
>>> import torch
>>> from lightning_modules.autoencoder import AutoEncoderModuleV2
>>> from omegaconf import OmegaConf
>>> cfg = OmegaConf.create(open('<PATH TO FINAL CONFIG>').read())
>>> model = AutoEncoderModuleV2.load_from_checkpoint('<PATH TO CHECKPOINT FILE>', cfg=cfg)
>>> with torch.no_grad():
...     # a random image with batch size = 1
...     img = torch.rand(1,3,112,112)
...     out = model(img)
```

(Obviously, you would need to size your image to match the model and perform any image preparation instead of using a random tensor).

To extract the latent representation, you can use `model.encoder(img)`. The result will be the encoded representation (e.g. an array of size equal to your latent dimension size).