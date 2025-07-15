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

#### Preliminaries

We expect that relatively small images (e.g. 192x192 pixels) will be fed to the autoencoder model. The input SVS or layered TIFF images are significantly larger and cannot be used directly. Thus, we need to preprocess and extract regions of interest. To that end, we have a tile-extraction script which defines some routines for preprocessing and saving PNG images, which we will call tiles. 

First, we need an input metadata file which will define which images we want to process and potentially some other metadata about the slides. For the tile extraction script, we make use of the following columns:
- `image_id`: This contains a unique image ID which will be used to locate the SVS or TIFF file through a wildcard pattern. For example, if the column contains "ABC", we will search for "ABC.*" in the directory of slide images.
- (optional) `image_subdir`: In case we have many images and wish to extract many tiles from each, it can be helpful to store the images in subdirectories to avoid directories with an excessive number of PNG files. If this field is provided, it's expected that the slide image will be located in `<input dir>/<image_subdir>/` so the wildcard search for `image_id` will be executed in this directory.
- (optional) `shard`: Used for simple parallelization (e.g. on an HPC cluster), if `shard` is specified, then one or more slides can be associated with a shard. Then, when starting jobs for tile extraction, the shard number is specified and only images corresponding to the matching shard will be processed. 

By default, the `conf/tile_prep.yaml` defines parameters for the tile extraction process. This includes things like tile size, the number of tiles to extract, the resolution level in the SVS/TIFF image, and other constants. See that file for detailed descriptions of the parameters. As usual, we employ Hydra, so we can set or override parameters at the command line. For example: 
```
python -m data_preparation.create_tiling_sharded \
    ++image_metadata=<PATH TO CSV METADATA FILE> \
    ++input_dir=<PATH TO SLIDE DIRECTORY> \
    ++tile_extraction_style=1 \
    ++blur_threshold=700
```
By default, this will create a directory at the root of the project and place the extracted tiles under that. That can be adjusted by editing `conf/tile_prep.yaml` 

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