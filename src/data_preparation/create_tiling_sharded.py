# This script creates tiles into a pre-allocated
# directory tree. The location of the tiles is dictated
# by the metadata file which contains the following named columns:
# - image_id
#     - contains the unique identifier, no suffix/format
# - (optional) image_subdir
#     - if the number of images is large enough that we choose
#       to save images to subdirectories (so as not to overwhelm
#       the filesystem), then this will tell the script where they 
#       are located relative to the `-d/--input_dir` argument. Also
#       dictates where they are placed in the output directory.
# - (optional) shard
#     - if we have many large slide files and wish to perform a dumb
#       parallelization to save time, you can add this column to the metadata
#       file which then enables calling this script with the `-s/--shard`
#       argument. It could be the case that the shard is the same as the 
#       image_subdir, but that's not required.  
#
# If the input image is located in a particular subdir (of the input folder)
# then the output image will also be located in a subdir of the same name
# in the output folder.
#
# Note: due to the import of a sibling package (utils.image_utils)
# need to call with 
# python3 -m data_preparation.create_tiling_sharded <args>

import os
from pathlib import Path

import hydra

from utils.image_utils import TileInfo, \
    get_extractor_class


@hydra.main(version_base='1.3', config_path="../../conf", config_name="tile_prep")
def main(cfg):

    input_dir = Path(cfg.input_dir)

    # the run-specific output directory created by hydra:
    output_dir = Path(os.getcwd())

    n_tiles = cfg.num_tiles
    tile_size = cfg.tile_size
    level = cfg.level
    offset_mode = cfg.offset_mode
    shard = cfg.shard
    image_meta_path = cfg.image_metadata
    tile_extraction_class_key = cfg.tile_extraction_style

    tile_info = TileInfo(n_tiles, 
                         tile_size, 
                         level, 
                         offset_mode)
    
    # will raise an exception if not a valid key
    extractor_class = get_extractor_class(tile_extraction_class_key)

    # run the tile extraction
    extractor = extractor_class(tile_info)
    extractor.extract_and_save(image_meta_path, input_dir, output_dir, cfg, shard=shard)


if __name__ == '__main__':
    main()