from dataclasses import dataclass

import cv2
import numpy as np
import openslide
import pandas as pd
import skimage
from sklearn.decomposition import PCA


def get_extractor_class(key):
    match key:
        case 0:
            return DensityBasedTileExtractor
        case 1:
            return GiemsaTileExtractor
        case _:
            raise Exception('Did not specify an appropriate key'
                            ' for choosing the tile extractor.')
    

@dataclass
class TileInfo():
    n_tiles: int
    tile_size: int
    level: int
    offset_mode: int


class BaseTileExtractor(object):
    '''
    Base class which handles extraction of "informative" tiles
    from large SVS/TIFF images
    '''

    def __init__(self, tile_info):
        self.tile_info = tile_info

    def _create_output_dir(self, output_dir):

        # the name of the directory will give info about how the tiles were created
        tile_dir = (f'numtile-{self.tile_info.n_tiles}-tilesize-{self.tile_info.tile_size}'
                    f'-res-{self.tile_info.level}-mode-{self.tile_info.offset_mode}')
        output_dir = output_dir / tile_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _check_format(self, img_path):
        multilayer_formats = ['svs', 'tiff']
        if not img_path.suffix.lower()[1:] in multilayer_formats:
            raise Exception(f'Unexpected format for path: {img_path}')

    def _load_image(self, img_path):
        return self._read_multilayer(img_path)

    def _open_multilayer(self, img_path):
        self._check_format(img_path)
        return openslide.OpenSlide(img_path)

    def _read_patch(self, o, level, start_tuple, stop_tuple):
        return np.array(o.read_region(
            start_tuple,
            level,
            stop_tuple
        ))[:,:,:3]  # in case there is an alpha channel, we take only the RGB channels      

    def _read_multilayer(self, img_path):
        '''
        Opens a multilayer format file (e.g. SVS, TIFF)
        and returns a numpy array of the pixels at the
        prescribed level. Reads entire slide
        '''
        level = self.tile_info.level

        o = self._open_multilayer(img_path)

        # a tuple of tuples, like ((19991, 42055), (4997, 10513), (1249, 2628))
        level_dimensions = o.level_dimensions

        if level >= o.level_count:
            raise Exception(f'Requested a resolution {level} that does not'
                            f' exist for the image at {img_path}, which has'
                            f' {o.level_count} levels')

        return self._read_patch(o, level, (0, 0), level_dimensions[level])
    
    def _calculate_padding(self, w, h):
        '''
        Given (w,h) calculate the padding to add such that
        the total image will permit an even number of tiles in 
        both horizontal and vertical directions.

        Return a tuple giving the horizontal and vertical padding
        '''
        # for brevity:
        tile_size = self.tile_info.tile_size
        offset_mode = self.tile_info.offset_mode
        pad_h = (tile_size - h % tile_size) % tile_size + ((tile_size * offset_mode) // 2)
        pad_w = (tile_size - w % tile_size) % tile_size + ((tile_size * offset_mode) // 2)
        return pad_w, pad_h

    def _apply_pad(self, img, pad_left, pad_right, pad_top, pad_bottom):
        '''
        Given an array (`img`) and padding on the four edges, apply
        zero-padding
        '''
        return np.pad(img,
                [
                    [pad_top, pad_bottom],
                    [pad_left, pad_right],
                    [0,0]
                ],
                constant_values=255)
    
    def _apply_white_padding(self, img, pad_w, pad_h):
        return self._apply_pad(img,
                               pad_w // 2,
                               pad_w - pad_w // 2,
                               pad_h // 2,
                               pad_h - pad_h // 2)
    
    def _pad_image(self, img):
        '''
        Pad the numpy array such that the eventual
        tiling operations will create an integer number
        of tiles
        '''
        h, w, c = img.shape
        pad_w, pad_h = self._calculate_padding(w, h)
        return self._apply_white_padding(img, pad_w, pad_h)
    
    def _get_tiles_from_array(self, img, tile_index_filter=None):
        '''
        Given an array (i.e. an image), return a "stack" of 
        image tiles with shape (N, tile_size, tile_size, 3)

        If the optional `tile_index_filter` is passed, then
        we extract tiles correspondong to those indices. This 
        is useful for situations where we know we have all whitespace
        and don't need those tiles. 
        '''
        tile_size = self.tile_info.tile_size

        if tile_index_filter is not None:
            img_stack = []
            n_w = img.shape[1] // tile_size
            for idx in tile_index_filter:
                tile_row = idx // n_w
                tile_col = idx % n_w
                y0 = tile_row * tile_size
                x0 = tile_col * tile_size
                img_stack.append(img[y0:y0+tile_size,x0:x0+tile_size])
            img = np.stack(img_stack)
            return img
        else:
            # note that the reshape method first performs a 'ravel' 
            # (e.g. a flattening) before it chunks everything for the reshape.
            # Based on this, we get this somewhat awkward 5-tensor. This is
            # due to the way that the raveling and reshaping traverses the
            # original image
            img = img.reshape(img.shape[0] // tile_size,
                            tile_size,
                            img.shape[1] // tile_size,
                            tile_size,
                            3)

            # to ultimately create an array of (tile_size, tile_size, 3) tiles, we need to 
            # do this transpose and reshape. This gives array of shape 
            # (N, tile_size, tile_size, 3)
            # where N is the number of tiles (i.e. padded size // tile_size)
            return img.transpose(0,2,1,3,4).reshape(-1, tile_size, tile_size, 3)
        
    def _get_raw_tiles(self, img_path, tile_index_filter=None):
        '''
        This returns an array of shape (N, tile_size, tile_size, 3)
        where N is the number of tiles

        If `tile_index_filter` is not None, then specific tiles are
        selected based on their grid position. For example, if we are
        looking to tile an image and our tile size is such that we can
        fit (m,n) tiles in the vertical and horizontal directions, then
        an index filter of `k` will select/keep the tile at (k//n, k%n).
        This allows us to avoid tiling in regions of largely whitespace, etc.
        '''
        img = self._load_image(img_path)
        img = self._pad_image(img)
        return self._get_tiles_from_array(img, tile_index_filter=tile_index_filter)

    def _select_tiles(self, img_path, cfg):
        raise NotImplementedError('Need to implement a _select_tiles method'
                                  ' specific to your application.')
    
    def extract(self, img_path, cfg):
        '''
        Direct way to get tiles without the save functionality of the
        `extract_and_save` method
        '''
        return self._select_tiles(img_path, cfg)

    def _get_image_path(self, image_meta, input_root_dir):
        '''
        Given a pd.Series from the image metadata
        dataframe, find the image and return a path

        `input_root_dir` is a pathlib.Path which locates
        the top directory of the input directory tree
        '''
        image_id = image_meta['image_id']

        if 'image_subdir' in image_meta:
            input_image_dir = input_root_dir / str(image_meta['image_subdir'])
        else:
            input_image_dir = input_root_dir

        img_path = list(input_image_dir.glob(f'{image_id}.*'))
        if len(img_path) != 1:
            raise Exception('Could not locate a single file matching'
                            f' the pattern {image_id}.* in'
                            f' {input_image_dir}.')
        else:
            return img_path[0]
        
    def _get_final_output_dir(self, image_meta, output_root_dir):
        '''
        Given a pd.Series (`image_meta`) from the image metadata
        dataframe, return a path to the output folder

        `output_root_dir` is a pathlib.Path to the root of the output
        directory, which may contain subdirectories
        '''
        if 'image_subdir' in image_meta:
            output_dir = output_root_dir / str(image_meta['image_subdir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            return output_dir
        else:
            return output_root_dir

    def extract_and_save(self, image_metadata_path, 
                         input_root_dir, output_root_dir, cfg, shard=None):
        '''
        Orchestrates the reading, tile extraction, and saving of tiles for
        a full dataset
        '''
        metadata_df = pd.read_csv(image_metadata_path)

        if shard is not None:
            self.sharded = True
            self.shard_num = shard
            metadata_df = metadata_df.loc[metadata_df.shard == self.shard_num]
        else:
            self.sharded = False

        output_root_dir = self._create_output_dir(output_root_dir)

        for i, row in metadata_df.iterrows():
            img_path = self._get_image_path(row, input_root_dir)
            output_dir = self._get_final_output_dir(row, output_root_dir)
            tiles = self._select_tiles(img_path, cfg)

            for ii in range(tiles.shape[0]):
                t = tiles[ii, :].astype(np.uint8)
                fout = output_dir / f'{row['image_id']}.tile_{ii}.png'
                skimage.io.imsave(fout, t)


class DensityBasedTileMixin:
    '''
    Mixin class which helps with sorting and filtering tiles based on density.

    Will remove tiles that are low variance which likely represent black space
    near the edges of a slide image. Follows this by sorting tiles based on
    pixel density (e.g. those with the darkest overall colors)
    '''

    # used to remove dark tiles with very little variance
    # due to edges, etc. that are all black. Will also remove
    # largely white tiles which are uninformative, although
    # those can also be removed by sorting on pixel sums
    # and truncating the resulting sorted list
    VARIANCE_THRESHOLD = 30

    def _post_process_tiles(self, tile_array):

        tile_vars = tile_array.var(axis=3) # (N, tile_size, tile_size)
        tile_mean_vars = tile_vars.mean(axis=(1,2))
        low_variance = tile_mean_vars < DensityBasedTileMixin.VARIANCE_THRESHOLD
        tile_array = tile_array[~low_variance]

        # based on the sum of the pixels in each tile, we sort and take the
        # top num_tiles. Recall that np.argsort gives ascending order. We
        # want this since more 'informative' images will have more colored pixels
        # which are lower pixel values.
        # Majority white tiles will have very large sums and are likely not
        # very informative.
        if tile_array.shape[0] > 0:
            tile_sums = tile_array.reshape(tile_array.shape[0],-1).sum(-1)
            idxs = np.argsort(tile_sums)
            sorted_tile_sums = tile_sums[idxs]
            return tile_array[idxs], sorted_tile_sums
        else:
            return [], None
    

class DensityBasedTileExtractor(BaseTileExtractor, DensityBasedTileMixin):
    '''
    Returns tiles based on the pixel colors (e.g. darker is better)
    while attempting to remove low-variance dark regions likely to be
    slide artifacts
    '''

    def _select_tiles(self, img_path, cfg):
        '''
        This method for tile selection involves summing the RGB
        pixels for each tile and selecting those with the lowest values
        (since lower pixel values means darker images)
        '''
        num_tiles = self.tile_info.n_tiles

        tiles = self._get_raw_tiles(img_path)

        # remove low-variance tiles and sort such that
        # the tiles with the darkest pixels are at the 
        # top of the list:
        tiles, tile_pixel_sums = self._post_process_tiles(tiles)

        # if the total number of tiles in the full image was fewer than the number of requested
        # tiles, add the required number of fully white tiles
        if len(tiles) < num_tiles:
            tiles = np.pad(tiles,
                        [
                            [0, num_tiles - len(tiles)],
                            [0, 0],
                            [0, 0],
                            [0, 0]
                        ], constant_values=255)
        return tiles[:num_tiles]
    

class GiemsaTileExtractor(DensityBasedTileExtractor):

    NUM_SAMPLES = 50

    def _project_and_sort(self, tiles):
        '''
        This method uses PCA to project the RGB
        pixels into a 2-d subspace. From there, we sort by
        the "blue" content to prioritize tiles that have lots 
        of staining

        Accepts a (n,H,W,3) array and returns an array of the 
        same shape, but with the first dimension sorted to place
        blue tiles at the top 
        '''

        num_tiles = len(tiles)
        num_samples = min(GiemsaTileExtractor.NUM_SAMPLES, num_tiles)
        selected_img_idx = np.random.choice(num_tiles, size=num_samples, replace=False)
        selected_imgs = tiles[selected_img_idx]
        pca = PCA(n_components=2)
        pca.fit(selected_imgs.reshape(-1,3))
        
        # The projected matrix is [n,2] where n is the number of pixels
        # across ALL of the tiles (e.g. tile_size*tile*size*num_tiles)
        projected = pca.transform(tiles.reshape(-1,3))

        # want the mean of those projected pixels for each image.
        # Results in an array of size (num_tiles, 2)
        mean_pixels = np.mean(projected.reshape(num_tiles, -1, 2), axis=1)

        # get the principal components and figure out which
        # one is more aligned with a 'blue' signal.
        # This is the inner product of a pure red and pure blue with the PCs
        # (e.g. a pure blue pixel is (0,0,1)), which just selects the third
        # column of the PCA components matrix
        v = pca.components_[:, [0,2]]

        # for the projected pure blue signal, check which PC
        # has the higher projection - this direction is
        # more aligned with blue pixels. Note the abs since
        # the PCs can be flipped in orientation.
        blue_pc = np.argmax(np.abs(v[:,1]))

        # once we know which PC is more aligned with blue content,
        # sort the projections of the tile-level mean pixels. This way
        # we are sorting by the blue'ish content
        sorted_projections = np.argsort(mean_pixels[:, blue_pc])

        # again, if the orientation of the PC was such, we need
        # to reverse the sort order given that argsort only gives
        # ascending order
        if v[blue_pc, 1] > 0:
            # reverse the sort order
            sorted_projections = sorted_projections[::-1]

        return tiles[sorted_projections]

    def _select_tiles(self, img_path, cfg):

        tiles = super()._select_tiles(img_path, cfg)

        # sort by 'blueish' content:
        tiles = self._project_and_sort(tiles)

        # remove blurry tiles
        laplacian_vals = np.array(
            [cv2.Laplacian(
                cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), 
                cv2.CV_64F).var() for x in tiles])
        tiles = tiles[laplacian_vals > cfg.blur_threshold]
        return tiles
