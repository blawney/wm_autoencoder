import sys

from .autoencoder import AutoEncoderModuleV1, \
    AutoEncoderModuleV2

PL_MODULE_LIST = [
    AutoEncoderModuleV1,
    AutoEncoderModuleV2
]

AVAILABLE_MODULES = {x.NAME: x for x in PL_MODULE_LIST}


def load_pl_module(cfg):
    '''
    Loads/returns the LightningModule subclass dictated by the
    `cfg` configuration object.
    '''
    try:
        pl_module_class = AVAILABLE_MODULES[cfg.pl_module.name]
    except KeyError:
        sys.stderr.write('Could not locate a LightningModule subclass'
                         f'  identified by{cfg.pl_module.name}. Available'
                         f' names are {",".join(AVAILABLE_MODULES.keys())}')
        sys.exit(1)
    return pl_module_class(cfg)
