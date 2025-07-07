import sys

from lr_schedulers.cosine_annealing import CosineAnnealing


AVAILABLE_SCHEDULERS = {
    'cosine_annealing': CosineAnnealing
}


def load_scheduler(scheduler_cfg, optimizer, trainer):
    '''
    Creates/returns an instance of a learning rate scheduler.

    Since some schedulers depend on things like the training set
    size or epochs, we pass a pytorch_lightning.Trainer instance
    '''
    scheduler_name = scheduler_cfg.name
    try:
        lr_scheduler_cls = AVAILABLE_SCHEDULERS[scheduler_name]
    except KeyError as ex:
        sys.stderr.write('The learning rate scheduler identified by'
                         f' the name {scheduler_name} was not found')
        sys.exit(1)

    return lr_scheduler_cls(scheduler_cfg, optimizer, trainer)
