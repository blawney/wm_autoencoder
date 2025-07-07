from pytorch_lightning import LightningModule
import torch.nn as nn
import torch.nn.functional as F

from optimizers import load_optimizer_and_lr_scheduler
from models.encoder import ResNet50Encoder
from models.decoder import ResNet50DecoderV1, \
    ResNet50DecoderV2, \
    DecoderBottleneckV1, \
    DecoderBottleneckV2


class AutoEncoderModule(LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.latent_dim = self.config.pl_module.latent_dim
        self._prep_model()
    

    def _prep_model(self):
        '''
        See child methods which implement a different decoder
        '''
        self.encoder = ResNet50Encoder()
        
        # linear layer to go from the encoder to the latent representation
        self.latent_proj1 = nn.Linear(self.encoder.num_output_features, 
                                      self.encoder.num_output_features // 2)
        self.latent_proj2 = nn.Linear(self.encoder.num_output_features // 2, 
                                      self.latent_dim)
        

    def forward(self, x):
        x = self.encoder(x)
        x = self.latent_proj1(x)
        x = self.latent_proj2(x)
        return self.decoder(x)


    def _batchstep(self, imgs):
        '''
        A "private" method which performs a forward pass and
        returns the reconstruction loss in a dict. 
        
        Used by the training/validation/test step method overrides

        `imgs` is a batch of images (size B)
        '''
        #print('in:')
        #print(imgs.min(), imgs.max())
        x_hat = self.forward(imgs)
        #print('out:')
        #print(x_hat.min(), x_hat.max())
        reconstruction_loss = F.mse_loss(x_hat, imgs)
        
        log_dict = {
            'reconstruction_loss': reconstruction_loss,
        }

        # although it seems redundant to return both, we do this
        # for the ability to add more metrics in the `log_dict`
        # in the future
        return reconstruction_loss, log_dict


    def training_step(self, batch, batch_idx):
        '''
        This is the standard method to override when executing
        training.

        `batch` is a tensor of shape (B,3,H,W).
        '''
        loss, log_dict = self._batchstep(batch)

        # reformat the names in the log_dict:
        log_dict = {f'train_{x}':y for x,y in log_dict.items()}
        self.log_dict(log_dict, on_step=True, on_epoch=False)
        return loss


    def validation_step(self, batch, batch_idx):
        '''
        This is the standard method to override when executing
        validation.

        `batch` is a tuple of inputs and targets.
        '''
        loss, log_dict = self._batchstep(batch)

        # reformat the names in the log_dict:
        log_dict = {f'val_{x}':y for x,y in log_dict.items()}
        self.log_dict(log_dict, on_step=False, on_epoch=True)


    def configure_optimizers(self):
        optimizer, lr_scheduler = load_optimizer_and_lr_scheduler(
            self.parameters(), self.config, self.trainer)

        if lr_scheduler is not None:
            additional_params = {}
            if 'scheduler_config' in self.config.lr_scheduler:
                additional_params = self.config.lr_scheduler.scheduler_config
            scheduler_dict = {
                "scheduler": lr_scheduler,
                **additional_params
            }
            return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
        
        return optimizer


class AutoEncoderModuleV1(AutoEncoderModule):
    NAME = 'resnet50_autoencoder_v1'

    def _prep_model(self):
        super()._prep_model()
        self.decoder = ResNet50DecoderV1(DecoderBottleneckV1, 
                                      self.latent_dim, 
                                      self.config.dataset.final_image_size)


class AutoEncoderModuleV2(AutoEncoderModule):
    NAME = 'resnet50_autoencoder_v2'

    def _prep_model(self):
        super()._prep_model()
        self.decoder = ResNet50DecoderV2(DecoderBottleneckV2, 
                                       self.latent_dim, 
                                       self.config.dataset.final_image_size)