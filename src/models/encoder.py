import torch.nn as nn
from torchvision.models import resnet50


# Used to replace the fully-connected final layer
# in the default ResNet
class Identity(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x


class ResNet50Encoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.net = resnet50(weights='IMAGENET1K_V2')

        # default conv1 has a 7x7 kernel
        self.net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # the default resnet50 has a final fully-connected
        # layer to go from 2048 to 1000 for classification.
        # we don't want that here, so we override
        # Cache this dimension for later
        self.num_output_features = self.net.fc.in_features
        self.net.fc = Identity()

    def forward(self, x):
        return self.net(x)
