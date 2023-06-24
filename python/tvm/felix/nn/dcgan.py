# https://github.com/Natsu6767/DCGAN-PyTorch/blob/master/dcgan.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the Generator Network
class Generator(nn.Module):
    def __init__(self, params):
        super().__init__()

        # Input is the latent vector Z.
        self.tconv1 = nn.ConvTranspose2d(
            params["nz"], params["ngf"] * 8, kernel_size=4, stride=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(params["ngf"] * 8)

        # Input Dimension: (ngf*8) x 4 x 4
        self.tconv2 = nn.ConvTranspose2d(params["ngf"] * 8, params["ngf"] * 4, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(params["ngf"] * 4)

        # Input Dimension: (ngf*4) x 8 x 8
        self.tconv3 = nn.ConvTranspose2d(params["ngf"] * 4, params["ngf"] * 2, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(params["ngf"] * 2)

        # Input Dimension: (ngf*2) x 16 x 16
        self.tconv4 = nn.ConvTranspose2d(params["ngf"] * 2, params["ngf"], 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(params["ngf"])

        # Input Dimension: (ngf) * 32 * 32
        self.tconv5 = nn.ConvTranspose2d(params["ngf"], params["nc"], 4, 2, 1, bias=False)
        # Output Dimension: (nc) x 64 x 64

    def forward(self, x):
        x = torch.relu(self.bn1(self.tconv1(x)))
        x = torch.relu(self.bn2(self.tconv2(x)))
        x = torch.relu(self.bn3(self.tconv3(x)))
        x = torch.relu(self.bn4(self.tconv4(x)))
        x = torch.tanh(self.tconv5(x))
        return x


# Define the Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, params):
        super().__init__()

        # Input Dimension: (nc) x 64 x 64
        self.conv1 = nn.Conv2d(params["nc"], params["ndf"], 4, 2, 1, bias=False)

        # Input Dimension: (ndf) x 32 x 32
        self.conv2 = nn.Conv2d(params["ndf"], params["ndf"] * 2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(params["ndf"] * 2)

        # Input Dimension: (ndf*2) x 16 x 16
        self.conv3 = nn.Conv2d(params["ndf"] * 2, params["ndf"] * 4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(params["ndf"] * 4)

        # Input Dimension: (ndf*4) x 8 x 8
        self.conv4 = nn.Conv2d(params["ndf"] * 4, params["ndf"] * 8, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(params["ndf"] * 8)

        # Input Dimension: (ndf*8) x 4 x 4
        self.conv5 = nn.Conv2d(params["ndf"] * 8, 1, 4, 1, 0, bias=False)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2, True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2, True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2, True)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2, True)
        x = torch.sigmoid(self.conv5(x))
        return x


DEFAULT_PARAMS = {
    "bsize": 128,  # Batch size during training.
    "imsize": 64,  # Spatial size of training images. All images will be resized to this size during preprocessing.
    "nc": 3,  # Number of channles in the training images. For coloured images this is 3.
    "nz": 100,  # Size of the Z latent vector (the input to the generator).
    "ngf": 64,  # Size of feature maps in the generator. The depth will be multiples of this.
    "ndf": 64,  # Size of features maps in the discriminator. The depth will be multiples of this.
    "nepochs": 10,  # Number of training epochs.
    "lr": 0.0002,  # Learning rate for optimizers
    "beta1": 0.5,  # Beta1 hyperparam for Adam optimizer
    "save_epoch": 2,  # Save step.
}


def dcgan():
    model = nn.Sequential(Generator(DEFAULT_PARAMS), Discriminator(DEFAULT_PARAMS))
    return model, (DEFAULT_PARAMS["nz"], 1, 1)
