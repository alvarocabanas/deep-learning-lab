import os
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
from torchvision import transforms, datasets, utils

seed = 123
np.random.seed(seed)
_ = torch.manual_seed(seed)
_ = torch.cuda.manual_seed(seed)
_ = torch.mps.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
  device = torch.device("mps")

print(device)

hparams = {
    'batch_size':128,
    'num_epochs':30,
    'channels':64,
    'latent_dims':2,
    'variational_beta':1,
    'learning_rate':1e-3,
    'weight_decay':1e-5
}

transforms = transforms.Compose([
    transforms.ToTensor(),
])

# Dataset initializations

mnist_trainset = datasets.MNIST(
    root='data',
    train=True,
    download=True,
    transform=transforms
)

mnist_testset = datasets.MNIST(
    root='data',
    train=False,
    download=True,
    transform=transforms
)

# Dataloders initialization

train_dataloader = torch.utils.data.DataLoader(
    dataset=mnist_trainset,
    batch_size=hparams['batch_size'],
    shuffle=True,
    drop_last=True,
)

test_dataloader = torch.utils.data.DataLoader(
    dataset=mnist_testset,
    batch_size=hparams['batch_size'],
    shuffle=False,
    drop_last=True,
)

class Encoder(nn.Module):
    def __init__(
            self,
            channels: int,
            latent_dims: int,
            ) -> None:

        super(Encoder, self).__init__()

        self.c = channels
        # TODO: Complete with the appropriate dimensions.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.c, kernel_size=4, stride=2, padding=1) # out: (c, 14, 14)
        self.conv2 = nn.Conv2d(in_channels=self.c, out_channels=self.c*2, kernel_size=4, stride=2, padding=1) # out: 2* (c, 7, 7)
        self.fc_mu = nn.Linear(in_features=self.c*2*7*7, out_features=latent_dims)
        self.fc_logvar = nn.Linear(in_features=self.c*2*7*7, out_features=latent_dims)

    def forward(self, x: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:

        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = out.view(out.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors

        # TODO: Obtain the mean and covariance matrices from the output of the linear layers
        x_mu = self.fc_mu(out)
        x_logvar = self.fc_logvar(out)

        return x_mu, x_logvar

class Decoder(nn.Module):
    def __init__(
            self,
            channels: int,
            latent_dims: int
            ) -> None:

        super(Decoder, self).__init__()
        self.c = channels
        # TODO
        self.fc = nn.Linear(in_features=latent_dims, out_features=self.c*2*7*7)
        self.conv2 = nn.ConvTranspose2d(in_channels=self.c*2, out_channels=self.c, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=self.c, out_channels=1, kernel_size=4, stride=2, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:

        out = self.fc(z)
        out = out.view(out.size(0), self.c*2, 7, 7) # unflatten batch

        # TODO
        out = F.relu(self.conv2(out))
        out = torch.sigmoid(self.conv1(out))
        return out

class VariationalAutoencoder(nn.Module):
    def __init__(
            self,
            z_dims: int,
            n_ch: int,
            ) -> None:

        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(channels=n_ch,latent_dims=z_dims).to(device)
        self.decoder = Decoder(channels=n_ch, latent_dims=z_dims).to(device)

    def reparametrize(
            self,
            mu:torch.Tensor,
            logvar:torch.Tensor,
            ) -> torch.Tensor:
        # Given mean and logvar returns z
        # reparameterization trick: instead of sampling from Q(z|X), sample epsilon = N(0,I)
        # mu, logvar: mean and log of variance of Q(z|X)

        # The factor 1/2 in the exponent ensures that the distribution has unit variance
        std = torch.exp(0.5 * logvar)
        # Random sample
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x: torch.Tensor) -> Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor]:
        # TODO
        latent_mu, latent_logvar = self.encoder(x)
        z = self.reparametrize(latent_mu, latent_logvar)
        x_recon = self.decoder(z)

        return x_recon, latent_mu, latent_logvar

def vae_loss(
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        variational_beta: int=1,
        ) -> float:
    # recon_x is the probability of a multivariate Bernoulli distribution p.
    # -log(p(x)) is then the pixel-wise binary cross-entropy.

    recon_loss = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
    kldivergence = variational_beta * (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))
    mean_batch_loss = (recon_loss +  kldivergence)/x.shape[0]

    return mean_batch_loss

def train_batch(
        image_batch: torch.Tensor,
        vae: torch.nn.Module,
        vae_loss: torch.nn.Module,
        optimizer: torch.optim,
        ) -> float:

    image_batch = image_batch.to(device)

    # TODO: Get vae reconstruction and loss
    image_batch_recon, latent_mu, latent_logvar = vae(image_batch)
    loss = vae_loss(image_batch_recon, image_batch, latent_mu, latent_logvar)

    # backpropagation
    optimizer.zero_grad()
    loss.backward()

    # one step of the optmizer (using the gradients from backpropagation)
    optimizer.step()

    return loss.item()

start_time=time.time()
# TODO: Instantiate optimizer and model here
vae_2z = VariationalAutoencoder(hparams['latent_dims'], hparams['channels'])
optimizer = torch.optim.Adam(params=vae_2z.parameters(), lr=hparams['learning_rate'], weight_decay=hparams['weight_decay'])
# This is the number of parameters used in the model
num_params = sum(p.numel() for p in vae_2z.parameters() if p.requires_grad)
print(f'Number of parameters: {num_params}')

# set to training mode
vae_2z.train()

train_loss_avg = []

print('Training ...')
for epoch in range(hparams['num_epochs']):
    train_loss_avg.append(0)
    num_batches = 0

    for i,(image_batch,_) in enumerate(train_dataloader):

        loss_batch = train_batch(image_batch, vae_2z, vae_loss, optimizer)
        train_loss_avg[-1] += loss_batch

    train_loss_avg[-1] /= i
    print('Epoch [%d / %d] average reconstruction error: %f' % (epoch+1, hparams['num_epochs'], train_loss_avg[-1]))

print("--- TOTAL TIME: %s min ---" % (round((time.time() - start_time) / 60, 3)))

fig = plt.figure()
plt.plot(train_loss_avg)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# set to evaluation mode
vae_2z.eval()

test_loss_avg, num_batches = 0, 0
for image_batch, _ in test_dataloader:

    with torch.no_grad():
        image_batch = image_batch.to(device)

        # vae reconstruction
        image_batch_recon, latent_mu, latent_logvar = vae_2z(image_batch)
        # reconstruction error
        loss = vae_loss(image_batch_recon, image_batch, latent_mu, latent_logvar)

        test_loss_avg += loss.item()
        num_batches += 1

test_loss_avg /= num_batches
print('average reconstruction error: %f' % (test_loss_avg))

# This function takes as an input the images to reconstruct
# and the name of the model with which the reconstructions
# are performed
def to_img(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0, 1)
    return x

def show_image(img: torch.Tensor) -> None:
    img = to_img(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

@torch.no_grad()
def visualise_output(
        images: torch.Tensor,
        model: nn.Module,
        device: torch.device,
        ) -> None:
    images = images.to(device)
    model.to(device)
    images, _, _ = model(images)
    images = images.cpu()
    images = to_img(images)
    np_imagegrid = torchvision.utils.make_grid(images[1:50], 10, 5).numpy()
    plt.imshow(np.transpose(np_imagegrid, (1, 2, 0)))
    plt.show()

vae_2z.eval()
images, labels = next(iter(test_dataloader))

# First visualise the original images
print('Original images')
show_image(torchvision.utils.make_grid(images[1:50],10,5))
plt.show()

# Reconstruct and visualise the images using the vae
print('VAE reconstruction:')
visualise_output(images, vae_2z, device)

# this is how the VAE parameters can be saved:
torch.save(vae_2z.state_dict(), './my_vae_2z.pth')