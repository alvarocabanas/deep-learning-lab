import torch
from torch import nn, optim
from torchvision import transforms, datasets, utils
from PIL import Image
import numpy as np
import math
from IPython.display import display
from tqdm import tqdm
from itertools import cycle
from typing import Tuple

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
    'num_epochs':1, #20
    'learning_rate':0.0002,
    'betas':(0.5, 0.999),
    'noise_size':100,
    'num_val_samples':25,
    'num_classes':10,
    'num_input_channels':1,
}

train_transforms = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(
    root='data',
    train=True,
    download=True,
    transform=train_transforms,
)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=hparams['batch_size'],
    shuffle=True,
)

class Generator(torch.nn.Module):

    def __init__(self, noise_size: int, num_input_channels: int):
        super().__init__()

        # TODO: Create the Fully connected layer using nn.Linear
        self.fc = nn.Linear(noise_size, 512 * 4 * 4)

        # TODO: Create the first block
        self.convt1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False), # (B, 256, 8, 8)
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # TODO: Create the second block
        self.convt2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False), # (B, 128, 16, 16)
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # TODO: Create the third block using nn.Sequential with ConvTranspose2d, and activation
        self.convt3 = nn.Sequential(
            nn.ConvTranspose2d(128, num_input_channels, 4, stride=2, padding=1, bias=False), # (B, num_input_channels, 32, 32)
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # TODO: Define the forward of the generator, x are random noise vectors (B, noise_size)
        x = self.fc(x)

        x = x.reshape(-1, 512, 4, 4) # (B, channels, height, width)

        x = self.convt1(x)
        x = self.convt2(x)
        x = self.convt3(x)

        return x

class Discriminator(torch.nn.Module):

    def __init__(self, num_input_channels: int):
        super().__init__()

        # TODO: Create the first block
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_input_channels, 128, 4, stride=2, padding=1, bias=False), # (B, 128, 16, 16)
            nn.LeakyReLU(0.2),
        )

        # TODO: Create the second block
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False), # (B, 256, 8, 8)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        # TODO: Create the third block
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=False), # (B, 512, 4, 4)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )

        # TODO: Create the fully connected block using nn.Sequential with Linear and activation
        self.fc = nn.Sequential(
            nn.Linear(512*4*4, 1),
            nn.Sigmoid(), # Binary classification (real vs false)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # TODO: Define the forward of the discriminator, x are images (B, num_input_channels, 32, 32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        B, input, h, w = x.size()
        x = x.reshape(B, input * h * w) # (B, channels * height * width)
        x = self.fc(x)

        return x

generator = Generator(hparams['noise_size'], hparams['num_input_channels']).to(device)
optimizer_g = torch.optim.Adam(generator.parameters(), lr=hparams['learning_rate'], betas=hparams['betas'])

discriminator = Discriminator(hparams['num_input_channels']).to(device)
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=hparams['learning_rate'], betas=hparams['betas'])

criterion = nn.BCELoss()

def init_weights(m):
    if type(m) in {nn.Conv2d, nn.ConvTranspose2d, nn.Linear}:
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias != None:
            torch.nn.init.constant_(m.bias, 0.0)
    if type(m) == nn.BatchNorm2d:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)

generator.apply(init_weights)
discriminator.apply(init_weights);

def train_batch(
        real_samples: torch.Tensor,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        optimizer_g: torch.optim,
        optimizer_d: torch.optim,
        ) -> Tuple[float, float]:

    generator.train()
    discriminator.train()

    bsz = real_samples.shape[0]

    # TODO: Define the labels for the real (ones) and fake (zeros) images of size (bsz, 1)
    label_real = torch.ones(bsz, 1)
    label_fake = torch.zeros(bsz, 1)

    label_real = label_real.to(device)
    label_fake = label_fake.to(device)

    ####################
    # OPTIMIZE GENERATOR
    ####################

    # Reset gradients
    optimizer_g.zero_grad()

    # Generate fake samples
    z = torch.randn(bsz, hparams['noise_size'], device=device)
    fake_samples = generator(z)

    # Evaluate the generated samples with the discriminator
    predictions_g_fake = discriminator(fake_samples)
    # Calculate error with respect to what the generator wants
    loss_g = criterion(predictions_g_fake, label_real)

    # Backpropagate
    loss_g.backward()

    # Update weights (do a step in the optimizer)
    optimizer_g.step()

    ########################
    # OPTIMIZE DISCRIMINATOR
    ########################

    fake_samples = fake_samples.detach() # Let's detach them to freeze the generator

    # Reset gradients
    optimizer_d.zero_grad()

    # Calculate discriminator prediction for real samples
    predictions_d_real = discriminator(real_samples)

    # Calculate error with respect to what the discriminator wants
    loss_d_real = criterion(predictions_d_real, label_real)

    # Calculate discriminator loss for fake samples
    predictions_d_fake = discriminator(fake_samples)

    # Calculate error with respect to what the discriminator wants
    loss_d_fake = criterion(predictions_d_fake, label_fake)

    # Total discriminator loss
    loss_d = (loss_d_real + loss_d_fake) / 2
    loss_d.backward()

    # Update weights (do a step in the optimizer)
    optimizer_d.step()

    return loss_g.item(), loss_d.item()

@torch.no_grad()
def evaluate(generator: torch.nn.Module, z_val: torch.Tensor) -> Image.Image:

    generator.eval()
    fake_samples = generator(z_val).cpu()
    # select a sample or create grid if img is a batch
    nrows = int(math.sqrt(fake_samples.shape[0]))
    img = utils.make_grid(fake_samples, nrow=nrows)

    # unnormalize
    img = (img*0.5 + 0.5)*255

    # to numpy
    image_numpy = img.numpy().astype(np.uint8)
    image_numpy = np.transpose(image_numpy, (1, 2, 0))

    return Image.fromarray(image_numpy)

z_val = torch.randn(hparams['num_val_samples'], hparams['noise_size'], device=device)

for epoch in range(hparams['num_epochs']):

    for i, (real_samples, labels) in enumerate(dataloader):
        real_samples = real_samples.to(device)
        loss_g, loss_d = train_batch(real_samples, generator, discriminator, optimizer_g, optimizer_d)

        if (i+1) % 200 == 0:
            print(f"\nEpoch: {epoch+1}/{hparams['num_epochs']}, batch: {i+1}/{len(dataloader)},"
                  +f" G_loss: {loss_g}, D_loss: {loss_d}")

            fake_images = evaluate(generator, z_val)
            display(fake_images)

    print(f"\nEpoch: {epoch+1}/{hparams['num_epochs']}, batch: {i+1}/{len(dataloader)},"
          +f" G_loss: {loss_g}, D_loss: {loss_d}")

    fake_images = evaluate(generator, z_val)
    display(fake_images)
    URL = 'https://invibe.net/cgi-bin/index.cgi/ImageGallery?action=AttachFile&do=get&target=raster_plot.png'
