import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, Dataset
import math
import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
from typing import Dict, List, Tuple
import tqdm
from torchvision.datasets import video_utils

seed = 22
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.mps.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
  device = torch.device("mps")


# Download and dataset preparation
transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ])

train_data: Dataset = torchvision.datasets.MNIST(
    root='./content/data/',
    train= True,
    transform=transforms,
    download= True
  )

test_data: Dataset = torchvision.datasets.MNIST(
    root='./content/data/',
    train= False,
    transform=None,
    download= True,

  )

def ddpm_schedules(betas: torch.tensor) -> Dict[str,torch.tensor]:
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=-1)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    return {
        "alphas": alphas,
        "betas": betas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
    }

def linear_beta_schedule(
      T: int = 500,
      beta1: float = 1e-4,
      beta2: float = 0.02
      ) -> torch.tensor:
    """
    linear schedule, proposed in original ddpm paper
    """
    timesteps = T
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    betas = torch.linspace(
        beta_start,
        beta_end,
        timesteps,
        dtype=torch.float32)
    return betas


def cosine_beta_schedule(
      T: int,
      s: float = 0.008
      ) -> torch.tensor:
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps=torch.linspace(0, T, steps=T+1, dtype=torch.float32)
    f_t=torch.cos(((steps/T+s)/(1.0+s))*math.pi*0.5)**2
    betas=torch.clip(1.0-f_t[1:]/f_t[:T],0.0,0.999)
    return betas

class ForwardDiffusionProcess(nn.Module):
    def __init__(
          self,
          ddpm_schedules: Dict[str,torch.tensor],
          device: torch.device) -> None:

        super().__init__()

        # Register buffers with ddpm schedules
        for k, v in ddpm_schedules.items():
            self.register_buffer(k, v)

        self.n_T = self.alphas.shape[0]
        self.device = device

    def apply_noise(self, x, noise, ts):
        # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps

        a = self.sqrt_alphas_cumprod.gather(-1, ts).reshape(x.shape[0], 1, 1, 1)
        b = self.sqrt_one_minus_alphas_cumprod.gather(-1, ts).reshape(x.shape[0], 1, 1, 1)
        x_t = a * x + b * noise

        print(x_t.shape)

        return x_t

    def forward(self, x):
        """
        this method is used in training, so samples t and noise randomly
        """

        # t ~ Uniform(0, n_T)
        # TODO: sample uniformly a timestep
        _ts = torch.randint(0,self.n_T, size=(x.shape[0],)).to(self.device)

        # eps ~ N(0, 1)
        # TODO: get the random noise
        # randn like
        noise = torch.randn_like(x)
        #noise = torch.normal(0.0, 1.0, size=(x.shape[0],)).to(self.device)

        x_t = self.apply_noise(x, noise, _ts)

        return x_t, noise, _ts

# Define number of steps
n_T=100

# Define beta schedulers
betas_linear = linear_beta_schedule(T=n_T)
betas_cosine = cosine_beta_schedule(T=n_T)


# Generate ddpm schedulers based on the given betas
ddpm_linear = ddpm_schedules(betas_linear)
ddpm_cosine = ddpm_schedules(betas_cosine)


# Create the ForwardDiffusionProcess objects
forward_diffusion_process_linear = ForwardDiffusionProcess(
    ddpm_schedules=ddpm_linear,
    device=device
).to(device) # Linear

forward_diffusion_process_cosine = ForwardDiffusionProcess(
    ddpm_schedules=ddpm_cosine,
    device=device
).to(device) # Cosine

plt.figure(figsize = (8, 4))
plt.plot(forward_diffusion_process_linear.alphas_cumprod.cpu().numpy(), label='linear')
plt.plot(forward_diffusion_process_cosine.alphas_cumprod.cpu().numpy(), label='cosine')
plt.legend(loc="upper right")

plt.title("Linear combination between a sample and noise")
plt.xlabel("Timesteps")
plt.ylabel("Amount of original image")
#plt.show()

# Pick a MNIST sample
image = train_data[0][0].clone().unsqueeze(0).to(device) # Shape: [1, 1, 28, 28]
plt.axis('off')
plt.imshow(image.permute(0,2,3,1)[0].cpu().numpy(), cmap='Greys')

# Return Forward Diffusion Process
def get_image_from_linear(noise: torch.tensor, image: torch.tensor) -> torch.tensor:
    image_linear = forward_diffusion_process_linear.apply_noise(
      image,
      noise,
      torch.tensor(t).to(device)
    )
    return image_linear


def get_image_from_cosine(noise: torch.tensor, image: torch.tensor) -> torch.tensor:
    image_cosine = forward_diffusion_process_cosine.apply_noise(
        image,
        noise,
        torch.tensor(t).to(device)
    )
    return image_cosine

# noise the image over timesteps using the two schedulers
image_linear = image.clone()
image_cosine = image.clone()

images_linear = []
images_cosine = []

for t in tqdm.tqdm(range(n_T)):
    noise = torch.randn_like(image_linear)

    # Add noise from linear scheduler
    image_linear = get_image_from_linear(noise, image_linear)
    images_linear.append(image_linear.detach().cpu())

    # Add noise from cosine scheduler
    image_cosine = get_image_from_cosine(noise, image_cosine)
    images_cosine.append(image_cosine.detach().cpu())

plt.figure(figsize = (16, 8))
plt.axis('off')
indices = [0, 5,10,15,20,25, 30, 35, 40, 45, 50]
img_row_1 = torch.cat([images_linear[i] for i in indices], dim=3).permute(0,2,3,1)[0] # Linear beta scheduler
img_row_2 = torch.cat([images_cosine[i] for i in indices], dim=3).permute(0,2,3,1)[0] # Cosine beta scheduler
img_row_1_2 = torch.cat([img_row_1, img_row_2])
np.clip(img_row_1_2, 0, 1)
plt.imshow(img_row_1_2, interpolation='nearest', cmap='Greys')
plt.show()