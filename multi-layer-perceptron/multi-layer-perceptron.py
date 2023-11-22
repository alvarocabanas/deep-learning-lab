import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple
from torchvision import datasets, transforms
from torchvision.utils import make_grid

# To ensure reproducibility of the experiments, we can set the seed to a fixed number
seed = 123
np.random.seed(seed)
_ = torch.manual_seed(seed)
_ = torch.cuda.manual_seed(seed)

# we select to work on GPU if it is available in the machine, otherwise
# will run on CPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# whenever we send something to the selected device (X.to(device)) we already use
# either CPU or CUDA (GPU). Importantly...
# The .to() operation is in-place for nn.Module's, so network.to(device) suffices
# The .to() operation is NOT in.place for tensors, so we must assign the result
# to some tensor, like: X = X.to(device)

# Let's define some hyper-parameters
hparams = {
    'batch_size': 128,
    'num_epochs': 10,
    'test_batch_size': 64,
    'hidden_size': 128,
    'num_classes': 10,
    'num_inputs': 784,
    'learning_rate': 1e-4,
    'log_interval': 100,
}

# Each of the datasets, mnist_trainset and mnist_testset, is composed by images and labels. 
# The model will be trained with the former and evaluated with the latter. 
# Our images are encoded as Numpy arrays, and the labels are simply an array of digits, 
# ranging from 0 to 9. There is a one-to-one correspondence between the images and the labels.

transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


# Dataset initializations

mnist_trainset = datasets.MNIST(
    root='data',
    train=True,
    download=True,
    transform=transforms)

mnist_testset = datasets.MNIST(
    root='data',
    train=False,
    download=True,
    transform=transforms
)

# Dataloders initialization

train_loader = torch.utils.data.DataLoader(
    dataset=mnist_trainset,
    batch_size=hparams['batch_size'],
    shuffle=True,
    drop_last=True,
)

test_loader = torch.utils.data.DataLoader(
    dataset=mnist_testset,
    batch_size=hparams['test_batch_size'],
    shuffle=False,
    drop_last=True,
)

# We can retrieve a sample from the dataset by simply indexing it
img, label = mnist_trainset[0]
print('Img shape: ', img.shape)
print('Label: ', label)

# Similarly, we can sample a BATCH from the dataloader by running over its iterator
iter_ = iter(train_loader)
bimg, blabel = next(iter_)
print('Batch Img shape: ', bimg.shape)
print('Batch Label shape: ', blabel.shape)
print(f'The Batched tensors return a collection of {bimg.shape[0]} grayscale images ({bimg.shape[1]} channel, {bimg.shape[2]} height pixels, {bimg.shape[3]} width pixels)')
print(f'In the case of the labels, we obtain {blabel.shape[0]} batched integers, one per image')

# make_grid is a function from the torchvision package that transforms a batch
# of images to a grid of images
img_grid = make_grid(bimg)

# show the MNIST images
# plt.figure(figsize=(8, 8))
# plt.imshow(img_grid.permute(1, 2, 0), interpolation='nearest')
# Opens a window with the image
# plt.show()

# Linear layer: y=xA^T+b
network = torch.nn.Sequential(
    nn.Linear(hparams['num_inputs'], hparams['hidden_size']),
    nn.ReLU(),
    nn.Linear(hparams['hidden_size'], hparams['num_classes']),
    nn.LogSoftmax(dim=0),
    )

network.to(device)


def get_nn_nparams(net: torch.nn.Module) -> int:
    """
    Function that returns all parameters regardless of the require_grad value.
    https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/6
    """
    return sum([torch.numel(p) for p in list(net.parameters())])


print(network)
print('Num params: ', get_nn_nparams(network))


# https://pytorch.org/docs/stable/nn.functional.html#nll-loss
criterion = F.nll_loss

# https://pytorch.org/docs/stable/optim.html#torch.optim.RMSprop
optimizer = torch.optim.RMSprop(network.parameters(), lr=hparams['learning_rate'])


def compute_accuracy(predicted_batch: torch.Tensor, label_batch: torch.Tensor) -> int:
    """
    Parameters:
    -----------
    predicted_batch: torch.Tensor shape: [BATCH_SIZE, N_CLASSES]
        Batch of predictions
    label_batch: torch.Tensor shape: [BATCH_SIZE, 1]
        Batch of labels / ground truths.
    """
    pred = predicted_batch.argmax(dim=1, keepdim=True)  # get the index of the max log-probability (predicted class)
    acum = pred.eq(label_batch.view_as(pred)).sum().item()  # compare predicted class index with tthe label_batch
    return acum


def train_epoch(
        train_loader: torch.utils.data.DataLoader,
        network: torch.nn.Module,
        optimizer: torch.optim,
        criterion: torch.nn.functional,
        log_interval: int,
        ) -> Tuple[float, float]:

    # Activate the train=True flag inside the model
    network.train()

    avg_loss = None
    acc = 0.
    avg_weight = 0.1
    for batch_idx, (data, target) in enumerate(train_loader):

        # Move input data and labels to the device
        data, target = data.to(device), target.to(device)

        # Set network gradients to 0.
        optimizer.zero_grad()

        # Rearrange the data dimension
        data = torch.reshape(data, (hparams['batch_size'], -1))

        # Forward batch of images through the network
        output = network(data)

        # Compute loss
        loss = criterion(output, target)

        # Compute backpropagation
        loss.backward()

        # Update parameters of the network
        optimizer.step()

        # Compute metrics
        acc += compute_accuracy(output, target)
        if avg_loss:
            avg_loss = avg_weight * loss.item() + (1 - avg_weight) * avg_loss
        else:
            avg_loss = loss.item()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    avg_acc = 100. * acc / len(train_loader.dataset)

    return avg_loss, avg_acc

@torch.no_grad() # decorator: avoid computing gradients
def test_epoch(
        test_loader: torch.utils.data.DataLoader,
        network: torch.nn.Module,
        ) -> Tuple[float, float]:

    # Dectivate the train=True flag inside the model
    network.eval()

    test_loss = 0
    acc = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        data = torch.reshape(data, (hparams['test_batch_size'], -1))
        # or data = data.view(data.shape[0], -1)

        output = network(data)

        # Apply the loss criterion and accumulate the loss
        test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss

        # WARNING: If you are using older Torch versions, the previous call may need to be replaced by
        # test_loss += criterion(output, target, size_average=False).item()

        # compute number of correct predictions in the batch
        acc += compute_accuracy(output, target)

    test_loss /= len(test_loader.dataset)
    # Average accuracy across all correct predictions batches now
    test_acc = 100. * acc / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, acc, len(test_loader.dataset), test_acc,
        ))
    return test_loss, test_acc


# Init lists to save the evolution of the training & test losses/accuracy.
train_losses = []
test_losses = []
train_accs = []
test_accs = []

# For each epoch
for epoch in range(hparams['num_epochs']):

    # Compute & save the average training loss for the current epoch
    train_loss, train_acc = train_epoch(train_loader, network, optimizer, criterion, hparams["log_interval"])
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # Compute & save the average test loss & accuracy for the current epoch
    test_loss, test_accuracy = test_epoch(test_loader, network)

    test_losses.append(test_loss)
    test_accs.append(test_accuracy)

# Plot the plots of the learning curves
plt.figure(figsize=(10, 8))
plt.subplot(2,1,1)
plt.xlabel('Epoch')
plt.ylabel('NLLLoss')
plt.plot(train_losses, label='train')
plt.plot(test_losses, label='test')
plt.legend()
plt.subplot(2,1,2)
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy [%]')
plt.plot(train_accs, label='train')
plt.plot(test_accs, label='test')
plt.show()
