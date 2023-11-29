import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from typing import Optional
from Logger import Logger
from utils import TaskType
from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger(Logger):

    def __init__(
        self, 
        task: TaskType, 
    ):
        # Define the folder where we will store all the tensorboard logs
        logdir = os.path.join("logs", f"{task}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")

        # TODO: Initialize Tensorboard Writer with the previous folder 'logdir'
        self.writer = SummaryWriter(log_dir=logdir)


    def log_reconstruction_training(
        self, 
        model: nn.Module, 
        epoch: int, 
        train_loss_avg: np.ndarray,
        val_loss_avg: np.ndarray,
        reconstruction_grid: Optional[torch.Tensor] = None,
    ):

        # TODO: Log train reconstruction loss to tensorboard.
        #  Tip: use "Reconstruction/train_loss" as tag
        self.writer.add_scalar("Reconstruction/train_loss", train_loss_avg, epoch)


        # TODO: Log validation reconstruction loss to tensorboard.
        #  Tip: use "Reconstruction/val_loss" as tag
        self.writer.add_scalar("Reconstruction/val_loss", val_loss_avg, epoch)


        # TODO: Log a batch of reconstructed images from the validation set.
        #  Use the reconstruction_grid variable returned above.
        self.writer.add_image("Reconstruction/grid", reconstruction_grid, epoch)

        # TODO: Log the weights values and grads histograms.
        #  Tip: use f"{name}/value" and f"{name}/grad" as tags
        for name, weight in model.encoder.named_parameters():
            self.writer.add_histogram(f"{name}/value", weight, epoch)
            self.writer.add_histogram(f"{name}/grad", weight.grad, epoch)



    def log_classification_training(
        self, 
        epoch: int,
        train_loss_avg: np.ndarray,
        val_loss_avg: np.ndarray,
        train_acc_avg: np.ndarray,
        val_acc_avg: np.ndarray,
        fig: plt.Figure,
    ):
        # TODO: Log confusion matrix figure to tensorboard
        self.writer.add_figure("Confusion_Matrix", fig, epoch)

        # TODO: Log validation loss to tensorboard.
        #  Tip: use "Classification/val_loss" as tag
        self.writer.add_scalar("Classification/val_loss", val_loss_avg, epoch)


        # TODO: Log validation accuracy to tensorboard.
        #  Tip: use "Classification/val_acc" as tag
        self.writer.add_scalar("Classification/val_acc", val_acc_avg, epoch)

        # TODO: Log training loss to tensorboard.
        #  Tip: use "Classification/train_loss" as tag
        self.writer.add_scalar("Classification/train_loss", train_loss_avg, epoch)


        # TODO: Log training accuracy to tensorboard.
        #  Tip: use "Classification/train_acc" as tag
        self.writer.add_scalar("Classification/train_acc", train_acc_avg, epoch)


    def log_model_graph(
        self, 
        model: nn.Module, 
        train_loader: torch.utils.data.DataLoader,
    ):
        batch, _ = next(iter(train_loader))
        """
        TODO:
        We are going to log the graph of the model to Tensorboard. For that, we need to
        provide an instance of the model and a batch of images, like you'd
        do in a forward pass.
        """
        self.writer.add_graph(model, batch)




    def log_embeddings(
        self, 
        model: nn.Module, 
        train_loader: torch.utils.data.DataLoader,
    ):
        list_latent = []
        list_images = []
        for i in range(10):
            batch, _ = next(iter(train_loader))

            # forward batch through the encoder
            list_latent.append(model.encoder(batch))
            list_images.append(batch)

        latent = torch.cat(list_latent)
        images = torch.cat(list_images)

        # TODO: Log latent representations (embeddings) with their corresponding labels (images)

        self.writer.add_embedding(mat=latent, label_img=images)

        # Be patient! Projector logs can take a while

