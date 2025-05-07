import torch
from torch import nn
import torch.utils.data
from torch.nn import functional as F
import timm

from torchvision.transforms import v2 as transforms

import numpy as np

from utils import *
import argparse
import os

from logger import MLLogger
from dataset import KrispyDataset
from gpu_metric_tracker import GPUMetrics

import random

import tqdm


class TrainingApplication():
    """
    The config for training our model
    """
    
    def __init__(self) -> None:
        # The CLU
        args = self._parse_args()

        # Store the arguments
        self.task_name: str      = args.task_name
        self.dataset_pth: str    = args.dataset_path
        self.seed: int           = args.seed
        self.workers: int        = args.workers
        self.lr: float           = args.learning_rate
        self.epochs: int         = args.epochs
        self.weight_decay: float = args.weight_decay
        self.batch_size: int     = args.batch_size

        # Time-optimisation
        torch.set_num_threads(self.workers)
        torch.autograd.set_detect_anomaly(True)

        # Setup a logger
        self.logger = MLLogger(task_name=self.task_name)
        self.gpu_metrics = GPUMetrics(self.logger)
        self.iteration = 0
        self.epoch     = 0

        # Setup torch things
        self._set_seeds(self.seed)
        # torch.distributed.init_process_group(
        #     backend="nccl",
        #     rank=0,
        #     world_size=torch.cuda.device_count()
        # )

        twins = timm.create_model('twins_svt_base.in1k', pretrained=True, num_classes=90, in_chans=256)
        self.model = nn.parallel.DataParallel(twins.cuda())

        # How to transform the data
        self.augmentation_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),               # Randomly flip 50% of the time
            transforms.RandomRotation(degrees=45.0, expand=False) # Randomly rotate ±45° keeping the same dimensions
        ])

        # Data loaders
        self.train_dataset = KrispyDataset(os.path.join(self.dataset_pth, "train"), transform=self.augmentation_transforms)
        self.val_dataset   = KrispyDataset(os.path.join(self.dataset_pth, "val"))

        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, shuffle=True, batch_size=self.batch_size)
        self.val_loader = torch.utils.data.DataLoader(  dataset=self.val_dataset, shuffle=False, batch_size=1)

        # Grad
        self.optimiser = torch.optim.Adam(self.model.parameters(recurse=True), lr=self.lr, weight_decay=self.weight_decay)
        self.loss_func = nn.CrossEntropyLoss().cuda()

        self.scaler = torch.amp.GradScaler(device="cuda")

    def train(self) -> None:
        
        for epoch in range(self.epochs):
            self.epoch = epoch
            
            print(f"Begin epoch {epoch}")
            losses = []
            
            tqdm_trainloader = tqdm.tqdm(self.train_loader, desc=f"Epoch {self.epoch} batches start")
            for i, batch in enumerate(tqdm_trainloader):
                minput = batch['image'].cuda()
                class_id = batch['class_id']
                labels   = batch['labels'].cuda()
                short_names = batch['short_name']

                # Use AMP
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    output = self.model(minput)
                    loss = self.loss_func(output, labels)
                    losses.append(loss.item())

                mean_loss = np.mean(losses)

                self.logger.report_train_loss(current=loss.item(), mean=mean_loss, iteration=self.iteration)
                self.gpu_metrics.gpu_power_report(iteration=self.iteration)
                # Update the TQDM message
                tqdm_trainloader.set_description(f"Train loss {loss.item():0.3f} [{mean_loss:0.3f}]")

                
                # Come out of AMP for backprop
                # Required to scale the loss
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimiser)
                self.scaler.update()

                self.iteration += 1

            torch.save(self.model.state_dict(), f"weights_e_{self.epoch}_i_{self.iteration}.pth")
            print("Starting validation")
            val_metric = self.validation()
            self.logger.report_val_metric(val_metric, self.iteration)
            print(f"Val {val_metric}")

    @torch.no_grad()
    def validation(self) -> float:
        val_metrics = []
        
        tqdm_valloader = tqdm.tqdm(self.val_loader, desc="Validation")
        
        for i, batch in enumerate(tqdm_valloader):
            minput = batch['image'].cuda()
            class_id = batch['class_id']
            labels   = batch['labels'].cuda()
            short_names = batch['short_name']
            
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                output = self.model(minput)
                _, pred_class = torch.max(output, dim=1, keepdim=True)
                val_metrics.append(1.0 if pred_class[0].detach().cpu()==class_id else 0.0)

            self.gpu_metrics.gpu_power_tick()

            # Update the TQDM instance
            tqdm_valloader.set_description(f"Validation {np.mean(val_metrics) * 100:0.1f}%")

        # Get the accuracy
        return np.mean(val_metrics)
            

    def _set_seeds(self, seed) -> None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _parse_args(self) -> argparse.Namespace:
        args = argparse.ArgumentParser(
            prog="KrispiesTrainer",
            description="The training script",
            epilog="Please cite our rice!"
        )

        args.add_argument(
            "--task_name", "-n",
            help="The name of the task (used in logging and saving checkpoints)",
            type=lambda n: n if (n.replace("_", "").isalnum()) else raise_(argparse.ArgumentTypeError(f"Task name '{n}' must be better")),
            default="hyperkrispies")

        args.add_argument(
            "--dataset_path", "-d",
            help="The directory of the dataset to be used for training (and validation)",
            type=lambda d: d if (os.path.isdir(d)) else raise_(NotADirectoryError(d)),
            default="../hyperrice")

        args.add_argument(
            "--seed", "-s",
            help="The seed used to initalise many states",
            type=int,
            default=1)

        args.add_argument(
            "--workers", "-w",
            help="Data loaer workers",
            type=int,
            default=12)

        args.add_argument(
            "--learning_rate", "-lr",
            type=float,
            default=1e-5,
            help="Set the initial* learning rate")

        args.add_argument(
            "--epochs", "-e",
            type=int,
            default=50,
            help="Stop training after e epochs")

        args.add_argument(
            "--batch_size", "-bs",
            type=int,
            default=510,
            help="Batch size to use")

        args.add_argument(
            "--weight_decay", "-wd",
            type=float,
            default=5e-4,
            help="Set the weight decay for the optimiser")

        return args.parse_args()

