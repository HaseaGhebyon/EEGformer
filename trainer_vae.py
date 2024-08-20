import os
import torch
import torchvision
from tqdm import tqdm
import numpy as np
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from torch.utils.data import DataLoader, Dataset, Subset
import matplotlib.pyplot as plt

import torchvision
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder

from torch.utils.tensorboard import SummaryWriter

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from util.config import get_config, get_eegnpy_train_file, get_labelnpy_train_file, get_eegnpy_test_file, get_labelnpy_test_file, get_logging_folder, get_weights_file_path, get_imgdataset_dir

from model_vae import VAE


config = get_config()

def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def latest_weights_file_path():
    model_folder = f"./VAE_weights"
    model_filename = f"vaemodel*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None

    latest_file = ""
    latest_epoch = 0
    for file in weights_files:
        splitted = str(file).split("_")
        if (int(splitted[-1]) > latest_epoch):
            latest_epoch = int(splitted[-1])
            latest_file = file
    return str(latest_file)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.data = data
        self.optimizer = optimizer
        self.save_every = save_every
        
        self.epochs_run = 0
        self.global_step = 0

        self.preload = config["preload"]

        self.model_filename = latest_weights_file_path() 

        if self.model_filename:
            if (self.gpu_id == 0):
                print(f'Preloading model {self.model_filename}')
            self._load_snapshot(self.model_filename)
        else:
            if (self.gpu_id == 0):
                print("No model to preload, starting from scratch")
        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def kl_divergence_loss(self, q_dist):
        return kl_divergence(
            q_dist, Normal(torch.zeros_like(q_dist.mean), torch.ones_like(q_dist.stddev))
        ).sum(-1)

    def _load_snapshot(self, model_filename):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(model_filename, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        self.global_step = snapshot["GLOBAL_STEP"]

        
    def _save_snapshot(self, epoch):
        model_filename = f"./VAE_weights/vae_{epoch}"
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
            "GLOBAL_STEP" : 0
        }
        torch.save(snapshot, model_filename)
        print(f"Epoch {epoch} | Training snapshot saved at {model_filename}")

    def _save_image(self, epoch, recon_images, source_images, num_images=config["batch_size"], size=(1, 28, 28)):
        plt.subplot(1,2,1)
        recon_image_unflat = recon_images.detach().cpu()
        recon_image_grid = make_grid(recon_image_unflat[:num_images], nrow=8)
        plt.axis('off')
        plt.imshow(recon_image_grid.permute(1, 2, 0).squeeze())
        plt.title("Reconstructed")
        
        plt.subplot(1,2,2)
        source_image_unflat = source_images.detach().cpu()
        source_image_grid = make_grid(source_image_unflat[:num_images], nrow=8)
        plt.axis('off')
        plt.imshow(source_image_grid.permute(1, 2, 0).squeeze())
        plt.title("Original")

        plt.savefig(f'./VAE_generated/{epoch}.png', bbox_inches='tight')


    def train(self, max_epochs: int):

        Path(f"./VAE_weights").mkdir(parents=True, exist_ok=True)
        reconstruction_loss = nn.BCELoss(reduction='sum').to(self.gpu_id)

        for epoch in range(self.epochs_run, max_epochs):
            saved =False
            self.data.sampler.set_epoch(epoch)
            iterator = tqdm(self.data, desc=f"Processing Epoch {epoch:02d}", position=0, leave=True)
            for source, target in iterator:
                source = source.to(self.gpu_id)
                self.optimizer.zero_grad()
                recon_images, encoding = self.model(source)
                # print(source[0])
                # print(recon_images[0])
                loss = reconstruction_loss(recon_images, source) + self.kl_divergence_loss(encoding).sum()
                loss.backward()
                self.optimizer.step()
                if (self.gpu_id == 0):
                    iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

                if not saved:
                    if self.gpu_id == 0 and epoch % self.save_every == 0:
                        self._save_image(epoch, recon_images, source)
                        saved = True
            
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
            
def load_train_objs():
    # LOAD PAIRED IMAGE DATA (RANDOM ASSIGNMENTS)
    transform = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.ToTensor()
        ])

    image_dataset = ImageFolder(root=get_imgdataset_dir(config), transform=transform)
    
    model = VAE()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    
    return image_dataset, model, optimizer

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

def main(save_every: int, total_epochs: int, batch_size: int):
    ddp_setup()
    
    dataset, model, optimizer = load_train_objs()
    data = prepare_dataloader(dataset, batch_size)
   
    print("\nConfigure training process...")
    trainer = Trainer(model, data, optimizer, save_every)
    trainer.train(total_epochs)
 
    destroy_process_group()

if __name__ == "__main__":
    import argparse
    assert torch.cuda.is_available(), "Training on CPU is not supported"
    parser = argparse.ArgumentParser(description='Distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    main(
        args.save_every, 
        args.total_epochs,
        args.batch_size
    )
