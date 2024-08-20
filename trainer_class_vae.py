import os
import torch
import torchvision
from tqdm import tqdm
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader 
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import Accuracy, Precision, Recall, ConfusionMatrix

from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from focal_loss.focal_loss import FocalLoss


from EEGImageDataset import EEGImageDataset
from util.config import get_config, get_eegnpy_train_file, get_labelnpy_train_file, get_eegnpy_test_file, get_labelnpy_test_file, get_logging_folder, latest_weights_file_path, get_weights_file_path, get_imgdataset_dir, get_imgnpy_train_file, get_imgnpy_test_file

from model import build_eegformer
from model_vae import VAE

config = get_config()

class ProjectionLayer(nn.Module):
    def __init__(self):
        super(ProjectionLayer, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(7*2662, 1024)
        self.linear2 = nn.Linear(1024, 256)
        self.linear3 = nn.Linear(256, 32)

        
    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x
    

def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

class Trainer:
    def __init__(
        self,
        model_projection: torch.nn.Module,
        model_eeg: torch.nn.Module,
        model_vae: torch.nn.Module,
        data: DataLoader,
        test_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model_projection = model_projection.to(self.gpu_id)
        self.model_eeg = model_eeg.to(self.gpu_id)
        self.model_vae = model_vae.to(self.gpu_id)
        
        self.data = data
        self.test_data = test_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.global_step = 0

        self.writer = SummaryWriter(get_logging_folder(config))
        self.preload = config["preload"]

        # LOAD PRETRAINED CLASSIFICATION EEG
        self.model_filename = latest_weights_file_path(config) if self.preload == 'latest' else get_weights_file_path(config, self.preload) if self.preload else None
        if self.model_filename:
            print(f'Preloading model {self.model_filename}')
            self._load_snapshot_classification(self.model_filename)
        else:
            print("Please Insert the pretrained EEGformer")
        
        # LOAD PRETRAINED VAE
        self.model_vae_filename = "./VAE_weights/vae_190"
        if self.model_vae_filename:
            self._load_snapshot_vae(self.model_vae_filename)
        else:
            print("Please Insert the pretrained VAE")

        # LOAD PROJECTION SNAPSHOT
        self.model_projection_filename = None
        if self.model_projection_filename:
            self._load_snapshot_proj(self.model_projection_filename)
        else:
            if (self.gpu_id == 0):
                print("No model to preload, starting from scratch")

        self.model_projection = DDP(self.model_projection, device_ids=[self.gpu_id])
        

    def _load_snapshot_classification(self, model_filename):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(model_filename, map_location=loc)
        self.model_eeg.load_state_dict(snapshot["MODEL_STATE"])
        
    def _load_snapshot_vae(self, model_filename):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(model_filename, map_location=loc)
        self.model_vae.load_state_dict(snapshot["MODEL_STATE"])

    def _load_snapshot_proj(self, model_filename):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(model_filename, map_location=loc)
        self.model_projection.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        self.global_step = snapshot["GLOBAL_STEP"]
        

    def _save_snapshot(self, epoch):
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        snapshot = {
            "MODEL_STATE": self.model_proj.module.state_dict(),
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

        plt.savefig(f'./Result_generated/epoch_{epoch}.png', bbox_inches='tight')

    def train(self, max_epochs: int):

        Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)
        loss_fn = nn.MSELoss()

        for epoch in range(self.epochs_run, max_epochs):
            self.data.sampler.set_epoch(epoch)
            # saved=False
            train_iterator = tqdm(self.data, desc=f"Processing Epoch {epoch:02d}", position=0, leave=True)
            for source_eeg, source_img, targets in train_iterator:
                self.optimizer.zero_grad()
                
                if source_eeg.shape[0] != config['batch_size']:
                    continue
                source_eeg = source_eeg.to(torch.float32).to(self.gpu_id)
                source_img = source_img.to(torch.float32).to(self.gpu_id)
                targets = targets.to(self.gpu_id)


                # EEG FORMER
                source_eeg = source_eeg.squeeze()
                output1 = self.model_eeg.construct3D(source_eeg)
                output2 = self.model_eeg.encode(output1)

                latent_space = self.model_vae.encode(source_img)
                
                output3 = self.model_projection(output2)
                
                loss = loss_fn(output3, latent_space)

                if (self.gpu_id == 0):
                    train_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
                
                # self.writer.add_scalar(f'GPU[{self.gpu_id}] TRAIN LOSS', loss.item(), self.global_step)
                # self.writer.flush()
                
                loss.backward()
                self.optimizer.step()

                # if not saved:
                #     if self.gpu_id == 0 and epoch % self.save_every == 0:
                #         recon_images = self.model_vae.decode(output3)
                #         self._save_image(epoch, recon_images, source_img)
                #         saved = True
            
            for source_eeg, source_img, targets in self.test_data:
                with torch.no_grad():
                    if source_eeg.shape[0] != config['batch_size']:
                        continue
                    source_eeg = source_eeg.to(torch.float32).to(self.gpu_id)
                    source_img = source_img.to(torch.float32).to(self.gpu_id)
                    targets = targets.to(self.gpu_id)

                    # EEG FORMER
                    source_eeg = source_eeg.squeeze()
                    output1 = self.model_eeg.construct3D(source_eeg)
                    output2 = self.model_eeg.encode(output1)

                    latent_space = self.model_vae.encode(source_img)
                
                    output3 = self.model_projection(output2)

                    if self.gpu_id == 0 and epoch % self.save_every == 0:
                        recon_images = self.model_vae.decode(output3)
                        self._save_image(epoch, recon_images, source_img)
                        break
            # if self.gpu_id == 0 and epoch % self.save_every == 0:
            #     self._save_snapshot(epoch)

def load_train_objs():
    transform_eeg = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    transform_img = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    

    dataset = EEGImageDataset(
        eeg_path=get_eegnpy_train_file(config),
        images_path=get_imgnpy_train_file(config),
        labels_path=get_labelnpy_train_file(config), 
        transform_eeg=transform_eeg,
        transform_img=transform_img
    )
    test_dataset = EEGImageDataset(
        eeg_path=get_eegnpy_test_file(config),
        images_path=get_imgnpy_test_file(config),
        labels_path=get_labelnpy_test_file(config), 
        transform_eeg=transform_eeg,
        transform_img=transform_img
    )
    
    model_eegformer = build_eegformer(
        channel_size=len(config["selected_channel"]),
        seq_len=config["seq_len"],
        N=config["transformer_size"],
        feature_onedcnn=120,
        kernel_size=9,
        h_regional=6,
        h_sync=6,
        h_temp=11,
        dropout=0.1,
        sub_matrices=6,
        feature_decoder=2,
        num_cls=5,
        scaler_ffn=4
    )
    model_vae = VAE()

    model_proj = ProjectionLayer()

    optimizer = torch.optim.Adam(model_proj.parameters(), lr=config['learning_rate'], eps=1e-9)
    
    return dataset, test_dataset, model_eegformer, model_vae, model_proj, optimizer

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
    
    dataset, test_dataset, model_eegformer, model_vae, model_proj, optimizer = load_train_objs()
    data = prepare_dataloader(dataset, batch_size)
    data_test = prepare_dataloader(test_dataset, batch_size)
    # test_data = prepare_dataloader(test_dataset, 1)
    print("\nConfigure training process...")
    trainer = Trainer(
        model_proj,
        model_eegformer, 
        model_vae, 
        data,
        data_test,
        optimizer, 
        save_every
    )
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