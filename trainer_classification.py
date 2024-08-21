import os
import torch
import torchvision
from tqdm import tqdm
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader 
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import Accuracy, Precision, Recall, ConfusionMatrix

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from focal_loss.focal_loss import FocalLoss

from EEGDataset import EEGDataset
from util.config import get_config, get_eegnpy_train_file, get_labelnpy_train_file, get_eegnpy_test_file, get_labelnpy_test_file, get_logging_folder, latest_weights_file_path, get_weights_file_path
from model import build_eegformer


config = get_config()

def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        test_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        if (self.gpu_id == 0):
            print("\nConfigure training process...")
        
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.global_step = 0
        
        self.writer = SummaryWriter(get_logging_folder(config))
        self.preload = config["preload"]


        self.model_filename = latest_weights_file_path(config) if self.preload == 'latest' else get_weights_file_path(config, self.preload) if self.preload else None

        if self.model_filename:
            if (self.gpu_id == 0):
                print(f'Preloading model {self.model_filename}\n')
            self._load_snapshot(self.model_filename)
        else:
            if (self.gpu_id == 0):
                print("No model to preload, starting from scratch")
                Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)
        self.model = DDP(self.model, device_ids=[self.gpu_id])



    def _load_snapshot(self, model_filename):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(model_filename, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"] + 1
        self.global_step = snapshot["GLOBAL_STEP"]

        
    def _save_snapshot(self, epoch):
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
            "GLOBAL_STEP" : 0
        }
        torch.save(snapshot, model_filename)
        print(f"Epoch {epoch} | Training snapshot saved at {model_filename}")


    def train(self, max_epochs: int):
        loss_fn = FocalLoss(gamma=0.7, weights=torch.FloatTensor([3.25, 2.22, 7.95, 7.49, 8.17]).to(self.gpu_id)).to(self.gpu_id)

        for epoch in range(self.epochs_run, max_epochs):
            print(f"[GPU {self.gpu_id}] Training Epoch {epoch}\n")
            
            self.train_data.sampler.set_epoch(epoch)
            train_iterator = tqdm(self.train_data, desc=f"[GPU {self.gpu_id}] Processing Epoch {epoch:02d}", position=0, leave=True) if self.gpu_id == 0 else self.train_data
            
            for source, targets in train_iterator:
                if source.shape[0] != config['batch_size']:
                    continue
                source = source.to(torch.float32).to(self.gpu_id)
                targets = targets.to(self.gpu_id)
                
                self.optimizer.zero_grad()
                source = source.squeeze()
                output1 = self.model.module.construct3D(source)
                output2 = self.model.module.encode(output1)
                output3 = self.model.module.decode(output2)
                loss = loss_fn(output3, targets)

                if (self.gpu_id == 0):
                    train_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
                
                self.writer.add_scalar(f'GPU[{self.gpu_id}] TRAIN LOSS', loss.item(), self.global_step)
                self.writer.flush()
                
                loss.backward()
                self.optimizer.step()
            
            print(f"\n[GPU {self.gpu_id}] Finised Training Epoch {epoch}")
            print(f"[GPU {self.gpu_id}] Testing Epoch {epoch}")
            
            self.test()
            
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
            
    def test(self):
        self.model.eval()
        test_iterator = tqdm(self.test_data, desc=f"[GPU {self.gpu_id}] Processing Test", position=0, leave=True) if self.gpu_id == 0 else self.test_data

        with torch.no_grad():
            evoutputs = torch.zeros(len(test_iterator))
            evlabel = torch.zeros(len(test_iterator))
            idx = 0

            for batch in (test_iterator):
                onedcnn_input = batch[0].squeeze(1).to(torch.float32).to(self.gpu_id)
                output_onedcnn = self.model.module.construct3D(onedcnn_input)
                output_encoder = self.model.module.encode(output_onedcnn)
                output_decoder = self.model.module.decode(output_encoder)
                
                evlabel[idx] = batch[1].detach().cpu().to(torch.int64)
                evoutputs[idx] = torch.argmax(output_decoder.detach().cpu(), dim=-1)
                idx += 1
            
            confMat = ConfusionMatrix(task='multiclass', num_classes=config["num_cls"])
            conf = confMat(evoutputs, evlabel)
            
            metricAcc = Accuracy(task='multiclass', num_classes=config["num_cls"])
            acc = metricAcc(evoutputs, evlabel)
            
            metricPrec = Precision(task='multiclass', num_classes=config["num_cls"])
            prec = metricPrec(evoutputs, evlabel)
            
            metricRec = Recall(task='multiclass', num_classes=config["num_cls"])
            rec = metricRec(evoutputs, evlabel)

            print(
                f"[GPU {self.gpu_id}] Finished Test | Acc: {acc} \t Precision: {prec} \t Recall: {rec}"
            )

def load_train_objs():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    train_dataset = EEGDataset(
        eeg_path=get_eegnpy_train_file(config), 
        labels_path=get_labelnpy_train_file(config), 
        transform_eeg=transform
    )
    test_dataset = EEGDataset(
        eeg_path=get_eegnpy_test_file(config), 
        labels_path=get_labelnpy_test_file(config), 
        transform_eeg=transform
    )
    model = build_eegformer(
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
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], eps=1e-9)
    
    return train_dataset, test_dataset, model, optimizer

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
    
    train_dataset, test_dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(train_dataset, batch_size)
    test_data = prepare_dataloader(test_dataset, 1)
    
    trainer = Trainer(model, train_data, test_data, optimizer, save_every)
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
