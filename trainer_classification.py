import os
import torch
import torchvision
from tqdm import tqdm
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader 
from torch.utils.tensorboard import SummaryWriter

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
            print(f'Preloading model {self.model_filename}')
            self._load_snapshot(self.model_filename)
        else:
            print("No model to preload, starting from scratch")
        self.model = DDP(self.model, device_ids=[self.gpu_id])



    def _load_snapshot(self, model_filename):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(model_filename, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
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

        Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)
        loss_fn = FocalLoss(gamma=0.6, weights=torch.FloatTensor([0.2, 0.2, 0.5, 0.5, 0.5]))

        for epoch in range(self.epochs_run, max_epochs):
            b_sz = len(next(iter(self.train_data))[0])
            print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
            self.train_data.sampler.set_epoch(epoch)
            train_iterator = tqdm(self.train_data, desc=f"Processing Epoch {epoch:02d}", position=0, leave=True)
            for source, targets in train_iterator:
                source = source.to(self.gpu_id)
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
            
            self._test()
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
            
    def _test(self):
        self.model.eval()
        test_iterator = tqdm(self.test_data)
        with torch.no_grad():
            evoutputs = torch.zeros(len(test_iterator))
            evlabel = torch.zeros(len(test_iterator))
            idx = 0
            for batch in (test_iterator):
                evlabel[idx] = batch[1].to(self.gpu_id).to(torch.int64)
                onedcnn_input = batch[0].to(self.gpu_id).squeeze(1)
                output_onedcnn = self.model.module.construct3D(onedcnn_input)
                output_encoder = self.model.module.encode(output_onedcnn)
                output_decoder = self.model.module.decode(output_encoder)
                evoutputs[idx] = torch.argmax(output_decoder)

            acc, prec, rec = self.calculate_dcsm(evlabel, evoutputs)
            if self.writer:
                self.writer.add_scalar(f'GPU[{self.gpu_id}] ACCURACY', acc, self.global_step)
                self.writer.flush()
                self.writer.add_scalar(f'GPU[{self.gpu_id}] PRECISION', prec, self.global_step)
                self.writer.flush()
                self.writer.add_scalar(f'GPU[{self.gpu_id}] RECALL', rec, self.global_step)
                self.writer.flush()

    def calculate_dcsm(self):
        actuals = actuals.cpu().detach().numpy()
        predictions = predictions.cpu().detach().numpy()
        actual_predict = np.zeros((5,5), dtype=int)
        for act, pred in zip(actuals, predictions):
            actual_predict[int(act), int(pred)] += 1
        tp_all = 0
        for i in range(5):
            tp_all += actual_predict[i, i]
        accuracy = tp_all/len(actuals)

        # [] : TP, FN, FP, TN for every class
        conf_mat = np.zeros((5,4), dtype=int)
        for cls in range(5):
            conf_mat[cls, 0] = actual_predict[cls, cls]
            for i in range(1,5,1):
                conf_mat[cls, 1] += actual_predict[cls, i]
            for i in range(5):
                if (i != cls):
                    conf_mat[cls, 2] += actual_predict[i, cls]
            for i in range(5):
                for j in range(5):
                    if (i != cls and j != cls):
                        conf_mat[cls, 3] += actual_predict[i, j]
        
        precision = 0
        recall = 0
        for i in range(5):
            temp_prec = conf_mat[i, 0]/(conf_mat[i, 0] + conf_mat[i, 2])
            precision += temp_prec
            temp_rec = conf_mat[i, 0]/(conf_mat[i, 0] + conf_mat[i, 1])
            recall += temp_rec
        precision = precision/5
        recall = recall/5

        if (self.gpu_id == 0):
            print(f"\nAcc : {accuracy} \t Prec : {precision} \t Recall : {recall}")
            return accuracy, precision, recall

def load_train_objs():
    print("Retrieving dataset from database ...\n")
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
        config=len(config["selected_channel"]),
        seq_len=config["seq_len"],
        N=config["transformer_size"],
        feature_onedcnn=120,
        kernel_size=10,
        h_regional=5,
        h_sync=5,
        h_temp=10,
        dropout=0.1,
        sub_matrices=10,
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
    print("\nConfigure training process...")
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