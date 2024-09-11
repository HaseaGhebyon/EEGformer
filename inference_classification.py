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

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

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
        test_data: DataLoader
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        if (self.gpu_id == 0):
            print("\nConfigure training process...")
        
        self.model = model.to(self.gpu_id)
        self.test_data = test_data
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

            TP = torch.diag(conf)
            FP = torch.sum(conf, dim=0) - TP
            FN = torch.sum(conf, dim=1) - TP
            accuracy = torch.sum(TP) / torch.sum(conf)
            recall = TP / (TP + FN)
            precision = TP / (TP + FP)
            f1score = 2 * (precision * recall) / (precision + recall)
            average_accuracy = accuracy.mean()
            average_recall = recall.mean()
            average_precision = precision.mean()
            average_f1score = f1score.mean()
            print("Recall per class:")
            print(recall)
            print("Precision per class:")
            print(precision)
            print("F1-score per class:")
            print(f1score)
            print(
                f"[GPU {self.gpu_id}] Average | Acc: {average_accuracy} \t Precision: {average_precision} \t Recall: {average_recall} \t F1-Score {average_f1score}"
            )

            print(conf)
def load_train_objs():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

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
    return test_dataset, model

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

def main():
    ddp_setup()
    
    test_dataset, model = load_train_objs()
    test_data = prepare_dataloader(test_dataset, 1)
    
    trainer = Trainer(model, test_data)
    trainer.test()
 
    destroy_process_group()

if __name__ == "__main__":
    import argparse
    assert torch.cuda.is_available(), "Testing on CPU is not supported"
    main()
