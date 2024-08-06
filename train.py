import warnings
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torcheeg import model_selection
from torcheeg.datasets import NumpyDataset
from torcheeg.transforms import MinMaxNormalize, Select, ToTensor, Compose


from config import get_config, get_database_name, get_weights_file_path, latest_weights_file_path
from model import build_eegformer

def dcsm(actuals, predictions):
    # Class = 0,1,2,3,4 ()
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
    print(f"\nAcc : {accuracy} \t Prec : {precision} \t Recall : {recall}")
    return accuracy, precision, recall

def run_validation(model, validation_loader, device, global_step, writer):
    model.eval()
    validation_iterator = tqdm(validation_loader)
    with torch.no_grad():
        evoutputs = torch.zeros(len(validation_loader)).to(device)
        evlabel = torch.zeros(len(validation_loader)).to(device)
        idx = 0
        for batch in (validation_iterator):
            evlabel[idx] = batch[1].to(device)
            onedcnn_input = batch[0].to(device)
            output_onedcnn = model.construct3D(onedcnn_input)
            output_encoder = model.encode(output_onedcnn)
            output_decoder = model.decode(output_encoder)
            evoutputs[idx] = torch.argmax(output_decoder)
            idx += 1
        acc, prec, rec = dcsm(evlabel, evoutputs)
        if writer:
            writer.add_scalar('Accuracy', acc, global_step)
            writer.flush()
            writer.add_scalar('Precision', prec, global_step)
            writer.flush()
            writer.add_scalar('Recall', rec, global_step)
            writer.flush()

def get_dataset(config):
    print("Retrieving dataset from database ...\n")
    X = np.random.randn(100, 32, 128)
    y = {
        'valence': np.random.randint(10, size=100),
        'arousal': np.random.randint(10, size=100)
    }

    dataset = NumpyDataset(X=X,
                       y=y,
                       io_path=config["datasource"],
                       online_transform=ToTensor(),
                       label_transform=Compose([
                           Select('label')
                       ]),
                       num_worker=1,
                       num_samples_per_worker=50)

    train_dataset, val_dataset = model_selection.train_test_split(dataset=dataset, test_size=0.3, random_state=7)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader

def get_model(config):   
    return build_eegformer(
        len(config["selected_channel"]),
        config["seq_len"],
        num_cls = config["num_cls"]
    )

def train_model(config):
    print("\nConfigure training process...")

    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader = get_dataset(config)
    model = get_model(config).to(device)

    writer = SummaryWriter(config['datasource'] + "/" + config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    preload = config["preload"]
    
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        global_step = state['global_step']
    else:
        print("No model to preload, starting from scratch")
    
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epoch']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}", position=0, leave=True)
        for batch in batch_iterator:
            label = batch[1].to(device)
            onedcnn_input = batch[0].to(device) # (Batch, EEGChannel, Seq_Len) or (B, S, L)
            # print("Input Shape : ", onedcnn_input.shape)
            
            output_onedcnn = model.construct3D(onedcnn_input) # (Batch, EEGChannel, ONEDCNNfeature, Seq_Len") or (B, S, C, Le)
            # print("Output 1DCNN Shape : ", output_onedcnn.shape)

            output_encoder = model.encode(output_onedcnn)
            # print("Output Encoder Shape: ", output_encoder.shape)

            output_decoder = model.decode(output_encoder)
            # print("Output Decoder Shape: ", output_decoder.shape)

            loss = loss_fn(output_decoder, label)
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
        
        run_validation(model, val_dataloader, device, global_step)
        
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
if (device == 'cuda'):
    print(f"Device name: {torch.cuda.get_device_name(device.index)}")
    print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
device = torch.device(device)
config = get_config()
train_model(config)