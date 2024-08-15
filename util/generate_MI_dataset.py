import os
import scipy.io
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn import model_selection
from datetime import datetime
from scipy.signal import butter, lfilter

import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset

from config import get_config, get_database_path, get_eegnpy_train_file, get_eegnpy_test_file,  get_labelnpy_train_file, get_labelnpy_test_file, get_root_database, get_imgdataset_dir, get_imgnpy_train_file, get_imgnpy_test_file

LIST_SUBJECT = {
    "A" : {"gender" : "MALE", "age" : "20-25"},
    "B" : {"gender" : "MALE", "age" : "20-25"},
    "C" : {"gender" : "MALE", "age" : "25-30"},
    "D" : {"gender" : "MALE", "age" : "20-25"},
    "E" : {"gender" : "FEMALE", "age" : "20-25"},
    "F" : {"gender" : "FEMALE", "age" : "20-25"},
    "G" : {"gender" : "MALE", "age" : "30-35"},
    "H" : {"gender" : "MALE", "age" : "20-25" },
    "I" : {"gender" : "FEMALE", "age" : "20-25"},
    "J" : {"gender" : "FEMALE", "age" : "20-25"},
    "K" : {"gender" : "MALE", "age" : "20-25"},
    "L" : {"gender" : "FEMALE", "age" : "20-25"},
    "J" : {"gender" : "FEMALE", "age" : "20-25"},
    "M" : {"gender" : "FEMALE","age" : "20-25"},
}
PARADIGM = {
    "5F" : "FIVE FINGERS",
    "CLA" : "CLASSIC",
    "FREEFORM" : "FREE STYLE 5F",
    "HALT" : "HAND LEG TONGUE",
    "NOMT" : "NO MOTOR"
}

def get_channel_size(config):
    return len(config["channel_order"])

def get_experiment_spec(filename: str):
    list_spec = filename.upper().replace('.','-').split('-')
    paradigm_code = list_spec[0]
    subject_name = list_spec[1]
    subject_detail = LIST_SUBJECT[subject_name[-1]]
    date_str = list_spec[2]
    date_object = datetime.strptime(date_str, '%y%m%d').date()
    total_states = int(list_spec[3][0])
    
    if "HFREQ" in list_spec:
        sampfreq = 1000
    else:
        sampfreq = 200 #default

    custom_bci = False
    if "INTER" in list_spec:
        custom_bci = True
    
    return {
        "subject" : {
            "name" : subject_name,
            "detail" : subject_detail
        },
        "task" : PARADIGM[paradigm_code],
        "meas_date" : date_object,
        "custom_bci" : custom_bci,
        "total_states" : total_states,
        "samp_freq" : sampfreq
    }

def get_files_mat(config):
    files = []
    for file in os.listdir(config["dataset_dir"]):
        # check the files which are end with specific extension
        if file.endswith(".mat"):
            # print path name of selected files
            files.append(os.path.join(config["dataset_dir"], file))
    return files

def load_mat_file(filename):
    return scipy.io.loadmat(filename)


def butter_bandpass(lowcut, highcut, fs, order=1):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=1):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


# GENERATING DATASET

X = []
y = []

config = get_config()
datafile = get_files_mat(config)
file_iterator = tqdm(datafile, desc=f"Retrieving Data", position=0, leave=True)
for file_path in file_iterator:
    fn = os.path.basename(file_path).split('\\')[-1]
    experiment = get_experiment_spec(fn)
    mat = load_mat_file(file_path)
    
    # Freeform tidak di handle pada preprocessing untuk meningkatkan generalisasi model
    # Labeling 5F bentrok dengan paradigma lain
    # Jika 5F, Berikan labeling yang bersifat global
    if experiment['task'] == "FIVE FINGERS" or experiment['task'] == "NO MOTOR":
        continue

    eeg_data = mat['o']['data'][0][0].transpose()
    label_data = mat['o']['marker'][0][0].transpose()

    # Drop channel synchronization X3 atau X5 <- pada index terakbir
    if eeg_data.shape[0] == 22:
        eeg_data = np.delete(eeg_data, 21, 0)
    
    channel_select_idx = [i for i, n in enumerate(config["channel_order"]) if n in config["selected_channel"]]
    eeg_data = eeg_data[channel_select_idx, :]
    
    # Lakukan Filter Bandpass jika diperlukan. Harus dilakukan sebelum Resampling (Downsampling)
    for idx, channel in enumerate(eeg_data):
        eeg_data[idx] = butter_bandpass_filter(channel, 0.57, 70.0, 200, 1)
    # Downsampling sinyal EEG HFREQ (1000Hz menjadi 200 Hz)
    if experiment["samp_freq"] == 1000:
        eeg_data = eeg_data[:,::5]
        label_data = label_data[:,::5]

    
    # Data dengan BCI yang berbeda (Inter) <- Asumsi di Handle oleh Normalisasi data
    '''
    Inter are the data recorded for the indicated interaction paradigm by using the custom interactive brain-computer interface software. That data differs from the other files by a different signal resolution and signal's dynamic range, namely the signal resolution of 0.133uV vs. 0.01uV in the other data, and the dynamic range of +/-121.6uV vs. better than +/-2mV in the rest of the data.
    '''
    # Apply FFT
    # eeg_data = np.abs(np.fft.fft(eeg_data, axis=1))
    # # Normalize the FFT data by channel
    # eeg_data = (eeg_data - np.mean(eeg_data, axis=1, keepdims=True)) / np.std(eeg_data, axis=1, keepdims=True)

    # Sampling 277 data point, pemilihan 277 berdasarkan model yang dibentuk
    # 77 data sebelum mulai, 200 data setelah mulai
    idx_state_change = []
    
    # From passive state to active state
    for i in range(len(label_data[0]) - 1):
        chosen_active_state = [1, 2, 4, 5, 6]
        if label_data[0][i] != label_data[0][i+1] and label_data[0][i] not in chosen_active_state and label_data[0][i+1] in chosen_active_state: # <- passive state
            idx_state_change.append(i+1)

    for idx in idx_state_change:
        sample = eeg_data[:, idx:idx+200]
        label = label_data[0][idx]

        # Lakukan fast fourier transform disini

        if label == 1:
            label = 0
        elif label == 2:
            label = 1
        elif label == 4:
            label = 2
        elif label ==  5:
            label = 3
        elif label == 6:
            label = 4

        X.append(sample)
        y.append(label)


print("\nFinished Retrieving Sample.\n")
print("Split train and test dataset...")
Xtrain, Xtest, ytrain, ytest = model_selection.train_test_split(X, y, test_size=0.3, stratify=y)

print("Composition :")
print("Train Data size : ", len(ytrain))
print("Test Data size : ", len(ytest))


new_Xtrain = []
new_ytrain = []
for sample, label in tqdm(zip(Xtrain, ytrain), desc=f"Sliding across temporal train data", position=0, leave=True):
    temp_chan = []
    for channel in sample:
        all_patches = np.lib.stride_tricks.sliding_window_view(channel, window_shape=(config["seq_len"],))
        stride_indices = np.arange(0, all_patches.shape[0], config["sliding_step"])
        temp_chan.append(all_patches[stride_indices])
    for i in range(len(stride_indices)):
        new_ytrain.append(label)
    temp_chan = np.array(temp_chan)
    new_Xtrain += np.transpose(temp_chan, [1,0,2]).tolist()
Xtrain = []
ytrain = []

new_Xtest = []
new_ytest = []
for sample, label in tqdm(zip(Xtest, ytest), desc=f"Sliding across temporal test data", position=0, leave=True):
    temp_chan = []
    for channel in sample:
        all_patches = np.lib.stride_tricks.sliding_window_view(channel, window_shape=(config["seq_len"],))
        stride_indices = np.arange(0, all_patches.shape[0], config["sliding_step"])
        temp_chan.append(all_patches[stride_indices])
    for i in range(len(stride_indices)):
        new_ytest.append(label)
    temp_chan = np.array(temp_chan)
    new_Xtest += np.transpose(temp_chan, [1,0,2]).tolist()
Xtest = []
ytest = []


unique, counts = np.unique(y, return_counts=True)
print("Composition Data After Sliding Window ")
print("Train Data size : ", len(new_ytrain))
print("Test Data size : ", len(new_ytest))
print(dict(zip(unique, counts)))



root_db = get_root_database(config)
path_db = get_database_path(config)
path_eegnpy_train = get_eegnpy_train_file(config)
path_eegnpy_test = get_eegnpy_test_file(config)
path_labelnpy_train = get_labelnpy_train_file(config)
path_labelnpy_test = get_labelnpy_test_file(config)


Path(root_db).mkdir(parents=True, exist_ok=True)
Path(path_db).mkdir(parents=True, exist_ok=True)

Path(str(Path(path_db) / "eegtrain")).mkdir(parents=True, exist_ok=True)
Path(str(Path(path_db) / "eegtest")).mkdir(parents=True, exist_ok=True)


new_Xtrain = np.array(new_Xtrain)
np.save(path_eegnpy_train, new_Xtrain)

new_Xtest = np.array(new_Xtest)
np.save(path_eegnpy_test, new_Xtest)


new_ytrain = np.array(new_ytrain)
np.save(path_labelnpy_test, new_ytrain)

new_ytest = np.array(new_ytest)
np.save(path_labelnpy_test, new_ytest)


# path_imgnpy_train = get_imgnpy_train_file(config)
# path_imgnpy_test = get_imgnpy_test_file(config)


# # LOAD PAIRED IMAGE DATA (RANDOM ASSIGNMENTS)
# transform = torchvision.transforms.Compose([
#         torchvision.transforms.Grayscale(num_output_channels=1),
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize((0.5,), (0.5,)),
#     ])

# image_dataset = ImageFolder(root=get_imgdataset_dir(config), transform=transform)
# result = []
# for label in tqdm(new_ytrain):
#     loader = DataLoader(image_dataset, batch_size=1, shuffle=True)
#     for image, lbl in loader:
#         if lbl == label:
#             result.append(image.squeeze(dim=0))
#             break
# Path(str(Path(path_db) / "imgtrain")).mkdir(parents=True, exist_ok=True)
# img = np.array(result)
# np.save(path_imgnpy_train, img)


# image_dataset = ImageFolder(root=get_imgdataset_dir(config), transform=transform)
# result = []
# for label in tqdm(new_ytest):
#     loader = DataLoader(image_dataset, batch_size=1, shuffle=True)
#     for image, lbl in loader:
#         if lbl == label:
#             result.append(image.squeeze(dim=0))
#             break
# Path(str(Path(path_db) / "imgtest")).mkdir(parents=True, exist_ok=True)
# img = np.array(result)
# np.save(path_imgnpy_test, img)
