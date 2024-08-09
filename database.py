import os
import numpy as np
from tqdm import tqdm
import pywt

import scipy.io
from scipy.signal import butter, lfilter

from sklearn import model_selection
from torcheeg.datasets import NumpyDataset
from torcheeg import transforms
from imblearn.under_sampling import RandomUnderSampler

def calcOnset(position, sampRate=200):
    return position/sampRate

def calcDuration(relativePos, sampRate=200):
    return relativePos/sampRate

def getAnnotationComponent(marker, sampRate):
    onset = []
    duration = [] # Dalam rencana Record menggunakan 1 second
    description= []
    count = 0
    count_current = 0
    prev = -1
    for event_id in (marker):
        event_id = int(event_id[0])
        if (prev in [1,2,4,5,6]):
            if (event_id in [1,2,4,5,6]):
                if (prev == event_id):
                    count_current += 1
                else:
                    pass
            else:
                duration.append(calcDuration(count_current, sampRate))
        else:
            if (event_id in [1,2,4,5,6]):
                onset.append(calcOnset(count, sampRate))
                description.append(event_id)
                count_current = 0
            else:
                count_current = 0

        prev = event_id
        count +=1
    return onset, duration, description

# GET SAMPLING DATA WITH SAME SIZE + SAME SAMPLING RATE : 200 HZ
def getData(config, marker, data, sampling_freq=200):
    prev = -1
    list_idx = []
    label = []
    temp = []

    # print("Data : \n", data)
    # Apply FFT
    # fft_data = np.abs(np.fft.fft(data, axis=1))
    # print("FFT : \n",fft_data)
    # Normalize the FFT data by channel
    # normalized_data = (fft_data - np.mean(fft_data, axis=1, keepdims=True)) / np.std(fft_data, axis=1, keepdims=True)
    # print("Normalized : \n",normalized_data)


    for idx, event_id in enumerate(marker):      
        event_id = int(event_id[0])
        if (prev in [1,2,4,5,6]):
            if (event_id in [1,2,4,5,6]):
                if (prev == event_id):
                    pass
                else:
                    pass
            else:
                temp.append(temp[0] + config['seq_len'])
                list_idx.append(temp)
                temp = []
                label.append(prev)
        else:
            if (event_id in [1,2,4,5,6]):
                temp.append(idx)
            else:
                pass
        prev = event_id
    
    X = []
    y = []
    for i in range (len(label)):
        if sampling_freq == 200:
            sample = data[:,list_idx[i][0]:list_idx[i][0]+config['seq_len']]
        else: # sampling_freq == 1000
            sample = data[:,list_idx[i][0]:list_idx[i][0]+2000:5]
        sample = sample[:, :config['seq_len']]
            
        
        X.append(sample)
        y.append(label[i])

    X = np.array(X)
    y = np.array(y)
    
    return X, y


def butter_bandpass(lowcut, highcut, fs, order=1):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=1):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def generate_dataset(config):
    # ==== LISTING ALL DATASET database
    files = []
    for file in os.listdir(config["ds_dir"]):
        # check the files which are end with specific extension
        if file.endswith(".mat"):
            files.append(os.path.join(config["ds_dir"], file))
    
    # ===== INITIALIZE EMPTY NP ARRAY
    X = np.empty([1, 21, config['seq_len']], dtype="float64")
    X = np.delete(X, 0, 0)

    lab = np.empty([1], dtype="int")
    lab = np.delete(lab, 0, 0)
    y = {
        'label' : lab
    }

    # ===== INSERTING DATA
    for fn in tqdm(files):

        # SUBJECT IDENTIFIER
        sub_idx = fn.find("Subject")
        subject = fn[sub_idx:sub_idx+8]
        
        # SAMPLING RATE
        if "HFREQ" in fn:
            sampling_freq = 1000
        else:
            sampling_freq = 200
        
        mat = scipy.io.loadmat(fn)

        # Filter Data from Channel X3 or X5
        data =  mat['o']['data'][0][0].transpose()
        if len(data) == 22:
            data = np.delete(data,21,0)

        x_data, y_data = getData(config, mat['o']['marker'][0][0], data, sampling_freq=sampling_freq)
        
        # AVOID IMPORT 
        if not np.any(np.isnan(x_data)):
            X = np.append(X, x_data, axis=0)
            y['label'] = np.append(y['label'], y_data, 0)

    # ====== UNDER SAMPLING DATA
    num_samples, num_timesteps, num_features = X.shape
    X_reshaped = X.reshape((num_samples, num_timesteps * num_features))
    y_temp = y['label']
    undersample = RandomUnderSampler(
        sampling_strategy={1: 5522, 
                        2: 5522, 
                        4: 5522,
                        5: 5522,
                        6: 5522})
    X_res, y_res = undersample.fit_resample(X_reshaped, y_temp)

    X_res_reshaped = X_res.reshape((X_res.shape[0], num_timesteps, num_features))

    # ====== CHANNEL FILTERING
    select_idx = [i for i, n in enumerate(config["channel_order"]) if n in config["selected_channel"]]
    X_res_chan_rem = X_res_reshaped[:, select_idx]
    X_res_chan_rem.shape

    X_res_band_filt = np.copy(X_res_chan_rem)
    for i, sample in enumerate(X_res_chan_rem):
        for j, chan in enumerate(sample):
            X_res_band_filt[i,j] = butter_bandpass_filter(chan, config['low_cut'], config['high_cut'], config['samp_freq'], config["bandpass_order"])

    # ===== SORTED INDEXING FOR LABEL
    y_res_lab_change = np.copy(y_res)
    y_res_lab_change[y_res_lab_change == 1] = 0
    y_res_lab_change[y_res_lab_change == 2] = 1
    y_res_lab_change[y_res_lab_change == 4] = 2
    y_res_lab_change[y_res_lab_change == 5] = 3
    y_res_lab_change[y_res_lab_change == 6] = 4
    y_res_lab_change


    X_train_dataset, X_eval_dataset, y_train_dataset, y_eval_dataset = model_selection.train_test_split(X_res_band_filt, y_res_lab_change, test_size=0.2, stratify=y_res)

    
    X_new = np.concatenate((X_train_dataset, X_eval_dataset))
    y_temp = np.concatenate((y_train_dataset, y_eval_dataset))
    y['label'] = y_temp
    
    return X_new, y
    
    
if __name__ == '__main__':
    from config import get_config, get_database_name
    config = get_config()
    X, y = generate_dataset(config)

    output_name = get_database_name(config, "")

    database = NumpyDataset(X=X,
                        y=y,
                        io_path=f"./database/{output_name}",
                        online_transform=transforms.ToTensor(),
                        label_transform=transforms.Compose([
                            transforms.Select('label')
                        ]),
                        num_worker=1,
                        num_samples_per_worker=50)
    print(database[0])