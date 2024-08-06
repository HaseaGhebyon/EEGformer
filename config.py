from pathlib import Path

def get_config():
    return {
        "batch_size" : 8,
        "num_epoch" : 200,
        "learning_rate" : 1e-05,
        "epsilon" : 1e-9,
        "datasource" : "/content/drive/Shareddrives/Tugas Akhir/3chan_bpf_377dp",
        "model_folder" : "weights",
        "model_basename" : "eegformer_model",
        "preload" : "latest",
        "num_cls" : 5,
        "experiment_name": "runs/eegformermodel",
        
        # DATASET CONFIGURATION 
        "ds_dir" : "E:Ghebyon's/Dataset/Motor Imagery",
        "seq_len" : 377,
        "channel_order" : ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'A1', 'A2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'],
        "selected_channel" : ['C3', 'C4', 'Cz'],
        "low_cut" : 0.57, #Hz
        "high_cut" : 70.0,
        "bandpass_order" : 1,
        "samp_freq" : 200.0,
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}_{epoch}"
    return str(Path('.') / model_folder / model_filename)

def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])

def get_database_name(config, additional:str=None):
    chan_size = len(config["selected_channel"])
    seq_len = config["seq_len"]
    return f"{chan_size}chan_{seq_len}dp_{additional}"