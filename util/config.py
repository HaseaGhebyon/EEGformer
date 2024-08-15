from pathlib import Path

def get_config():
    return {
        # MODEL & TRAINER CONFIGURATION
        "batch_size" : 32,
        "num_epoch" : 1000,
        "learning_rate" : 1e-04,
        "epsilon" : 1e-9,
        "num_cls" : 5,
        "transformer_size" : 1,

        "root" : "/raid/data/m13520079/EEGformer",
        "datasource" : "medium_3chan_5st_277dp",
        "model_folder" : "weights",
        "model_basename" : "eegformer_model",
        "experiment_name": "runs/eegformermodel",
        "preload" : "latest",

        "channel_order" : ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'A1', 'A2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'],        
        "selected_channel" : ['C3', 'Cz', 'C4'],
        "seq_len" :120,
        "sliding_step": 1,

        # DATASET CONFIGURATION
        "dataset_dir" : "E:\Ghebyon's\Dataset\Motor Imagery",#"./dataset",
        "low_cut" : 0.57, #Hz
        "high_cut" : 70.0,
        "samp_freq" : 200.0,
        "bandpass_order" : 1,
        "datasample_per_label" : 200,
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

def get_root_database(config):
    return str(Path(config["root"]) / "database")

def get_database_path(config):
    return str(Path(config["root"]) / "database"/ config["datasource"])

def get_eegnpy_file(config):
    return str(Path(config["root"]) / "database" / config["datasource"] / "eeg" / "eeg_data.npy")

def get_imgnpy_file(config):
    return str(Path(config["root"]) / "database" /config["datasource"] / "img" / "imgeeg.npy")

def get_labelnpy_file(config):
    return str(Path(config["root"]) / "database" / config["datasource"] / "labeleeg.npy")