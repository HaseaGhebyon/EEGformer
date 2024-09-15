from pathlib import Path

def get_config():
    return {
        # MODEL & TRAINER CONFIGURATION
        "batch_size" : 32,
        "num_epoch" : 500,
        "learning_rate" : 1e-04,
        "epsilon" : 1e-9,
        "num_cls" : 5,
        "transformer_size" : 1,

        "root" : "/raid/data/m13520079/new_env/EEGformer",
        "datasource" : "./database/21chan_5st_120dp_1step",
        "model_folder" : "weights",
        "model_basename" : "eegformer_model",
        "experiment_name": "runs/eegformermodel",
        "preload" : "latest",

        "channel_order" : ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'A1', 'A2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'],        
        "selected_channel" :  ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'A1', 'A2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'],
        "seq_len" :120,
        "sliding_step": 1,

        # DATASET CONFIGURATION
        "dataset_dir" : "/raid/data/m13520079/new_env/EEGformer/dataset",
        "target_subject" : "GENERAL",
        "low_cut" : 0.57, #Hz
        "high_cut" : 70.0,
        "samp_freq" : 200.0,
        "bandpass_order" : 1,
        "datasample_per_label" : 200,
        "img_dataset_dir" : "imgdataset"
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

    latest_file = ""
    latest_epoch = -1
    for file in weights_files:
        splitted = str(file).split("_")
        if (int(splitted[-1]) > latest_epoch):
            latest_epoch = int(splitted[-1])
            latest_file = file
    return str(latest_file)

def get_root_database(config):
    return str(Path(config["root"]) )

def get_database_path(config):
    return str(Path(config["root"]) / config["datasource"])

def get_logging_folder(config):
    return str(Path(get_database_path(config)) / config["experiment_name"])

def get_imgdataset_dir(config):
    return str(Path(config["root"]) / config["img_dataset_dir"] )


# PATH TO PREPROCESEED DATASET
def get_eegnpy_train_file(config):
    return str(Path(config["root"])  / config["datasource"] / "eegtrain" / "eeg_data.npy")

def get_eegnpy_test_file(config):
    return str(Path(config["root"])  / config["datasource"] / "eegtest" / "eeg_data.npy")

def get_imgnpy_train_file(config):
    return str(Path(config["root"])  /config["datasource"] / "imgtrain" / "imgeeg.npy")

def get_imgnpy_test_file(config):
    return str(Path(config["root"])  /config["datasource"] / "imgtest" / "imgeeg.npy")

def get_labelnpy_train_file(config):
    return str(Path(config["root"])  / config["datasource"] / "labeleeg_train.npy")

def get_labelnpy_test_file(config):
    return str(Path(config["root"])  / config["datasource"] / "labeleeg_test.npy")

