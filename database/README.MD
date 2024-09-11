This folder contains datasets that have been processed into npy files. Both EEG, Image and label data are also differentiated based on train files and test files

Notes : 
The weight folder will be automatically created when you train the dataset via the file ```trainer_classification.py```

## Directory Structure
```
database/
├── <<generated_database_1>>
│   │
│   ├── eegtrain
│   │   └──eeg_data.npy
│   │
│   ├── eegtest
│   │   └──eeg_data.npy
│   │
│   ├── imgtest
│   │   └──imgeeg.npy
│   │
│   ├── imgtest
│   │   └──imgeeg.npy
│   │
│   ├── labeleeg_train.npy
│   └── labeleeg_test.npy
│
├── <<generated_database_weights_1>>
│
│
...
│
├── <<generated_database_N>>
└── <<generated_database_weights_N>>
```
<br>

```./generated_database_N``` is a database folder generated from the ```./dataset``` folder. Follow the instructions <a href="../README.md">README</a>

## Quick Start

If you only want the quick start of the model. Rather than you download all of the dataset and generate, I provide the preproceed database here : ... 


