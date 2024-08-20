Folder ini berisi dataset yang sudah di-olah menjadi file npy. Baik data EEG, Image, dan label.Juga dibedakan berdasarkan file train dan file test 

### Struktur Folder
```
database/
├── <<generated_dataset_1>>
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
├── <<generated_dataset_weights_1>>
│
│
...
│
├── <<generated_dataset_N>>
└── <<generated_dataset_weights_N>>
```

Anda dapat melakukan generasi dataset yang telah diolah dengan melakukan running file <a href="./util/generate_MI_dataset.py">./util/generate_MI_dataset.py</a> pada folder <a href="./util">./util</a>

```sh
cd ./util
python3 generate_MI_dataset.py
```

Catatan : 
Untuk folder weight akan secara otomatis dibuatkan ketika Anda melakukan train pada dataset melalui file ```trainer_classification.py```