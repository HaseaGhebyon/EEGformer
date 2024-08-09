from setuptools import setup, find_packages

setup(
    name='eeg_former',                  # Nama paket
    version='0.1.0',                    # Versi paket
    author='Ghebyon Tohada Nainggolan',                 # Nama penulis
    author_email='ghebyon.tohada@gmail.com',  # Email penulis
    url='https://github.com/HaseaGhebyon/EEGformer',  # URL repositori proyek
    packages=find_packages(),           # Temukan semua paket dalam direktori
    classifiers=[                      # Klasifikasi paket (opsional)
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    install_requires=[                # Daftar ketergantungan paket
        "scipy==1.10.1",
        "numpy",
        "pandas",
        "torch",
        "torcheeg==1.0.11",
        "torch_geometric",
        "tensorboard",
        "imblearn"
    ],
    python_requires='>=3.6',          # Versi Python yang diperlukan
)