#!/usr/bin/env python
# coding: utf-8

# In[ ]:


__version__ = "1.0"
__author__ = "Johnny Dorigo Jones"

import requests
import os
from pathlib import Path
from Global21cmLSTM import __path__

PACKAGE_PATH = __file__[: -len("__init__.py")] #__path__[0] + "/"
BASE_PATH = f"{os.environ.get('HOME')}/.Global21cmLSTM/"

if not os.path.exists(BASE_PATH):
   os.makedirs(BASE_PATH)

if not os.path.exists(BASE_PATH + 'models/'):
   os.makedirs(BASE_PATH + 'models/')
    
if not Path(BASE_PATH + "dataset_21cmGEM.h5").exists():
    print("Downloading data set for 21cmGEM (Cohen et al. 2020) used in Dorigo Jones et al. 2024 and also Bye et al. 2022 and Bevins et al. 2021.")
    r1 = requests.get("https://zenodo.org/record/5084114/files/dataset_21cmVAE.h5?download=1")
    with open(BASE_PATH + "dataset_21cmGEM.h5", "wb") as f1:
        f1.write(r1.content)

if not Path(BASE_PATH + "dataset_ARES.h5").exists():
    print("Downloading data set for ARES used and generated in Dorigo Jones et al. 2024.")
    r2 = requests.get("https://zenodo.org/record/13840725/files/dataset_ARES.h5?download=1")
    with open(BASE_PATH + "dataset_ARES.h5", "wb") as f2:
        f2.write(r2.content)

if not Path(BASE_PATH + "models/emulator_21cmGEM.h5").exists():
    print("Transferring instance of 21cmLSTM trained on 21cmGEM to auxiliary folder.")
    r3 = open(PACKAGE_PATH + "models/emulator_21cmGEM.h5", 'w')
    with open(BASE_PATH + "models/emulator_21cmGEM.h5", "wb") as f3:
        f3.write(r3.content)

if not Path(BASE_PATH + "models/emulator_ARES.h5").exists():
    print("Transferring instance of 21cmLSTM trained on ARES to auxiliary folder.")
    r4 = open(PACKAGE_PATH + "models/emulator_ARES.h5", 'w')
    with open(BASE_PATH + "models/emulator_ARES.h5", "wb") as f4:
        f4.write(r4.content)

if not Path(BASE_PATH + "models/train_maxs_21cmGEM.txt").exists():
    print("Transferring preprocessing files (21cmGEM and ARES training set mins and maxs) to auxiliary folder.")
    r5 = open(PACKAGE_PATH + "models/train_maxs_21cmGEM.txt", 'w')
    with open(BASE_PATH + "models/train_maxs_21cmGEM.txt", "wb") as f5:
        f5.write(r5.content)

if not Path(BASE_PATH + "models/train_mins_21cmGEM.txt").exists():
    r6 = open(PACKAGE_PATH + "models/train_mins_21cmGEM.txt", 'w')
    with open(BASE_PATH + "models/train_mins_21cmGEM.txt", "wb") as f6:
        f6.write(r6.content)

if not Path(BASE_PATH + "models/train_maxs_ARES.txt").exists():
    r7 = open(PACKAGE_PATH + "models/train_maxs_ARES.txt", 'w')
    with open(BASE_PATH + "models/train_maxs_ARES.txt", "wb") as f7:
        f7.write(r7.content)

if not Path(BASE_PATH + "models/train_mins_ARES.txt").exists():
    r8 = open(PACKAGE_PATH + "models/train_mins_ARES.txt", 'w')
    with open(BASE_PATH + "models/train_mins_ARES.txt", "wb") as f8:
        f8.write(r8.content)

from Global21cmLSTM import emulator_21cmGEM
from Global21cmLSTM import emulator_ARES
from Global21cmLSTM import preprocess_21cmGEM
from Global21cmLSTM import preprocess_ARES

