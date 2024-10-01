#!/usr/bin/env python
# coding: utf-8

# In[ ]:


__version__ = "1.0"
__author__ = "Johnny Dorigo Jones"

from pathlib import Path
import os
import requests
# from Global21cmLSTM import __path__
# PATH = __path__[0] + "/"

BASE_PATH = f"{os.environ.get('HOME')}/.Global21cmLSTM/"
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

#os.getcwd()

from Global21cmLSTM import emulator_21cmGEM
from Global21cmLSTM import emulator_ARES
from Global21cmLSTM import preprocess_21cmGEM
from Global21cmLSTM import preprocess_ARES

