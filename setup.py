#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import requests
import shutil
from pathlib import Path
from setuptools import setup, find_packages

setup(
    name='21cmLSTM',
    version='1.0.0',
    description='21cmLSTM: A Fast Memory-based Emulator of the Global 21 cm Signal with Unprecedented Accuracy',
    long_description=open('README.md'),
    author='Johnny Dorigo Jones',
    author_email='johnny.dorigojones@colorado.edu',
    url='https://github.com/jdorigojones/21cmLSTM',
    packages=find_packages(),
    install_requires=open('requirements.txt').read().splitlines(),
    license='MIT',
    classifiers=[
               'Intended Audience :: Science/Research',
               'Natural Language :: English',
               'License :: OSI Approved :: MIT License',
               'Programming Language :: Python :: 3',
               'Topic :: Scientific/Engineering',
               'Topic :: Scientific/Engineering :: Astronomy',
               'Topic :: Scientific/Engineering :: Physics',
    ],
)

# the below code creates a hidden folder in $HOME to store needed auxiliary H5 and txt files (i.e., data sets, trained models, training set min and max values)
# and copies/downloads those files to the folder
# the location of this folder can be changed here if the user's $HOME has storage limitations, but note that if the location of this folder is changed here
# then the specified PATH must be updated in the emulator.py, preprocess.py, and eval.py scripts for the respective model/data
BASE_PATH = f"{os.environ.get('HOME')}/.Global21cmLSTM/"
MODELS_INSTALL_PATH = os.path.dirname(os.path.realpath(__file__))+'/Global21cmLSTM/models/'
emulator_files_list = ['emulator_21cmGEM.h5', 'emulator_ARES.h5', 'train_maxs_21cmGEM.txt',\
                       'train_mins_21cmGEM.txt', 'train_maxs_ARES.txt', 'train_mins_ARES.txt']

if not os.path.exists(BASE_PATH):
   os.makedirs(BASE_PATH)

if not os.path.exists(BASE_PATH+'models/'):
   os.makedirs(BASE_PATH+'models/')

if not Path(BASE_PATH + 'dataset_21cmGEM.h5').exists():
    print('Downloading data set for 21cmGEM (Cohen et al. 2020) used in Dorigo Jones et al. 2024 and also Bye et al. 2022 and Bevins et al. 2021.')
    r1 = requests.get('https://zenodo.org/record/5084114/files/dataset_21cmVAE.h5?download=1')
    with open(BASE_PATH + 'dataset_21cmGEM.h5', 'wb') as f1:
        f1.write(r1.content)

if not Path(BASE_PATH + 'dataset_ARES.h5').exists():
    print('Downloading data set for ARES used and generated in Dorigo Jones et al. 2024.')
    r2 = requests.get('https://zenodo.org/record/13840725/files/dataset_ARES.h5?download=1')
    with open(BASE_PATH + 'dataset_ARES.h5', 'wb') as f2:
        f2.write(r2.content)

for item in emulator_files_list:
    shutil.copyfile(MODELS_INSTALL_PATH+item, BASE_PATH+'models/'+item)

